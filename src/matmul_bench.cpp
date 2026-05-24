/**
 * matmul_bench.cpp
 *
 * Benchmark harness for progressively optimized matrix multiplication.
 * Measures wall-clock time AND hardware performance counters via
 * perf_event_open(2): cycles, instructions (→ IPC), L1D misses, LLC misses.
 *
 * Build:
 *   g++ -O2 -march=native -std=c++17 -o matmul_bench matmul_bench.cpp
 *
 * Run:
 *   ./matmul_bench [M N K [reps]]      # default: 1024 1024 1024 5
 *
 * Hardware counters require a real Linux machine with PMU access.
 * Inside Docker / VMs without an exposed PMU the harness auto-detects this
 * and falls back to timing-only mode — no source changes needed.
 *
 * Correctness check:
 *   Uses mixed absolute+relative tolerance:
 *     |a - b| ≤ atol + rtol * max(|a|,|b|)
 *   where atol = 8 * sqrt(K) * eps_f32 * ||C||_inf
 *   This accounts for catastrophic cancellation on near-zero elements that
 *   is an inherent property of float32 summation order differences, not a bug.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Linux perf headers
#include <cerrno>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "../include/CustardFlow.h"

// ─────────────────────────────────────────────────────────────────────────────
// perf_event_open wrapper + counter group
// ─────────────────────────────────────────────────────────────────────────────

static long perf_event_open(perf_event_attr* attr, pid_t pid,
                             int cpu, int group_fd, unsigned long flags) {
    return syscall(SYS_perf_event_open, attr, pid, cpu, group_fd, flags);
}

struct PerfCounter { const char* label; uint32_t type; uint64_t config; };

// Counters we want — group leader is index 0.
// HW_CACHE encoding: cache_id | (op << 8) | (result << 16)
static const PerfCounter COUNTERS[] = {
    { "cycles",
      PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES },
    { "insns",
      PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS },
    { "L1D-miss",
      PERF_TYPE_HW_CACHE,
      uint64_t(PERF_COUNT_HW_CACHE_L1D)
    | uint64_t(PERF_COUNT_HW_CACHE_OP_READ)     << 8
    | uint64_t(PERF_COUNT_HW_CACHE_RESULT_MISS)  << 16 },
    { "LLC-acc",
      PERF_TYPE_HW_CACHE,
      uint64_t(PERF_COUNT_HW_CACHE_LL)
    | uint64_t(PERF_COUNT_HW_CACHE_OP_READ)       << 8
    | uint64_t(PERF_COUNT_HW_CACHE_RESULT_ACCESS)  << 16 },
    { "LLC-miss",
      PERF_TYPE_HW_CACHE,
      uint64_t(PERF_COUNT_HW_CACHE_LL)
    | uint64_t(PERF_COUNT_HW_CACHE_OP_READ)      << 8
    | uint64_t(PERF_COUNT_HW_CACHE_RESULT_MISS)   << 16 },
};
static constexpr int NC = sizeof(COUNTERS) / sizeof(COUNTERS[0]);

/**
 * RAII counter group.  All fds stay -1 when the PMU is unavailable;
 * read_counts() returns zeros — callers never need to branch.
 */
struct PerfGroup {
    int  fds[NC];
    bool available = false;

    PerfGroup() {
        for (int i = 0; i < NC; ++i) fds[i] = -1;

        // Open group leader
        perf_event_attr attr{};
        attr.size           = sizeof(attr);
        attr.type           = COUNTERS[0].type;
        attr.config         = COUNTERS[0].config;
        attr.disabled       = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv     = 1;
        attr.read_format    = PERF_FORMAT_GROUP;  // atomic snapshot of whole group

        fds[0] = perf_event_open(&attr, 0, -1, -1, 0);
        if (fds[0] < 0) return;

        // Open remaining counters as members of the leader's group
        for (int i = 1; i < NC; ++i) {
            attr.type     = COUNTERS[i].type;
            attr.config   = COUNTERS[i].config;
            attr.disabled = 0;  // members start/stop with the leader
            fds[i] = perf_event_open(&attr, 0, -1, fds[0], 0);
            if (fds[i] < 0) {
                for (int j = 0; j < i; ++j) { close(fds[j]); fds[j] = -1; }
                return;
            }
        }
        available = true;
    }

    ~PerfGroup() {
        for (int i = 0; i < NC; ++i) if (fds[i] >= 0) close(fds[i]);
    }

    void reset_and_start() {
        if (!available) return;
        ioctl(fds[0], PERF_EVENT_IOC_RESET,  PERF_IOC_FLAG_GROUP);
        ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }

    void stop() {
        if (!available) return;
        ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    }

    /**
     * Atomically read all counters.
     * Kernel layout for PERF_FORMAT_GROUP:  uint64_t nr;  uint64_t values[nr];
     */
    void read_counts(uint64_t out[NC]) {
        memset(out, 0, NC * sizeof(uint64_t));
        if (!available) return;
        uint64_t buf[1 + NC];
        if (::read(fds[0], buf, sizeof(buf)) < (ssize_t)sizeof(buf)) return;
        uint64_t nr = buf[0];
        for (uint64_t i = 0; i < nr && i < (uint64_t)NC; ++i)
            out[i] = buf[1 + i];
    }

    PerfGroup(const PerfGroup&)            = delete;
    PerfGroup& operator=(const PerfGroup&) = delete;
};

// ─────────────────────────────────────────────────────────────────────────────
// Timing
// ─────────────────────────────────────────────────────────────────────────────

using Clock = std::chrono::high_resolution_clock;

template <typename Fn>
double time_once(Fn&& fn) {
    auto t0 = Clock::now();
    fn();
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix helpers
// ─────────────────────────────────────────────────────────────────────────────

float* alloc_matrix(size_t rows, size_t cols) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, rows * cols * sizeof(float)) != 0) {
        std::cerr << "posix_memalign failed\n"; std::exit(1);
    }
    memset(ptr, 0, rows * cols * sizeof(float));
    return static_cast<float*>(ptr);
}

void fill_random(float* M, size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (size_t i = 0; i < n; ++i) M[i] = dist(rng);
}

/**
 * Mixed absolute+relative tolerance check.
 *
 * float32 dot products of length K accumulate rounding error of order
 * sqrt(K) * eps_f32 * ||C||_inf.  Near-zero elements can have large *relative*
 * error due to catastrophic cancellation — that is expected and correct.
 *
 * Tolerance:  |a - b| <= atol + rtol * max(|a|, |b|)
 * where  atol = safety * sqrt(K) * eps_f32 * ||C||_inf
 */
bool matrices_close(const float* ref, const float* got,
                    size_t M, size_t N, size_t K) {
    // Find ||C||_inf
    float cmax = 0.f;
    for (size_t i = 0; i < M * N; ++i) cmax = std::max(cmax, std::abs(ref[i]));

    // Noise floor: generous 8× safety factor over theoretical rounding
    const float eps_f32 = 1.2e-7f;
    float atol = 8.f * std::sqrt(float(K)) * eps_f32 * cmax;
    float rtol = 1e-5f;

    int fails = 0;
    for (size_t i = 0; i < M * N; ++i) {
        float diff   = std::abs(ref[i] - got[i]);
        float thresh = atol + rtol * std::max(std::abs(ref[i]), std::abs(got[i]));
        if (diff > thresh) {
            if (++fails <= 3)
                std::cerr << "  MISMATCH @ " << i
                          << ": ref=" << ref[i] << " got=" << got[i]
                          << " diff=" << diff << " thresh=" << thresh << "\n";
        }
    }
    return fails == 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// IMPLEMENTATIONS  — add new variants at the bottom of this section
// ─────────────────────────────────────────────────────────────────────────────

using MatmulFn = std::function<void(const float*, const float*, float*,
                                    size_t, size_t, size_t)>;

// 1. Naive ijk ----------------------------------------------------------------
// void naive_matmul(const float* A, const float* B, float* C,
//                   size_t M, size_t N, size_t K) {
//     for (size_t i = 0; i < M; i++)
//         for (size_t j = 0; j < N; j++)
//             for (size_t k = 0; k < K; k++)
//                 C[i*N+j] += A[i*K+k] * B[k*N+j];
// }

// // 2. Loop-order ikj (A[i][k] hoisted; B and C both stride sequentially) ------
// void ikj_matmul(const float* A, const float* B, float* C,
//                 size_t M, size_t N, size_t K) {
//     for (size_t i = 0; i < M; i++)
//         for (size_t k = 0; k < K; k++) {
//             float a = A[i*K+k];
//             for (size_t j = 0; j < N; j++)
//                 C[i*N+j] += a * B[k*N+j];
//         }
// }

// // 3. Register-tile width=4 (4 C accumulators in registers per inner loop) ----
// void tiled_reg_matmul(const float* A, const float* B, float* C,
//                       size_t M, size_t N, size_t K) {
//     constexpr size_t W = 4;
//     for (size_t i = 0; i < M; i++) {
//         size_t j = 0;
//         for (; j + W <= N; j += W) {
//             float c0=0, c1=0, c2=0, c3=0;
//             for (size_t k = 0; k < K; k++) {
//                 float a = A[i*K+k];
//                 c0 += a * B[k*N+j+0];
//                 c1 += a * B[k*N+j+1];
//                 c2 += a * B[k*N+j+2];
//                 c3 += a * B[k*N+j+3];
//             }
//             C[i*N+j+0] += c0; C[i*N+j+1] += c1;
//             C[i*N+j+2] += c2; C[i*N+j+3] += c3;
//         }
//         for (; j < N; j++) {
//             float c = 0;
//             for (size_t k = 0; k < K; k++) c += A[i*K+k] * B[k*N+j];
//             C[i*N+j] += c;
//         }
//     }
// }

// // 4. Cache-blocked 64³ (tiles sized to stay in L2) ----------------------------
// void blocked_matmul(const float* A, const float* B, float* C,
//                     size_t M, size_t N, size_t K) {
//     constexpr size_t BR = 64, BC = 64, BK = 64;
//     for (size_t i0 = 0; i0 < M; i0 += BR)
//     for (size_t k0 = 0; k0 < K; k0 += BK)
//     for (size_t j0 = 0; j0 < N; j0 += BC) {
//         size_t iE = std::min(i0+BR, M);
//         size_t kE = std::min(k0+BK, K);
//         size_t jE = std::min(j0+BC, N);
//         for (size_t i = i0; i < iE; i++)
//         for (size_t k = k0; k < kE; k++) {
//             float a = A[i*K+k];
//             for (size_t j = j0; j < jE; j++)
//                 C[i*N+j] += a * B[k*N+j];
//         }
//     }
// }

// ── Add your next variant here ────────────────────────────────────────────────
// void my_matmul(const float* A, const float* B, float* C,
//                size_t M, size_t N, size_t K) { ... }

// ─────────────────────────────────────────────────────────────────────────────
// Results and reporting
// ─────────────────────────────────────────────────────────────────────────────

struct BenchResult {
    std::string name;
    double      seconds;
    double      gflops;
    bool        correct;
    uint64_t    ctr[NC];    // raw hardware counter values (averaged over reps)
};

static double calc_gflops(size_t M, size_t N, size_t K, double s) {
    return (2.0 * M * N * K) / s / 1e9;
}

static void hline(int w) { std::cout << std::string(w, '-') << "\n"; }

void print_results(const std::vector<BenchResult>& results, bool has_pmu) {
    using std::cout; using std::left; using std::right;
    using std::setw; using std::fixed; using std::setprecision;

    double t0 = results.empty() ? 1.0 : results[0].seconds;
    const int W = 76;

    // ── Timing table (always shown) ──────────────────────────────────────────
    cout << "\n";
    hline(W);
    cout << left  << setw(26) << "Implementation"
         << right << setw(10) << "Time (s)"
                  << setw(10) << "GFLOP/s"
                  << setw(10) << "Speedup"
                  << setw(10) << "Correct"
         << "\n";
    hline(W);
    for (const auto& r : results) {
        cout << left  << setw(26) << r.name
             << right << fixed << setprecision(4) << setw(10) << r.seconds
                               << setprecision(2) << setw(10) << r.gflops
             << setw(9) << (t0 / r.seconds) << "x"
             << setw(10) << (r.correct ? "YES" : "FAIL")
             << "\n";
    }
    hline(W);

    if (!has_pmu) {
        cout << "\n  ⚠  PMU unavailable (running inside a container/VM).\n"
             << "     On bare-metal Linux this table also shows:\n"
             << "     cycles · IPC · L1D-misses · LLC-misses · LLC miss-rate\n\n";
        return;
    }

    // ── PMU table ────────────────────────────────────────────────────────────
    // ctr[]: 0=cycles 1=insns 2=L1D-miss 3=LLC-acc 4=LLC-miss
    cout << "\n";
    hline(W);
    cout << left  << setw(26) << "Implementation"
         << right << setw(12) << "Cycles"
                  << setw(7)  << "IPC"
                  << setw(12) << "L1D-miss"
                  << setw(12) << "LLC-miss"
                  << setw(9)  << "LLC-MR%"
         << "\n";
    hline(W);
    for (const auto& r : results) {
        uint64_t cyc  = r.ctr[0];
        uint64_t ins  = r.ctr[1];
        uint64_t l1dm = r.ctr[2];
        uint64_t llca = r.ctr[3];
        uint64_t llcm = r.ctr[4];

        double ipc    = cyc  ? double(ins)  / cyc  : 0.0;
        double llc_mr = llca ? 100.0 * llcm / llca : 0.0;

        cout << left  << setw(26) << r.name
             << right << fixed
             << setprecision(0) << setw(12) << cyc
             << setprecision(2) << setw(7)  << ipc
             << setprecision(0) << setw(12) << l1dm
                                << setw(12) << llcm
             << setprecision(1) << setw(8)  << llc_mr << "%"
             << "\n";
    }
    hline(W);
    cout << "\n"
         << "  IPC      — instructions retired per cycle (higher = better pipeline use)\n"
         << "  L1D-miss — L1 data-cache read misses      (lower  = better data locality)\n"
         << "  LLC-miss — last-level cache read misses   (lower  = working-set fits in L2/L3)\n"
         << "  LLC-MR%  — LLC miss rate = LLC-miss / LLC-access\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    size_t M = 1024, N = 1024, K = 1024;
    int    reps = 5;
    if (argc >= 4) { M = std::stoul(argv[1]); N = std::stoul(argv[2]); K = std::stoul(argv[3]); }
    if (argc >= 5) reps = std::stoi(argv[4]);

    // Probe PMU availability once at startup
    PerfGroup probe;
    bool has_pmu = probe.available;

    std::cout << "=== MatMul Benchmark ===\n"
              << "  A: " << M << "×" << K
              << "  B: " << K << "×" << N
              << "  C: " << M << "×" << N << "\n"
              << "  Reps (median timing): " << reps << "\n"
              << "  FLOPs/call: " << std::fixed << std::setprecision(3)
              << (2.0*M*N*K/1e9) << " GFLOP\n"
              << "  PMU counters: "
              << (has_pmu ? "available ✓" : "unavailable (container/VM)") << "\n";

    // Allocate inputs
    float* A     = alloc_matrix(M, K); fill_random(A, M*K, 1);
    float* B     = alloc_matrix(K, N); fill_random(B, K*N, 2);
    float* C_ref = alloc_matrix(M, N);
    float* C_got = alloc_matrix(M, N);

    // Register implementations
    struct Entry { std::string name; MatmulFn fn; };
    std::vector<Entry> impls = {
        { "naive (ijk)",          naive_matmul     },
        // { "my_avx_matmul",     my_avx_matmul    },
    };

    // Build reference (naive, single call on zero-initialised C_ref)
    naive_matmul(A, B, C_ref, M, N, K);

    // ── Benchmark loop ────────────────────────────────────────────────────────
    std::vector<BenchResult> results;

    for (const auto& e : impls) {
        std::cout << "  Running: " << e.name << " ...\r" << std::flush;

        std::vector<double> times(reps);
        uint64_t sum_ctr[NC] = {};

        for (int r = 0; r < reps; ++r) {
            memset(C_got, 0, M*N*sizeof(float));

            // Fresh counter group each rep so counts don't accumulate
            PerfGroup pg;
            pg.reset_and_start();
            auto t0 = Clock::now();

            e.fn(A, B, C_got, M, N, K);

            times[r] = std::chrono::duration<double>(Clock::now() - t0).count();
            pg.stop();

            uint64_t c[NC];
            pg.read_counts(c);
            for (int i = 0; i < NC; ++i) sum_ctr[i] += c[i];
        }

        // Median wall-clock time
        std::sort(times.begin(), times.end());
        double med = times[reps / 2];

        // Average counters across reps
        uint64_t avg_ctr[NC];
        for (int i = 0; i < NC; ++i) avg_ctr[i] = sum_ctr[i] / reps;

        // Correctness: compare last rep's C_got to reference
        bool ok = (e.name == impls[0].name)
               || matrices_close(C_ref, C_got, M, N, K);

        std::cout << "                              \r";  // clear progress line

        BenchResult res;
        res.name    = e.name;
        res.seconds = med;
        res.gflops  = calc_gflops(M, N, K, med);
        res.correct = ok;
        memcpy(res.ctr, avg_ctr, sizeof(avg_ctr));
        results.push_back(res);
    }

    print_results(results, has_pmu);

    // Best summary
    double best_gf = 0; std::string best_name;
    for (const auto& r : results)
        if (r.correct && r.gflops > best_gf) { best_gf = r.gflops; best_name = r.name; }

    std::cout << "Best: \"" << best_name << "\"  →  "
              << std::fixed << std::setprecision(2) << best_gf << " GFLOP/s  ("
              << best_gf / results[0].gflops << "× over naive)\n\n";

    free(A); free(B); free(C_ref); free(C_got);
}
