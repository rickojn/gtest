/**
 * matmul_bench.cpp
 *
 * Benchmark harness for progressively optimized matrix multiplication.
 * Add new implementations in the "IMPLEMENTATIONS" section and register
 * them in main() — the harness handles timing, verification, and reporting.
 *
 * Build:
 *   g++ -O2 -march=native -std=c++17 -o matmul_bench matmul_bench.cpp
 *
 * (Use -O2, not -O3, so the compiler doesn't auto-vectorize your hand-rolled
 *  versions in ways that mask the real difference between implementations.)
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Timing helpers
// ─────────────────────────────────────────────────────────────────────────────

using Clock = std::chrono::high_resolution_clock;

/** Returns elapsed seconds for one call of fn(). */
template <typename Fn>
double time_once(Fn&& fn) {
    auto t0 = Clock::now();
    fn();
    auto t1 = Clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

/** Runs fn() `reps` times, returns the *median* elapsed seconds. */
template <typename Fn>
double time_median(Fn&& fn, int reps = 5) {
    std::vector<double> samples(reps);
    for (int i = 0; i < reps; ++i) samples[i] = time_once(fn);
    std::sort(samples.begin(), samples.end());
    return samples[reps / 2];
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Allocate a zero-initialised, 64-byte-aligned matrix. */
float* alloc_matrix(size_t rows, size_t cols) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, rows * cols * sizeof(float)) != 0) {
        std::cerr << "posix_memalign failed\n";
        std::exit(1);
    }
    std::memset(ptr, 0, rows * cols * sizeof(float));
    return static_cast<float*>(ptr);
}

/** Fill with random values in (-1, 1). */
void fill_random(float* M, size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) M[i] = dist(rng);
}

/**
 * Check that two result matrices agree to within a relative tolerance.
 * Returns true if they match (or if ref is nullptr — skip check).
 */
bool matrices_close(const float* ref, const float* got,
                    size_t M, size_t N, float rtol = 1e-4f) {
    if (!ref) return true;
    for (size_t i = 0; i < M * N; ++i) {
        float a = ref[i], b = got[i];
        float denom = std::max(1e-5f, std::max(std::abs(a), std::abs(b)));
        if (std::abs(a - b) / denom > rtol) {
            std::cerr << "  MISMATCH at index " << i
                      << ": ref=" << a << " got=" << b << "\n";
            return false;
        }
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// IMPLEMENTATIONS — add yours below
// ─────────────────────────────────────────────────────────────────────────────

// Signature all implementations must match.
using MatmulFn = std::function<void(const float*, const float*, float*,
                                    size_t, size_t, size_t)>;

// --- 1. Naive (ijk) ----------------------------------------------------------
void naive_matmul(const float* A, const float* B, float* C,
                  size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < K; k++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

// --- 2. Loop-order swap: ikj (better B-access locality) ----------------------
void ikj_matmul(const float* A, const float* B, float* C,
                size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++) {
            float a = A[i * K + k];
            for (size_t j = 0; j < N; j++)
                C[i * N + j] += a * B[k * N + j];
        }
}

// --- 3. Register-tile (small unrolled inner loop) ----------------------------
//
// Accumulates a 4-wide strip of C[i][j..j+3] in registers so the compiler can
// keep them in SIMD lanes without repeated store-load round-trips.
void tiled_reg_matmul(const float* A, const float* B, float* C,
                      size_t M, size_t N, size_t K) {
    constexpr size_t TILE = 4;

    for (size_t i = 0; i < M; i++) {
        size_t j = 0;
        for (; j + TILE <= N; j += TILE) {
            float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            for (size_t k = 0; k < K; k++) {
                float a = A[i * K + k];
                c0 += a * B[k * N + j + 0];
                c1 += a * B[k * N + j + 1];
                c2 += a * B[k * N + j + 2];
                c3 += a * B[k * N + j + 3];
            }
            C[i * N + j + 0] += c0;
            C[i * N + j + 1] += c1;
            C[i * N + j + 2] += c2;
            C[i * N + j + 3] += c3;
        }
        // Remainder columns
        for (; j < N; j++) {
            float c = 0;
            for (size_t k = 0; k < K; k++) c += A[i * K + k] * B[k * N + j];
            C[i * N + j] += c;
        }
    }
}

// --- 4. Cache-blocking (L2-friendly tiles) -----------------------------------
//
// Choose tile sizes so that the three sub-tiles fit in L2 (~256 KB).
// A_tile : BROW x BKAY  floats
// B_tile : BKAY x BCOL  floats
// C_tile : BROW x BCOL  floats  (hot, stays in L1)
void blocked_matmul(const float* A, const float* B, float* C,
                    size_t M, size_t N, size_t K) {
    constexpr size_t BROW = 64;   // i-tile
    constexpr size_t BCOL = 64;   // j-tile
    constexpr size_t BKAY = 64;   // k-tile

    for (size_t i0 = 0; i0 < M; i0 += BROW)
    for (size_t k0 = 0; k0 < K; k0 += BKAY)
    for (size_t j0 = 0; j0 < N; j0 += BCOL) {
        size_t iEnd = std::min(i0 + BROW, M);
        size_t kEnd = std::min(k0 + BKAY, K);
        size_t jEnd = std::min(j0 + BCOL, N);

        for (size_t i = i0; i < iEnd; i++)
        for (size_t k = k0; k < kEnd; k++) {
            float a = A[i * K + k];
            for (size_t j = j0; j < jEnd; j++)
                C[i * N + j] += a * B[k * N + j];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reporting
// ─────────────────────────────────────────────────────────────────────────────

/** GFLOP/s for a general matmul: 2*M*N*K FLOPs. */
double gflops(size_t M, size_t N, size_t K, double seconds) {
    return (2.0 * M * N * K) / seconds / 1e9;
}

struct BenchResult {
    std::string name;
    double seconds;
    double gflops;
    bool correct;
};

void print_header() {
    std::cout << "\n"
              << std::left  << std::setw(28) << "Implementation"
              << std::right << std::setw(12) << "Time (s)"
              << std::setw(12) << "GFLOP/s"
              << std::setw(10) << "Speedup"
              << std::setw(10) << "Correct"
              << "\n"
              << std::string(72, '-') << "\n";
}

void print_results(const std::vector<BenchResult>& results) {
    double baseline = results.empty() ? 1.0 : results[0].seconds;
    for (const auto& r : results) {
        std::cout << std::left  << std::setw(28) << r.name
                  << std::right << std::fixed << std::setprecision(4)
                  << std::setw(12) << r.seconds
                  << std::setw(12) << r.gflops
                  << std::setw(9)  << std::setprecision(2) << (baseline / r.seconds) << "x"
                  << std::setw(10) << (r.correct ? "YES" : "FAIL")
                  << "\n";
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // Matrix dimensions — square by default; override via command line.
    size_t M = 512, N = 512, K = 512;
    int    reps = 5;

    if (argc >= 4) {
        M = std::stoul(argv[1]);
        N = std::stoul(argv[2]);
        K = std::stoul(argv[3]);
    }
    if (argc >= 5) reps = std::stoi(argv[4]);

    std::cout << "=== MatMul Benchmark ===\n"
              << "  A: " << M << " x " << K << "\n"
              << "  B: " << K << " x " << N << "\n"
              << "  C: " << M << " x " << N << "\n"
              << "  Reps (median): " << reps << "\n"
              << "  Peak FLOPs per call: "
              << std::fixed << std::setprecision(3)
              << (2.0 * M * N * K / 1e9) << " GFLOP\n";

    // Allocate and initialise inputs
    float* A = alloc_matrix(M, K);  fill_random(A, M * K, 1);
    float* B = alloc_matrix(K, N);  fill_random(B, K * N, 2);
    float* C_ref = alloc_matrix(M, N);
    float* C_got = alloc_matrix(M, N);

    // ── Register implementations ──────────────────────────────────────────
    struct Entry { std::string name; MatmulFn fn; };
    std::vector<Entry> impls = {
        { "naive (ijk)",          naive_matmul     },
        { "loop-order ikj",       ikj_matmul       },
        { "register-tile (w=4)",  tiled_reg_matmul },
        { "cache-blocked 64³",    blocked_matmul   },
        // Add your next implementation here:
        // { "my_fast_matmul", my_fast_matmul },
    };

    // ── Build reference with naive ────────────────────────────────────────
    std::memset(C_ref, 0, M * N * sizeof(float));
    naive_matmul(A, B, C_ref, M, N, K);

    // ── Run benchmarks ────────────────────────────────────────────────────
    print_header();
    std::vector<BenchResult> results;

    for (const auto& e : impls) {
        std::cout << "  Benchmarking: " << e.name << " ... " << std::flush;

        double t = time_median([&] {
            std::memset(C_got, 0, M * N * sizeof(float));
            e.fn(A, B, C_got, M, N, K);
        }, reps);

        // Verify against naive reference (skip for naive itself)
        bool ok = (e.name == impls[0].name)
                  || matrices_close(C_ref, C_got, M, N);

        std::cout << "\r";
        results.push_back({ e.name, t, gflops(M, N, K, t), ok });
    }

    print_results(results);

    // Summary
    if (results.size() > 1) {
        double best_gf = 0;
        std::string best_name;
        for (const auto& r : results) {
            if (r.correct && r.gflops > best_gf) {
                best_gf   = r.gflops;
                best_name = r.name;
            }
        }
        std::cout << "Best: \"" << best_name << "\"  →  "
                  << std::fixed << std::setprecision(2)
                  << best_gf << " GFLOP/s  ("
                  << std::setprecision(2)
                  << (best_gf / results[0].gflops) << "x over naive)\n\n";
    }

    free(A); free(B); free(C_ref); free(C_got);
    return 0;
}
