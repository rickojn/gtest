#include "../include/CustardFlow.h"
// matmul_benchmark.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



int main(int argc, char **argv) {
    int M = 1024;  // Matrix size
    int N = 1024;  // Matrix size
    int K = 1024;  // Matrix size
    int warmup = 5;
    int iterations = 10;
    
    // Allocate matrices
    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];
    
    // Initialize with random values
    srand(42);
    for (int i = 0; i < M * K; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (float)rand() / RAND_MAX;
    }
    
    // Define test cases
    struct {
        const char *name;
        void (*func)(const float*, const float*, float*, size_t, size_t, size_t);
    } tests[] = {
        {"naive",   naive_matmul},
        {"ikj",     ikj_matmul}
    };
    
    // Run each implementation
    for (int t = 0; t < 2; t++) {
        printf("\n=== %s ===\n", tests[t].name);
        
        // Warmup
        for (int i = 0; i < warmup; i++)
            tests[t].func(A, B, C, M, N, K);
        
        // Timed runs (for sanity check)
        clock_t start = clock();
        for (int i = 0; i < iterations; i++)
            tests[t].func(A, B, C, M, N, K);
        clock_t end = clock();
        
        double time = (double)(end - start) / CLOCKS_PER_SEC / iterations;
        printf("Time: %.4f ms\n", time * 1000);
        
        // Optional: verify results match
        // verify(C, reference);
    }
    
    delete[] A; delete[] B; delete[] C;
    return 0;
}