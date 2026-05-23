#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include  "../include/CustardFlow.h"

#define M 10000
#define N 288
#define K 784
#define TILE 128
#define INNER_TILE 32
#define NAIVE 1
#define OUTER 0
#define L1 0
#define TILED 0
#define SIMD 1
#define ONLY_LARGE 1
#define  SIMD_M 8
#define SIMD_N 8



void initialise_large_matrices(float *A_large, float * B_large, float * C_large){
    srand(42);
    for (size_t i = 0; i < M * K; i++){
        A_large[i] = rand() % 100;
        // A_large[i] = 1.0;
    }
    for (size_t i = 0; i < K * N; i++){
        B_large[i] = rand() % 100;
        // B_large[i] = 1.0;
    }
    for (size_t i = 0; i < M * N; i++){
        C_large[i] = 0.0f;
    }

}

void check_result(const float * ref_result, const float * result, size_t m, size_t n){

    for (size_t idx = 0; idx < m * n; idx++){
        if (ref_result[idx] != result[idx]){
            printf("result does not aggree with naive matmul\n");
            printf("ref_result[%zu] = %f  result[%zu] = %f\n", idx, ref_result[idx], idx, result[idx]);
            return;
        }
    }
    printf("result agrees with naive matmul\n");
}


void col_to_row_major(const float *src_matrix, float *dest_matrix, size_t num_rows, size_t num_cols){
    for (size_t r = 0; r < num_rows; r++){
        for (size_t c = 0; c < num_cols; c++){            
            int offset_dest = r * num_cols + c; //[i][j] 
            int offset_source = c * num_rows + r; //[j][i] 
            dest_matrix[offset_dest] = src_matrix[offset_source];
            int db_copied_value = dest_matrix[offset_dest];
            int db = 0;
        }
    }
}


void row_to_col_major(const float *src_matrix, float *dest_matrix, size_t num_rows, size_t num_cols){
    for (size_t r = 0; r < num_rows; r++){
        for (size_t c = 0; c < num_cols; c++){            
            int offset_dest = c * num_rows + r; //[j][i] 
            int offset_source = r * num_cols + c; //[i][j] 
            dest_matrix[offset_dest] = src_matrix[offset_source];
            int db_copied_value = dest_matrix[offset_dest];
            int db = 0;
        }
    }
}

void print_row_major_matrix(const float *A, size_t rows, size_t cols){
    for (size_t i = 0; i < rows; i++){
        printf("\n");
        for (size_t j = 0; j < cols; j++){
            printf("%f\t", A[i * cols + j]);
        }
        printf("\n");
    }
}

void print_column_major_matrix(const float *A, size_t rows, size_t cols){
    for (size_t i = 0; i < rows; i++){
        printf("\n");
        for (size_t j = 0; j < cols; j++){
            printf("%f\t", A[j * rows + i]);
        }
        printf("\n");
    }
}

int mainx(){
    float matrix [6] = {1,4,2,5,3,6}; 
    float transposed_matrix [6] = {};
    
    transpose_matrix(matrix, transposed_matrix, 3, 2);

    print_column_major_matrix(&matrix[0],2,3);
    print_row_major_matrix(&transposed_matrix[0],2,3);
}

int main() {
    /*
    1111  1234
    2222  1234
    3333  1234
    4444  1234

    4,8,12,16
    8,16,24,32
    12,24,36,48
    16,32,48,64
    */


    if (!ONLY_LARGE && NAIVE){
        printf("naive matmul ... \n");
        float A[] = {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4}; // row major
        float B[] = {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4}; // column major
        float C[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        naive_matmul(A, B, C, 4, 4, 4);
    
        for (size_t i = 0; i < 4; i++)
        {
            printf("\n");
            for (size_t j = 0; j < 4; j++){
                printf("%f\t", C[i * 4 + j]);
            }
            printf("\n");
        }
    
        printf("\n");
    }

    if (!ONLY_LARGE && OUTER){
        printf("outer product matmul ... \n");
        float AO[] = {1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4}; // column major
        float BO[] = {1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4}; // row major
        float CO[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        outer_product_matmul(AO, BO, CO, 4, 4, 4);
    
        for (size_t i = 0; i < 4; i++)
        {
            printf("\n");
            for (size_t j = 0; j < 4; j++){
                printf("%f\t", CO[i * 4 + j]);
            }
            printf("\n");
        }
    
        printf("\n");
    }
    



    if (!ONLY_LARGE && TILED){
        printf("\n");
        printf("tiled matmul:\n");
    
        float ATP[] = {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4}; // row major
        float BTP[] = {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4}; // column major
        float CTP[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; // row major
        
        tiled_matmul(ATP, BTP, CTP, 4, 4, 4, 2);
    
        printf("\n");
        for (size_t i = 0; i < 4; i++)
        {
            printf("\n");   
            for (size_t j = 0; j < 4; j++){
                printf("%f\t", CTP[i * 4 + j]);
            }
            printf("\n");
        }
    }

    if (!ONLY_LARGE && L1){
        printf("\n");
        printf("l1 tiled matmul:\n");
    
        float ATL[] = {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4}; // row major
        float BTL[] = {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4}; // column major
        float CTL[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; // row major
        
        l1_tiled_matmul(ATL, BTL, CTL, 4, 4, 4, 2, 1);
    
        printf("\n");
        for (size_t i = 0; i < 4; i++)
        {
            printf("\n");   
            for (size_t j = 0; j < 4; j++){
                printf("%f\t", CTL[i * 4 + j]);
            }
            printf("\n");
        }
        
    }


    
    
    // exit(0);

    printf("initialising matrices ....\n");

    float * LA = malloc(M * K * sizeof(float));
    float * LB = malloc(K * N * sizeof(float));
    float * LC = malloc(M * N * sizeof(float));
    float * TLA = malloc(K * M * sizeof(float));
    float * TLB = malloc(N * K * sizeof(float));
    float * TLC = malloc(M * N * sizeof(float));
    initialise_large_matrices(LA, LB, LC);
    float * ref_C = calloc(M * N, sizeof(float));


    float * ref_grads_A = calloc(M * K, sizeof(float));
    float * ref_grads_B = calloc(K * N, sizeof(float));
    float * grads_C = malloc(M * N * sizeof(float));
    for (size_t i = 0; i < M * N; i++){
        grads_C[i] = (float)rand() / (float)RAND_MAX;
        if (i % 2 == 0){
            grads_C[i] = grads_C[i] - 1;
        }
    }
    clock_t start, end;
    double time_spent;

    if (NAIVE)
    {
        printf("Naive .. \n");
        start = clock();
        naive_matmul(LA, LB, ref_C, M, N, K);
        end = clock();
        time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Time spent on matmul: %f seconds\n", time_spent);        

        start = clock();
        matmul_backwards(grads_C, LB, LA, ref_grads_B, ref_grads_A, M, N, K);
        end = clock();
        time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Time spent on matmul back: %f seconds\n", time_spent);        
    }

    if (TILED){
        printf("executing dot product matmul now with tile %d ...\n", TILE);
        initialise_large_matrices(LA, LB, LC);
        start = clock();
        tiled_matmul(LA, LB, LC, M, N, K, TILE);
        end = clock();
        time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Time spent on tiled matmul: %f seconds\n", time_spent);
        if (NAIVE){
            check_result(ref_C, LC, 1024, 1024);
        }
    }

    if (OUTER){
        printf("executing outer product matmul now ...\n");
        initialise_large_matrices(LA, LB, LC);
        start = clock();
        outer_product_matmul(LA, LB, LC, M, N, K);
        end = clock();
        time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Time spent on outer matmul: %f seconds\n", time_spent);
        if (NAIVE){
            check_result(ref_C, LC, 1024, 1024);
        }
    }

    if (L1){
        printf("executing l1 matmul now with tile %d and inner tile %d ...\n", TILE, INNER_TILE);
        initialise_large_matrices(LA, LB, LC);
        start = clock();
        l1_tiled_matmul(LA, LB, LC, M, N, K, TILE, INNER_TILE);
        end = clock();
        time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Time spent on l1 matmul: %f seconds\n", time_spent);
        if (NAIVE){
            check_result(ref_C, LC, 1024, 1024);
        }
    }

    if (SIMD){
        printf("executing simd matmul now ...\n");
        //initialise_large_matrices(LA, LB, LC);
        row_to_col_major(LA, TLA, M, K);
        col_to_row_major(LB, TLB, K, N);        
        start = clock();
        // simd_matmul(TLA, TLB, LC, M, N, K);
        end = clock();
        time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Time spent on simd matmul: %f seconds\n", time_spent);
        col_to_row_major(LC, TLC, M, N);
        if (NAIVE){
            check_result(ref_C, TLC, M, N);
        }
    }






    free(LA);
    free(LB);
    free(LC);
    free(TLA);
    free(TLB);
    free(TLC);
    free(ref_C);
    return 0;

}


// gcc -o main main.c -lm
// ./main
/*
perf stat -e L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.references,l2_rqsts.miss ./main



*/