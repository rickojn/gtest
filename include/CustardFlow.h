#ifndef CUSTARD_FLOW_H
#define CUSTARD_FLOW_H


#include <stdio.h>
#include <stdlib.h>
#include <time.h>





// Function declarations
int min(int a, int b);

void naive_matmul(const float *A, const float *B, float *C, size_t m, size_t n, size_t k);
void ikj_matmul(const float *A, const float *B, float *C, size_t m, size_t n, size_t k);
void outer_product_matmul(const float *A, const float *B, float *C, size_t rows, size_t cols, size_t inner_dim);
void tiled_matmul(const float *A, const float *B, float *C, size_t rows, size_t cols, size_t inner_dim, size_t tile_size);
void l1_tiled_matmul(const float *A, const float *B, float *C, size_t rows, size_t cols, size_t inner_dim, size_t tile_size, size_t inner_tile_size);
void initialise_large_matrices(float *A, float *B, float *C);
void check_result(const float *ref_C, const float *C, size_t rows, size_t cols);
void simd_matmul(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);
void matmul_backwards(const float *grads_C, const float *B, const float *A, float *grads_B, float *grads_A, size_t M, size_t N, size_t K);
void transpose_matrix(const float *src_matrix, float *dest_matrix, size_t src_num_rows, size_t src_num_cols);
void simd_matmul_backwards(const float *grads_C, const float *B, const float *A, float *grads_B, float *grads_A, size_t M, size_t N, size_t K);
float cross_entropy_forward(const float *logits, const long *targets, float * log_probs, size_t batch_size, size_t num_classes);
void loss_backward(const float *logits, const long *targets, float *grad_logits, size_t batch_size, size_t num_classes);
void layer_normalization_forward(const float *input, float *output, size_t batch_size, size_t num_features,
float * gamma, float * beta);
void layer_normalization_backward(const float *inputs, const float *grad_outputs, float *grad_inputs, size_t size_batch, size_t num_features,
const float * gammas, float * grad_gammas, float * grad_betas);
void relu_forward(float *activations, size_t num_features, size_t size_batch);
void relu_backward(const float *input, float *gradients, const long *labels, size_t num_features, size_t size_batch);
void softmax_forward(float *activations, size_t num_classes, size_t size_batch);
void softmax_backward(const float *probs, const float *gradients_output, float *gradients_input, size_t num_classes, size_t size_batch);
void loss_softmax_backward(const float *activations_output, float *gradients_output, const long *labels, size_t num_neurons, size_t size_batch);
void attention_forward(const float *input, const float *weights_query, const float *weights_key, const float *weights_value, 
    const float *weights_output, float *output, float * db_matrix, size_t size_batch, size_t size_sequence, size_t dim_model, size_t num_heads);
void attention_forward_mask(const float *input, const float *weights_query, const float *weights_key, const float *weights_value, 
    const float *weights_output, float *output, float * db_matrix, size_t size_batch, size_t size_sequence, size_t dim_model, size_t num_heads);




#endif // CUSTARD_FLOW_H