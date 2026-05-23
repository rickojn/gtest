#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <stddef.h>

#ifndef LN_EPS
#define LN_EPS 1e-5f
#endif





#ifndef LN_EPS
#define LN_EPS 1e-5f
#endif



int min(int a, int b){
    return a < b ? a : b;
}

void transpose_matrix(const float *src_matrix, float *dest_matrix, size_t src_num_rows, size_t src_num_cols){
    for (size_t idx_row = 0; idx_row < src_num_rows; idx_row++){
        for (size_t idx_col = 0; idx_col < src_num_cols; idx_col++){
            dest_matrix[idx_col * src_num_rows + idx_row] = src_matrix[idx_row * src_num_cols + idx_col];
        }
    }
}


void naive_matmul(const float* A, const float *B, float * C, size_t M, size_t N, size_t K){
    for (size_t i = 0; i < M; i++ ){
        for (size_t j = 0; j < N; j++){
            for (size_t k = 0; k < K; k++){            
                // C[i][j] += A[i][k] * B[k][j]
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}




void ikj_matmul(const float* A, const float *B, float * C, size_t M, size_t N, size_t K){   
    for (size_t i = 0; i < M; i++ ){
        for (size_t k = 0; k < K; k++){
            for (size_t j = 0; j < N; j++){            
                /* 
                C[i][j] += A[i][k] * B[k][j] 
                Since j is being incremented in the innermost loop, the elements of B being accessed are contiguous in memory
                , which can improve cache performance.
                */
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}



     /*
     C = A * B
     grads_C is dL/dC
     grads_B is dL/dB
     grads_A is dL/dA 
     
     A is M x K 
     B is K x N 
     C is M x N 
     
     
     K: input dimension
     M: batch size
     N: output dimension


     */


void matmul_backwards(const float * grads_C, const float * B, const float * A, float * grads_B, float * grads_A, size_t M, size_t N, size_t K){
    
    float * A_transpose = (float *)malloc(M * K * sizeof(float));
    transpose_matrix(A, A_transpose, M, K);

    naive_matmul(A_transpose, grads_C, grads_B, K, N, M); // grads_B = A-transpose * grads_C


    float * B_transpose = (float *)malloc(K * N * sizeof(float));
    transpose_matrix(B, B_transpose, K, N);

    naive_matmul(grads_C, B_transpose, grads_A, M, N, K); // grads_A = grads_C * B-transpose

    free(A_transpose);
    free(B_transpose);
}



void simd_matmul_backwards(const float *grads_C, const float *B, const float *A,
                           float *grads_B, float *grads_A,
                           size_t M, size_t N, size_t K)
{
    float *A_transpose = (float *)malloc(M * K * sizeof(float));
    transpose_matrix(A, A_transpose, M, K);

    // grads_B = A^T * grads_C   => (K x M) * (M x N) = (K x N)
    // simd_matmul(A_transpose, grads_C, grads_B, K, N, M);
    naive_matmul(A_transpose, grads_C, grads_B, K, N, M); 

    if (grads_A == NULL) {
        free(A_transpose);
        return;
    }

    float *B_transpose = (float *)malloc(K * N * sizeof(float));
    transpose_matrix(B, B_transpose, K, N);

    // grads_A = grads_C * B^T   => (M x N) * (N x K) = (M x K)
    // simd_matmul(grads_C, B_transpose, grads_A, M, K, N);
    naive_matmul(grads_C, B_transpose, grads_A, M, K, N);
    free(A_transpose);
    free(B_transpose);
}


float cross_entropy_forward(const float *logits, const long *targets, float *log_probs, size_t batch_size, size_t num_classes)
{
    float loss = 0.0f;
    for (size_t idx_sample = 0; idx_sample < batch_size; idx_sample++)
    {
        float logit_sum = 0.0f;
        // subtract logit max for numerical stability
        float max_logit = logits[idx_sample * num_classes];
        for (size_t idx_class = 1; idx_class < num_classes; idx_class++)
        {
            if (logits[idx_sample * num_classes + idx_class] > max_logit)
            {
                max_logit = logits[idx_sample * num_classes + idx_class];
            }
        }
        
        for (size_t idx_class = 0; idx_class < num_classes; idx_class++)
        {
            logit_sum += expf(logits[idx_sample * num_classes + idx_class] - max_logit);
        }
        for (size_t idx_class = 0; idx_class < num_classes; idx_class++)
        {
            float log_prob = logits[idx_sample * num_classes + idx_class] - max_logit - logf(logit_sum);
            log_probs[idx_sample * num_classes + idx_class] = log_prob;
            if (idx_class == targets[idx_sample])
            {
                loss -= log_prob; // accumulate loss only for the target class
            }
        }
    }

    return loss/batch_size;
}

void loss_backward(const float *logits, const long *targets, float *grad_logits, size_t size_batch, size_t num_classes)
{
    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++)
    {
        float max_logit = logits[idx_sample * num_classes];
        for (size_t idx_class = 0; idx_class < num_classes; idx_class++)
        {
            if (logits[idx_sample * num_classes + idx_class] > max_logit)
            {
                max_logit = logits[idx_sample * num_classes + idx_class];
            }
        }
        float logit_sum = 0.0f;
        for (size_t idx_class = 0; idx_class < num_classes; idx_class++)
        {
            logit_sum += expf(logits[idx_sample * num_classes + idx_class] - max_logit);
        }

        for (size_t idx_class = 0; idx_class < num_classes; idx_class++)
        {
            float prob = expf(logits[idx_sample * num_classes + idx_class] - max_logit) / logit_sum;
            float target = (idx_class == targets[idx_sample]) ? 1.0f : 0.0f;
            grad_logits[idx_sample * num_classes + idx_class] = (prob - target) / size_batch;
        }
    }
}


void layer_normalization_forward(const float *input, float *output, size_t batch_size, size_t num_features,
float * gamma, float * beta)
{
    for (size_t idx_sample = 0; idx_sample < batch_size; idx_sample++)
    {
        float mean = 0.0f;
        float variance = 0.0f;

        // Calculate mean
        for (size_t idx_feature = 0; idx_feature < num_features; idx_feature++)
        {
            mean += input[idx_sample * num_features + idx_feature];
        }
        mean /= num_features;

        // Calculate variance
        for (size_t idx_feature = 0; idx_feature < num_features; idx_feature++)
        {
            float diff = input[idx_sample * num_features + idx_feature] - mean;
            variance += diff * diff;
        }
        variance /= num_features;

        // Normalize and apply scale and shift
        for (size_t idx_feature = 0; idx_feature < num_features; idx_feature++)
        {
            output[idx_sample * num_features + idx_feature] = gamma[idx_feature] * 
                (input[idx_sample * num_features + idx_feature] - mean) / sqrtf(variance + 1e-5f) + beta[idx_feature];
        }
    }
}

/*
    Layer Normalization (per sample)
    K: number of features
    B: batch size
    x_k: input feature k
    y_k: output feature k
    gamma_k: scale parameter for feature k
    beta_k: shift parameter for feature k

    mean = 1/K * sum_j x_j
    x_centered_k = x_k - mean
    variance = 1/K * sum_j (x_centered_j)^2
    x_hat_k = x_centered_k / sqrt(variance + eps)
    y_k = gamma_k * x_hat_k + beta_k

    Gradients:

    dL/dgamma_k = 1 / B * sum_over_samples( dL/dy_k * x_hat_k )
    dL/dbeta_k  = 1 / B * sum_over_samples( dL/dy_k )

    dL/dx_k = 1 / sqrt(variance + eps) *
        ( dL/dy_k
          - 1/K * sum_j(dL/dy_j * gamma_j)
          - x_hat_k * 1/K * sum_j(gamma_j * dL/dy_j * x_hat_j) )
*/


void layer_normalization_backward(const float *inputs,
                                  const float *grad_outputs,
                                  float *grad_inputs,
                                  size_t size_batch,
                                  size_t num_features,
                                  const float *gammas,
                                  float *grad_gammas,
                                  float *grad_betas)
{
    // Initialize accumulators for grad_gamma and grad_beta
    for (size_t idx_feature = 0; idx_feature < num_features; ++idx_feature) {
        grad_gammas[idx_feature] = 0.0f;
        grad_betas[idx_feature]  = 0.0f;
    }

    for (size_t idx_sample = 0; idx_sample < size_batch; ++idx_sample)
    {
        const size_t offset_sample = idx_sample * num_features;

        // Mean
        float x_mean = 0.0f;
        for (size_t idx_feature = 0; idx_feature < num_features; ++idx_feature) {
            x_mean += inputs[offset_sample + idx_feature];
        }
        x_mean /= (float)num_features;
        
        // Variance
        float variance = 0.0f;
        for (size_t idx_feature = 0; idx_feature < num_features; ++idx_feature) {
            float diff = inputs[offset_sample + idx_feature] - x_mean;
            variance += diff * diff;
        }
        variance /= (float)num_features;


        // Inverse standard deviation plus epsilon for numerical stability
        const float inv_stddev = 1.0f / sqrtf(variance + LN_EPS);

        float mean_dl_dy_gamma = 0.0f;
        float mean_dl_dy_gamma_xhat = 0.0f;

        // Precompute sums for input gradient calculation
        for (size_t idx_feature = 0; idx_feature < num_features; ++idx_feature)
        {
            const float x = inputs[offset_sample + idx_feature];
            const float grad_output = grad_outputs[offset_sample + idx_feature];
            const float x_hat = (x - x_mean) * inv_stddev;
            mean_dl_dy_gamma += grad_output * gammas[idx_feature];
            mean_dl_dy_gamma_xhat += grad_output * gammas[idx_feature] * x_hat;
        }
        mean_dl_dy_gamma /= (float)num_features;
        mean_dl_dy_gamma_xhat /= (float)num_features;

        // Input gradients and accumulate parameter gradients
        for (size_t idx_feature = 0; idx_feature < num_features; ++idx_feature)
        {
            const float x = inputs[offset_sample + idx_feature];
            const float grad_output = grad_outputs[offset_sample + idx_feature];
            const float x_hat = (x - x_mean) * inv_stddev;
            
            grad_inputs[offset_sample + idx_feature] = inv_stddev * 
                (grad_output * gammas[idx_feature]
                 - mean_dl_dy_gamma
                 - x_hat * mean_dl_dy_gamma_xhat);

            grad_gammas[idx_feature] += grad_output * x_hat;  
            grad_betas[idx_feature]  += grad_output;

        }
    }

}

void relu_forward(float *activations, size_t num_features, size_t size_batch)
{
    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++) {
        for (size_t idx_feature = 0; idx_feature < num_features; idx_feature++) {
            activations[idx_sample * num_features + idx_feature] = fmaxf(0.0f, activations[idx_sample * num_features + idx_feature]);
        }
    }
}

void relu_backward(const float *input, float *gradients, const long *labels, size_t num_features, size_t size_batch)
{
    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++) {
        for (size_t idx_feature = 0; idx_feature < num_features; idx_feature++) {
            gradients[idx_sample * num_features + idx_feature] = 
            (input[idx_sample * num_features + idx_feature] > 0.0f) ? gradients[idx_sample * num_features + idx_feature] : 0.0f;
        }
    }
}

void softmax_forward(float *activations, size_t num_classes, size_t size_batch)
{
    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++) {
        float max_logit = activations[idx_sample * num_classes];
        for (size_t idx_neuron = 1; idx_neuron < num_classes; idx_neuron++) {
            if (activations[idx_sample * num_classes + idx_neuron] > max_logit) {
                max_logit = activations[idx_sample * num_classes + idx_neuron];
            }
        }

        float sum_exp = 0.0f;
        for (size_t idx_neuron = 0; idx_neuron < num_classes; idx_neuron++) {
            activations[idx_sample * num_classes + idx_neuron] = expf(activations[idx_sample * num_classes + idx_neuron] - max_logit);
            sum_exp += activations[idx_sample * num_classes + idx_neuron];
        }

        for (size_t idx_neuron = 0; idx_neuron < num_classes; idx_neuron++) {
            activations[idx_sample * num_classes + idx_neuron] /= sum_exp;
        }
    }
}

void loss_softmax_backward(const float *probs, float *gradients_output, const long *labels, size_t num_neurons, size_t size_batch)
{
    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++){
        for (size_t idx_logit = 0; idx_logit < num_neurons; idx_logit++){
            float label = idx_logit == labels[idx_sample] ? 1.0 : 0.0;
            size_t offset_logit = idx_sample * num_neurons + idx_logit;
            gradients_output[offset_logit] = probs[offset_logit] - label;
        }
    }    
}

/*
Decoder Attention.

Input - Sequence of length T of C-dimensional embeddings

Each of these are multiplied by matrices Q, K and V of shape C x A to produce A-dimensional vectors q,k and v.
For each element of the sequence:
	Calculate attention score for that and previous elements by getting the dot product of that elements q vector with the the k vectors of that and prev elements
	Devide each of the scores by sqrt of A.
	Get attention weights by getting softmax of the scaled scores
	Multiply vs for that element and prev by the attention weight.
	For each element of the sequence so far the output is the sum of weighted value vectors of that and prev elements.

*/

void attention_forward(const float *input, const float *weights_query, const float *weights_key, 
    const float *weights_value, const float *weights_output, float *output, float *db_matrix, size_t size_batch, size_t size_sequence, size_t dim_model, size_t num_heads)
{
    // zero output
    memset(output, 0, size_batch * size_sequence * dim_model * sizeof(float));
    /*
    input: batch_size x sequence_length x model_dim
    weights_query: model_dim x model_dim
    weights_key: model_dim x model_dim
    weights_value: model_dim x model_dim
    output: batch_size x sequence_length x model_dim
    */
    // allocate memory for q, k and v
    // q, k and v: batch_size x sequence_length x model_dim
    float *queries = (float *)malloc(size_batch * size_sequence * dim_model * sizeof(float));
    float *keys = (float *)malloc(size_batch * size_sequence * dim_model * sizeof(float));
    float *values = (float *)malloc(size_batch * size_sequence * dim_model * sizeof(float));

    // allocate max memory for attention weights
    float *attention_weights = (float *)malloc(size_sequence * sizeof(float));

    // compute q, k and v
    for (size_t idx_sequence = 0; idx_sequence < size_batch; idx_sequence++){ // each sample is a sequence of embeddings
        // simd_matmul(&input[idx_sequence * size_sequence * dim_model], weights_query, &queries[idx_sequence * size_sequence * dim_model], size_sequence, dim_model, dim_model);
        // simd_matmul(&input[idx_sequence * size_sequence * dim_model], weights_key, &keys[idx_sequence * size_sequence * dim_model], size_sequence, dim_model, dim_model);
        // simd_matmul(&input[idx_sequence * size_sequence * dim_model], weights_value, &values[idx_sequence * size_sequence * dim_model], size_sequence, dim_model, dim_model);
    }

    // memset(db_matrix, 0, size_sequence * size_sequence * sizeof(float));
    // compute attention 
    for (size_t idx_sequence = 0; idx_sequence < size_batch; idx_sequence++){ // each seqence of embeddings
        for (size_t idx_embedding = 0; idx_embedding < size_sequence; idx_embedding++){ // each embedding in the sequence
            for (size_t idx_head = 0; idx_head < num_heads; idx_head++){  // each head in the embedding
                size_t offset_query = (idx_sequence * size_sequence + idx_embedding) * dim_model + idx_head * (dim_model / num_heads);
                for (size_t idx_prefix = 0; idx_prefix <= idx_embedding; idx_prefix++){
                    size_t offset_key = (idx_sequence * size_sequence + idx_prefix) * dim_model + idx_head * (dim_model / num_heads);
                    float attention_score = 0.0f;
                    // attention score is dot product of q and k vectors of the head for the embedding and prefix embedding
                    for (size_t idx_dim = 0; idx_dim < dim_model / num_heads; idx_dim++){
                        attention_score += queries[offset_query + idx_dim] * keys[offset_key + idx_dim];
                    }
                    //scale attention score by sqrt of dimension of head
                    attention_score /= sqrtf((float)(dim_model / num_heads));                   
                    attention_weights[idx_prefix] = attention_score;
                }
                softmax_forward(attention_weights, idx_embedding + 1, 1);
                // memcpy(&db_matrix[idx_embedding * size_sequence], attention_weights, (idx_embedding + 1) * sizeof(float)); // copy attention weights to db_matrix for debugging
                // compute output as sum of weighted value vectors for the head for the embedding and prefix embedding
                size_t offset_v = (idx_sequence * size_sequence + idx_embedding) * dim_model + idx_head * (dim_model / num_heads);
                for (size_t idx_prefix = 0; idx_prefix <= idx_embedding; idx_prefix++){
                    size_t offset_v_prefix = (idx_sequence * size_sequence + idx_prefix) * dim_model + idx_head * (dim_model / num_heads);
                    size_t offset_attention_weight = idx_prefix;
                    for (size_t idx_dim = 0; idx_dim < dim_model / num_heads; idx_dim++){
                        output[offset_v + idx_dim] += attention_weights[offset_attention_weight] * values[offset_v_prefix + idx_dim];
                    }
                }
            }
        }
    }

    memset(db_matrix, 0, size_batch * size_sequence * size_sequence * sizeof(float));
    memcpy(db_matrix, output, size_batch * size_sequence * size_sequence * sizeof(float));


    // aggregate heads by multiplying with weights_output
    float *output_aggregated = (float *)malloc(size_batch * size_sequence * dim_model * sizeof(float));
    // simd_matmul(output, weights_output, output_aggregated, size_batch * size_sequence, dim_model, dim_model);
    memcpy(output, output_aggregated, size_batch * size_sequence * dim_model * sizeof(float));
    free(output_aggregated);


    // free q, k and v 
    free(queries);
    free(keys);
    free(values);
    // free attention weights
    free(attention_weights);
}

void attention_forward_mask(const float *input, const float *weights_query, const float *weights_key,
    const float *weights_value, const float *weights_output, 
    float *output, float * db_matrix, size_t size_batch, size_t size_sequence, size_t dim_model, size_t num_heads)
{
    // zero output
    memset(output, 0, size_batch * size_sequence * dim_model * sizeof(float));
    /*
    B: batch size
    T: sequence length
    C: model dimension
    input: B x T x C
    weights_query: C x C
    weights_key: C x C
    weights_value: C x C
    attention_weights: B x T x T
    output: B x T x C
    */
    // allocate memory for q, k and v
    // q, k and v: B x T x C
    float *queries = (float *)malloc(size_batch * size_sequence * dim_model * sizeof(float));
    float *keys = (float *)malloc(size_batch * size_sequence * dim_model * sizeof(float));
    float *values = (float *)malloc(size_batch * size_sequence * dim_model * sizeof(float));

    // allocate memory for attention weights
    float *attention_weights = (float *)malloc(size_batch * size_sequence * size_sequence * sizeof(float));
    // zero attention weights
    memset(attention_weights, 0, size_batch * size_sequence * size_sequence * sizeof(float));

    // compute q, k and v
    for (size_t idx_sequence = 0; idx_sequence < size_batch; idx_sequence++){ // each sample is a sequence of embeddings
        // simd_matmul(&input[idx_sequence * size_sequence * dim_model], weights_query, &queries[idx_sequence * size_sequence * dim_model], size_sequence, dim_model, dim_model);
        // simd_matmul(&input[idx_sequence * size_sequence * dim_model], weights_key, &keys[idx_sequence * size_sequence * dim_model], size_sequence, dim_model, dim_model);
        // simd_matmul(&input[idx_sequence * size_sequence * dim_model], weights_value, &values[idx_sequence * size_sequence * dim_model], size_sequence, dim_model, dim_model);
    }


    /*
    For each sequence we want a T x T attention score matrix, each row is the attention scores for the embedding at 
    that position to all the embeddings in the sequence including itself. This will be obtained by doing B matrix 
    multiplications between q and k transpose chopped up into 2 x B smaller matrices of shapes T x C and C x T. 
    e.g T = 4 C = 6
q1    x x x x x x
q2    x x x x x x
q3    x x x x x x
q4    x x x x x x
            @   
                kkkk
                1234

                xxxx
                xxxx
                xxxx
                xxxx
                xxxx
                xxxx
            =
                kkkk
                1234
            q1  xxxx
            q2  xxxx
            q3  xxxx
            q4  xxxx

    */

    // allocate memory for one B dimension of k transpose
    float *keys_transpose = (float *)malloc(dim_model * size_sequence * sizeof(float));
    // allocate memory a copy of attention scores for debugging
    float *attention_scores_copy = (float *)malloc(size_batch * size_sequence * size_sequence * sizeof(float));
    // tbc

    
    // populate attention weights tensor with attention scores by multiplying q with k transpose for each sequence in the batch
    for (size_t idx_sequence = 0; idx_sequence < size_batch; idx_sequence++)
    {
        // transpose_matrix(&keys[idx_sequence * size_sequence * dim_model], keys_transpose, size_sequence, dim_model);
        // simd_matmul(&queries[idx_sequence * size_sequence * dim_model], keys_transpose, 
        //             &attention_weights[idx_sequence * size_sequence * size_sequence], 
        //             size_sequence, size_sequence, dim_model);
    }

    // scale attention scores by sqrt of dimension of head
    for (size_t idx_sequence = 0; idx_sequence < size_batch; idx_sequence++)
    {
        for (size_t idx_row = 0; idx_row < size_sequence; idx_row++)
        {
            for (size_t idx_col = 0; idx_col < size_sequence; idx_col++)
            {                
                attention_weights[idx_sequence * size_sequence * size_sequence + idx_row * size_sequence + idx_col] /= sqrtf((float)(dim_model / num_heads));
            }
        }
    }

    // apply causal mask to attention weights
    for (size_t idx_sequence = 0; idx_sequence < size_batch; idx_sequence++)
    {
        for (size_t idx_row = 0; idx_row < size_sequence; idx_row++)
        {
            for (size_t idx_col = 0; idx_col < size_sequence; idx_col++)
            {
                if (idx_col > idx_row)
                {
                    attention_weights[idx_sequence * size_sequence * size_sequence + idx_row * size_sequence + idx_col] = -INFINITY; // causal mask
                }
            }
        }
    }

    // apply softmax to attention weights
    for (size_t idx_sequence = 0; idx_sequence < size_batch; idx_sequence++)
    {
        for (size_t idx_row = 0; idx_row < size_sequence; idx_row++)
        {            
            softmax_forward(&attention_weights[idx_sequence * size_sequence * size_sequence + idx_row * size_sequence], size_sequence, 1);
        } 
    }

    // memset(db_matrix, 0, size_batch * size_sequence * size_sequence * sizeof(float));
    // memcpy(db_matrix, attention_weights, size_batch * size_sequence * size_sequence * sizeof(float)); // copy attention weights to db_matrix for debugging

    // weighted sum of attention values

    float *v_transpose = (float *)malloc(size_batch * size_sequence * dim_model * sizeof(float));
    transpose_matrix(values, v_transpose, size_batch * size_sequence, dim_model);

    for (size_t idx_sequence = 0; idx_sequence < size_batch; idx_sequence++)
    {
        // simd_matmul(&attention_weights[idx_sequence * size_sequence * size_sequence], &values[idx_sequence * size_sequence * dim_model], 
        //             &output[idx_sequence * size_sequence * dim_model], 
        //             size_sequence, dim_model, size_sequence);
    }

    memset(db_matrix, 0, size_batch * size_sequence * size_sequence * sizeof(float));
    memcpy(db_matrix, output, size_batch * size_sequence * size_sequence * sizeof(float));

    // aggregate heads by multiplying with weights_output
    float *output_aggregated = (float *)malloc(size_batch * size_sequence * dim_model * sizeof(float));
    // simd_matmul(output, weights_output, output_aggregated, size_batch * size_sequence, dim_model, dim_model);
    memcpy(output, output_aggregated, size_batch * size_sequence * dim_model * sizeof(float));
    free(output_aggregated);


}


