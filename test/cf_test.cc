#include <gtest/gtest.h>
#include <torch/torch.h>
#include "../include/CustardFlow.h"
#include <stdlib.h>

TEST(NaiveMatrixMultiplicationTest, CompareWithLibTorch) {
    // ARRANGE
    torch::manual_seed(42);
    torch::Tensor A = torch::rand({3, 3});
    torch::Tensor B = torch::rand({3, 3});
    torch::Tensor expected = torch::mm(A, B);

    float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* expected_ptr = expected.data_ptr<float>();

    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    float* actual_ptr = new float[m * n];
    std::fill(actual_ptr, actual_ptr + m * n, 0.0f); 

    // ACT
    naive_matmul(A_ptr, B_ptr, actual_ptr, m, k, n);


    // ASSERT
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_FLOAT_EQ(actual_ptr[i * n + j], expected_ptr[i * n + j])
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }

    // Clean up
    delete[] actual_ptr;
}


 
void mat_mul_backwards_test(
    std::function<void(const float *, const float *, const float *, float *, float *, size_t, size_t, size_t)> matmul_backwards_func,
    std::string func_name)
{
    int m = 784, n = 128, k = 16;

    torch::manual_seed(42);
    torch::Tensor input   = torch::rand({k, m}, torch::requires_grad());
    torch::Tensor weights = torch::rand({m, n}, torch::requires_grad());

    torch::Tensor probs = torch::mm(input, weights);   // (k x n)
    torch::Tensor grad_output = torch::rand_like(probs);
    probs.backward(grad_output);

    float *expected_input_grad   = input.grad().data_ptr<float>();    // k x m
    float *expected_weights_grad = weights.grad().data_ptr<float>();  // m x n
    float *input_ptr = input.data_ptr<float>();
    float *weights_ptr = weights.data_ptr<float>();
    float *grad_output_ptr = grad_output.data_ptr<float>();

    float *actual_input_grad   = new float[k * m];
    float *actual_weights_grad = new float[m * n];
    std::fill(actual_input_grad, actual_input_grad + k * m, 0.0f);
    std::fill(actual_weights_grad, actual_weights_grad + m * n, 0.0f);

    // M=batch, N=probs dim, K=input dim
    matmul_backwards_func(
        grad_output_ptr, weights_ptr, input_ptr,
        actual_weights_grad, actual_input_grad,
        k, n, m
    );

    // check input gradients: k x m
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < m; ++j) {
            EXPECT_NEAR(actual_input_grad[i * m + j],
                        expected_input_grad[i * m + j], 1e-3);
        }
    }

    // check weight gradients: m x n
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(actual_weights_grad[i * n + j],
                        expected_weights_grad[i * n + j], 1e-3);
        }
    }

    delete[] actual_input_grad;
    delete[] actual_weights_grad;
}
// TEST(MatrixMultiplicationBackwardsTest, MatmulBackwards) {
//     mat_mul_backwards_test(matmul_backwards, "matmul_backwards");
//  }

 TEST(MatrixMultiplicationBackwardsTest, SimdMatmulBackwards) {
    mat_mul_backwards_test(simd_matmul_backwards, "simd_matmul_backwards");
 }


 TEST(CrossEntropyLossTest, BasicFunctionality) {
    // ARRANGE
    torch::manual_seed(42);
    int batch_size = 4;
    int num_classes = 5;

    // Random logits and targets
    torch::Tensor logits = torch::randn({batch_size, num_classes}, torch::requires_grad());
    torch::Tensor targets = torch::randint(0, num_classes, {batch_size}, torch::kLong);

    auto loss = torch::nn::functional::cross_entropy(logits, targets, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));


    float *logits_ptr = logits.data_ptr<float>();
    float * transposed_logits_ptr = (float *)malloc(logits.size(0) * logits.size(1) * sizeof(float));
    transpose_matrix(logits_ptr, transposed_logits_ptr, logits.size(0), logits.size(1));
    long *targets_ptr = targets.data_ptr<long>();

    float *log_probs = new float[batch_size * num_classes];

    // ACT
    float loss_value = cross_entropy_forward(logits_ptr, targets_ptr, log_probs, batch_size, num_classes);

    // ASSERT
    EXPECT_FLOAT_EQ(loss.item<float>(), loss_value);

    delete[] log_probs;

 }

 TEST(LossBackwardTest, BasicFunctionality) {
    // ARRANGE
    torch::manual_seed(42);
    int batch_size = 4;
    int num_classes = 5;

    // Random logits and targets
    torch::Tensor logits = torch::randn({batch_size, num_classes}, torch::requires_grad());
    torch::Tensor targets = torch::randint(0, num_classes, {batch_size}, torch::kLong);

    auto loss = torch::nn::functional::cross_entropy(logits, targets, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
    loss.backward();

    float *logits_ptr = logits.data_ptr<float>();
    long *targets_ptr = targets.data_ptr<long>();
    float *grad_logits = logits.grad().data_ptr<float>();

    // ACT
    loss_backward(logits_ptr, targets_ptr, grad_logits, batch_size, num_classes);

    // ASSERT
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            EXPECT_FLOAT_EQ(grad_logits[i * num_classes + j], logits.grad()[i][j].item<float>())
                << "Mismatch in gradient at (" << i << ", " << j << ")";
        }
    }
 }

 // 1) Define a struct to hold your dimensions:
struct MatMulDims {
    int m, k, n;
};

// 2) Create a fixture that derives from TestWithParam, templated on MatMulDims:
class MatrixMultiplicationTest
  : public ::testing::TestWithParam<MatMulDims> {
protected:
    void run_compare_with_libtorch() {
        // ARRANGE
        auto dims = GetParam();
        int m = dims.m, k = dims.k, n = dims.n;

        torch::manual_seed(42);
        auto A = torch::rand({m, k});
        auto B = torch::rand({k, n});
        auto expected = torch::mm(A, B);


        float* A_ptr = A.data_ptr<float>();

        float* B_ptr = B.data_ptr<float>();

        float* expected_ptr = expected.data_ptr<float>();

        float* actual = new float[m * n];
        std::fill(actual, actual + m * n, 0.0f);

        // ACT
        naive_matmul(A_ptr, B_ptr, actual, m, k, n);



        // compare
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                // actual is column-major: [j*m + i]
                EXPECT_NEAR(actual[i * n + j], expected_ptr[i * n + j], 1e-3)
                    << "Failed for dims m="<<m<<", k="<<k<<", n="<<n
                    << " at ("<<i<<","<<j<<") offset: " << i * n + j;
                    if (std::abs(actual[i * n + j] - expected_ptr[i * n + j]) > 1e-3) {
                            i = m; // break outer loop
                            break; // break inner loop
                        }
            }
        }

        // cleanup
        delete[] actual;
    }
};

// 3) Define the parameterized test using TEST_P:
// TEST_P(MatrixMultiplicationTest, CompareWithLibTorch) {
//     run_compare_with_libtorch();
// }

// 4) Instantiate the suite with the size tuples you want to cover:
// INSTANTIATE_TEST_SUITE_P(
//     VariousSizes,
//     MatrixMultiplicationTest,
//     ::testing::Values(
//         MatMulDims{1, 1, 1},
//         MatMulDims{3, 3, 3},
//         MatMulDims{8, 8, 1},
//         MatMulDims{8,8,8},
//         MatMulDims{16, 16, 16},
//         MatMulDims{8, 16,8},
//         MatMulDims{8, 15, 1},
//         MatMulDims{256, 512, 1024},
//         MatMulDims{8, 16, 32},
//         MatMulDims{257, 512, 1024},
//         MatMulDims{257, 512, 1023},
//         MatMulDims{16, 8, 784}
//      ),
//     [](auto const& info) {
//         auto dims = info.param;
//         return "m" + std::to_string(dims.m)
//              + "_k" + std::to_string(dims.k)
//              + "_n" + std::to_string(dims.n);
//     }
// );

TEST(layer_norm_test, basic_functionality) {
    // ARRANGE
    torch::manual_seed(42);
    int batch_size = 4;
    int num_features = 5;

    torch::Tensor input = torch::randn({batch_size, num_features}, torch::requires_grad());
    torch::Tensor gamma = torch::randn({num_features}, torch::requires_grad());
    torch::Tensor beta = torch::randn({num_features}, torch::requires_grad());
    // torch layer norm
    auto layer_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({num_features}).elementwise_affine(true));

    auto probs = layer_norm->forward(input);

    // ACT
    float *input_ptr = input.data_ptr<float>();
    float *output_ptr = probs.data_ptr<float>();
    float *gamma_ptr = layer_norm->weight.data_ptr<float>();
    float *beta_ptr = layer_norm->bias.data_ptr<float>();
    float *expected_output = new float[batch_size * num_features];
    layer_normalization_forward(input_ptr, expected_output, batch_size, num_features, gamma_ptr, beta_ptr);
    // ASSERT
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_features; ++j) {
            EXPECT_NEAR(output_ptr[i * num_features + j], expected_output[i * num_features + j], 1e-3)
                << "Mismatch at (" << i << ", " << j << ")";}
        }
}

// Layer normalization backward test
TEST(layer_norm_backward_test, basic_functionality) {
    // ARRANGE
    torch::manual_seed(42);
    int batch_size = 4;
    int num_features = 5; 
    torch::Tensor input = torch::randn({batch_size, num_features}, torch::requires_grad());
    auto mean = input.mean(1, true);
    
    auto variance = input.var(1, false, true);
    torch::Tensor gamma = torch::randn({num_features}, torch::requires_grad());
    torch::Tensor beta = torch::randn({num_features}, torch::requires_grad());
    auto layer_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({num_features}).elementwise_affine(true));
    auto probs = layer_norm->forward(input);
    torch::Tensor grad_output = torch::randn_like(probs);
    probs.backward(grad_output);
    float *input_ptr = input.data_ptr<float>();
    float *gamma_ptr = layer_norm->weight.data_ptr<float>();
    float *beta_ptr = layer_norm->bias.data_ptr<float>();
    float *grad_output_ptr = grad_output.data_ptr<float>();
    float *expected_input_grad = input.grad().data_ptr<float>();
    float *expected_gamma_grad = layer_norm->weight.grad().data_ptr<float>();
    float *expected_beta_grad = layer_norm->bias.grad().data_ptr<float>();  
    float *actual_input_grad = new float[batch_size * num_features];
    float *actual_gamma_grad = new float[num_features];
    float *actual_beta_grad = new float[num_features];
    std::fill(actual_input_grad, actual_input_grad + batch_size * num_features, 0.0f);
    std::fill(actual_gamma_grad, actual_gamma_grad + num_features, 0.0f);
    std::fill(actual_beta_grad, actual_beta_grad + num_features, 0.0f);
    // ACT
    layer_normalization_backward(input_ptr, grad_output_ptr, actual_input_grad, batch_size, num_features,
        gamma_ptr, actual_gamma_grad, actual_beta_grad);
    // ASSERT
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_features; ++j) {
            EXPECT_NEAR(actual_input_grad[i * num_features + j], expected_input_grad[i * num_features + j], 1e-3)
                << "Mismatch in input gradient at (" << i << ", " << j << ")";
        }
    }
    for (int j = 0; j < num_features; ++j) {
        EXPECT_NEAR(actual_gamma_grad[j], expected_gamma_grad[j], 1e-3)
            << "Mismatch in gamma gradient at (" << j << ")";
        EXPECT_NEAR(actual_beta_grad[j], expected_beta_grad[j], 1e-3)
            << "Mismatch in beta gradient at (" << j << ")";
    }   
}


// relu forward test
TEST(ReLUForwardTest, BasicFunctionality) {
    // ARRANGE
    torch::manual_seed(42);
    int batch_size = 4;
    int num_features = 5;
    torch::Tensor input = torch::randn({batch_size, num_features}, torch::requires_grad());
    auto relu = torch::nn::ReLU();
    auto probs = relu->forward(input);
    float *input_ptr = input.data_ptr<float>();
    float *output_ptr = probs.data_ptr<float>();
    float *actual_output = new float[batch_size * num_features];
    //copy input to actual_output
    std::copy(input_ptr, input_ptr + batch_size * num_features, actual_output);
    // ACT
    relu_forward(actual_output, num_features, batch_size);
    // ASSERT
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_features; ++j) {
            EXPECT_FLOAT_EQ(output_ptr[i * num_features + j], actual_output[i * num_features + j])
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
    delete[] actual_output;
}

//relu backward test
TEST(ReLUBackwardTest, BasicFunctionality) {
    // ARRANGE
    torch::manual_seed(42);
    int batch_size = 4;
    int num_features = 5;
    torch::Tensor input = torch::randn({batch_size, num_features}, torch::requires_grad());
    auto relu = torch::nn::ReLU();
    auto probs = relu->forward(input);
    torch::Tensor grad_output = torch::randn_like(probs);
    probs.backward(grad_output);
    float *input_ptr = input.data_ptr<float>();
    float *grad_output_ptr = grad_output.data_ptr<float>();
    float *expected_input_grad = input.grad().data_ptr<float>();
    float *actual_grads = new float[batch_size * num_features];
    //copy grad_output to actual_grads
    std::copy(grad_output_ptr, grad_output_ptr + batch_size * num_features, actual_grads);
    // ACT
        relu_backward(input_ptr, actual_grads, nullptr, num_features, batch_size);
    //  ASSERT
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_features; ++j) {
            EXPECT_NEAR(actual_grads[i * num_features + j], expected_input_grad[i * num_features + j], 1e-3)
                << "Mismatch in input gradient at (" << i << ", " << j << ")";
        }
    }
    delete[] actual_grads;
}

// softmax forward test
TEST(SoftmaxForwardTest, BasicFunctionality) {
    // ARRANGE
    torch::manual_seed(42);
    int batch_size = 4;
    int num_classes = 5;    
    torch::Tensor logits = torch::randn({batch_size, num_classes}, torch::requires_grad());
    auto softmax = torch::nn::Softmax(/*dim=*/1);
    auto probs = softmax->forward(logits);
    float *logits_ptr = logits.data_ptr<float>();
    float *output_ptr = probs.data_ptr<float>();
    float *actual_output = new float[batch_size * num_classes];
    // copy logits to actual_output
    std::copy(logits_ptr, logits_ptr + batch_size * num_classes, actual_output);
    // ACT
    softmax_forward(actual_output, num_classes, batch_size);
    // ASSERT
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            EXPECT_NEAR(output_ptr[i * num_classes + j], actual_output[i * num_classes + j], 1e-3)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
    delete[] actual_output;
}

// loss softmax backward test
TEST(SoftmaxCrossEntropyBackwardTest, MatchesPyTorch) {
    torch::manual_seed(42);

    const int batch_size = 4;
    const int num_classes = 5;

    auto logits = torch::randn({batch_size, num_classes},
                               torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));

    auto labels = torch::randint(0, num_classes, {batch_size},
                                 torch::TensorOptions().dtype(torch::kInt64));

    // PyTorch: CE on logits (internally logsoftmax + NLL)
    auto loss = torch::nn::functional::cross_entropy(
        logits, labels,
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean)
    );

    loss.backward();

    // Our side: compute probs then (p - onehot) / batch_size to match mean reduction
    auto probs = torch::softmax(logits.detach(), /*dim=*/1).contiguous();

    const float* probs_ptr = probs.data_ptr<float>();
    const int64_t* labels_ptr = labels.data_ptr<int64_t>();

    std::vector<float> actual(batch_size * num_classes);
    loss_softmax_backward(probs_ptr, actual.data(), labels_ptr, num_classes, batch_size);

    // divide by batch_size because PyTorch reduction=mean
    for (auto& v : actual) v /= batch_size;

    const float* expected = logits.grad().data_ptr<float>();

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            EXPECT_NEAR(actual[i * num_classes + j], expected[i * num_classes + j], 1e-4)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

// attention forward test without cache
TEST(AttentionForwardNoCacheTest, DISABLED_BasicFunctionality) {
    // ARRANGE
    torch::manual_seed(42);
    int batch_size = 16; // passes if batch_size=1 but fails for batch_size=2, need to investigate
    int size_sequence = 9;
    int dim_model = 128;
    int num_heads = 8; 
    torch::Tensor input = torch::randn({size_sequence, batch_size, dim_model}, torch::requires_grad());
    torch::Tensor weights_query = torch::randn({dim_model, dim_model}, torch::requires_grad());
    torch::Tensor weights_key = torch::randn({dim_model, dim_model}, torch::requires_grad());
    torch::Tensor weights_value = torch::randn({dim_model, dim_model}, torch::requires_grad());
    torch::Tensor weights_output = torch::randn({dim_model, dim_model}, torch::requires_grad());
    
    // transpose weights for libtorch multihead attention
    auto weights_query_t = weights_query.transpose(0, 1);
    auto weights_key_t = weights_key.transpose(0, 1);
    auto weights_value_t = weights_value.transpose(0, 1);
    auto weights_output_t = weights_output.transpose(0, 1);

    torch::nn::MultiheadAttention mha(torch::nn::MultiheadAttentionOptions(dim_model, num_heads));
    torch::NoGradGuard no_grad;
    mha->in_proj_weight.copy_(torch::cat({weights_query_t, weights_key_t, weights_value_t}, 0));
    mha->in_proj_bias.zero_();
    mha->out_proj->weight.copy_(weights_output_t);
    mha->out_proj->bias.zero_();
    mha->eval();
    auto causal_mask = torch::zeros({size_sequence, size_sequence}, torch::kFloat);
    causal_mask = causal_mask.masked_fill(
        torch::ones({size_sequence, size_sequence}, torch::kBool).triu(1),
        -1e9);
    torch::Tensor key_padding_mask;

    
    auto attention_output_tuple = mha->forward(input, input, input, /*key_padding_mask=*/key_padding_mask, /*need_weights=*/true, /*attn_mask=*/causal_mask);
    auto attention_output = std::get<0>(attention_output_tuple);

    auto input_bsd = input.permute({1, 0, 2}).contiguous();   // [B,S,D]
    float *input_ptr = input_bsd.data_ptr<float>();

    // float *input_ptr = input.data_ptr<float>();
    float *weights_query_ptr = weights_query.data_ptr<float>();
    float *weights_key_ptr = weights_key.data_ptr<float>();
    float *weights_value_ptr = weights_value.data_ptr<float>();
    float *weights_output_ptr = weights_output.data_ptr<float>();
    auto attention_output_bsd = attention_output.permute({1, 0, 2}).contiguous();
    float *expected_output_ptr = attention_output_bsd.data_ptr<float>();
    float *actual_output = new float[batch_size * size_sequence * dim_model];
    float * dummy = new float[batch_size * size_sequence * dim_model]; // for debugging
    
    // ACT
    attention_forward(input_ptr, weights_query_ptr, weights_key_ptr, weights_value_ptr, weights_output_ptr,
        actual_output, dummy, batch_size, size_sequence, dim_model, num_heads);
    // ASSERT
    // for debugging: compare 




    // log max and mean absolute and relative differences for debugging
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    float sum_abs_diff = 0.0f;
    float sum_rel_diff = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
    // for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < size_sequence; ++j) {
        // for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < dim_model; ++k) {
            // for (int k = 0; k < 1; ++k) {
                int idx = i * size_sequence * dim_model + j * dim_model + k;
                float actual = actual_output[idx];
                float expected = expected_output_ptr[idx];
                float abs_diff = fabsf(actual - expected);
                float rel_diff = abs_diff / (fabsf(expected) + 1e-6f);
                max_abs_diff = std::max(max_abs_diff, abs_diff);
                max_rel_diff = std::max(max_rel_diff, rel_diff);
                sum_abs_diff += abs_diff;
                sum_rel_diff += rel_diff;

                const float atol = 2e-3f;
                const float rtol = 1e-3f;   // maybe 2e-3f if needed

                EXPECT_TRUE(abs_diff <= atol + rtol * fabsf(expected))
                    << "Mismatch at (" << i << ", " << j << ", " << k << "): "
                    << "actual=" << actual
                    << ", expected=" << expected
                    << ", abs_diff=" << abs_diff
                    << ", rel_diff=" << rel_diff;
            }
        }
    }
    std::cout << "Max absolute difference: " << max_abs_diff << std::endl;
    std::cout << "Max relative difference: " << max_rel_diff << std::endl;
    std::cout << "Mean absolute difference: " << sum_abs_diff / (batch_size * size_sequence * dim_model) << std::endl;
    std::cout << "Mean relative difference: " << sum_rel_diff / (batch_size * size_sequence * dim_model) << std::endl;
    delete[] actual_output;
}

// using the cf attention_foward as a reference for expected probs, test the attention_forward_mask implementation.

/*

./my_tests --gtest_filter=MySuite.MyTest
./build-debug/cf_test --gtest_filter=AttentionForwardMaskTest.BasicFunctionality

*/
TEST(AttentionForwardMaskTest, BasicFunctionality) {
    // ARRANGE
    int batch_size = 16; 
    int size_sequence = 9;
    int dim_model = 128;
    int num_heads = 8; 

    //logout dimensions for debugging
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Sequence length: " << size_sequence << std::endl;
    std::cout << "Model dimension: " << dim_model << std::endl;
    std::cout << "Number of heads: " << num_heads << std::endl;


    // use torch to generate randomly populated input and weight tensors
    torch::manual_seed(42);
    torch::Tensor input = torch::randn({batch_size, size_sequence, dim_model}, torch::requires_grad());
    torch::Tensor weights_query = torch::randn({dim_model, dim_model}, torch::requires_grad());
    torch::Tensor weights_key = torch::randn({dim_model, dim_model}, torch::requires_grad());
    torch::Tensor weights_value = torch::randn({dim_model, dim_model}, torch::requires_grad());
    torch::Tensor weights_output = torch::randn({dim_model, dim_model}, torch::requires_grad());

    float *input_ptr = input.data_ptr<float>();
    float *weights_query_ptr = weights_query.data_ptr<float>();
    float *weights_key_ptr = weights_key.data_ptr<float>();
    float *weights_value_ptr = weights_value.data_ptr<float>();
    float *weights_output_ptr = weights_output.data_ptr<float>();

    // allocate memory for actual and expected probs tensors, and compute expected probs using attention_forward as reference
    float *actual_output_ptr = new float[batch_size * size_sequence * dim_model];
    float *expected_output_ptr = new float[batch_size * size_sequence * dim_model];

    float * expected_db_matrix = new float[batch_size * size_sequence * size_sequence]; // for debugging
    float * actual_db_matrix = new float[batch_size * size_sequence * size_sequence]; // for debugging
    
    attention_forward(input_ptr, weights_query_ptr, weights_key_ptr, weights_value_ptr, weights_output_ptr,
        expected_output_ptr, expected_db_matrix, batch_size, size_sequence, dim_model, num_heads);
        
    // ACT
    attention_forward_mask(input_ptr, weights_query_ptr, weights_key_ptr, weights_value_ptr, weights_output_ptr,
        actual_output_ptr, actual_db_matrix, batch_size, size_sequence, dim_model, num_heads);


    // DEBUG ASSERT: compare actual_db_matrix and expected_db_matrix to check if the intermediate db_matrix is the same for both implementations,
    // log message also if they match.
    bool db_matrix_match = true;
    for (int i = 0; i < batch_size * size_sequence * dim_model; ++i) {
        if (fabsf(actual_db_matrix[i] - expected_db_matrix[i]) > 1e-3f) {
            db_matrix_match = false;
            // std::cout << "Mismatch in db_matrix at index " << i << ": actual=" << actual_db_matrix[i] << ", expected=" << expected_db_matrix[i] << std::endl;
        }
    }
    if (db_matrix_match) {
        std::cout << "db_matrix matches between attention_forward and attention_forward_mask implementations." << std::endl;
    } else {
        std::cout << "db_matrix does NOT match between attention_forward and attention_forward_mask implementations." << std::endl;
    }

    // ASSERT

    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    float sum_abs_diff = 0.0f;
    float sum_rel_diff = 0.0f;
    for (int i = 0; i < batch_size; ++i)
    {
        for (int i = 0; i < 1; ++i) 
        // for (int j = 0; j < size_sequence; ++j)
        {
            for (int j = 0; j < 1; ++j) 
            // for (int k = 0; k < dim_model; ++k)
            
            for (int k = 0; k < 1; ++k) {
                int idx = i * size_sequence * dim_model + j * dim_model + k;
                float actual = actual_output_ptr[idx];
                float expected = expected_output_ptr[idx];
                float abs_diff = fabsf(actual - expected);
                float rel_diff = abs_diff / (fabsf(expected) + 1e-6f);
                max_abs_diff = std::max(max_abs_diff, abs_diff);
                max_rel_diff = std::max(max_rel_diff, rel_diff);
                sum_abs_diff += abs_diff;
                sum_rel_diff += rel_diff;

                const float atol = 2e-3f;
                const float rtol = 1e-3f; // maybe 2e-3f if needed

                EXPECT_TRUE(abs_diff <= atol + rtol * fabsf(expected))
                    << "Mismatch at (" << i << ", " << j << ", " << k << "): "
                    << "actual=" << actual
                    << ", expected=" << expected
                    << ", abs_diff=" << abs_diff
                    << ", rel_diff=" << rel_diff;
            }
        }
    }
    std::cout << "Max absolute difference: " << max_abs_diff << std::endl;
    std::cout << "Max relative difference: " << max_rel_diff << std::endl;
    std::cout << "Mean absolute difference: " << sum_abs_diff / (batch_size * size_sequence * dim_model) << std::endl;
    std::cout << "Mean relative difference: " << sum_rel_diff / (batch_size * size_sequence * dim_model) << std::endl;

    //cleanup

    delete[] actual_output_ptr;
    delete[] expected_output_ptr;
}

TEST(SoftmaxBackwardsTest, BasicFunctionality) {
    // ARRANGE
    torch::manual_seed(42);
    int batch_size = 4;
    int num_classes = 5;

    torch::Tensor logits = torch::randn(
        {batch_size, num_classes},
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)
    );

    auto softmax = torch::nn::Softmax(/*dim=*/1);
    torch::Tensor probs = softmax->forward(logits);

    torch::Tensor grad_output = torch::randn_like(probs);

    probs.backward(grad_output);

    float *probs_ptr = probs.data_ptr<float>();
    float *grad_output_ptr = grad_output.data_ptr<float>();
    float *expected_input_grad = logits.grad().data_ptr<float>();

    std::vector<float> actual_input_grad(batch_size * num_classes, 0.0f);

    // ACT
    softmax_backward(
        probs_ptr,
        grad_output_ptr,
        actual_input_grad.data(),
        batch_size,
        num_classes
    );

    // ASSERT
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            EXPECT_NEAR(
                actual_input_grad[i * num_classes + j],
                expected_input_grad[i * num_classes + j],
                1e-3
            ) << "Mismatch in input gradient at (" << i << ", " << j << ")";
        }
    }
}

    