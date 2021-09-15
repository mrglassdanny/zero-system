#pragma once

#include "Tensor.cuh"
#include "Batch.cuh"

#include <vector>

enum ActivationFunctionId
{
    None,
    ReLU,
    Sigmoid,
    Tanh
};

enum CostFunctionId
{
    MSE,
    CrossEntropy
};

class NN
{
private:
    std::vector<Tensor *> neurons;
    std::vector<Tensor *> weights;
    std::vector<Tensor *> biases;
    std::vector<Tensor *> weight_derivatives;
    std::vector<Tensor *> bias_derivatives;

    ActivationFunctionId hidden_layer_activation_func_id;
    ActivationFunctionId output_layer_activation_func_id;

    CostFunctionId cost_func_id;

    float learning_rate;

public:
    NN(std::vector<int> layer_config, ActivationFunctionId hidden_layer_activation_func_id,
       ActivationFunctionId output_layer_activation_func_id, CostFunctionId cost_func_id, float learning_rate);
    ~NN();

    void feed_forward(Tensor *x);
    float get_cost(Tensor *y);
    void back_propagate(Tensor *y);
    void optimize(int batch_size);
    void check_gradient(Tensor *x, Tensor *y, bool print_flg);
    void train(Batch *batch);
    void validate(Batch *batch);
    void test(Batch *batch);
};