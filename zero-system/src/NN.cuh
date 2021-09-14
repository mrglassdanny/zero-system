
#include "Tensor.cuh"

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
    int layer_cnt;
    int *layer_neuron_cnts;

    Tensor **neurons;
    Tensor **weights;
    Tensor **biases;
    Tensor **weight_derivatives;
    Tensor **bias_derivatives;

    ActivationFunctionId hidden_layer_activation;
    ActivationFunctionId output_layer_activation;
    CostFunctionId cost;

    float learning_rate;
    float dropout_rate;

public:
    NN(int layer_cnt, int *layer_neuron_cnts, ActivationFunctionId hidden_layer_activation, ActivationFunctionId output_layer_activation,
       CostFunctionId cost, float learning_rate, float dropout_rate);
    ~NN();

    Tensor *feed_forward(Tensor *x);
    float get_cost(Tensor *y);
    void back_propagate(Tensor *y);
    void optimize();
};