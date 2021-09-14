#include "NN.cuh"

NN::NN(int layer_cnt, int *layer_neuron_cnts, ActivationFunctionId hidden_layer_activation, ActivationFunctionId output_layer_activation,
       CostFunctionId cost, float learning_rate, float dropout_rate)
{
    this->layer_cnt = layer_cnt;
    this->layer_neuron_cnts = (int *)malloc(sizeof(int) * layer_cnt);
    memcpy(this->layer_neuron_cnts, layer_neuron_cnts, sizeof(int) * layer_cnt);

    this->neurons = (Tensor **)malloc(sizeof(Tensor *));
    this->weights = (Tensor **)malloc(sizeof(Tensor *));
    this->biases = (Tensor **)malloc(sizeof(Tensor *));
    this->weight_derivatives = (Tensor **)malloc(sizeof(Tensor *));
    this->bias_derivatives = (Tensor **)malloc(sizeof(Tensor *));

    // Leave first layer neurons NULL since we will just set them during feed_forward.
    this->neurons[0] = NULL;

    for (int i = 0; i < layer_cnt - 1; i++)
    {
        this->neurons[i + 1] = new Tensor(1, layer_neuron_cnts[i + 1], true);
        this->weights[i] = new Tensor(layer_neuron_cnts[i + 1], layer_neuron_cnts[i], true);
        this->biases[i] = new Tensor(layer_neuron_cnts[i + 1], 1, true);
        this->weight_derivatives[i] = new Tensor(layer_neuron_cnts[i + 1], layer_neuron_cnts[i], true);
        this->bias_derivatives[i] = new Tensor(layer_neuron_cnts[i + 1], 1, true);

        this->neurons[i + 1]->set_all(0.0f);
        this->weights[i]->set_all_rand(1.0f / sqrt(layer_neuron_cnts[i]));
        this->biases[i]->set_all(0.0f);
        this->weight_derivatives[i]->set_all(0.0f);
        this->bias_derivatives[i]->set_all(0.0f);
    }

    this->hidden_layer_activation = hidden_layer_activation;
    this->output_layer_activation = output_layer_activation;
    this->cost = cost;

    this->learning_rate = learning_rate;
    this->dropout_rate = dropout_rate;
}

NN::~NN()
{
    for (int i = 0; i < layer_cnt - 1; i++)
    {
        delete this->neurons[i + 1];
        delete this->weights[i];
        delete this->biases[i];
        delete this->weight_derivatives[i];
        delete this->bias_derivatives[i];
    }

    delete this->neurons;
    delete this->weights;
    delete this->biases;
    delete this->weight_derivatives;
    delete this->bias_derivatives;
}