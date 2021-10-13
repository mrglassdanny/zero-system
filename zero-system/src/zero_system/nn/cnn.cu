#include "cnn.cuh"

#define THREADS_PER_BLOCK 32

using namespace zero::core;
using namespace zero::nn;

// Device functions:

__device__ float d_relu(float val)
{
    return val > 0.0f ? val : 0.0f;
}

__device__ float d_derive_relu(float val)
{
    return val > 0.0f ? 1.0f : 0.0f;
}

__device__ float d_sigmoid(float val)
{
    return (1.0 / (1.0 + exp(-val)));
}

__device__ float d_derive_sigmoid(float val)
{
    return (val) * (1.0 - val);
}

__device__ float d_tanh(float val)
{
    return ((exp(val) - exp(-val)) / (exp(val) + exp(-val)));
}

__device__ float d_derive_tanh(float val)
{
    return (1 - (val * val));
}

__device__ float d_sine(float val)
{
    return sin(val);
}

__device__ float d_derive_sine(float val)
{
    return cos(val);
}

__device__ float d_cosine(float val)
{
    return cos(val);
}

__device__ float d_derive_cosine(float val)
{
    return -sin(val);
}

__device__ float d_mse_cost(float n_val, float y_val)
{
    return ((n_val - y_val) * (n_val - y_val));
}

__device__ float d_derive_mse_cost(float n_val, float y_val)
{
    return 2.0f * (n_val - y_val);
}

__device__ float d_cross_entropy_cost(float n_val, float y_val)
{
    return (float)((y_val * log(n_val)) + ((1.0 - y_val) * log(1.0 - n_val)));
}

__device__ float d_derive_cross_entropy_cost(float n_val, float y_val)
{
    return (n_val - y_val);
}

// Kernel functions:

__global__ void k_set_arr(float *arr, int cnt, float val)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        arr[tid] = val;
    }
}

__global__ void k_convolute(float *n_arr, float *f_arr, float *b_arr, float *nxt_n_arr, int n_col_cnt,
                            int f_row_cnt, int f_col_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int f_idx = tid / (f_row_cnt * f_col_cnt);

    int nxt_n_row_idx = tid / f_col_cnt;
    int nxt_n_col_idx = tid % f_col_cnt;

    for (int f_row_idx = 0, f_rot_row_idx = f_row_cnt - 1; f_row_idx < f_row_cnt; f_row_idx++, f_rot_row_idx--)
    {
        for (int f_col_idx = 0, f_rot_col_idx = f_col_cnt - 1; f_col_idx < f_col_cnt; f_col_idx++, f_rot_col_idx--)
        {
            int x_row_idx = nxt_n_row_idx + f_row_idx;
            int x_col_idx = nxt_n_col_idx + f_col_idx;

            float val = n_arr[x_row_idx * n_col_cnt + x_col_idx];
            val *= f_arr[(f_idx * f_row_cnt) + (f_rot_row_idx * f_col_cnt) + f_rot_col_idx];
            val += b_arr[nxt_n_row_idx * f_col_cnt + nxt_n_col_idx];

            nxt_n_arr[nxt_n_row_idx * f_col_cnt + nxt_n_col_idx] += val;
        }
    }
}

// CNNLayerConfiguration member functions:

CNNLayerConfiguration::CNNLayerConfiguration()
{
}

CNNLayerConfiguration::CNNLayerConfiguration(int channel_cnt, int neuron_row_cnt, int neuron_col_cnt,
                                             int filter_cnt, int filter_row_cnt, int filter_col_cnt,
                                             ActivationFunctionId activation_func_id)
{
    this->channel_cnt = channel_cnt;
    this->neuron_row_cnt = neuron_row_cnt;
    this->neuron_col_cnt = neuron_col_cnt;
    this->filter_cnt = filter_cnt;
    this->filter_row_cnt = this->filter_row_cnt;
    this->filter_col_cnt = filter_col_cnt;
    this->activation_func_id = activation_func_id;
}

CNNLayerConfiguration::~CNNLayerConfiguration()
{
}

// CNN member functions:

CNN::CNN(CostFunctionId cost_func_id, float learning_rate)
{
    this->nn = new NN(cost_func_id, learning_rate);
}

CNN::~CNN()
{
    int lyr_cnt = this->layer_configurations.size();
    for (int lyr_idx = 0; lyr_idx < lyr_cnt; lyr_idx++)
    {
        delete this->neurons[lyr_idx];
        delete this->filters[lyr_idx];
        delete this->biases[lyr_idx];
        delete this->filter_derivatives[lyr_idx];
        delete this->bias_derivatives[lyr_idx];
    }

    delete this->nn;
}

void CNN::add_layer(int channel_cnt, int neuron_row_cnt, int neuron_col_cnt,
                    int filter_cnt, int filter_row_cnt, int filter_col_cnt,
                    ActivationFunctionId activation_func_id)
{
    this->layer_configurations.push_back(CNNLayerConfiguration(channel_cnt, neuron_row_cnt, neuron_col_cnt,
                                                               filter_cnt, filter_row_cnt, filter_col_cnt, activation_func_id));
}

NN *CNN::fully_connected()
{
    return this->nn;
}

void CNN::compile()
{
    int lyr_cnt = this->layer_configurations.size();
    for (int lyr_idx = 0; lyr_idx < lyr_cnt; lyr_idx++)
    {
        CNNLayerConfiguration *lyr_cfg = &this->layer_configurations[lyr_idx];

        int output_row_cnt = lyr_cfg->neuron_row_cnt - lyr_cfg->filter_row_cnt + 1;
        int output_col_cnt = lyr_cfg->neuron_col_cnt - lyr_cfg->filter_col_cnt + 1;

        this->neurons[lyr_idx] = new Tensor(lyr_cfg->channel_cnt * lyr_cfg->neuron_row_cnt, lyr_cfg->neuron_col_cnt, Gpu);
        this->filters[lyr_idx] = new Tensor(lyr_cfg->channel_cnt * lyr_cfg->filter_cnt * lyr_cfg->filter_row_cnt, lyr_cfg->filter_col_cnt, Gpu);
        this->biases[lyr_idx] = new Tensor(lyr_cfg->filter_cnt * output_row_cnt, output_col_cnt, Gpu);
        this->filter_derivatives[lyr_idx] = new Tensor(lyr_cfg->channel_cnt * lyr_cfg->filter_cnt * lyr_cfg->filter_row_cnt, lyr_cfg->filter_col_cnt, Gpu);
        this->bias_derivatives[lyr_idx] = new Tensor(lyr_cfg->filter_cnt * output_row_cnt, output_col_cnt, Gpu);
    }

    this->nn->compile();
}

void CNN::feed_forward(Tensor *x, bool train_flg)
{
    // Need to set input neurons before we do anything.
    this->neurons[0]->set_arr(x->get_arr(Gpu), Gpu);

    int lyr_cnt = this->layer_configurations.size();
    int lst_lyr_idx = lyr_cnt - 1;

    for (int lyr_idx = 0; lyr_idx < lst_lyr_idx; lyr_idx++)
    {
        CNNLayerConfiguration *lyr_cfg = &this->layer_configurations[lyr_idx];
        CNNLayerConfiguration *nxt_lyr_cfg = &this->layer_configurations[lyr_idx + 1];

        int nxt_n_cnt = (nxt_lyr_cfg->channel_cnt * nxt_lyr_cfg->neuron_row_cnt * nxt_lyr_cfg->neuron_col_cnt);

        Tensor *n = this->neurons[lyr_idx];
        Tensor *f = this->filters[lyr_idx];
        Tensor *b = this->biases[lyr_idx];
        Tensor *nxt_n = this->neurons[lyr_idx + 1];

        // Need to reset next layer neurons before we do anything:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((nxt_n_cnt / threads_per_block) + 1);
            k_set_arr<<<num_blocks, threads_per_block>>>(nxt_n->get_arr(Gpu), nxt_n_cnt, 0.0f);
        }

        // Convolution:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((nxt_n_cnt / threads_per_block) + 1);

            for (int chan_idx = 0; chan_idx < lyr_cfg->channel_cnt; chan_idx++)
            {
                k_convolute<<<num_blocks, threads_per_block>>>(n->get_slice(chan_idx * lyr_cfg->neuron_row_cnt, Gpu), f->get_arr(Gpu), b->get_arr(Gpu), nxt_n->get_arr(Gpu),
                                                               lyr_cfg->neuron_col_cnt, lyr_cfg->filter_row_cnt, lyr_cfg->filter_col_cnt);
            }
        }
    }
}