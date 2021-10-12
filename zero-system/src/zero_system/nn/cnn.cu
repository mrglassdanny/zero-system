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

__global__ void k_cross_correlate(float *x_arr, float *f_arr, float *b_arr, float *y_arr, int x_col_cnt,
                                  int f_row_cnt, int f_col_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int y_row_idx = tid / f_col_cnt;
    int y_col_idx = tid % f_col_cnt;

    for (int f_row_idx = 0, f_rot_row_idx = f_row_cnt - 1; f_row_idx < f_row_cnt; f_row_idx++, f_rot_row_idx--)
    {
        for (int f_col_idx = 0, f_rot_col_idx = f_col_cnt - 1; f_col_idx < f_col_cnt; f_col_idx++, f_rot_col_idx--)
        {
            int x_row_idx = y_row_idx + f_row_idx;
            int x_col_idx = y_col_idx + f_col_idx;

            float val = x_arr[x_row_idx * x_col_cnt + x_col_idx];
            val *= f_arr[f_rot_row_idx * f_col_cnt + f_rot_col_idx];
            val += b_arr[y_row_idx * f_col_cnt + y_col_idx];

            y_arr[y_row_idx * f_col_cnt + y_col_idx] += val;
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

CNN::CNN()
{
}

CNN::~CNN()
{
}

void add_layer(int channel_cnt, int neuron_row_cnt, int neuron_col_cnt,
               int filter_cnt, int filter_row_cnt, int filter_col_cnt,
               ActivationFunctionId activation_func_id)
{
}