#include "layer.cuh"

using namespace zero_v2::core;
using namespace zero_v2::nn;

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

// Kernel functions:

__global__ void k_dot(float *n_arr, float *w_arr, float *nxt_n_arr, int n_cnt, int nxt_n_cnt)
{
    __shared__ float temp[CUDA_THREADS_PER_BLOCK];
    memset(temp, 0, CUDA_THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int w_cnt = n_cnt * nxt_n_cnt;

    int n_idx = tid % n_cnt;
    int w_idx = tid;

    if (w_idx < w_cnt)
    {
        temp[threadIdx.x] = n_arr[n_idx] * w_arr[w_idx];
    }

    __syncthreads();

    if (threadIdx.x == 0) // threadIdx MUST be 0 for below logic to work!
    {
        /*
        The goal here is to try to minimize atomic adds. If the neuron count is
        greater than or equal to the threads per block, a maximum of 2 atomic adds
        is necessary for this block. However, most of the time we can get away with just 1.

        If the threads per block is greater than the neuron count, we just play it safe
        and incur an atomic add for each thread in the block.
        */

        int lower_idx = tid / n_cnt;
        int upper_idx = ((tid + CUDA_THREADS_PER_BLOCK) - 1) / n_cnt;

        if (n_cnt >= CUDA_THREADS_PER_BLOCK)
        {
            if (lower_idx == upper_idx)
            {
                float sum = 0.0f;

#pragma unroll
                for (int i = 0; i < CUDA_THREADS_PER_BLOCK; i++)
                {
                    sum += temp[i];
                }

                atomicAdd(&nxt_n_arr[lower_idx], sum);
            }
            else
            {
                float sums[2] = {0.0f, 0.0f};

#pragma unroll
                for (int i = 0; i < CUDA_THREADS_PER_BLOCK; i++)
                {
                    if ((tid + i) / n_cnt == lower_idx)
                    {
                        sums[0] += temp[i];
                    }
                    else
                    {
                        sums[1] += temp[i];
                    }
                }

                atomicAdd(&nxt_n_arr[lower_idx], sums[0]);
                if (upper_idx < nxt_n_cnt)
                {
                    atomicAdd(&nxt_n_arr[upper_idx], sums[1]);
                }
            }
        }
        else
        {

#pragma unroll
            for (int i = 0; i < CUDA_THREADS_PER_BLOCK; i++)
            {
                atomicAdd(&nxt_n_arr[(tid + i) / n_cnt], temp[i]);
            }
        }
    }
}

__global__ void k_add_bias(float *b_arr, float *nxt_n_arr, int nxt_n_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nxt_n_cnt)
    {
        nxt_n_arr[tid] += b_arr[tid];
    }
}

__global__ void k_derive_z_and_increment_weight_derivative(float *dc_arr, float *n_arr, float *dw_arr, int dc_cnt, int n_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int w_cnt = dc_cnt * n_cnt;

    int nxt_n_idx = tid % n_cnt;
    int n_idx = tid / n_cnt;
    int w_idx = n_idx * n_cnt + nxt_n_idx;

    if (w_idx < w_cnt)
    {
        dw_arr[w_idx] += (dc_arr[n_idx] * n_arr[nxt_n_idx]);
    }
}

__global__ void k_derive_z_and_increment_bias_derivative(float *dc_arr, float *db_arr, int dc_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dc_cnt)
    {
        db_arr[tid] += (dc_arr[tid]);
    }
}

__global__ void k_derive_z_and_aggregate_derivatives(float *agg_derivatives_arr, float *nxt_w_arr, float *nxt_agg_derivatives_arr, int n_cnt, int nxt_n_cnt)
{
    __shared__ float temp[CUDA_THREADS_PER_BLOCK];
    memset(temp, 0, CUDA_THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int w_cnt = nxt_n_cnt * n_cnt;

    // Transpose the weights "matrix".
    int n_idx = tid % n_cnt;
    int nxt_n_idx = tid / n_cnt;
    int w_idx = n_idx * nxt_n_cnt + nxt_n_idx;

    if (w_idx < w_cnt)
    {
        temp[threadIdx.x] = (agg_derivatives_arr[n_idx] * nxt_w_arr[w_idx]);
    }

    __syncthreads();

    if (threadIdx.x == 0) // threadIdx MUST be 0 for below logic to work!
    {
        /*
        The goal here is to try to minimize atomic adds. If the neuron count is
        greater than or equal to the threads per block, a maximum of 2 atomic adds
        is necessary for this block. However, most of the time we can get away with just 1.

        If the threads per block is greater than the neuron count, we just play it safe
        and incur an atomic add for each thread in the block.
        */

        int lower_idx = tid / n_cnt;
        int upper_idx = ((tid + CUDA_THREADS_PER_BLOCK) - 1) / n_cnt;

        if (n_cnt >= CUDA_THREADS_PER_BLOCK)
        {
            if (lower_idx == upper_idx)
            {
                float sum = 0.0f;

#pragma unroll
                for (int i = 0; i < CUDA_THREADS_PER_BLOCK; i++)
                {
                    sum += temp[i];
                }
                atomicAdd(&nxt_agg_derivatives_arr[lower_idx], sum);
            }
            else
            {
                float sums[2] = {0.0f, 0.0f};

#pragma unroll
                for (int i = 0; i < CUDA_THREADS_PER_BLOCK; i++)
                {
                    if ((tid + i) / n_cnt == lower_idx)
                    {
                        sums[0] += temp[i];
                    }
                    else
                    {
                        sums[1] += temp[i];
                    }
                }

                atomicAdd(&nxt_agg_derivatives_arr[lower_idx], sums[0]);
                if (upper_idx < nxt_n_cnt)
                {
                    atomicAdd(&nxt_agg_derivatives_arr[upper_idx], sums[1]);
                }
            }
        }
        else
        {

#pragma unroll
            for (int i = 0; i < CUDA_THREADS_PER_BLOCK; i++)
            {
                atomicAdd(&nxt_agg_derivatives_arr[(tid + i) / n_cnt], temp[i]);
            }
        }
    }
}

__global__ void k_adjust_weight(float *w_arr, float *dw_arr, int batch_size, float learning_rate, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        w_arr[tid] -= ((dw_arr[tid] * learning_rate) / (float)batch_size);
        dw_arr[tid] = 0.0f;
    }
}

__global__ void k_adjust_bias(float *b_arr, float *db_arr, int batch_size, float learning_rate, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        b_arr[tid] -= ((db_arr[tid] * learning_rate) / (float)batch_size);
        db_arr[tid] = 0.0f;
    }
}

__global__ void k_activate(float *n_arr, float *nxt_n_arr, int n_cnt, ActivationFunction activation_fn)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (activation_fn)
        {
        case ActivationFunction::ReLU:
            nxt_n_arr[tid] = d_relu(n_arr[tid]);
            break;
        case ActivationFunction::Sigmoid:
            nxt_n_arr[tid] = d_sigmoid(n_arr[tid]);
            break;
        case ActivationFunction::Tanh:
            nxt_n_arr[tid] = d_tanh(n_arr[tid]);
            break;
        case ActivationFunction::Sine:
            nxt_n_arr[tid] = d_sine(n_arr[tid]);
            break;
        case ActivationFunction::Cosine:
            nxt_n_arr[tid] = d_cosine(n_arr[tid]);
            break;
        default:
            // None
            break;
        }
    }
}

__global__ void k_derive_activation(float *n_arr, float *dc_arr, int n_cnt, ActivationFunction activation_fn)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (activation_fn)
        {
        case ActivationFunction::ReLU:
            dc_arr[tid] *= d_derive_relu(n_arr[tid]);
            break;
        case ActivationFunction::Sigmoid:
            dc_arr[tid] *= d_derive_sigmoid(n_arr[tid]);
            break;
        case ActivationFunction::Tanh:
            dc_arr[tid] *= d_derive_tanh(n_arr[tid]);
            break;
        case ActivationFunction::Sine:
            dc_arr[tid] *= d_derive_sine(n_arr[tid]);
            break;
        case ActivationFunction::Cosine:
            dc_arr[tid] *= d_derive_cosine(n_arr[tid]);
            break;
        default:
            // None
            break;
        }
    }
}

// Layer functions:

Layer::Layer()
{
    this->n = nullptr;
}

Layer::~Layer()
{
    if (this->n != nullptr)
    {
        delete this->n;
    }
}

// LearnableLayer functions:

LearnableLayer::LearnableLayer()
{
    this->w = nullptr;
    this->b = nullptr;
    this->dw = nullptr;
    this->db = nullptr;
}

LearnableLayer::~LearnableLayer()
{
    if (this->w != nullptr)
    {
        delete this->w;
    }

    if (this->b != nullptr)
    {
        delete this->b;
    }

    if (this->dw != nullptr)
    {
        delete this->dw;
    }

    if (this->db != nullptr)
    {
        delete this->db;
    }
}

// LinearLayer functions:

LinearLayer::LinearLayer(int n_cnt, int nxt_n_cnt, InitializationFunction init_fn)
    : LearnableLayer()
{
    this->n = new Tensor(Device::Cuda, n_cnt);
    this->n->reset();

    this->w = new Tensor(Device::Cuda, nxt_n_cnt, n_cnt);
    Initializer::initialize(init_fn, this->w);

    this->b = new Tensor(Device::Cuda, nxt_n_cnt);
    Initializer::initialize(init_fn, this->b);

    this->dw = new Tensor(Device::Cuda, n_cnt, nxt_n_cnt);
    this->dw->reset();

    this->db = new Tensor(Device::Cuda, nxt_n_cnt);
    this->db->reset();
}

LinearLayer::~LinearLayer()
{
}

void LinearLayer::evaluate(Tensor *nxt_n)
{
    int n_cnt = this->n->get_cnt();
    int nxt_n_cnt = nxt_n->get_cnt();

    // Dot product:
    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = ((n_cnt * nxt_n_cnt) / threads_per_block) + 1;
        k_dot<<<num_blocks, threads_per_block>>>(this->n->get_arr(), w->get_arr(),
                                                 nxt_n->get_arr(), n_cnt, nxt_n_cnt);
    }

    // Add biases:
    {
        int threads_per_block(CUDA_THREADS_PER_BLOCK);
        int num_blocks((nxt_n_cnt / threads_per_block) + 1);
        k_add_bias<<<num_blocks, threads_per_block>>>(this->b->get_arr(), nxt_n->get_arr(),
                                                      nxt_n_cnt);
    }
}

void LinearLayer::derive(Tensor *dc)
{
    int dc_cnt = dc->get_cnt();
    int n_cnt = this->n->get_cnt();

    // Weights:
    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = ((dc_cnt * n_cnt) / threads_per_block) + 1;
        k_derive_z_and_increment_weight_derivative<<<num_blocks, threads_per_block>>>(dc->get_arr(),
                                                                                      this->n->get_arr(),
                                                                                      this->dw->get_arr(),
                                                                                      dc_cnt, n_cnt);
    }

    // Biases:
    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (dc_cnt / threads_per_block) + 1;
        k_derive_z_and_increment_bias_derivative<<<num_blocks, threads_per_block>>>(dc->get_arr(), this->db->get_arr(), n_cnt);
    }

    Tensor *nxt_dc = new Tensor(Device::Cuda, n_cnt);
    nxt_dc->reset();

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = ((n_cnt * dc->get_cnt()) / threads_per_block) + 1;
        k_derive_z_and_aggregate_derivatives<<<num_blocks, threads_per_block>>>(dc->get_arr(), this->w->get_arr(),
                                                                                nxt_dc->get_arr(),
                                                                                dc->get_cnt(), n_cnt);
    }

    delete dc;
    dc = nxt_dc;
}

void LinearLayer::step(int batch_size, float learning_rate)
{
    // Weights:
    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (this->w->get_cnt() / threads_per_block) + 1;
        k_adjust_weight<<<num_blocks, threads_per_block>>>(this->w->get_arr(), this->dw->get_arr(), batch_size, learning_rate,
                                                           (this->w->get_cnt()));
    }

    // Biases:
    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (this->b->get_cnt() / threads_per_block) + 1;
        k_adjust_bias<<<num_blocks, threads_per_block>>>(this->b->get_arr(), this->db->get_arr(), batch_size, learning_rate, this->b->get_cnt());
    }
}

ActivationLayer::ActivationLayer(int n_cnt, ActivationFunction activation_fn)
    : Layer()
{
    this->n = new Tensor(Device::Cuda, n_cnt);
    this->n->reset();

    this->activation_fn = activation_fn;
}

ActivationLayer::~ActivationLayer()
{
}

void ActivationLayer::evaluate(Tensor *nxt_n)
{
    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (this->n->get_cnt() / threads_per_block) + 1;
        k_activate<<<num_blocks, threads_per_block>>>(this->n->get_arr(), nxt_n->get_arr(), this->n->get_cnt(), this->activation_fn);
    }
}

void ActivationLayer::derive(Tensor *dc)
{
    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (this->n->get_cnt() / threads_per_block) + 1;
        k_derive_activation<<<num_blocks, threads_per_block>>>(this->n->get_arr(), dc->get_arr(), this->n->get_cnt(), this->activation_fn);
    }
}
