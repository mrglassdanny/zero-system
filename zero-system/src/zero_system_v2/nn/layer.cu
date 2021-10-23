#include "layer.cuh"

using namespace zero_v2::core;
using namespace zero_v2::nn;

// Device functions:

__device__ float d_relu(float val)
{
    return val > 0.0f ? val : 0.0f;
}

__device__ float d_derive_relu(float relu_val)
{
    return relu_val > 0.0f ? 1.0f : 0.0f;
}

__device__ float d_sigmoid(float val)
{
    return (1.0 / (1.0 + exp(-val)));
}

__device__ float d_derive_sigmoid(float sigmoid_val)
{
    return (sigmoid_val) * (1.0 - sigmoid_val);
}

__device__ float d_tanh(float val)
{
    return ((exp(val) - exp(-val)) / (exp(val) + exp(-val)));
}

__device__ float d_derive_tanh(float tanh_val)
{
    return (1 - (tanh_val * tanh_val));
}

__device__ float d_sine(float val)
{
    return sin(val);
}

__device__ float d_derive_sine(float sine_val)
{
    return cos(sine_val);
}

__device__ float d_cosine(float val)
{
    return cos(val);
}

__device__ float d_derive_cosine(float cosine_val)
{
    return -sin(cosine_val);
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

__global__ void k_lin_derive_z_and_increment_weight_derivative(float *dc_arr, float *n_arr, float *dw_arr, int dc_cnt, int n_cnt)
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

__global__ void k_lin_derive_z_and_increment_bias_derivative(float *dc_arr, float *db_arr, int dc_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dc_cnt)
    {
        db_arr[tid] += (dc_arr[tid]);
    }
}

__global__ void k_lin_derive_z_and_aggregate_derivatives(float *dc_arr, float *w_arr, float *nxt_dc_arr, int n_cnt, int nxt_n_cnt)
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
        temp[threadIdx.x] = (dc_arr[n_idx] * w_arr[w_idx]);
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
                atomicAdd(&nxt_dc_arr[lower_idx], sum);
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

                atomicAdd(&nxt_dc_arr[lower_idx], sums[0]);
                if (upper_idx < nxt_n_cnt)
                {
                    atomicAdd(&nxt_dc_arr[upper_idx], sums[1]);
                }
            }
        }
        else
        {

#pragma unroll
            for (int i = 0; i < CUDA_THREADS_PER_BLOCK; i++)
            {
                atomicAdd(&nxt_dc_arr[(tid + i) / n_cnt], temp[i]);
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

__global__ void k_convolve(float *n_arr, float *w_arr, float *b_arr, float *nxt_n_arr, int chan_cnt, int n_row_cnt, int n_col_cnt,
                           int w_row_cnt, int w_col_cnt, int nxt_n_row_cnt, int nxt_n_col_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int nxt_n_idx = tid;

    if (nxt_n_idx < (nxt_n_row_cnt * nxt_n_col_cnt))
    {
        int nxt_n_row_idx = nxt_n_idx / nxt_n_col_cnt;
        int nxt_n_col_idx = nxt_n_idx % nxt_n_col_cnt;

#pragma unroll
        for (int chan_idx = 0; chan_idx < chan_cnt; chan_idx++)
        {
#pragma unroll
            for (int w_row_idx = 0; w_row_idx < w_row_cnt; w_row_idx++)
            {
#pragma unroll
                for (int w_col_idx = 0; w_col_idx < w_col_cnt; w_col_idx++)
                {
                    int n_row_idx = nxt_n_row_idx + w_row_idx;
                    int n_col_idx = nxt_n_col_idx + w_col_idx;

                    int w_rot_val_idx = (w_row_cnt * w_col_cnt) - (w_row_idx * w_col_cnt + w_col_idx);

                    float val = n_arr[(chan_idx * n_row_cnt * n_col_cnt) + (n_row_idx * n_col_cnt) + n_col_idx];
                    val *= w_arr[(chan_idx * w_row_cnt * w_col_cnt) + w_rot_val_idx];
                    nxt_n_arr[nxt_n_idx] += val;
                }
            }
        }

        nxt_n_arr[nxt_n_idx] += b_arr[nxt_n_idx];
    }
}

__global__ void k_conv_derive_z_and_increment_filter_derivative(float *dc_arr, float *n_arr, float *dw_arr, int chan_cnt, int n_row_cnt, int n_col_cnt,
                                                                int w_row_cnt, int w_col_cnt, int dc_row_cnt, int dc_col_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int w_global_val_idx = tid;
    int w_global_cnt = (chan_cnt * w_row_cnt * w_col_cnt);

    if (w_global_val_idx < w_global_cnt)
    {
        int w_cnt = (w_row_cnt * w_col_cnt);
        int chan_idx = w_global_val_idx / w_cnt;
        int w_val_idx = w_global_val_idx - (chan_idx * w_cnt);
        int w_rot_val_idx = w_cnt - w_val_idx;
        int w_row_idx = w_val_idx / w_col_cnt;
        int w_col_idx = w_val_idx % w_col_cnt;
        int w_global_rot_val_idx = (chan_idx * w_cnt) + w_rot_val_idx;

        float val = 0.0f;

#pragma unroll
        for (int dc_row_idx = 0; dc_row_idx < dc_row_cnt; dc_row_idx++)
        {
            int n_row_idx = (dc_row_idx + w_row_idx);

#pragma unroll
            for (int dc_col_idx = 0; dc_col_idx < dc_col_cnt; dc_col_idx++)
            {
                int n_col_idx = (dc_col_idx + w_col_idx);

                val += (dc_arr[(dc_row_idx * dc_col_cnt + dc_col_idx)] * n_arr[(chan_idx * n_row_cnt * n_col_cnt) + (n_row_idx * n_col_cnt) + n_col_idx]);
            }
        }

        dw_arr[w_global_rot_val_idx] += val;
    }
}

__global__ void k_conv_derive_z_and_increment_bias_derivative(float *dc_arr, float *db_arr, int dc_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dc_cnt)
    {
        db_arr[tid] += (dc_arr[tid]);
    }
}

__global__ void k_conv_derive_z_and_aggregate_derivatives(float *dc_arr, float *w_arr, float *nxt_dc_arr,
                                                          int dc_row_cnt, int dc_col_cnt, int w_row_cnt, int w_col_cnt,
                                                          int chan_cnt, int nxt_dc_row_cnt, int nxt_dc_col_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int nxt_dc_global_idx = tid;
    int nxt_dc_global_cnt = (chan_cnt * nxt_dc_row_cnt * nxt_dc_col_cnt);

    if (nxt_dc_global_idx < nxt_dc_global_cnt)
    {
        int nxt_dc_cnt = (nxt_dc_row_cnt * nxt_dc_col_cnt);
        int chan_idx = nxt_dc_global_idx / nxt_dc_cnt;
        int nxt_dc_idx = nxt_dc_global_idx - (chan_idx * nxt_dc_cnt);
        int nxt_dc_row_idx = nxt_dc_idx / nxt_dc_col_cnt;
        int nxt_dc_col_idx = nxt_dc_idx % nxt_dc_col_cnt;
        int w_cnt = w_row_cnt * w_col_cnt;

        float val = 0.0f;

#pragma unroll
        for (int w_row_idx = 0; w_row_idx < w_row_cnt; w_row_idx++)
        {
#pragma unroll
            for (int w_col_idx = 0; w_col_idx < w_col_cnt; w_col_idx++)
            {
                int dc_row_idx = nxt_dc_row_idx - w_row_idx;
                int dc_col_idx = nxt_dc_col_idx - w_col_idx;

                int w_rot_val_idx = w_cnt - (w_row_idx * w_col_cnt + w_col_idx);

                if (dc_row_idx >= 0 && dc_row_idx < dc_row_cnt && dc_col_idx >= 0 && dc_col_idx < dc_col_cnt)
                {
                    val += (dc_arr[dc_row_idx * dc_col_cnt + dc_col_idx] * w_arr[(chan_idx * w_row_cnt * w_col_cnt) + w_rot_val_idx]);
                }
            }
        }

        nxt_dc_arr[nxt_dc_global_idx] += val;
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
            nxt_n_arr[tid] = n_arr[tid];
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
            dc_arr[tid] *= d_derive_relu(d_relu(n_arr[tid]));
            break;
        case ActivationFunction::Sigmoid:
            dc_arr[tid] *= d_derive_sigmoid(d_sigmoid(n_arr[tid]));
            break;
        case ActivationFunction::Tanh:
            dc_arr[tid] *= d_derive_tanh(d_tanh(n_arr[tid]));
            break;
        case ActivationFunction::Sine:
            dc_arr[tid] *= d_derive_sine(d_sine(n_arr[tid]));
            break;
        case ActivationFunction::Cosine:
            dc_arr[tid] *= d_derive_cosine(d_cosine(n_arr[tid]));
            break;
        default:
            // None
            break;
        }
    }
}

__global__ void k_set_dropout_mask(float *dropout_mask_arr, int dropout_mask_cnt, float dropout_rate)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dropout_mask_cnt)
    {
        curandState state;
        curand_init(clock64(), tid, 0, &state);

        if (curand_uniform(&state) <= dropout_rate)
        {
            dropout_mask_arr[tid] = 0.0f;
        }
        else
        {
            dropout_mask_arr[tid] = 1.0f;
        }
    }
}

__global__ void k_dropout(float *n_arr, float *dropout_mask_arr, float *nxt_n_arr, int n_cnt, float dropout_rate)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        nxt_n_arr[tid] = (n_arr[tid] * dropout_mask_arr[tid]) * (1.0f / (1.0f - dropout_rate));
    }
}

__global__ void k_derive_dropout(float *dc_arr, float *dropout_mask_arr, int n_cnt, float dropout_rate)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        dc_arr[tid] *= (dropout_mask_arr[tid] * (1.0f / (1.0f - dropout_rate)));
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

void Layer::evaluate(Tensor *nxt_n, bool train_flg)
{
    nxt_n->reset();
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

LinearLayer::LinearLayer()
    : LearnableLayer() {}

LinearLayer::LinearLayer(int n_cnt, int nxt_n_cnt, InitializationFunction init_fn)
    : LearnableLayer()
{
    this->n = new Tensor(Device::Cuda, n_cnt);
    this->n->reset();

    this->w = new Tensor(Device::Cuda, nxt_n_cnt, n_cnt);
    Initializer::initialize(init_fn, this->w, n_cnt, nxt_n_cnt);

    this->b = new Tensor(Device::Cuda, nxt_n_cnt);
    Initializer::initialize(init_fn, this->b, nxt_n_cnt, 0);

    this->dw = new Tensor(Device::Cuda, nxt_n_cnt, n_cnt);
    this->dw->reset();

    this->db = new Tensor(Device::Cuda, nxt_n_cnt);
    this->db->reset();
}

LinearLayer::LinearLayer(std::vector<int> n_shape, int nxt_n_cnt, InitializationFunction init_fn)
    : LearnableLayer()
{
    int n_cnt = Tensor::get_cnt(n_shape);

    this->n = new Tensor(Device::Cuda, n_cnt);
    this->n->reset();

    this->w = new Tensor(Device::Cuda, nxt_n_cnt, n_cnt);
    Initializer::initialize(init_fn, this->w, n_cnt, nxt_n_cnt);

    this->b = new Tensor(Device::Cuda, nxt_n_cnt);
    Initializer::initialize(init_fn, this->b, nxt_n_cnt, 0);

    this->dw = new Tensor(Device::Cuda, nxt_n_cnt, n_cnt);
    this->dw->reset();

    this->db = new Tensor(Device::Cuda, nxt_n_cnt);
    this->db->reset();
}

LinearLayer::~LinearLayer() {}

LayerType LinearLayer::get_type()
{
    return LayerType::Linear;
}

std::vector<int> LinearLayer::get_input_shape()
{
    return this->n->get_shape();
}

std::vector<int> LinearLayer::get_output_shape()
{
    return this->b->get_shape();
}

void LinearLayer::evaluate(Tensor *nxt_n, bool train_flg)
{
    Layer::evaluate(nxt_n, train_flg);

    int n_cnt = this->n->get_cnt();
    int nxt_n_cnt = nxt_n->get_cnt();

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = ((n_cnt * nxt_n_cnt) / threads_per_block) + 1;
        k_dot<<<num_blocks, threads_per_block>>>(this->n->get_arr(), w->get_arr(),
                                                 nxt_n->get_arr(), n_cnt, nxt_n_cnt);
    }

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (nxt_n_cnt / threads_per_block) + 1;
        k_add_bias<<<num_blocks, threads_per_block>>>(this->b->get_arr(), nxt_n->get_arr(),
                                                      nxt_n_cnt);
    }
}

Tensor *LinearLayer::derive(Tensor *dc)
{
    int dc_cnt = dc->get_cnt();
    int n_cnt = this->n->get_cnt();

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = ((dc_cnt * n_cnt) / threads_per_block) + 1;
        k_lin_derive_z_and_increment_weight_derivative<<<num_blocks, threads_per_block>>>(dc->get_arr(),
                                                                                          this->n->get_arr(),
                                                                                          this->dw->get_arr(),
                                                                                          dc_cnt, n_cnt);
    }

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (dc_cnt / threads_per_block) + 1;
        k_lin_derive_z_and_increment_bias_derivative<<<num_blocks, threads_per_block>>>(dc->get_arr(), this->db->get_arr(), dc_cnt);
    }

    Tensor *nxt_dc = new Tensor(Device::Cuda, n_cnt);
    nxt_dc->reset();

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = ((n_cnt * dc_cnt) / threads_per_block) + 1;
        k_lin_derive_z_and_aggregate_derivatives<<<num_blocks, threads_per_block>>>(dc->get_arr(), this->w->get_arr(),
                                                                                    nxt_dc->get_arr(),
                                                                                    dc_cnt, n_cnt);
    }

    delete dc;
    dc = nxt_dc;

    return dc;
}

void LinearLayer::step(int batch_size, float learning_rate)
{
    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (this->w->get_cnt() / threads_per_block) + 1;
        k_adjust_weight<<<num_blocks, threads_per_block>>>(this->w->get_arr(), this->dw->get_arr(), batch_size, learning_rate,
                                                           (this->w->get_cnt()));
    }

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (this->b->get_cnt() / threads_per_block) + 1;
        k_adjust_bias<<<num_blocks, threads_per_block>>>(this->b->get_arr(), this->db->get_arr(), batch_size, learning_rate, this->b->get_cnt());
    }
}

void LinearLayer::load(FILE *file_ptr)
{
    int w_cnt = this->w->get_cnt();
    int b_cnt = this->b->get_cnt();

    float *w_buf = (float *)malloc(sizeof(float) * w_cnt);
    fread(w_buf, sizeof(float), w_cnt, file_ptr);
    this->w->set_arr(w_buf);
    this->w->to(Device::Cuda);
    free(w_buf);

    float *b_buf = (float *)malloc(sizeof(float) * b_cnt);
    fread(b_buf, sizeof(float), b_cnt, file_ptr);
    this->b->set_arr(b_buf);
    this->b->to(Device::Cuda);
    free(b_buf);
}

void LinearLayer::save(FILE *file_ptr)
{
    fwrite(this->w->get_arr(Device::Cpu), sizeof(float), this->w->get_cnt(), file_ptr);
    fwrite(this->b->get_arr(Device::Cpu), sizeof(float), this->b->get_cnt(), file_ptr);
}

// ConvolutionalLayer functions:

ConvolutionalLayer::ConvolutionalLayer()
    : LearnableLayer() {}

ConvolutionalLayer::ConvolutionalLayer(int chan_cnt, int n_row_cnt, int n_col_cnt,
                                       int fltr_cnt, int w_row_cnt, int w_col_cnt,
                                       InitializationFunction init_fn)
    : LearnableLayer()
{
    int nxt_n_row_cnt = n_row_cnt - w_row_cnt + 1;
    int nxt_n_col_cnt = n_col_cnt - w_col_cnt + 1;

    this->n = new Tensor(Device::Cuda, chan_cnt, n_row_cnt, n_col_cnt);
    this->n->reset();

    this->w = new Tensor(Device::Cuda, fltr_cnt, chan_cnt, w_row_cnt, w_col_cnt);
    Initializer::initialize(init_fn, this->w, n_row_cnt * n_col_cnt, 0);

    this->b = new Tensor(Device::Cuda, fltr_cnt, nxt_n_row_cnt, nxt_n_col_cnt);
    Initializer::initialize(init_fn, this->b, n_row_cnt * n_col_cnt, 0);

    this->dw = new Tensor(Device::Cuda, fltr_cnt, chan_cnt, w_row_cnt, w_col_cnt);
    this->dw->reset();

    this->db = new Tensor(Device::Cuda, fltr_cnt, nxt_n_row_cnt, nxt_n_col_cnt);
    this->db->reset();
}

ConvolutionalLayer::ConvolutionalLayer(std::vector<int> n_shape,
                                       int fltr_cnt, int w_row_cnt, int w_col_cnt,
                                       InitializationFunction init_fn)
    : LearnableLayer()
{
    int chan_cnt = n_shape[0];
    int n_row_cnt = n_shape[1];
    int n_col_cnt = n_shape[2];
    int nxt_n_row_cnt = n_row_cnt - w_row_cnt + 1;
    int nxt_n_col_cnt = n_col_cnt - w_col_cnt + 1;

    this->n = new Tensor(Device::Cuda, n_shape);
    this->n->reset();

    this->w = new Tensor(Device::Cuda, fltr_cnt, chan_cnt, w_row_cnt, w_col_cnt);
    Initializer::initialize(init_fn, this->w, n_row_cnt * n_col_cnt, 0);

    this->b = new Tensor(Device::Cuda, fltr_cnt, nxt_n_row_cnt, nxt_n_col_cnt);
    Initializer::initialize(init_fn, this->b, n_row_cnt * n_col_cnt, 0);

    this->dw = new Tensor(Device::Cuda, fltr_cnt, chan_cnt, w_row_cnt, w_col_cnt);
    this->dw->reset();

    this->db = new Tensor(Device::Cuda, fltr_cnt, nxt_n_row_cnt, nxt_n_col_cnt);
    this->db->reset();
}

ConvolutionalLayer::~ConvolutionalLayer()
{
}

LayerType ConvolutionalLayer::get_type()
{
    return LayerType::Convolutional;
}

std::vector<int> ConvolutionalLayer::get_input_shape()
{
    return this->n->get_shape();
}

std::vector<int> ConvolutionalLayer::get_output_shape()
{
    return this->b->get_shape();
}

void ConvolutionalLayer::evaluate(Tensor *nxt_n, bool train_flg)
{

    Layer::evaluate(nxt_n, train_flg);

    int fltr_cnt = this->w->get_shape()[0];
    int chan_cnt = this->w->get_shape()[1];
    int w_row_cnt = this->w->get_shape()[2];
    int w_col_cnt = this->w->get_shape()[3];
    int n_row_cnt = this->n->get_shape()[1];
    int n_col_cnt = this->n->get_shape()[2];
    int nxt_n_row_cnt = nxt_n->get_shape()[1];
    int nxt_n_col_cnt = nxt_n->get_shape()[2];

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = ((nxt_n->get_cnt() / fltr_cnt) / threads_per_block) + 1;

        for (int fltr_idx = 0; fltr_idx < fltr_cnt; fltr_idx++)
        {
            float *n_arr = this->n->get_arr();
            float *w_arr = &this->w->get_arr()[fltr_idx * chan_cnt * w_row_cnt * w_col_cnt];
            float *b_arr = &this->w->get_arr()[fltr_idx * nxt_n_row_cnt * nxt_n_col_cnt];
            float *nxt_n_arr = &nxt_n->get_arr()[fltr_idx * nxt_n_row_cnt * nxt_n_col_cnt];

            k_convolve<<<num_blocks, threads_per_block>>>(n_arr, w_arr, b_arr, nxt_n_arr,
                                                          chan_cnt, n_row_cnt, n_col_cnt, w_row_cnt, w_col_cnt,
                                                          nxt_n_row_cnt, nxt_n_col_cnt);
        }
    }
}

Tensor *ConvolutionalLayer::derive(Tensor *dc)
{
    int fltr_cnt = this->w->get_shape()[0];
    int chan_cnt = this->w->get_shape()[1];
    int w_row_cnt = this->w->get_shape()[2];
    int w_col_cnt = this->w->get_shape()[3];
    int n_row_cnt = this->n->get_shape()[1];
    int n_col_cnt = this->n->get_shape()[2];
    int dc_row_cnt = dc->get_shape()[1];
    int dc_col_cnt = dc->get_shape()[2];

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = ((chan_cnt * w_row_cnt * w_col_cnt) / threads_per_block) + 1;

        for (int fltr_idx = 0; fltr_idx < fltr_cnt; fltr_idx++)
        {
            float *dc_arr = &dc->get_arr()[fltr_idx * dc_row_cnt * dc_col_cnt];
            float *n_arr = this->n->get_arr();
            float *dw_arr = &this->dw->get_arr()[fltr_idx * chan_cnt * w_row_cnt * w_col_cnt];

            k_conv_derive_z_and_increment_filter_derivative<<<num_blocks, threads_per_block>>>(dc_arr, n_arr, dw_arr,
                                                                                               chan_cnt, n_row_cnt, n_col_cnt,
                                                                                               w_row_cnt, w_col_cnt, dc_row_cnt, dc_col_cnt);
        }
    }

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = ((dc_row_cnt * dc_col_cnt) / threads_per_block) + 1;

        for (int fltr_idx = 0; fltr_idx < fltr_cnt; fltr_idx++)
        {
            float *dc_arr = &dc->get_arr()[fltr_idx * dc_row_cnt * dc_col_cnt];
            float *db_arr = &this->db->get_arr()[fltr_idx * dc_row_cnt * dc_col_cnt];

            k_conv_derive_z_and_increment_bias_derivative<<<num_blocks, threads_per_block>>>(dc_arr, db_arr, dc_row_cnt * dc_col_cnt);
        }
    }

    {
        Tensor *nxt_dc = new Tensor(Device::Cuda, chan_cnt, n_row_cnt, n_col_cnt);
        nxt_dc->reset();

        int nxt_dc_row_cnt = nxt_dc->get_shape()[1];
        int nxt_dc_col_cnt = nxt_dc->get_shape()[2];

        {
            int threads_per_block = CUDA_THREADS_PER_BLOCK;
            int num_blocks = (this->n->get_cnt() / threads_per_block) + 1;

            for (int fltr_idx = 0; fltr_idx < fltr_cnt; fltr_idx++)
            {
                float *dc_arr = &dc->get_arr()[fltr_idx * dc_row_cnt * dc_col_cnt];
                float *w_arr = &this->w->get_arr()[fltr_idx * chan_cnt * w_row_cnt * w_col_cnt];
                float *nxt_dc_arr = nxt_dc->get_arr();

                k_conv_derive_z_and_aggregate_derivatives<<<num_blocks, threads_per_block>>>(dc_arr, w_arr, nxt_dc_arr,
                                                                                             dc_row_cnt, dc_col_cnt,
                                                                                             w_row_cnt, w_col_cnt,
                                                                                             chan_cnt, nxt_dc_row_cnt, nxt_dc_col_cnt);
            }
        }

        delete dc;
        dc = nxt_dc;
    }

    return dc;
}

void ConvolutionalLayer::step(int batch_size, float learning_rate)
{
    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (this->w->get_cnt() / threads_per_block) + 1;
        k_adjust_weight<<<num_blocks, threads_per_block>>>(this->w->get_arr(), this->dw->get_arr(), batch_size, learning_rate,
                                                           (this->w->get_cnt()));
    }

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (this->b->get_cnt() / threads_per_block) + 1;
        k_adjust_bias<<<num_blocks, threads_per_block>>>(this->b->get_arr(), this->db->get_arr(), batch_size, learning_rate, this->b->get_cnt());
    }
}

void ConvolutionalLayer::load(FILE *file_ptr)
{
    int w_cnt = this->w->get_cnt();
    int b_cnt = this->b->get_cnt();

    float *w_buf = (float *)malloc(sizeof(float) * w_cnt);
    fread(w_buf, sizeof(float), w_cnt, file_ptr);
    this->w->set_arr(w_buf);
    this->w->to(Device::Cuda);
    free(w_buf);

    float *b_buf = (float *)malloc(sizeof(float) * b_cnt);
    fread(b_buf, sizeof(float), b_cnt, file_ptr);
    this->b->set_arr(b_buf);
    this->b->to(Device::Cuda);
    free(b_buf);
}

void ConvolutionalLayer::save(FILE *file_ptr)
{
    fwrite(this->w->get_arr(Device::Cpu), sizeof(float), this->w->get_cnt(), file_ptr);
    fwrite(this->b->get_arr(Device::Cpu), sizeof(float), this->b->get_cnt(), file_ptr);
}

// ActivationLayer functions:

ActivationLayer::ActivationLayer()
    : Layer() {}

ActivationLayer::ActivationLayer(int n_cnt, ActivationFunction activation_fn)
    : Layer()
{
    this->n = new Tensor(Device::Cuda, n_cnt);
    this->n->reset();

    this->activation_fn = activation_fn;
}

ActivationLayer::ActivationLayer(std::vector<int> n_shape, ActivationFunction activation_fn)
    : Layer()
{
    this->n = new Tensor(Device::Cuda, n_shape);
    this->n->reset();

    this->activation_fn = activation_fn;
}

ActivationLayer::~ActivationLayer()
{
}

LayerType ActivationLayer::get_type()
{
    return LayerType::Activation;
}

std::vector<int> ActivationLayer::get_input_shape()
{
    return this->n->get_shape();
}

std::vector<int> ActivationLayer::get_output_shape()
{
    return this->n->get_shape();
}

void ActivationLayer::evaluate(Tensor *nxt_n, bool train_flg)
{
    Layer::evaluate(nxt_n, train_flg);

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (this->n->get_cnt() / threads_per_block) + 1;
        k_activate<<<num_blocks, threads_per_block>>>(this->n->get_arr(), nxt_n->get_arr(), this->n->get_cnt(), this->activation_fn);
    }
}

Tensor *ActivationLayer::derive(Tensor *dc)
{
    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (this->n->get_cnt() / threads_per_block) + 1;
        k_derive_activation<<<num_blocks, threads_per_block>>>(this->n->get_arr(), dc->get_arr(), this->n->get_cnt(), this->activation_fn);
    }

    return dc;
}

void ActivationLayer::load(FILE *file_ptr)
{
    fread(&this->activation_fn, sizeof(ActivationFunction), 1, file_ptr);
}

void ActivationLayer::save(FILE *file_ptr)
{
    fwrite(&this->activation_fn, sizeof(ActivationFunction), 1, file_ptr);
}

// DropoutLayer functions:

DropoutLayer::DropoutLayer()
    : Layer() {}

DropoutLayer::DropoutLayer(int n_cnt, float dropout_rate)
    : Layer()
{
    this->n = new Tensor(Device::Cuda, n_cnt);
    this->n->reset();

    this->dropout_rate = dropout_rate;

    this->dropout_mask = new Tensor(Device::Cuda, n_cnt);
}

DropoutLayer::DropoutLayer(std::vector<int> n_shape, float dropout_rate)
    : Layer()
{
    this->n = new Tensor(Device::Cuda, n_shape);
    this->n->reset();

    this->dropout_rate = dropout_rate;

    this->dropout_mask = new Tensor(Device::Cuda, n_shape);
}

DropoutLayer::~DropoutLayer()
{
    delete this->dropout_mask;
}

LayerType DropoutLayer::get_type()
{
    return LayerType::Dropout;
}

std::vector<int> DropoutLayer::get_input_shape()
{
    return this->n->get_shape();
}

std::vector<int> DropoutLayer::get_output_shape()
{
    return this->n->get_shape();
}

void DropoutLayer::evaluate(Tensor *nxt_n, bool train_flg)
{
    Layer::evaluate(nxt_n, train_flg);

    if (train_flg)
    {
        {
            int threads_per_block = CUDA_THREADS_PER_BLOCK;
            int num_blocks = (this->dropout_mask->get_cnt() / threads_per_block) + 1;
            k_set_dropout_mask<<<num_blocks, threads_per_block>>>(this->dropout_mask->get_arr(), this->dropout_mask->get_cnt(),
                                                                  this->dropout_rate);
        }

        if (this->dropout_rate > 0.0f)
        {
            int threads_per_block = CUDA_THREADS_PER_BLOCK;
            int num_blocks((nxt_n->get_cnt() / threads_per_block) + 1);
            k_dropout<<<num_blocks, threads_per_block>>>(this->n->get_arr(), this->dropout_mask->get_arr(), nxt_n->get_arr(),
                                                         nxt_n->get_cnt(), this->dropout_rate);
        }
    }
}

Tensor *DropoutLayer::derive(Tensor *dc)
{
    {
        if (this->dropout_rate > 0.0f)
        {
            int threads_per_block = CUDA_THREADS_PER_BLOCK;
            int num_blocks = (this->n->get_cnt() / threads_per_block) + 1;
            k_derive_dropout<<<num_blocks, threads_per_block>>>(dc->get_arr(), this->dropout_mask->get_arr(),
                                                                this->n->get_cnt(), this->dropout_rate);
        }
    }

    return dc;
}

void DropoutLayer::load(FILE *file_ptr)
{
    fread(&this->dropout_rate, sizeof(float), 1, file_ptr);
}

void DropoutLayer::save(FILE *file_ptr)
{
    fwrite(&this->dropout_rate, sizeof(float), 1, file_ptr);
}