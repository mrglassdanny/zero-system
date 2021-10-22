#include "model.cuh"

using namespace zero_v2::core;
using namespace zero_v2::nn;

// Device functions:

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

__global__ void k_cost(float *n_arr, float *y_arr, float *cost, int n_cnt, CostFunction cost_fn)
{
    __shared__ float temp[CUDA_THREADS_PER_BLOCK];
    memset(temp, 0, CUDA_THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (cost_fn)
        {
        case MSE:
            temp[threadIdx.x] = d_mse_cost(n_arr[tid], y_arr[tid]);
            break;
        case CrossEntropy:
            temp[threadIdx.x] = d_cross_entropy_cost(n_arr[tid], y_arr[tid]);
            break;
        default:
            break;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        float sum = 0.0f;

#pragma unroll
        for (int i = 0; i < CUDA_THREADS_PER_BLOCK; i++)
        {
            sum += temp[i];
        }

        atomicAdd(cost, sum);
    }
}

__global__ void k_derive_cost(float *n_arr, float *y_arr, float *agg_derivatives_arr, int n_cnt, CostFunction cost_fn)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (cost_fn)
        {
        case MSE:
            agg_derivatives_arr[tid] *= d_derive_mse_cost(n_arr[tid], y_arr[tid]);
            break;
        case CrossEntropy:
            agg_derivatives_arr[tid] *= d_derive_cross_entropy_cost(n_arr[tid], y_arr[tid]);
            break;
        default:
            break;
        }
    }
}

// Model functions:

Model::Model(CostFunction cost_fn, float learning_rate)
{
    this->cost_fn = cost_fn;
    this->learning_rate = learning_rate;

    cudaMalloc(&this->d_cost, sizeof(float));
    cudaMemset(this->d_cost, 0, sizeof(float));
}

Model::~Model()
{
    for (Layer *lyr : this->layers)
    {
        delete lyr;
    }

    cudaFree(this->d_cost);
}

void Model::load(const char *path)
{
    FILE *file_ptr = fopen(path, "rb");

    fread(&this->cost_fn, sizeof(CostFunction), 1, file_ptr);
    fread(&this->learning_rate, sizeof(float), 1, file_ptr);

    int lyr_cnt = 0;
    fread(&lyr_cnt, sizeof(int), 1, file_ptr);

    for (int i = 0; i < lyr_cnt; i++)
    {
        LayerType lyr_typ;
        fread(&lyr_typ, sizeof(LayerType), 1, file_ptr);

        Layer *lyr = nullptr;

        switch (lyr_typ)
        {
        case LayerType::Linear:
            lyr = new LinearLayer();
            break;
        case LayerType::Activation:
            lyr = new ActivationLayer();
            break;
        case LayerType::Dropout:
            lyr = new DropoutLayer();
            break;
        default:
            break;
        }

        lyr->load(file_ptr);
        this->add_layer(lyr);
    }

    cudaMalloc(&this->d_cost, sizeof(float));
    cudaMemset(this->d_cost, 0, sizeof(float));
}

void Model::save(const char *path)
{
    FILE *file_ptr = fopen(path, "wb");

    fwrite(&this->cost_fn, sizeof(CostFunction), 1, file_ptr);
    fwrite(&this->learning_rate, sizeof(float), 1, file_ptr);

    int lyr_cnt = this->layers.size();
    fwrite(&lyr_cnt, sizeof(int), 1, file_ptr);

    for (Layer *lyr : this->layers)
    {
        LayerType lyr_typ = lyr->get_type();
        fwrite(&lyr_typ, sizeof(LayerType), 1, file_ptr);

        lyr->save(file_ptr);
    }

    fclose(file_ptr);
}

void Model::add_layer(Layer *lyr)
{
    this->layers.push_back(lyr);
}

Tensor *Model::forward(Tensor *x)
{
    int lst_lyr_idx = this->layers.size() - 1;

    Layer *fst_lyr = this->layers[0];
    Layer *lst_lyr = this->layers[lst_lyr_idx];

    fst_lyr->n->copy(x);

    for (int i = 0; i < lst_lyr_idx; i++)
    {
        Layer *lyr = this->layers[i];
        Layer *nxt_lyr = this->layers[i + 1];

        lyr->evaluate(nxt_lyr->n);
    }

    Tensor *pred = new Tensor(Device::Cuda, lst_lyr->n->get_shape());
    lst_lyr->evaluate(pred);

    return pred;
}

float Model::cost(Tensor *pred, Tensor *y)
{
    float h_cost = 0.0f;

    {
        int threads_per_block(CUDA_THREADS_PER_BLOCK);
        int num_blocks((pred->get_cnt() / threads_per_block) + 1);

        k_cost<<<num_blocks, threads_per_block>>>(pred->get_arr(), y->get_arr(),
                                                  this->d_cost, pred->get_cnt(), this->cost_fn);
    }

    cudaMemcpy(&h_cost, this->d_cost, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemset(this->d_cost, 0, sizeof(float));

    return h_cost;
}

void Model::backward(Tensor *pred, Tensor *y)
{
    Tensor *dc = new Tensor(Device::Cuda, pred->get_shape());
    dc->set_all(1.0f);

    // Derive cost (with respect to activation):
    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (pred->get_cnt() / threads_per_block) + 1;
        k_derive_cost<<<num_blocks, threads_per_block>>>(pred->get_arr(),
                                                         y->get_arr(), dc->get_arr(), pred->get_cnt(), this->cost_fn);
    }

    int lst_lyr_idx = this->layers.size() - 1;

    for (int i = lst_lyr_idx; i > 0; i--)
    {
        Layer *lyr = this->layers[i];
        lyr->derive(dc);
    }

    delete dc;
}

void Model::step(int batch_size)
{
    for (Layer *lyr : this->layers)
    {
        if (LearnableLayer *lrn_lyr = dynamic_cast<LearnableLayer *>(lyr))
        {
            lrn_lyr->step(batch_size, this->learning_rate);
        }
    }
}
