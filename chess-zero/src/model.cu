#include "model.cuh"

using namespace zero::core;
using namespace zero::nn;

// Device functions:

__device__ float d_chess_mse_cost(float n_val, float y_val)
{
    return ((n_val - y_val) * (n_val - y_val));
}

__device__ float d_derive_chess_mse_cost(float n_val, float y_val)
{
    return 2.0f * (n_val - y_val);
}

__device__ float d_chess_softmax(float val, float *arr, int cnt)
{
    float e_sum_val = 0.0f;

    for (int i = 0; i < cnt; i++)
    {
        e_sum_val += exp(arr[i]);
    }

    return exp(val) / e_sum_val;
}

__device__ float d_chess_cross_entropy_cost(float n_val, float y_val, float *n_arr, int n_cnt)
{
    float np_val = d_chess_softmax(n_val, n_arr, n_cnt);
    return -(y_val * log(np_val));
}

__device__ float d_derive_chess_cross_entropy_cost(float n_val, float y_val, float *n_arr, int n_cnt)
{
    float np_val = d_chess_softmax(n_val, n_arr, n_cnt);
    return np_val - y_val;
}

// Kernel functions:

__global__ void k_chess_cost(float *n_arr, float *y_arr, float *cost_val, int n_cnt, CostFunction cost_fn)
{
    __shared__ float temp[CUDA_THREADS_PER_BLOCK];
    memset(temp, 0, CUDA_THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (cost_fn)
        {
        case MSE:
            temp[threadIdx.x] = d_chess_mse_cost(n_arr[tid], y_arr[tid]);
            break;
        case CrossEntropy:
            temp[threadIdx.x] = d_chess_cross_entropy_cost(n_arr[tid], y_arr[tid], n_arr, n_cnt);
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

        atomicAdd(cost_val, sum);
    }
}

__global__ void k_derive_chess_cost(float *n_arr, float *y_arr, float *dc_arr, int n_cnt, CostFunction cost_fn)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (cost_fn)
        {
        case MSE:
            dc_arr[tid] *= d_derive_chess_mse_cost(n_arr[tid], y_arr[tid]);
            break;
        case CrossEntropy:
            dc_arr[tid] *= d_derive_chess_cross_entropy_cost(n_arr[tid], y_arr[tid], n_arr, n_cnt);
            break;
        default:
            break;
        }
    }
}

// ChessModel functions:

ChessModel::ChessModel(CostFunction cost_fn, float learning_rate)
    : Model(cost_fn, learning_rate)
{
}

ChessModel::ChessModel(const char *path)
    : Model(path)
{
}

Tensor *ChessModel::forward(Tensor *x, bool train_flg)
{
    // Chess legality mask:
    x->to(Device::Cpu);
    reverse_one_hot_encode_board(x->get_arr(), this->board);
    get_src_legality_mask(this->board, true, this->src_legality_mask);

    // Convert to Cuda.
    x->to(Device::Cuda);

    int lst_lyr_idx = this->layers.size() - 1;

    Layer *frst_lyr = this->layers[0];
    Layer *lst_lyr = this->layers[lst_lyr_idx];

    frst_lyr->set_neurons(x);

    for (int i = 0; i < lst_lyr_idx; i++)
    {
        Layer *lyr = this->layers[i];
        Layer *nxt_lyr = this->layers[i + 1];

        lyr->evaluate(nxt_lyr->get_neurons(), train_flg);
    }

    Tensor *pred = new Tensor(Device::Cuda, lst_lyr->get_output_shape());
    lst_lyr->evaluate(pred, train_flg);

    // Chess legality dropout START
    pred->to(Device::Cpu);
    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        pred->get_arr()[i] *= this->src_legality_mask[i];
    }
    pred->to(Device::Cuda);
    // Chess legality dropout END

    return pred;
}

void ChessModel::backward(Tensor *pred, Tensor *y)
{

    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        printf("%f\n", this->src_legality_mask[i]);
    }

    // Convert to Cuda.
    y->to(Device::Cuda);

    Tensor *dc = new Tensor(Device::Cuda, pred->get_shape());
    dc->set_all(1.0f);

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (pred->get_cnt() / threads_per_block) + 1;
        k_derive_chess_cost<<<num_blocks, threads_per_block>>>(pred->get_arr(),
                                                               y->get_arr(), dc->get_arr(), pred->get_cnt(), this->cost_fn);
    }

    // Chess legality dropout START
    dc->to(Device::Cpu);
    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        dc->get_arr()[i] *= this->src_legality_mask[i];
    }
    dc->to(Device::Cuda);
    // Chess legality dropout END

    int lst_lyr_idx = this->layers.size() - 1;

    for (int i = lst_lyr_idx; i >= 0; i--)
    {
        Layer *lyr = this->layers[i];
        dc = lyr->derive(dc);
    }

    delete dc;
}
