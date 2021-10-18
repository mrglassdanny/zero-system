#include "tensor.cuh"

#define TENSOR_GPU_THREADS_PER_BLOCK 32

using namespace zero_v2::core;

// Device functions:

// Kernel functions:

__global__ void k_set_arr(float *arr, int cnt, float val)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        arr[tid] = val;
    }
}

__global__ void k_set_arr_rand(float *arr, int cnt, float mean, float stddev)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        curandState state;
        curand_init(clock64(), tid, 0, &state);
        arr[tid] = curand_log_normal(&state, mean, stddev);
    }
}

// Tensor functions:

Tensor::Tensor(TensorType typ)
{
    this->typ = typ;
    this->shape.push_back(1);

    if (typ == TensorType::Cpu)
    {
        this->arr = (float *)malloc(sizeof(float));
    }
    else if (typ == TensorType::Gpu)
    {
        cudaMalloc(&this->arr, sizeof(float));
    }

    this->reset();
}

Tensor::Tensor(TensorType typ, int cnt)
{
    this->typ = typ;
    this->shape.push_back(cnt);

    if (typ == TensorType::Cpu)
    {
        this->arr = (float *)malloc(sizeof(float) * cnt);
    }
    else if (typ == TensorType::Gpu)
    {
        cudaMalloc(&this->arr, sizeof(float) * cnt);
    }

    this->reset();
}

Tensor::Tensor(TensorType typ, int row_cnt, int col_cnt)
{
    this->typ = typ;
    this->shape.push_back(row_cnt);
    this->shape.push_back(col_cnt);

    if (typ == TensorType::Cpu)
    {
        this->arr = (float *)malloc(sizeof(float) * row_cnt * col_cnt);
    }
    else if (typ == TensorType::Gpu)
    {
        cudaMalloc(&this->arr, sizeof(float) * row_cnt * col_cnt);
    }

    this->reset();
}

Tensor::Tensor(TensorType typ, int x_cnt, int y_cnt, int z_cnt)
{
    this->typ = typ;
    this->shape.push_back(x_cnt);
    this->shape.push_back(y_cnt);
    this->shape.push_back(z_cnt);

    if (typ == TensorType::Cpu)
    {
        this->arr = (float *)malloc(sizeof(float) * x_cnt * y_cnt * z_cnt);
    }
    else if (typ == TensorType::Gpu)
    {
        cudaMalloc(&this->arr, sizeof(float) * x_cnt * y_cnt * z_cnt);
    }

    this->reset();
}

Tensor::~Tensor()
{
    if (this->typ == TensorType::Cpu)
    {
        free(this->arr);
    }
    else if (this->typ == TensorType::Gpu)
    {
        cudaFree(this->arr);
    }
}

int Tensor::get_tot_cnt()
{
    int tot_cnt = 1;

    for (int i = 0; i < this->shape.size(); i++)
    {
        tot_cnt *= this->shape[i];
    }

    return tot_cnt;
}

float *Tensor::get_arr()
{
    return this->arr;
}

void Tensor::translate(TensorType typ)
{
    int tot_cnt = this->get_tot_cnt();

    if (typ == TensorType::Cpu)
    {
        if (this->typ == TensorType::Cpu)
        {
            return;
        }
        else if (this->typ == TensorType::Gpu)
        {
            float *h_arr = (float *)malloc(sizeof(float) * tot_cnt);
            cudaMemcpy(h_arr, this->arr, sizeof(float) * tot_cnt, cudaMemcpyDeviceToHost);
            cudaFree(this->arr);
            this->arr = h_arr;
        }
    }
    else if (typ == TensorType::Gpu)
    {
        if (this->typ == TensorType::Cpu)
        {
            float *d_arr;
            cudaMalloc(&d_arr, sizeof(float) * tot_cnt);
            cudaMemcpy(d_arr, this->arr, sizeof(float) * tot_cnt, cudaMemcpyHostToDevice);
            free(this->arr);
            this->arr = d_arr;
        }
        else if (this->typ == TensorType::Gpu)
        {
            return;
        }
    }
}

void Tensor::reset()
{
    int tot_cnt = this->get_tot_cnt();

    if (this->typ == TensorType::Cpu)
    {
        memset(this->arr, 0, sizeof(float) * tot_cnt);
    }
    else if (this->typ == TensorType::Gpu)
    {
        cudaMemset(this->arr, 0, sizeof(float) * tot_cnt);
    }
}

void Tensor::reset(float val)
{
    int tot_cnt = this->get_tot_cnt();

    if (this->typ == TensorType::Cpu)
    {
        for (int i = 0; i < tot_cnt; i++)
        {
            this->arr[i] = val;
        }
    }
    else if (this->typ == TensorType::Gpu)
    {
        {
            int threads_per_block = TENSOR_GPU_THREADS_PER_BLOCK;
            int num_blocks = (tot_cnt / TENSOR_GPU_THREADS_PER_BLOCK) + 1;
            k_set_arr<<<num_blocks, threads_per_block>>>(this->arr, tot_cnt, val);
        }
    }
}

void Tensor::reset_rand(float mean, float stddev)
{
    int tot_cnt = this->get_tot_cnt();

    if (this->typ == TensorType::Cpu)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int i = 0; i < tot_cnt; i++)
        {
            std::normal_distribution<float> d(mean, stddev);
            this->arr[i] = d(gen);
        }
    }
    else if (this->typ == TensorType::Gpu)
    {
        {
            int threads_per_block = TENSOR_GPU_THREADS_PER_BLOCK;
            int num_blocks = (tot_cnt / TENSOR_GPU_THREADS_PER_BLOCK) + 1;
            k_set_arr_rand<<<num_blocks, threads_per_block>>>(this->arr, tot_cnt, mean, stddev);
        }
    }
}

void Tensor::print()
{
    TensorType orig_typ = this->typ;

    this->translate(TensorType::Cpu);

    switch (this->shape.size())
    {
    case 1:
    {
        int cnt = this->shape[0];
        printf("[ ");
        for (int i = 0; i < cnt; i++)
        {
            if (i == cnt - 1)
            {
                printf("%f", this->arr[i]);
            }
            else
            {
                printf("%f, ", this->arr[i]);
            }
        }
        printf(" ]");
    }

    break;
    case 2:
    {
        int row_cnt = this->shape[0];
        int col_cnt = this->shape[1];

        printf("[");
        for (int i = 0; i < row_cnt; i++)
        {

            if (i == 0)
            {
                printf(" [ ");
            }
            else
            {
                printf("  [ ");
            }

            for (int j = 0; j < col_cnt; j++)
            {
                if (j == col_cnt - 1)
                {
                    printf("%f", this->arr[i * col_cnt + j]);
                }
                else
                {
                    printf("%f, ", this->arr[i * col_cnt + j]);
                }
            }

            if (i == row_cnt - 1)
            {
                printf(" ] ");
            }
            else
            {
                printf(" ],\n");
            }
        }
        printf("]\n");
    }
    break;
    case 3:
    {
        int x_cnt = this->shape[0];
        int y_cnt = this->shape[1];
        int z_cnt = this->shape[2];

        printf("[");
        for (int i = 0; i < x_cnt; i++)
        {

            if (i == 0)
            {
                printf(" [ ");
            }
            else
            {
                printf("  [ ");
            }

            for (int j = 0; j < y_cnt; j++)
            {

                if (j == 0)
                {
                    printf(" [ ");
                }
                else
                {
                    printf("  [ ");
                }

                for (int k = 0; k < z_cnt; k++)
                {
                    if (k == z_cnt - 1)
                    {
                        printf("%f", this->arr[(i * y_cnt * z_cnt) + (j * z_cnt) + k]);
                    }
                    else
                    {
                        printf("%f, ", this->arr[(i * y_cnt * z_cnt) + (j * z_cnt) + k]);
                    }
                }

                if (j == y_cnt - 1)
                {
                    printf(" ] ");
                }
                else
                {
                    printf(" ],\n");
                }
            }

            if (i == x_cnt - 1)
            {
                printf(" ] ");
            }
            else
            {
                printf(" ],\n");
            }
        }
        printf("]\n");
    }
    break;
    default:
        printf("Cannot print Tensor: too many dimensions!\n");
        break;
    }

    this->translate(orig_typ);
}
