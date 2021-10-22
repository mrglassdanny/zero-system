#include "tensor.cuh"

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

Tensor *Tensor::one_hot_encode(Device device, int row_cnt, int col_cnt, float *cpu_arr)
{
    Tensor *tensor = new Tensor(device, row_cnt, col_cnt);
    tensor->set_all(0.0f);

    for (int row_idx = 0; row_idx < row_cnt; row_idx++)
    {
        int col_idx = (int)cpu_arr[row_idx];
        if (col_idx >= 0 && col_idx < col_cnt)
        {
            tensor->set_val(row_idx * col_cnt + col_idx, 1.0f);
        }
        // If column index is less than 0 or is greater than or equal to column count, skip it!
        // ^ this shouldn't happen...
    }

    return tensor;
}

Tensor::Tensor(Tensor &src)
{
    this->device = src.device;
    this->shape = src.shape;

    int cnt = this->get_cnt();

    if (src.device == Device::Cpu)
    {
        this->arr = (float *)malloc(sizeof(float) * cnt);
        memcpy(this->arr, src.arr, sizeof(float) * cnt);
    }
    else if (src.device == Device::Cuda)
    {
        cudaMalloc(&this->arr, sizeof(float) * cnt);
        cudaMemcpy(this->arr, src.arr, sizeof(float) * cnt, cudaMemcpyHostToDevice);
    }
}

Tensor::Tensor(Device device)
{
    this->device = device;
    this->shape.push_back(1);

    if (device == Device::Cpu)
    {
        this->arr = (float *)malloc(sizeof(float));
    }
    else if (device == Device::Cuda)
    {
        cudaMalloc(&this->arr, sizeof(float));
    }

    this->reset();
}

Tensor::Tensor(Device device, int cnt)
{
    this->device = device;
    this->shape.push_back(cnt);

    if (device == Device::Cpu)
    {
        this->arr = (float *)malloc(sizeof(float) * cnt);
    }
    else if (device == Device::Cuda)
    {
        cudaMalloc(&this->arr, sizeof(float) * cnt);
    }

    this->reset();
}

Tensor::Tensor(Device device, int row_cnt, int col_cnt)
{
    this->device = device;
    this->shape.push_back(row_cnt);
    this->shape.push_back(col_cnt);

    if (device == Device::Cpu)
    {
        this->arr = (float *)malloc(sizeof(float) * row_cnt * col_cnt);
    }
    else if (device == Device::Cuda)
    {
        cudaMalloc(&this->arr, sizeof(float) * row_cnt * col_cnt);
    }

    this->reset();
}

Tensor::Tensor(Device device, int x_cnt, int y_cnt, int z_cnt)
{
    this->device = device;
    this->shape.push_back(x_cnt);
    this->shape.push_back(y_cnt);
    this->shape.push_back(z_cnt);

    if (device == Device::Cpu)
    {
        this->arr = (float *)malloc(sizeof(float) * x_cnt * y_cnt * z_cnt);
    }
    else if (device == Device::Cuda)
    {
        cudaMalloc(&this->arr, sizeof(float) * x_cnt * y_cnt * z_cnt);
    }

    this->reset();
}

Tensor::Tensor(Device device, int a_cnt, int b_cnt, int c_cnt, int d_cnt)
{
    this->device = device;
    this->shape.push_back(a_cnt);
    this->shape.push_back(b_cnt);
    this->shape.push_back(c_cnt);
    this->shape.push_back(d_cnt);

    if (device == Device::Cpu)
    {
        this->arr = (float *)malloc(sizeof(float) * a_cnt * b_cnt * c_cnt * d_cnt);
    }
    else if (device == Device::Cuda)
    {
        cudaMalloc(&this->arr, sizeof(float) * a_cnt * b_cnt * c_cnt * d_cnt);
    }

    this->reset();
}

Tensor::Tensor(Device device, std::vector<int> shape)
{
    this->device = device;
    this->shape = shape;

    int cnt = this->get_cnt();

    if (device == Device::Cpu)
    {
        this->arr = (float *)malloc(sizeof(float) * cnt);
    }
    else if (device == Device::Cuda)
    {
        cudaMalloc(&this->arr, sizeof(float) * cnt);
    }

    this->reset();
}

Tensor::~Tensor()
{
    if (this->device == Device::Cpu)
    {
        free(this->arr);
    }
    else if (this->device == Device::Cuda)
    {
        cudaFree(this->arr);
    }
}

void Tensor::to(Device device)
{
    int cnt = this->get_cnt();

    if (device == Device::Cpu)
    {
        if (this->device == Device::Cpu)
        {
            return;
        }
        else if (this->device == Device::Cuda)
        {
            float *h_arr = (float *)malloc(sizeof(float) * cnt);
            cudaMemcpy(h_arr, this->arr, sizeof(float) * cnt, cudaMemcpyDeviceToHost);
            cudaFree(this->arr);
            this->arr = h_arr;
        }
    }
    else if (device == Device::Cuda)
    {
        if (this->device == Device::Cpu)
        {
            float *d_arr;
            cudaMalloc(&d_arr, sizeof(float) * cnt);
            cudaMemcpy(d_arr, this->arr, sizeof(float) * cnt, cudaMemcpyHostToDevice);
            free(this->arr);
            this->arr = d_arr;
        }
        else if (this->device == Device::Cuda)
        {
            return;
        }
    }
}

void Tensor::copy(Tensor *src)
{
    this->shape = src->shape;

    int cnt = this->get_cnt();

    if (src->device == Device::Cpu)
    {
        if (this->device == Device::Cpu)
        {
            this->arr = (float *)realloc(this->arr, sizeof(float) * cnt);
            memcpy(this->arr, src->arr, sizeof(float) * cnt);
        }
        else if (this->device == Device::Cuda)
        {
            cudaFree(this->arr);
            this->arr = (float *)malloc(sizeof(float) * cnt);
            memcpy(this->arr, src->arr, sizeof(float) * cnt);
        }
    }
    else if (src->device == Device::Cuda)
    {
        if (this->device == Device::Cpu)
        {
            free(this->arr);
            cudaMalloc(&this->arr, sizeof(float) * cnt);
            cudaMemcpy(this->arr, src->arr, sizeof(float) * cnt, cudaMemcpyHostToDevice);
        }
        else if (this->device == Device::Cuda)
        {
            cudaFree(this->arr);
            cudaMalloc(&this->arr, sizeof(float) * cnt);
            cudaMemcpy(this->arr, src->arr, sizeof(float) * cnt, cudaMemcpyHostToDevice);
        }
    }

    this->device = src->device;
}

void Tensor::print()
{
    Device orig_device = this->device;

    this->to(Device::Cpu);

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

    this->to(orig_device);
}

std::vector<int> Tensor::get_shape()
{
    return this->shape;
}

int Tensor::get_cnt()
{
    int cnt = 1;

    for (int i = 0; i < this->shape.size(); i++)
    {
        cnt *= this->shape[i];
    }

    return cnt;
}

float *Tensor::get_arr()
{
    return this->arr;
}

float *Tensor::get_arr(Device device)
{
    this->to(device);
    return this->arr;
}

void Tensor::set_arr(float *arr)
{
    cudaMemcpy(this->arr, arr, sizeof(float) * this->get_cnt(), cudaMemcpyDefault);
}

float Tensor::get_val(int idx)
{
    if (this->device == Device::Cuda)
    {
        float val;
        cudaMemcpy(&val, &this->arr[idx], sizeof(float), cudaMemcpyDeviceToHost);
        return val;
    }
    else
    {
        return this->arr[idx];
    }
}

void Tensor::set_val(int idx, float val)
{
    if (this->device == Device::Cuda)
    {
        cudaMemcpy(&this->arr[idx], &val, sizeof(float), cudaMemcpyHostToDevice);
    }
    else
    {
        this->arr[idx] = val;
    }
}

void Tensor::reset()
{
    int cnt = this->get_cnt();

    if (this->device == Device::Cpu)
    {
        memset(this->arr, 0, sizeof(float) * cnt);
    }
    else if (this->device == Device::Cuda)
    {
        cudaMemset(this->arr, 0, sizeof(float) * cnt);
    }
}

void Tensor::set_all(float val)
{
    int cnt = this->get_cnt();

    if (this->device == Device::Cpu)
    {
        for (int i = 0; i < cnt; i++)
        {
            this->arr[i] = val;
        }
    }
    else if (this->device == Device::Cuda)
    {
        {
            int threads_per_block = CUDA_THREADS_PER_BLOCK;
            int num_blocks = (cnt / CUDA_THREADS_PER_BLOCK) + 1;
            k_set_arr<<<num_blocks, threads_per_block>>>(this->arr, cnt, val);
        }
    }
}

// TODO: figure out what is up with CUDA random number generator!!!
// void Tensor::set_all_rand(float mean, float stddev)
// {
//     int cnt = this->get_cnt();

//     if (this->device == Device::Cpu)
//     {
//         std::random_device rd;
//         std::mt19937 gen(rd());

//         for (int i = 0; i < cnt; i++)
//         {
//             std::normal_distribution<float> d(mean, stddev);
//             this->arr[i] = d(gen);
//         }
//     }
//     else if (this->device == Device::Cuda)
//     {
//         {
//             int threads_per_block = CUDA_THREADS_PER_BLOCK;
//             int num_blocks = (cnt / CUDA_THREADS_PER_BLOCK) + 1;
//             k_set_arr_rand<<<num_blocks, threads_per_block>>>(this->arr, cnt, mean, stddev);
//         }
//     }
// }

void Tensor::set_all_rand(float mean, float stddev)
{
    int cnt = this->get_cnt();

    if (this->device == Device::Cpu)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int i = 0; i < cnt; i++)
        {
            std::normal_distribution<float> d(mean, stddev);
            this->arr[i] = d(gen);
        }
    }
    else if (this->device == Device::Cuda)
    {
        this->to(Device::Cpu);

        std::random_device rd;
        std::mt19937 gen(rd());

        for (int i = 0; i < cnt; i++)
        {
            std::normal_distribution<float> d(mean, stddev);
            this->arr[i] = d(gen);
        }

        this->to(Device::Cuda);
    }
}
