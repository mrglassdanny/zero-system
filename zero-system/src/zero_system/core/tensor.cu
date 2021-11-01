#include "tensor.cuh"

using namespace zero::core;

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
        cudaMemcpy(this->arr, src.arr, sizeof(float) * cnt, cudaMemcpyDeviceToDevice);
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

// Geared toward small csv files (under 0.5 GB).
Tensor *Tensor::from_csv(const char *csv_file_name)
{
    FILE *file_ptr = fopen(csv_file_name, "rb");

    fseek(file_ptr, 0L, SEEK_END);
    long long file_size = FileUtils::get_file_size(csv_file_name);
    rewind(file_ptr);

    char *buf = (char *)malloc(file_size + 1);
    memset(buf, 0, file_size + 1);
    fread(buf, 1, file_size, file_ptr);

    fclose(file_ptr);

    int buf_idx = 0;

    int row_cnt = 0;
    int col_cnt = 0;

    while (buf[buf_idx] != '\n')
    {
        if (buf[buf_idx] == ',')
        {
            col_cnt++;
        }

        buf_idx++;
    }

    col_cnt++;
    buf_idx++;

    int lst_row_idx = 0;
    for (int i = buf_idx; i < file_size; i++)
    {
        if (buf[i] == '\n')
        {
            row_cnt++;
            lst_row_idx = i;
        }
    }

    // If file does not end in newline, add to the row count.
    if (lst_row_idx < file_size - 1)
    {
        row_cnt++;
    }

    Tensor *tensor = new Tensor(Device::Cpu, row_cnt, col_cnt);

    char temp_buf[64];
    memset(temp_buf, 0, 64);
    int temp_buf_idx = 0;
    int row_idx = 0;
    int col_idx = 0;

    for (; buf_idx < file_size; buf_idx++)
    {
        while (buf[buf_idx] != ',' && buf[buf_idx] != '\n' && buf_idx < file_size)
        {
            if (buf[buf_idx] != '"')
            {
                temp_buf[temp_buf_idx++] = buf[buf_idx];
            }

            buf_idx++;
        }

        if (buf[buf_idx] == ',')
        {
            tensor->set_val(row_idx * col_cnt + col_idx, (float)atof(temp_buf));
            memset(temp_buf, 0, 64);
            col_idx++;
            temp_buf_idx = 0;
        }
        else if (buf[buf_idx] == '\n')
        {
            tensor->set_val(row_idx * col_cnt + col_idx, (float)atof(temp_buf));
            memset(temp_buf, 0, 64);
            row_idx++;
            col_idx = 0;
            temp_buf_idx = 0;
        }
    }

    // Make sure to grab the last bit before we finish up!
    if (temp_buf_idx > 0)
    {
        tensor->set_val(row_idx * col_cnt + col_idx, (float)atof(temp_buf));
        memset(temp_buf, 0, 64);
        row_idx++;
        col_idx = 0;
        temp_buf_idx = 0;
    }

    free(buf);

    return tensor;
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

    this->device = device;
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
            cudaMemcpy(this->arr, src->arr, sizeof(float) * cnt, cudaMemcpyDeviceToDevice);
        }
        else if (this->device == Device::Cuda)
        {
            cudaFree(this->arr);
            cudaMalloc(&this->arr, sizeof(float) * cnt);
            cudaMemcpy(this->arr, src->arr, sizeof(float) * cnt, cudaMemcpyDeviceToDevice);
        }
    }

    this->device = src->device;
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
        for (int i = 0; i < this->get_cnt(); i++)
        {
            printf("%d: %f\n", i, this->arr[i]);
        }
        break;
    }

    printf("\n");

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

int Tensor::get_cnt(std::vector<int> shape)
{
    int cnt = 1;

    for (int i = 0; i < shape.size(); i++)
    {
        cnt *= shape[i];
    }

    return cnt;
}

int Tensor::get_dim_cnt()
{
    return this->shape.size();
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
    if (this->device == Device::Cpu)
    {
        return this->arr[idx];
    }
    else if (this->device == Device::Cuda)
    {
        float val;
        cudaMemcpy(&val, &this->arr[idx], sizeof(float), cudaMemcpyDeviceToHost);
        return val;
    }

    return 0.0f;
}

void Tensor::set_val(int idx, float val)
{
    if (this->device == Device::Cpu)
    {
        this->arr[idx] = val;
    }
    else if (this->device == Device::Cuda)
    {
        cudaMemcpy(&this->arr[idx], &val, sizeof(float), cudaMemcpyDefault);
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

TensorTuple Tensor::get_min()
{
    TensorTuple tup;

    tup.idx = 0;
    tup.val = FLT_MAX;

    for (int i = 0; i < this->get_cnt(); i++)
    {
        float cur_val = this->get_val(i);
        if (cur_val < tup.val)
        {
            tup.idx = i;
            tup.val = cur_val;
        }
    }

    return tup;
}

TensorTuple Tensor::get_max()
{
    TensorTuple tup;

    tup.idx = 0;
    tup.val = -FLT_MAX;

    for (int i = 0; i < this->get_cnt(); i++)
    {
        float cur_val = this->get_val(i);
        if (cur_val > tup.val)
        {
            tup.idx = i;
            tup.val = cur_val;
        }
    }

    return tup;
}

void Tensor::dump_to_csv(const char *csv_file_name)
{
    int dim_cnt = this->shape.size();

    if (dim_cnt == 1)
    {
        int cnt = this->shape[0];

        FILE *file_ptr = fopen(csv_file_name, "w");

        fprintf(file_ptr, "col\n");

        for (int i = 0; i < cnt; i++)
        {
            fprintf(file_ptr, "%f\n", this->get_val(i));
        }

        fclose(file_ptr);
    }
    else if (dim_cnt == 2)
    {

        int row_cnt = this->shape[0];
        int col_cnt = this->shape[1];

        FILE *file_ptr = fopen(csv_file_name, "w");

        for (int j = 0; j < col_cnt; j++)
        {

            if (j < col_cnt - 1)
            {
                fprintf(file_ptr, "col_%d,", j);
            }
            else
            {
                fprintf(file_ptr, "col_%d", j);
            }
        }
        fprintf(file_ptr, "\n");

        for (int i = 0; i < row_cnt; i++)
        {
            for (int j = 0; j < col_cnt; j++)
            {
                if (j < col_cnt - 1)
                {
                    fprintf(file_ptr, "%f,", this->get_val(i * col_cnt + j));
                }
                else
                {
                    fprintf(file_ptr, "%f", this->get_val(i * col_cnt + j));
                }
            }
            fprintf(file_ptr, "\n");
        }
        fclose(file_ptr);
    }
    else
    {
        return;
    }
}
