#include "Tensor.cuh"

Tensor::Tensor(int row_cnt, int col_cnt, bool gpu_flg)
{
    if (gpu_flg)
    {
        cudaMalloc(&this->arr, sizeof(float) * (row_cnt * col_cnt));
    }
    else
    {
        this->arr = (float *)malloc(sizeof(float) * (row_cnt * col_cnt));
    }

    this->row_cnt = row_cnt;
    this->col_cnt = col_cnt;
    this->gpu_flg = gpu_flg;
}

Tensor::Tensor(Tensor *src)
{
    if (src->gpu_flg)
    {
        cudaMalloc(&this->arr, sizeof(float) * (row_cnt * col_cnt));
        cudaMemcpy(this->arr, src->arr, sizeof(float) * (row_cnt * col_cnt), cudaMemcpyDeviceToDevice);
    }
    else
    {
        this->arr = (float *)malloc(sizeof(float) * (row_cnt * col_cnt));
        memcpy(this->arr, src->arr, sizeof(float) * (row_cnt * col_cnt));
    }

    this->row_cnt = src->row_cnt;
    this->col_cnt = src->col_cnt;
    this->gpu_flg = src->gpu_flg;
}

Tensor::Tensor(int row_cnt, int col_cnt, bool gpu_flg, float *cpu_arr)
{
    if (gpu_flg)
    {
        cudaMalloc(&this->arr, sizeof(float) * (row_cnt * col_cnt));
        cudaMemcpy(this->arr, cpu_arr, sizeof(float) * (row_cnt * col_cnt), cudaMemcpyHostToDevice);
    }
    else
    {
        this->arr = (float *)malloc(sizeof(float) * (row_cnt * col_cnt));
        memcpy(this->arr, cpu_arr, sizeof(float) * (row_cnt * col_cnt));
    }

    this->row_cnt = row_cnt;
    this->col_cnt = col_cnt;
    this->gpu_flg = gpu_flg;
}

Tensor::~Tensor()
{
    if (this->gpu_flg)
    {
        cudaFree(this->arr);
    }
    else
    {
        free(this->arr);
    }
}

void Tensor::translate(bool gpu_flg)
{
    if (gpu_flg)
    {
        if (!this->gpu_flg)
        {
            float *d_arr;
            cudaMalloc(&d_arr, sizeof(float) * (this->row_cnt * this->col_cnt));
            cudaMemcpy(d_arr, this->arr, sizeof(float) * (this->row_cnt * this->col_cnt), cudaMemcpyHostToDevice);
            free(this->arr);
            this->arr = d_arr;

            this->gpu_flg = true;
        }
    }
    else
    {
        if (this->gpu_flg)
        {
            float *h_arr = (float *)malloc(sizeof(float) * (this->row_cnt * this->col_cnt));
            cudaMemcpy(h_arr, this->arr, sizeof(float) * (this->row_cnt * this->col_cnt), cudaMemcpyDeviceToHost);
            cudaFree(this->arr);
            this->arr = h_arr;

            this->gpu_flg = false;
        }
    }
}

float Tensor::get_idx(int idx)
{
    if (this->gpu_flg)
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

float Tensor::get_rowcol(int row_idx, int col_idx)
{
    int idx = row_idx * this->col_cnt + col_idx;
    return this->get_idx(idx);
}

float *Tensor::get_arr(bool gpu_flg)
{
    this->translate(gpu_flg);
    return this->arr;
}

void Tensor::set_idx(int idx, float val)
{
    if (this->gpu_flg)
    {
        cudaMemcpy(&this->arr[idx], &val, sizeof(float), cudaMemcpyHostToDevice);
    }
    else
    {
        this->arr[idx] = val;
    }
}

void Tensor::set_rowcol(int row_idx, int col_idx, float val)
{
    int idx = row_idx * this->col_cnt + col_idx;
    return this->set_idx(idx, val);
}

void Tensor::set_arr(float *arr, bool gpu_flg, bool translate_flg)
{
    bool orig_gpu_flg = this->gpu_flg;

    this->translate(gpu_flg);

    if (gpu_flg)
    {
        cudaMalloc(&this->arr, sizeof(float) * (row_cnt * col_cnt));
        cudaMemcpy(this->arr, arr, sizeof(float) * (row_cnt * col_cnt), cudaMemcpyHostToDevice);
    }
    else
    {
        this->arr = (float *)malloc(sizeof(float) * (row_cnt * col_cnt));
        memcpy(this->arr, arr, sizeof(float) * (row_cnt * col_cnt));
    }

    if (translate_flg)
    {
        this->translate(orig_gpu_flg);
    }
}

void Tensor::set_all(float val)
{
    int tot_cnt = this->row_cnt * this->col_cnt;

    bool orig_gpu_flg = this->gpu_flg;

    this->translate(false);

    for (int i = 0; i < tot_cnt; i++)
    {
        this->arr[i] = val;
    }

    if (orig_gpu_flg)
    {
        this->translate(true);
    }
}

void Tensor::set_all_rand(float upper)
{
    int tot_cnt = this->row_cnt * this->col_cnt;

    bool orig_gpu_flg = this->gpu_flg;

    this->translate(false);

    for (int i = 0; i < tot_cnt; i++)
    {
        float val = (float)rand() / ((float)RAND_MAX);
        val *= (2 * upper);
        val -= upper;
        this->arr[i] = val;
    }

    if (orig_gpu_flg)
    {
        this->translate(true);
    }
}

void Tensor::print()
{
    bool orig_gpu_flg = this->gpu_flg;

    if (orig_gpu_flg == 1)
    {
        this->translate(false);
    }

    {
        printf("[");
        for (int i = 0; i < this->row_cnt; i++)
        {

            if (i == 0)
            {
                printf(" [ ");
            }
            else
            {
                printf("  [ ");
            }

            for (int j = 0; j < this->col_cnt; j++)
            {
                if (j == this->col_cnt - 1)
                {
                    printf("%f", this->arr[i * this->col_cnt + j]);
                }
                else
                {
                    printf("%f, ", this->arr[i * this->col_cnt + j]);
                }
            }

            if (i == this->row_cnt - 1)
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

    this->translate(orig_gpu_flg);
}