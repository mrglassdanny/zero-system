#include "Tensor.cuh"

using namespace zero::core;

Tensor *Tensor::one_hot_encode(int row_cnt, int col_cnt, TensorType typ, float *cpu_arr)
{
    Tensor *tensor = new Tensor(row_cnt, col_cnt, typ);
    tensor->set_all(0.0f);

    for (int i = 0; i < row_cnt; i++)
    {
        int col_idx = (int)cpu_arr[i];
        if (col_idx < col_cnt)
        {
            tensor->set_rowcol(i, col_idx, 1.0f);
        }
        // If column index is greater than or equal to column count, skip it!
        // ^ this shouldn't happen...
    }

    return tensor;
}

// Geared toward small csv files (under 0.5 GB).
Tensor *Tensor::from_csv(const char *csv_file_name)
{
    FILE *file_ptr = fopen(csv_file_name, "rb");

    fseek(file_ptr, 0L, SEEK_END);
    long file_size = ftell(file_ptr);
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

    Tensor *tensor = new Tensor(row_cnt, col_cnt, Cpu);

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
            tensor->set_rowcol(row_idx, col_idx, (float)atof(temp_buf));
            memset(temp_buf, 0, 64);
            col_idx++;
            temp_buf_idx = 0;
        }
        else if (buf[buf_idx] == '\n')
        {
            tensor->set_rowcol(row_idx, col_idx, (float)atof(temp_buf));
            memset(temp_buf, 0, 64);
            row_idx++;
            col_idx = 0;
            temp_buf_idx = 0;
        }
    }

    // Make sure to grab the last bit before we finish up!
    if (temp_buf_idx > 0)
    {
        tensor->set_rowcol(row_idx, col_idx, (float)atof(temp_buf));
        memset(temp_buf, 0, 64);
        row_idx++;
        col_idx = 0;
        temp_buf_idx = 0;
    }

    free(buf);

    return tensor;
}

Tensor::Tensor(int row_cnt, int col_cnt, TensorType typ)
{
    if (typ == Gpu)
    {
        cudaMalloc(&this->arr, sizeof(float) * (row_cnt * col_cnt));
    }
    else
    {
        this->arr = (float *)malloc(sizeof(float) * (row_cnt * col_cnt));
    }

    this->row_cnt = row_cnt;
    this->col_cnt = col_cnt;
    this->typ = typ;
}

Tensor::Tensor(const Tensor &src)
{
    if (src.typ == Gpu)
    {
        cudaMalloc(&this->arr, sizeof(float) * (src.row_cnt * src.col_cnt));
        cudaMemcpy(this->arr, src.arr, sizeof(float) * (src.row_cnt * src.col_cnt), cudaMemcpyDeviceToDevice);
    }
    else
    {
        this->arr = (float *)malloc(sizeof(float) * (src.row_cnt * src.col_cnt));
        memcpy(this->arr, src.arr, sizeof(float) * (src.row_cnt * src.col_cnt));
    }

    this->row_cnt = src.row_cnt;
    this->col_cnt = src.col_cnt;
    this->typ = src.typ;
}

Tensor::Tensor(const Tensor &src, TensorType typ)
{

    if (src.typ == Gpu && typ == Gpu)
    {
        cudaMalloc(&this->arr, sizeof(float) * (src.row_cnt * src.col_cnt));
        cudaMemcpy(this->arr, src.arr, sizeof(float) * (src.row_cnt * src.col_cnt), cudaMemcpyDeviceToDevice);
    }
    else if (src.typ == Cpu && typ == Cpu)
    {
        this->arr = (float *)malloc(sizeof(float) * (src.row_cnt * src.col_cnt));
        memcpy(this->arr, src.arr, sizeof(float) * (src.row_cnt * src.col_cnt));
    }
    else if (src.typ == Cpu && typ == Gpu)
    {
        cudaMalloc(&this->arr, sizeof(float) * (src.row_cnt * src.col_cnt));
        cudaMemcpy(this->arr, src.arr, sizeof(float) * (src.row_cnt * src.col_cnt), cudaMemcpyHostToDevice);
    }
    else if (src.typ == Gpu && typ == Cpu)
    {
        this->arr = (float *)malloc(sizeof(float) * (src.row_cnt * src.col_cnt));
        cudaMemcpy(this->arr, src.arr, sizeof(float) * (src.row_cnt * src.col_cnt), cudaMemcpyDeviceToHost);
    }

    this->row_cnt = src.row_cnt;
    this->col_cnt = src.col_cnt;
    this->typ = typ;
}

Tensor::Tensor(int row_cnt, int col_cnt, TensorType typ, float *cpu_arr)
{
    if (typ == Gpu)
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
    this->typ = typ;
}

Tensor::Tensor(int row_cnt, int col_cnt, TensorType typ, int *cpu_arr)
{

    if (typ == Gpu)
    {
        cudaMalloc(&this->arr, sizeof(float) * (row_cnt * col_cnt));

        for (int i = 0; i < row_cnt * col_cnt; i++)
        {
            float f = (float)cpu_arr[i];
            cudaMemcpy(&this->arr[i], &f, sizeof(float), cudaMemcpyHostToDevice);
        }
    }
    else
    {
        this->arr = (float *)malloc(sizeof(float) * (row_cnt * col_cnt));

        for (int i = 0; i < row_cnt * col_cnt; i++)
        {
            this->arr[i] = (float)cpu_arr[i];
        }
    }

    this->row_cnt = row_cnt;
    this->col_cnt = col_cnt;
    this->typ = typ;
}

Tensor::~Tensor()
{
    if (this->typ == Gpu)
    {
        cudaFree(this->arr);
    }
    else
    {
        free(this->arr);
    }
}

void Tensor::print()
{
    TensorType orig_typ = this->typ;

    this->translate(Cpu);

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

    this->translate(orig_typ);
}

void Tensor::dump_to_csv(const char *csv_file_name)
{
    FILE *file_ptr = fopen(csv_file_name, "w");

    for (int j = 0; j < this->col_cnt; j++)
    {

        if (j < this->col_cnt - 1)
        {
            fprintf(file_ptr, "col_%d,", j);
        }
        else
        {
            fprintf(file_ptr, "col_%d", j);
        }
    }
    fprintf(file_ptr, "\n");

    for (int i = 0; i < this->row_cnt; i++)
    {
        for (int j = 0; j < this->col_cnt; j++)
        {
            if (j < this->col_cnt - 1)
            {
                fprintf(file_ptr, "%f,", this->get_rowcol(i, j));
            }
            else
            {
                fprintf(file_ptr, "%f", this->get_rowcol(i, j));
            }
        }
        fprintf(file_ptr, "\n");
    }
    fclose(file_ptr);
}

void Tensor::translate(TensorType typ)
{
    if (typ == Gpu)
    {
        if (this->typ != Gpu)
        {
            float *d_arr;
            cudaMalloc(&d_arr, sizeof(float) * (this->row_cnt * this->col_cnt));
            cudaMemcpy(d_arr, this->arr, sizeof(float) * (this->row_cnt * this->col_cnt), cudaMemcpyHostToDevice);
            free(this->arr);
            this->arr = d_arr;

            this->typ = Gpu;
        }
    }
    else
    {
        if (this->typ == Gpu)
        {
            float *h_arr = (float *)malloc(sizeof(float) * (this->row_cnt * this->col_cnt));
            cudaMemcpy(h_arr, this->arr, sizeof(float) * (this->row_cnt * this->col_cnt), cudaMemcpyDeviceToHost);
            cudaFree(this->arr);
            this->arr = h_arr;

            this->typ = Cpu;
        }
    }
}

int Tensor::get_row_cnt()
{
    return this->row_cnt;
}

int Tensor::get_col_cnt()
{
    return this->col_cnt;
}

float Tensor::get_idx(int idx)
{
    if (this->typ == Gpu)
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

float *Tensor::get_arr(TensorType typ)
{
    this->translate(typ);
    return this->arr;
}

float *Tensor::get_slice(int idx, TensorType typ)
{
    this->translate(typ);
    return &this->arr[idx];
}

TensorTuple Tensor::get_min()
{
    TensorTuple tup;

    tup.idx = 0;
    tup.val = FLT_MAX;

    for (int i = 0; i < this->row_cnt * this->col_cnt; i++)
    {
        float cur_val = this->get_idx(i);
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

    for (int i = 0; i < this->row_cnt * this->col_cnt; i++)
    {
        float cur_val = this->get_idx(i);
        if (cur_val > tup.val)
        {
            tup.idx = i;
            tup.val = cur_val;
        }
    }

    return tup;
}

void Tensor::set_idx(int idx, float val)
{
    if (this->typ == Gpu)
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

void Tensor::set_all(float val)
{
    int tot_cnt = this->row_cnt * this->col_cnt;

    TensorType orig_typ = this->typ;

    this->translate(Cpu);

    for (int i = 0; i < tot_cnt; i++)
    {
        this->arr[i] = val;
    }

    this->translate(orig_typ);
}

void Tensor::set_all_rand(float upper)
{
    int tot_cnt = this->row_cnt * this->col_cnt;

    TensorType orig_typ = this->typ;

    this->translate(Cpu);

    for (int i = 0; i < tot_cnt; i++)
    {
        float val = (float)rand() / ((float)RAND_MAX);
        val *= (2 * upper);
        val -= upper;
        this->arr[i] = val;
    }

    this->translate(orig_typ);
}

void Tensor::set_arr(float *cpu_arr)
{
    int tot_cnt = this->row_cnt * this->col_cnt;

    TensorType orig_typ = this->typ;

    this->translate(Cpu);

    memcpy(this->arr, cpu_arr, sizeof(float) * tot_cnt);

    this->translate(orig_typ);
}
