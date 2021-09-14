#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

typedef enum TensorType
{
    Cpu,
    Gpu
} TensorType;

class Tensor
{
private:
    float *arr;
    int row_cnt;
    int col_cnt;
    TensorType typ;

public:
    Tensor(int row_cnt, int col_cnt, TensorType typ);
    Tensor(Tensor *src);
    Tensor(int row_cnt, int col_cnt, TensorType typ, float *cpu_arr);
    ~Tensor();

    void translate(TensorType typ);

    int get_row_cnt();
    int get_col_cnt();
    float get_idx(int idx);
    float get_rowcol(int row_idx, int col_idx);
    float *get_arr(TensorType typ);

    void set_idx(int idx, float val);
    void set_rowcol(int row_idx, int col_idx, float val);
    void set_arr(float *arr, TensorType typ, bool translate_flg);
    void set_all(float val);
    void set_all_rand(float upper);

    void print();
};