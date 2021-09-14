#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

class Tensor
{
    float *arr;
    int row_cnt;
    int col_cnt;
    bool gpu_flg;

public:
    Tensor(int row_cnt, int col_cnt, bool gpu_flg);
    Tensor(Tensor *src);
    Tensor::Tensor(int row_cnt, int col_cnt, bool gpu_flg, float *cpu_arr);
    ~Tensor();

    void translate(bool gpu_flg);

    float get_idx(int idx);
    float get_rowcol(int row_idx, int col_idx);
    float *get_arr(bool gpu_flg);

    void set_idx(int idx, float val);
    void set_rowcol(int row_idx, int col_idx, float val);
    void set_arr(float *arr, bool gpu_flg, bool translate_flg);
    void set_all(float val);
    void set_all_rand(float upper);

    void print();
};