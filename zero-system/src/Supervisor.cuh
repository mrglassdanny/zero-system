#pragma once

#include <vector>

#include "Tensor.cuh"
#include "Batch.cuh"

class Supervisor
{
private:
    std::vector<Tensor *> xs;
    std::vector<Tensor *> ys;

public:
    Supervisor();
    Supervisor(int row_cnt, int col_cnt, int output_col_cnt, float *x_arr, float *y_arr);
    ~Supervisor();

    void add(int col_cnt, int output_col_cnt, float *x_arr, float y_val);
    void add_all(int row_cnt, int col_cnt, int output_col_cnt, float *x_arr, float *y_arr);
    void clear();

    Batch *create_batch(int batch_size, int lower, int upper);
    Batch *create_train_batch(int batch_size);
    Batch *create_validation_batch(int batch_size);
    Batch *create_test_batch(int batch_size);
};