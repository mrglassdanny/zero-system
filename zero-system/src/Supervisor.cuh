#pragma once

#include <vector>

#include "Tensor.cuh"
#include "Batch.cuh"

#define SUPERVISOR_TRAIN_SPLIT 0.70f
#define SUPERVISOR_VALIDATION_SPLIT 0.15f
#define SUPERVISOR_TEST_SPLIT 0.15f

class Supervisor
{
private:
    std::vector<Tensor *> xs;
    std::vector<Tensor *> ys;

public:
    Supervisor();
    Supervisor(int row_cnt, int col_cnt, int one_hot_cnt, float *x_arr, float *y_arr, TensorType typ);
    ~Supervisor();

    void add(int col_cnt, int one_hot_cnt, float *x_arr, float y_val, TensorType typ);
    void add_all(int row_cnt, int col_cnt, int one_hot_cnt, float *x_arr, float *y_arr, TensorType typ);
    void clear();

    int get_cnt();

    Batch *create_batch();
    Batch *create_batch(int lower, int upper);
    Batch *create_batch(int batch_size, int lower, int upper);
    Batch *create_train_batch();
    Batch *create_train_batch(int batch_size);
    Batch *create_validation_batch();
    Batch *create_test_batch();
};