#pragma once

#include <vector>
#include <algorithm>
#include <random>

#include "../core/tensor.cuh"

#define SUPERVISOR_TRAIN_SPLIT 0.80f
#define SUPERVISOR_VALIDATION_SPLIT 0.10f
#define SUPERVISOR_TEST_SPLIT 0.10f

namespace zero
{
    using namespace core;

    namespace nn
    {
        class Record
        {

        public:
            // Record owns tensors!
            Tensor *x;
            Tensor *y;

            Record(Tensor *x, Tensor *y);
            ~Record();
        };

        class Batch
        {
        private:
            // Batch does NOT own records!
            std::vector<Record *> records;

        public:
            Batch(int batch_size);
            ~Batch();

            void add(Record *record);

            int get_size();

            Tensor *get_x(int idx);
            Tensor *get_y(int idx);
        };

        class Supervisor
        {
        private:
            // Supervisor owns records!
            std::vector<Record *> records;

        public:
            Supervisor();
            Supervisor(int row_cnt, int col_cnt, int one_hot_cnt, float *x_arr, float *y_arr, TensorType typ);
            ~Supervisor();

            void add(int col_cnt, int one_hot_cnt, float *x_arr, float y_val, TensorType typ);
            void add_all(int row_cnt, int col_cnt, int one_hot_cnt, float *x_arr, float *y_arr, TensorType typ);
            void clear();

            int get_cnt();

            void shuffle();

            Batch *create_batch();
            Batch *create_batch(int lower, int upper);
            Batch *create_batch(int batch_size, int lower, int upper);
            Batch *create_train_batch();
            Batch *create_train_batch(int batch_size);
            Batch *create_validation_batch();
            Batch *create_test_batch();
        };
    }
}
