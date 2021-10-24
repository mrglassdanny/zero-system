#pragma once

#include <vector>
#include <algorithm>
#include <random>

#include "../core/tensor.cuh"
#include "util.cuh"

#define SUPERVISOR_TRAIN_SPLIT 0.60f
#define SUPERVISOR_VALIDATION_SPLIT 0.20f
#define SUPERVISOR_TEST_SPLIT 0.20f

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
            bool owns_records_flg;
            std::vector<Record *> records;

        public:
            Batch(int batch_size);
            Batch(bool owns_records_flg, int batch_size);
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
            Supervisor(int row_cnt, int col_cnt, int one_hot_cnt, float *x_arr, float *y_arr, Device device);
            ~Supervisor();

            void add(int col_cnt, int one_hot_cnt, float *x_arr, float y_val, Device device);
            void add_all(int row_cnt, int col_cnt, int one_hot_cnt, float *x_arr, float *y_arr, Device device);
            void clear();

            int get_cnt();

            std::vector<int> get_x_shape();
            std::vector<int> get_y_shape();

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
