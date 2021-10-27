#pragma once

#include <vector>
#include <algorithm>
#include <random>

#include "../core/tensor.cuh"
#include "util.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        class Record
        {

        public:
            Tensor *x; // Record owns tensors!
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
            std::vector<Record *> records; // Supervisor owns records!
            float train_pct;
            float validation_pct;
            float test_pct;

        public:
            Supervisor();
            Supervisor(int row_cnt, int col_cnt, int one_hot_cnt, float *x_arr, float *y_arr, float train_pct, float test_pct, Device device);
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
