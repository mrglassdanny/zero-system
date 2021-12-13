#pragma once

#include <vector>
#include <algorithm>
#include <random>

#include "../core/tensor.cuh"
#include "nn_util.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        class Record
        {

        public:
            Tensor *x;
            Tensor *y;

            Record(Tensor *x, Tensor *y);
            ~Record();
        };

        class Batch
        {
        private:
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
            FILE *x_file_ptr;
            FILE *y_file_ptr;

            long long x_file_size;
            long long y_file_size;

            std::vector<int> x_shape;
            int y_one_hot_cnt;

            Batch *create_batch(int cnt, int lower, int upper, bool rand_flg);

        public:
            Supervisor(const char *x_path, const char *y_path, std::vector<int> x_shape, int y_one_hot_cnt);
            ~Supervisor();

            int get_cnt();

            std::vector<int> get_x_shape();
            std::vector<int> get_y_shape();

            Batch *create_batch();
            Batch *create_batch(int batch_size);
            Batch *create_batch(int lower, int upper);
            Batch *create_batch(int batch_size, int lower, int upper);
        };
    }
}
