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
        public:
            float train_pct;
            float validation_pct;
            float test_pct;

            Supervisor(float train_pct, float test_pct);
            ~Supervisor();

            float get_train_pct();
            float get_validation_pct();
            float get_test_pct();

            virtual int get_cnt() = 0;

            virtual std::vector<int> get_x_shape() = 0;
            virtual std::vector<int> get_y_shape() = 0;

            virtual Batch *create_batch() = 0;
            virtual Batch *create_batch(int lower, int upper) = 0;
            virtual Batch *create_batch(int batch_size, int lower, int upper) = 0;
            virtual Batch *create_train_batch() = 0;
            virtual Batch *create_train_batch(int batch_size) = 0;
            virtual Batch *create_validation_batch() = 0;
            virtual Batch *create_test_batch() = 0;
        };

        class InMemorySupervisor : Supervisor
        {
        private:
            std::vector<Record *> records; // InMemorySupervisor owns records!

            void add(int col_cnt, int one_hot_cnt, float *x_arr, float y_val, Device device);
            void add_all(int row_cnt, int col_cnt, int one_hot_cnt, float *x_arr, float *y_arr, Device device);
            void clear();

        public:
            InMemorySupervisor(float train_pct, float test_pct, int row_cnt, int col_cnt, int one_hot_cnt, float *x_arr, float *y_arr, Device device);
            ~InMemorySupervisor();

            virtual int get_cnt();

            virtual std::vector<int> get_x_shape();
            virtual std::vector<int> get_y_shape();

            virtual Batch *create_batch();
            virtual Batch *create_batch(int lower, int upper);
            virtual Batch *create_batch(int batch_size, int lower, int upper);
            virtual Batch *create_train_batch();
            virtual Batch *create_train_batch(int batch_size);
            virtual Batch *create_validation_batch();
            virtual Batch *create_test_batch();
        };

        class OnDiskSupervisor : Supervisor
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
            OnDiskSupervisor(float train_pct, float test_pct, const char *x_path, const char *y_path, std::vector<int> x_shape, int y_one_hot_cnt);
            ~OnDiskSupervisor();

            virtual int get_cnt();

            virtual std::vector<int> get_x_shape();
            virtual std::vector<int> get_y_shape();

            virtual Batch *create_batch();
            virtual Batch *create_batch(int lower, int upper);
            virtual Batch *create_batch(int batch_size, int lower, int upper);
            virtual Batch *create_train_batch();
            virtual Batch *create_train_batch(int batch_size);
            virtual Batch *create_validation_batch();
            virtual Batch *create_test_batch();
        };
    }
}
