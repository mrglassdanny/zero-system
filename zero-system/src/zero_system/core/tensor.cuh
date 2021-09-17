#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <conio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

namespace zero
{
    namespace core
    {
        enum TensorType
        {
            Cpu,
            Gpu
        };

        struct TensorTuple
        {
            int idx;
            float val;
        };

        class Tensor
        {
        private:
            float *arr;
            int row_cnt;
            int col_cnt;
            TensorType typ;

        public:
            static Tensor *one_hot_encode(int row_cnt, int col_cnt, TensorType typ, float *cpu_arr);
            static Tensor *from_csv(const char *csv_file_name);

            Tensor(int row_cnt, int col_cnt, TensorType typ);
            Tensor(const Tensor &src);
            Tensor(int row_cnt, int col_cnt, TensorType typ, float *cpu_arr);
            ~Tensor();

            void print();

            void dump_to_csv(const char *csv_file_name);

            void translate(TensorType typ);

            int get_row_cnt();
            int get_col_cnt();
            float get_idx(int idx);
            float get_rowcol(int row_idx, int col_idx);
            float *get_arr(TensorType typ);
            float *get_slice(int idx, TensorType typ);
            TensorTuple get_min();
            TensorTuple get_max();

            void set_idx(int idx, float val);
            void set_rowcol(int row_idx, int col_idx, float val);
            void set_all(float val);
            void set_all_rand(float upper);
            void set_arr(float *cpu_arr);
        };
    }
}
