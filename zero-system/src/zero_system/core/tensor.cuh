#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <conio.h>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "core_constants.cuh"
#include "core_util.cuh"

namespace zero
{
    namespace core
    {
        enum Device
        {
            Cpu,
            Cuda
        };

        struct TensorTuple
        {
            int idx;
            float val;
        };

        class Tensor
        {
        private:
            Device device;
            std::vector<int> shape;
            float *arr;

        public:
            Tensor(Tensor &src);
            Tensor(Device device);
            Tensor(Device device, int cnt);
            Tensor(Device device, int row_cnt, int col_cnt);
            Tensor(Device device, int x_cnt, int y_cnt, int z_cnt);
            Tensor(Device device, int a_cnt, int b_cnt, int c_cnt, int d_cnt);
            Tensor(Device device, std::vector<int> shape);
            ~Tensor();

            void to_device(Device device);

            void copy(Tensor *src);

            void reset();

            void print();

            bool equals(Tensor *other);

            void reshape(std::vector<int> shape);

            Device get_device();

            std::vector<int> get_shape();
            int get_cnt();

            int get_dim_cnt();

            float *get_arr();
            float *get_arr(Device device);
            void set_arr(float *arr);

            float get_val(int idx);
            void set_val(int idx, float val);

            void inc_val(int idx, float inc);
            void dec_val(int idx, float dec);

            void set_all(float val);
            void set_all_rand(float mean, float stddev);

            float get_sum();

            TensorTuple get_min();
            TensorTuple get_abs_min();
            TensorTuple get_max();
            TensorTuple get_abs_max();

            void scale_down();

            void add(Tensor *tensor);
            void subtract(Tensor *tensor);
            void subtract_abs(Tensor *tensor);

            static int get_cnt(std::vector<int> shape);

            static Tensor *one_hot_encode(Device device, int row_cnt, int col_cnt, float *cpu_arr);

            static Tensor *fr_csv(const char *path);
            static void to_csv(const char *path, Tensor *tensor);

            static void to_file(const char *path, Tensor *tensor);
        };
    }
}