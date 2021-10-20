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

namespace zero_v2
{
    namespace core
    {
        enum Device
        {
            Cpu,
            Cuda
        };

        class Tensor
        {
        private:
            float *arr;
            Device device;
            std::vector<int> shape;

        public:
            Tensor(Tensor &src);
            Tensor(Device device);
            Tensor(Device device, int cnt);
            Tensor(Device device, int row_cnt, int col_cnt);
            Tensor(Device device, int x_cnt, int y_cnt, int z_cnt);
            Tensor(Device device, std::vector<int> shape);
            ~Tensor();

            void to(Device device);

            void copy(Tensor *src);

            void print();

            std::vector<int> get_shape();
            int get_cnt();
            float *get_arr();

            float get_val(int idx);
            void set_val(int idx, float val);

            void reset();
            void set_all(float val);
            void set_all_rand(float mean, float stddev);
        };
    }
}