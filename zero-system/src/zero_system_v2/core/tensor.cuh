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
        enum TensorType
        {
            Cpu,
            Gpu
        };

        class Tensor
        {
        private:
            float *arr;
            TensorType typ;
            std::vector<int> shape;

        public:
            Tensor(TensorType typ);
            Tensor(TensorType typ, int cnt);
            Tensor(TensorType typ, int row_cnt, int col_cnt);
            Tensor(TensorType typ, int x_cnt, int y_cnt, int z_cnt);
            ~Tensor();

            void translate(TensorType typ);
            void reset();
            void reset(float val);
            void reset_rand(float mean, float stddev);
            void print();
        };
    }
}