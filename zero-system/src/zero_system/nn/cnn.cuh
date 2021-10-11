#pragma once

#include <vector>

#include "../core/tensor.cuh"
#include "nn.cuh"
#include "supervisor.cuh"

namespace zero
{
    namespace nn
    {
        class CNN
        {
        private:
            NN *nn;

        public:
            CNN();
            ~CNN();
        };
    }
}