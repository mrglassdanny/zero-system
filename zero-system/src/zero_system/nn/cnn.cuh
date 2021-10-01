#pragma once

#include <vector>

#include "../core/tensor.cuh"
#include "supervisor.cuh"
#include "nn.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        class CNN
        {

        public:
            CNN();
            ~CNN();
        };
    }
}
