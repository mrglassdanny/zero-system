#pragma once

#include <vector>

#include "../core/tensor.cuh"
#include "nn.cuh"
#include "supervisor.cuh"

namespace zero
{
    namespace nn
    {
        class ConvaLayer
        {
        public:
            std::vector<Tensor *> inputs;
            std::vector<std::vector<Tensor *>> filters;
            std::vector<Tensor *> outputs;
            ActivationFunctionId activation_func_id;

            ConvaLayer();
            ~ConvaLayer();
        };

        class CNN
        {
        private:
            std::vector<ConvaLayer> layers;
            NN *nn;

        public:
            CNN();
            ~CNN();
        };
    }
}