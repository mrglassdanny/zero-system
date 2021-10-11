#pragma once

#include <vector>

#include "../core/tensor.cuh"
#include "nn.cuh"
#include "supervisor.cuh"

namespace zero
{
    namespace nn
    {
        class ConvLayer
        {
        public:
            std::vector<Tensor *> inputs;
            std::vector<std::vector<Tensor *>> filters;
            std::vector<Tensor *> outputs;
            ActivationFunctionId activation_func_id;

            ConvLayer();
            ~ConvLayer();
        };

        class CNN
        {
        private:
            std::vector<ConvLayer> layers;
            NN *nn;

        public:
            CNN();
            ~CNN();
        };
    }
}