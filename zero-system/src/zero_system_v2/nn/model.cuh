#pragma once

#include "layer.cuh"

namespace zero_v2
{
    using namespace core;

    namespace nn
    {
        class Model
        {
        private:
            std::vector<Layer *> layers;

        public:
            Model();
            ~Model();

            void add_layer(Layer *lyr);

            void forward(Tensor *x);
            void backward(Tensor *y);
            void step(int batch_size);
        };
    }
}