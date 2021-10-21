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
            CostFunction cost_fn;
            float learning_rate;

            float *d_cost;

        public:
            Model(CostFunction cost_fn, float learning_rate);
            ~Model();

            void add_layer(Layer *lyr);

            Tensor *forward(Tensor *x);
            float cost(Tensor *pred, Tensor *y);
            void backward(Tensor *pred, Tensor *y);
            void step(int batch_size);
        };
    }
}