#pragma once

#include "layer.cuh"
#include "nn_util.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        class Embedding
        {
        private:
            std::vector<Layer *> layers;

        public:
            Embedding();
            Embedding(const char *path);
            ~Embedding();

            void save(const char *path);

            void linear(int nxt_n_cnt);
            void linear(int nxt_n_cnt, InitializationFunction init_fn);
            void linear(std::vector<int> n_shape, int nxt_n_cnt);
            void linear(int n_cnt, int nxt_n_cnt);
            void linear(std::vector<int> n_shape, int nxt_n_cnt, InitializationFunction init_fn);

            void activation(ActivationFunction activation_fn);
            void activation(std::vector<int> n_shape, ActivationFunction activation_fn);

            std::vector<int> get_input_shape();
            std::vector<int> get_output_shape();

            Tensor *forward(Tensor *x, bool train_flg);
            void backward(Tensor *pred, Tensor *y);
            void step(int batch_size);
        };
    }
}