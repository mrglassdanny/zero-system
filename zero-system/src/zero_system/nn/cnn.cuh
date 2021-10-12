#pragma once

#include <vector>

#include "../core/tensor.cuh"
#include "nn.cuh"
#include "supervisor.cuh"

namespace zero
{
    namespace nn
    {

        class CNNLayerConfiguration
        {
        public:
            int channel_cnt;
            int neuron_row_cnt;
            int neuron_col_cnt;
            int filter_cnt;
            int filter_row_cnt;
            int filter_col_cnt;
            ActivationFunctionId activation_func_id;

            CNNLayerConfiguration();
            CNNLayerConfiguration(int channel_cnt, int neuron_row_cnt, int neuron_col_cnt,
                                  int filter_cnt, int filter_row_cnt, int filter_col_cnt,
                                  ActivationFunctionId activation_func_id);
            ~CNNLayerConfiguration();
        };

        class CNN
        {
        private:
            std::vector<CNNLayerConfiguration> layer_configurations;

            std::vector<Tensor *> neurons;
            std::vector<Tensor *> filters;
            std::vector<Tensor *> biases;
            std::vector<Tensor *> filter_derivatives;
            std::vector<Tensor *> bias_derivatives;

            NN *nn;

        public:
            CNN();
            ~CNN();

            void add_layer(int channel_cnt, int neuron_row_cnt, int neuron_col_cnt,
                           int filter_cnt, int filter_row_cnt, int filter_col_cnt,
                           ActivationFunctionId activation_func_id);

            NN *fully_connected();

            void compile();

            void feed_forward(Tensor *x, bool train_flg);
            void back_propagate(Tensor *y);
            void optimize(int batch_size);
        };
    }
}