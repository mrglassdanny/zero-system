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
            float learning_rate;

            // Channels are stacked in single tensor.
            // Filters get their own vector(channels are still stacked).
            std::vector<Tensor *> neurons;
            std::vector<std::vector<Tensor *>> filters;
            std::vector<std::vector<Tensor *>> biases;
            std::vector<std::vector<Tensor *>> filter_derivatives;
            std::vector<std::vector<Tensor *>> bias_derivatives;

            NN *nn;

        public:
            CNN(CostFunctionId cost_func_id, float learning_rate);
            ~CNN();

            void add_layer(ActivationFunctionId activation_func_id);
            void add_layer(int filter_cnt, int filter_row_cnt, int filter_col_cnt,
                           ActivationFunctionId activation_func_id);
            void add_layer(int channel_cnt, int neuron_row_cnt, int neuron_col_cnt,
                           int filter_cnt, int filter_row_cnt, int filter_col_cnt);
            void add_layer(int channel_cnt, int neuron_row_cnt, int neuron_col_cnt,
                           int filter_cnt, int filter_row_cnt, int filter_col_cnt,
                           ActivationFunctionId activation_func_id);

            NN *fully_connected();

            void compile();

            void feed_forward(Tensor *x, bool train_flg);
            float get_cost(Tensor *y);
            void back_propagate(Tensor *y);
            void optimize(int batch_size);
        };
    }
}