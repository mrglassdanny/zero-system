#pragma once

#include <vector>

#include "../core/tensor.cuh"
#include "supervisor.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        class Report
        {
        public:
            float cost;
            int correct_cnt;
            int total_cnt;

            void print();
            void update_correct_cnt(Tensor *n, Tensor *y);
        };

        enum ActivationFunctionId
        {
            None,
            ReLU,
            Sigmoid,
            Tanh,
            Sine,
            Cosine
        };

        enum CostFunctionId
        {
            MSE,
            CrossEntropy
        };

        class NNLayerConfiguration
        {
        public:
            int neuron_cnt;
            ActivationFunctionId activation_func_id;
            float dropout_rate;

            NNLayerConfiguration();
            NNLayerConfiguration(int neuron_cnt, ActivationFunctionId activation_func_id, float dropout_rate);
            ~NNLayerConfiguration();
        };

        class NN
        {
        private:
            std::vector<NNLayerConfiguration> layer_configurations;
            CostFunctionId cost_func_id;
            float learning_rate;

            std::vector<Tensor *> neurons;
            std::vector<Tensor *> dropout_masks;
            std::vector<Tensor *> weights;
            std::vector<Tensor *> biases;
            std::vector<Tensor *> weight_derivatives;
            std::vector<Tensor *> bias_derivatives;

            float *d_cost;
            bool compiled_flg; // TODO

        public:
            static void write_csv_header(FILE *csv_file_ptr);
            static void write_to_csv(FILE *csv_file_ptr, int epoch, Report rpt);

            NN(CostFunctionId cost_func_id, float learning_rate);
            NN(const char *path);
            ~NN();

            void print();

            void dump(const char *path);

            void add_layer(int neuron_cnt);
            void add_layer(int neuron_cnt, ActivationFunctionId activation_func_id);
            void add_layer(int neuron_cnt, float dropout_rate);
            void add_layer(int neuron_cnt, ActivationFunctionId activation_func_id, float dropout_rate);

            void compile();

            void set_learning_rate(float learning_rate);

            void set_dropout_masks();
            void feed_forward(Tensor *x, bool train_flg);
            float get_cost(Tensor *y);
            Tensor *back_propagate(Tensor *y, bool keep_agg_derivatives_flg);
            void optimize(int batch_size);

            void check_gradient(Tensor *x, Tensor *y, bool print_flg);

            Report train(Batch *batch);
            Report validate(Batch *batch);
            Report test(Batch *batch);

            void train_and_test(Supervisor *supervisor, int train_batch_size, const char *csv_path);
            void all(Supervisor *supervisor, int train_batch_size, int validation_chk_freq, const char *csv_path);

            Tensor *predict(Tensor *x);
        };
    }
}
