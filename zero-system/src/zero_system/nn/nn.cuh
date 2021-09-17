#pragma once

#include <vector>

#include "../core/tensor.cuh"
#include "supervisor.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        enum ActivationFunctionId
        {
            None,
            ReLU,
            Sigmoid,
            Tanh
        };

        enum CostFunctionId
        {
            MSE,
            CrossEntropy
        };

        class ProgressReport
        {
        public:
            float cost;
            int crct_cnt;
            int tot_cnt;

            void print();
        };

        class NN
        {
        private:
            std::vector<Tensor *> neurons;
            std::vector<Tensor *> weights;
            std::vector<Tensor *> biases;
            std::vector<Tensor *> weight_derivatives;
            std::vector<Tensor *> bias_derivatives;

            ActivationFunctionId hidden_layer_activation_func_id;
            ActivationFunctionId output_layer_activation_func_id;

            CostFunctionId cost_func_id;

            float learning_rate;

            static void write_csv_header(FILE *csv_file_ptr);
            static void write_to_csv(FILE *csv_file_ptr, int epoch, ProgressReport rpt);

        public:
            NN(std::vector<int> layer_config, ActivationFunctionId hidden_layer_activation_func_id,
               ActivationFunctionId output_layer_activation_func_id, CostFunctionId cost_func_id, float learning_rate);
            NN(const char *path);
            ~NN();

            void print();

            void dump_to_file(const char *path);

            void feed_forward(Tensor *x);
            float get_cost(Tensor *y);
            void back_propagate(Tensor *y);
            void optimize(int batch_size);

            void check_gradient(Tensor *x, Tensor *y, bool print_flg);
            void check_performance(Tensor *x, Tensor *y);

            ProgressReport train(Batch *batch);
            ProgressReport validate(Batch *batch);
            ProgressReport test(Batch *batch);

            void all(Supervisor *supervisor, int train_batch_size, int validation_chk_freq, const char *csv_path);

            Tensor *predict(Tensor *x);
        };
    }
}
