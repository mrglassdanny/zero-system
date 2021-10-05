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

        class Report
        {
        public:
            float cost;
            int correct_cnt;
            int total_cnt;

            void print();
            void update_correct_cnt(Tensor *n, Tensor *y);
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

            float *d_cost;

        public:
            NN(std::vector<int> layer_config, ActivationFunctionId hidden_layer_activation_func_id,
               ActivationFunctionId output_layer_activation_func_id, CostFunctionId cost_func_id, float learning_rate);
            NN(const char *path);
            ~NN();

            static void write_csv_header(FILE *csv_file_ptr);
            static void write_to_csv(FILE *csv_file_ptr, int epoch, Report rpt);

            void print();

            void dump(const char *path);
            void dump(const char *path, int input_lyr_n_cnt);

            void set_learning_rate(float learning_rate);

            void feed_forward(Tensor *x, float dropout);
            float get_cost(Tensor *y);
            void back_propagate(Tensor *y);
            void optimize(int batch_size);

            void check_gradient(Tensor *x, Tensor *y, bool print_flg);

            Report train(Batch *batch, float dropout);
            Report validate(Batch *batch);
            Report test(Batch *batch);

            void all(Supervisor *supervisor, float dropout, int train_batch_size, int validation_chk_freq, const char *csv_path);

            Tensor *predict(Tensor *x);
        };
    }
}
