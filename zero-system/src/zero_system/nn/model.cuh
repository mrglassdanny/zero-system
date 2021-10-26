#pragma once

#include "layer.cuh"
#include "util.cuh"
#include "batch.cuh"

namespace zero
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

            float *d_cost_val;

        public:
            Model(CostFunction cost_fn, float learning_rate);
            Model(const char *path);
            ~Model();

            void save(const char *path);

            void add_layer(Layer *lyr);

            std::vector<int> get_input_shape();
            std::vector<int> get_output_shape();

            Tensor *forward(Tensor *x, bool train_flg);
            float cost(Tensor *pred, Tensor *y);
            void backward(Tensor *pred, Tensor *y);
            void step(int batch_size);

            void gradient_check(Tensor *x, Tensor *y, bool print_flg);

            Report train(Batch *batch);
            Report test(Batch *batch);

            void train_and_test(Supervisor *supervisor, int train_batch_size, int target_epoch_cnt, const char *csv_path);
            void all(Supervisor *supervisor, int train_batch_size, int target_epoch, const char *csv_path);

            Tensor *predict(Tensor *x);
        };
    }
}