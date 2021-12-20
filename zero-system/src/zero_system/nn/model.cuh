#pragma once

#include "layer.cuh"
#include "batch.cuh"
#include "nn_util.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        class Model
        {
        protected:
            std::vector<Layer *> layers;
            CostFunction cost_fn;
            float learning_rate;
            float *d_cost_val;

            void add_layer(Layer *lyr);

        public:
            Model(CostFunction cost_fn, float learning_rate);
            Model(const char *path);
            ~Model();

            void save(const char *path);

            void linear(int nxt_n_cnt);
            void linear(int nxt_n_cnt, InitializationFunction init_fn);
            void linear(std::vector<int> n_shape, int nxt_n_cnt);
            void linear(std::vector<int> n_shape, int nxt_n_cnt, InitializationFunction init_fn);

            void convolutional(int fltr_cnt, int w_row_cnt, int w_col_cnt);
            void convolutional(int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn);
            void convolutional(std::vector<int> n_shape, int fltr_cnt, int w_row_cnt, int w_col_cnt);
            void convolutional(std::vector<int> n_shape, int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn);

            void activation(ActivationFunction activation_fn);
            void activation(std::vector<int> n_shape, ActivationFunction activation_fn);

            void dropout(float dropout_rate);
            void dropout(std::vector<int> n_shape, float dropout_rate);

            void pooling(PoolingFunction pool_fn);
            void pooling(std::vector<int> n_shape, PoolingFunction pool_fn);

            std::vector<int> get_input_shape();
            std::vector<int> get_output_shape();

            void set_learning_rate(float learning_rate);

            Tensor *forward(Tensor *x, bool train_flg);
            float cost(Tensor *pred, Tensor *y);
            void backward(Tensor *pred, Tensor *y);
            void step(int batch_size);

            void gradient_check(Tensor *x, Tensor *y, bool print_flg);

            Report train(Batch *batch);
            Report test(Batch *batch);

            void fit(Batch *batch);
            void fit(Supervisor *supervisor, int batch_size, int target_epoch, const char *csv_path);

            Tensor *predict(Tensor *x);
        };
    }
}