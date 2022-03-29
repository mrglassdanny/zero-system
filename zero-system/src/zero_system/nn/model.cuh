#pragma once

#include "../core/mod.cuh"

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
            std::vector<Model *> embgs;
            std::vector<Range> embg_ranges;
            CostFunction cost_fn;
            float learning_rate;

            void add_layer(Layer *lyr);
            void add_embedding(Model *embg, Range embg_range);

        public:
            Model();
            Model(CostFunction cost_fn);
            Model(float learning_rate);
            Model(CostFunction cost_fn, float learning_rate);
            ~Model();

            void load(FILE *file_ptr);
            void load(const char *path);
            void save(FILE *file_ptr);
            void save(const char *path);

            void linear(int nxt_n_cnt);
            void linear(int nxt_n_cnt, InitializationFunction init_fn);
            void linear(std::vector<int> n_shape, int nxt_n_cnt);
            void linear(int n_cnt, int nxt_n_cnt);
            void linear(int n_cnt, int nxt_n_cnt, InitializationFunction init_fn);
            void linear(std::vector<int> n_shape, int nxt_n_cnt, InitializationFunction init_fn);
            void convolutional(int fltr_cnt, int w_row_cnt, int w_col_cnt);
            void convolutional(int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn);
            void convolutional(std::vector<int> n_shape, int fltr_cnt, int w_row_cnt, int w_col_cnt);
            void convolutional(std::vector<int> n_shape, int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn);
            void activation(ActivationFunction activation_fn);
            void activation(int n_cnt, ActivationFunction activation_fn);
            void activation(std::vector<int> n_shape, ActivationFunction activation_fn);
            void dropout(float dropout_rate);
            void pooling(PoolingFunction pool_fn);
            void aggregation();
            void custom(std::vector<int> (*get_output_shape_fn)(),
                        void (*forward_fn)(Tensor *n, Tensor *nxt_n, bool train_flg),
                        Tensor *(*backward_fn)(Tensor *n, Tensor *dc));
            void custom(int n_cnt,
                        std::vector<int> (*get_output_shape_fn)(),
                        void (*forward_fn)(Tensor *n, Tensor *nxt_n, bool train_flg),
                        Tensor *(*backward_fn)(Tensor *n, Tensor *dc));
            void custom(std::vector<int> n_shape,
                        std::vector<int> (*get_output_shape_fn)(),
                        void (*forward_fn)(Tensor *n, Tensor *nxt_n, bool train_flg),
                        Tensor *(*backward_fn)(Tensor *n, Tensor *dc));

            void embed(Model *embg);
            void embed(Model *embg, Range embg_range);

            std::vector<int> get_input_shape();
            std::vector<int> get_output_shape();
            std::vector<int> get_embedded_input_shape();

            std::vector<Layer *> get_layers();
            std::vector<Model *> get_embeddings();
            std::vector<Range> get_embedding_ranges();

            void set_learning_rate(float learning_rate);

            void share_parameters(Model *other_model);

            Tensor *forward(Tensor *x, bool train_flg);
            float cost(Tensor *pred, Tensor *y);
            Tensor *backward(Tensor *pred, Tensor *y);
            Tensor *embedding_backward(Tensor *dc, int embd_x_offset);
            void step(int batch_size);

            void grad_check(Tensor *x, Tensor *y, bool print_flg);
            void embedding_grad_check(Model *parent_embd_model, Tensor *x, Tensor *y,
                                      float *agg_ana_grad, float *agg_num_grad, float *agg_grad_diff,
                                      int embg_idx, bool print_flg);

            Report train(Batch *batch, UpdateResultFn fn);
            Report test(Batch *batch, UpdateResultFn fn);

            void fit(Batch *batch, UpdateResultFn fn);
            void fit(Supervisor *supervisor, int batch_size, int target_epoch, const char *csv_path, UpdateResultFn fn);

            Tensor *predict(Tensor *x);

            static std::vector<int> calc_embedded_input_shape(Model *model, std::vector<int> n_shape);
            static std::vector<int> calc_embedded_input_shape(Model *model, int n_cnt);
        };
    }
}