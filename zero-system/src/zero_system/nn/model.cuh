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
            CostFunction cost_fn;
            float learning_rate;

            void add_layer(Layer *lyr);

        public:
            Model();
            Model(CostFunction cost_fn, float learning_rate);
            ~Model();

            virtual void load(FILE *file_ptr);
            virtual void load(const char *path);
            virtual void save(FILE *file_ptr);
            virtual void save(const char *path);

            void linear(int nxt_n_cnt);
            void linear(int nxt_n_cnt, InitializationFunction init_fn);
            void linear(std::vector<int> n_shape, int nxt_n_cnt);
            void linear(int n_cnt, int nxt_n_cnt);
            void linear(int n_cnt, int nxt_n_cnt, InitializationFunction init_fn);
            void linear(std::vector<int> n_shape, int nxt_n_cnt, InitializationFunction init_fn);

            void activation(ActivationFunction activation_fn);

            void dropout(float dropout_rate);

            void aggregation(AggregationFunction agg_fn, int grp_cnt);
            void aggregation(int n_cnt, AggregationFunction agg_fn, int grp_cnt);
            void aggregation(std::vector<int> n_shape, AggregationFunction agg_fn, int grp_cnt);

            std::vector<int> get_input_shape();
            std::vector<int> get_output_shape();

            std::vector<Layer *> get_layers();

            void set_learning_rate(float learning_rate);

            virtual Tensor *forward(Tensor *x, bool train_flg);
            virtual float cost(Tensor *pred, Tensor *y);
            virtual Tensor *backward(Tensor *pred, Tensor *y);
            virtual void step(int batch_size);
            virtual void grad_check(Tensor *x, Tensor *y, bool print_flg);

            Report train(Batch *batch, UpdateResultFn fn);
            Report test(Batch *batch, UpdateResultFn fn);

            void fit(Batch *batch, UpdateResultFn fn);
            void fit(Supervisor *supervisor, int batch_size, int target_epoch, const char *csv_path, UpdateResultFn fn);

            Tensor *predict(Tensor *x);
        };

        class ConvNet : public Model
        {
        public:
            ConvNet();
            ConvNet(CostFunction cost_fn, float learning_rate);
            ~ConvNet();

            void convolutional(int fltr_cnt, int w_row_cnt, int w_col_cnt);
            void convolutional(int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn);
            void convolutional(std::vector<int> n_shape, int fltr_cnt, int w_row_cnt, int w_col_cnt);
            void convolutional(std::vector<int> n_shape, int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn);

            void pooling(PoolingFunction pool_fn);
        };

        class Embedding : public Model
        {
        public:
            Embedding();
            Embedding(CostFunction cost_fn, float learning_rate);
            ~Embedding();

            Tensor *embedding_backward(Tensor *dc, int embd_x_offset);
            void embedding_grad_check(EmbeddedModel *parent_embd_model, Tensor *x, Tensor *y,
                                      float *agg_ana_grad, float *agg_num_grad, float *agg_grad_diff,
                                      int embg_idx, bool print_flg);
        };

        class EmbeddedModel : public Embedding
        {
        protected:
            std::vector<Embedding *> embgs;
            std::vector<Range> embg_ranges;

            void add_embedding(Embedding *embg, Range embg_range);

        public:
            EmbeddedModel();
            EmbeddedModel(CostFunction cost_fn, float learning_rate);
            ~EmbeddedModel();

            virtual void load(FILE *file_ptr);
            virtual void load(const char *path);
            virtual void save(FILE *file_ptr);
            virtual void save(const char *path);

            std::vector<int> calc_embedded_input_shape(std::vector<int> n_shape);
            std::vector<int> calc_embedded_input_shape(int n_cnt);

            std::vector<Embedding *> get_embeddings();
            std::vector<Range> get_embedding_ranges();

            void embed(Embedding *embg);
            void embed(Embedding *embg, Range embg_range);

            virtual Tensor *forward(Tensor *x, bool train_flg);
            virtual Tensor *backward(Tensor *pred, Tensor *y);
            virtual void step(int batch_size);
            virtual void grad_check(Tensor *x, Tensor *y, bool print_flg);
        };
    }
}