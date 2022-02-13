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

            virtual void load(const char *path);
            virtual void save(const char *path);

            void linear(int nxt_n_cnt);
            void linear(int nxt_n_cnt, InitializationFunction init_fn);
            void linear(std::vector<int> n_shape, int nxt_n_cnt);
            void linear(int n_cnt, int nxt_n_cnt);
            void linear(std::vector<int> n_shape, int nxt_n_cnt, InitializationFunction init_fn);

            void activation(ActivationFunction activation_fn);
            void activation(std::vector<int> n_shape, ActivationFunction activation_fn);

            void dropout(float dropout_rate);
            void dropout(std::vector<int> n_shape, float dropout_rate);

            std::vector<int> get_input_shape();
            std::vector<int> get_output_shape();

            void set_learning_rate(float learning_rate);

            virtual Tensor *forward(Tensor *x, bool train_flg);
            float cost(Tensor *pred, Tensor *y);
            virtual Tensor *backward(Tensor *pred, Tensor *y);
            virtual void step(int batch_size);
            void check_grad(Tensor *x, Tensor *y, bool print_flg);

            Report train(Batch *batch);
            Report test(Batch *batch);

            void fit(Batch *batch);
            void fit(Supervisor *supervisor, int batch_size, int target_epoch, const char *csv_path);

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
            void pooling(std::vector<int> n_shape, PoolingFunction pool_fn);
        };

        class Embedding : public Model
        {
        protected:
            int beg_x_idx;
            int end_x_idx;

        public:
            Embedding();
            Embedding(CostFunction cost_fn, float learning_rate);
            Embedding(int x_idx);
            Embedding(int beg_x_idx, int end_x_idx);
            ~Embedding();

            // TODO: load & save

            int get_beg_x_idx();
            int get_end_x_idx();

            void emb_backward(Tensor *dc);
        };

        class EmbeddableModel : public Model
        {
        protected:
            std::vector<Embedding *> embeddings;

            void add_embedding(Embedding *emb);

        public:
            EmbeddableModel();
            EmbeddableModel(CostFunction cost_fn, float learning_rate);
            ~EmbeddableModel();

            // TODO: load & save

            void embed(Embedding *emb);

            virtual Tensor *forward(Tensor *x, bool train_flg);
            virtual Tensor *backward(Tensor *pred, Tensor *y);
            virtual void step(int batch_size);
        };
    }
}