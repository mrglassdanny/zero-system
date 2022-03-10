#pragma once

#include "../core/mod.cuh"

#include "nn_util.cuh"
#include "nn_constants.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        enum LayerType
        {
            Linear,
            Convolutional,
            Activation,
            Dropout,
            Pooling,
            Aggregation
        };

        class Layer
        {
        protected:
            Tensor *n;

        public:
            Layer();
            Layer(std::vector<int> n_shape);
            ~Layer();

            virtual LayerType get_type() = 0;

            virtual void load(FILE *file_ptr);
            virtual void save(FILE *file_ptr);

            virtual std::vector<int> get_input_shape();
            virtual std::vector<int> get_output_shape();

            virtual Tensor *get_neurons();
            virtual void set_neurons(Tensor *n);
            virtual void reshape_neurons(std::vector<int> shape);

            virtual void forward(Tensor *nxt_n, bool train_flg);
            virtual Tensor *backward(Tensor *dc) = 0;
        };

        class LearnableLayer : public Layer
        {
        protected:
            Tensor *w;
            Tensor *b;
            Tensor *dw;
            Tensor *db;

        public:
            LearnableLayer();
            LearnableLayer(std::vector<int> n_shape);
            ~LearnableLayer();

            virtual void load(FILE *file_ptr);
            virtual void save(FILE *file_ptr);

            virtual Tensor *get_weights();
            virtual Tensor *get_weight_derivatives();
            virtual Tensor *get_biases();
            virtual Tensor *get_bias_derivatives();

            virtual void set_weights(Tensor *w);
            virtual void set_weight_derivatives(Tensor *dw);
            virtual void set_biases(Tensor *b);
            virtual void set_bias_derivatives(Tensor *db);

            virtual void step(int batch_size, float learning_rate) = 0;
        };

        class LinearLayer : public LearnableLayer
        {
        public:
            LinearLayer();
            LinearLayer(std::vector<int> n_shape, int nxt_n_cnt, InitializationFunction init_fn);
            ~LinearLayer();

            virtual LayerType get_type();

            virtual void load(FILE *file_ptr);
            virtual void save(FILE *file_ptr);

            virtual std::vector<int> get_output_shape();

            virtual void forward(Tensor *nxt_n, bool train_flg);
            virtual Tensor *backward(Tensor *dc);

            virtual void step(int batch_size, float learning_rate);
        };

        class ConvolutionalLayer : public LearnableLayer
        {
            // NOTE: we only support stride of 1!
        public:
            ConvolutionalLayer();
            ConvolutionalLayer(std::vector<int> n_shape, int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn);
            ~ConvolutionalLayer();

            virtual LayerType get_type();

            virtual void load(FILE *file_ptr);
            virtual void save(FILE *file_ptr);

            virtual std::vector<int> get_output_shape();

            virtual void forward(Tensor *nxt_n, bool train_flg);
            virtual Tensor *backward(Tensor *dc);

            virtual void step(int batch_size, float learning_rate);
        };

        class ActivationLayer : public Layer
        {
        private:
            ActivationFunction activation_fn;

        public:
            ActivationLayer();
            ActivationLayer(std::vector<int> n_shape, ActivationFunction activation_fn);
            ~ActivationLayer();

            virtual LayerType get_type();

            virtual void load(FILE *file_ptr);
            virtual void save(FILE *file_ptr);

            virtual void forward(Tensor *nxt_n, bool train_flg);
            virtual Tensor *backward(Tensor *dc);
        };

        class DropoutLayer : public Layer
        {
        private:
            float dropout_rate;
            Tensor *dropout_mask;

        public:
            DropoutLayer();
            DropoutLayer(std::vector<int> n_shape, float dropout_rate);
            ~DropoutLayer();

            virtual LayerType get_type();

            virtual void load(FILE *file_ptr);
            virtual void save(FILE *file_ptr);

            virtual void forward(Tensor *nxt_n, bool train_flg);
            virtual Tensor *backward(Tensor *dc);
        };

        class PoolingLayer : public Layer
        {
        private:
            PoolingFunction pool_fn;
            int pool_row_cnt;
            int pool_col_cnt;

        public:
            PoolingLayer();
            PoolingLayer(std::vector<int> n_shape, PoolingFunction pool_fn);
            ~PoolingLayer();

            virtual LayerType get_type();

            virtual void load(FILE *file_ptr);
            virtual void save(FILE *file_ptr);

            virtual std::vector<int> get_output_shape();

            virtual void forward(Tensor *nxt_n, bool train_flg);
            virtual Tensor *backward(Tensor *dc);
        };

        class AggregationLayer : public Layer
        {
        private:
            AggregationFunction agg_fn;

        public:
            AggregationLayer();
            AggregationLayer(std::vector<int> n_shape, AggregationFunction agg_fn);
            ~AggregationLayer();

            virtual LayerType get_type();

            virtual void load(FILE *file_ptr);
            virtual void save(FILE *file_ptr);

            virtual std::vector<int> get_output_shape();

            virtual void forward(Tensor *nxt_n, bool train_flg);
            virtual Tensor *backward(Tensor *dc);
        };
    }
}