#pragma once

#include "../core/tensor.cuh"
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
            Pooling
        };

        class Layer
        {
        protected:
            Tensor *n;

        public:
            Layer(std::vector<int> n_shape);
            Layer(FILE *file_ptr);
            ~Layer();

            virtual LayerType get_type() = 0;
            virtual std::vector<int> get_input_shape();
            virtual std::vector<int> get_output_shape();
            virtual int get_adjusted_input_cnt();
            virtual Tensor *get_neurons();
            virtual void set_neurons(Tensor *n);
            virtual void forward(Tensor *nxt_n, bool train_flg);
            virtual Tensor *backward(Tensor *dc) = 0;
            virtual void save(FILE *file_ptr);
        };

        class LearnableLayer : public Layer
        {
        protected:
            Tensor *w;
            Tensor *b;
            Tensor *dw;
            Tensor *db;

        public:
            LearnableLayer(std::vector<int> n_shape);
            LearnableLayer(FILE *file_ptr);
            ~LearnableLayer();

            virtual Tensor *get_weights();
            virtual Tensor *get_weight_derivatives();
            virtual Tensor *get_biases();
            virtual Tensor *get_bias_derivatives();
            virtual void save(FILE *file_ptr);
            virtual void step(int batch_size, float learning_rate) = 0;
        };

        class LinearLayer : public LearnableLayer
        {
        public:
            LinearLayer(std::vector<int> n_shape, int nxt_n_cnt, InitializationFunction init_fn);
            LinearLayer(FILE *file_ptr);
            ~LinearLayer();

            virtual LayerType get_type();
            virtual std::vector<int> get_output_shape();
            virtual void forward(Tensor *nxt_n, bool train_flg);
            virtual Tensor *backward(Tensor *dc);
            virtual void save(FILE *file_ptr);
            virtual void step(int batch_size, float learning_rate);
        };

        class ConvolutionalLayer : public LearnableLayer
        {
            // NOTE: we only support stride of 1!
        public:
            ConvolutionalLayer(std::vector<int> n_shape, int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn);
            ConvolutionalLayer(FILE *file_ptr);
            ~ConvolutionalLayer();

            virtual LayerType get_type();
            virtual std::vector<int> get_output_shape();
            virtual int get_adjusted_input_cnt();
            virtual void forward(Tensor *nxt_n, bool train_flg);
            virtual Tensor *backward(Tensor *dc);
            virtual void save(FILE *file_ptr);
            virtual void step(int batch_size, float learning_rate);
        };

        class ActivationLayer : public Layer
        {
        private:
            ActivationFunction activation_fn;

        public:
            ActivationLayer(std::vector<int> n_shape, ActivationFunction activation_fn);
            ActivationLayer(FILE *file_ptr);
            ~ActivationLayer();

            virtual LayerType get_type();
            virtual void forward(Tensor *nxt_n, bool train_flg);
            virtual Tensor *backward(Tensor *dc);
            virtual void save(FILE *file_ptr);
        };

        class DropoutLayer : public Layer
        {
        private:
            float dropout_rate;
            Tensor *dropout_mask;

        public:
            DropoutLayer(std::vector<int> n_shape, float dropout_rate);
            DropoutLayer(FILE *file_ptr);
            ~DropoutLayer();

            virtual LayerType get_type();
            virtual void forward(Tensor *nxt_n, bool train_flg);
            virtual Tensor *backward(Tensor *dc);
            virtual void save(FILE *file_ptr);
        };

        class PoolingLayer : public Layer
        {
        private:
            PoolingFunction pool_fn;
            int pool_row_cnt;
            int pool_col_cnt;

        public:
            PoolingLayer(std::vector<int> n_shape, PoolingFunction pool_fn);
            PoolingLayer(FILE *file_ptr);
            ~PoolingLayer();

            virtual LayerType get_type();
            std::vector<int> get_output_shape();
            virtual void forward(Tensor *nxt_n, bool train_flg);
            virtual Tensor *backward(Tensor *dc);
            virtual void save(FILE *file_ptr);
        };
    }
}