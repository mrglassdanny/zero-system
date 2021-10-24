#pragma once

#include "../core/tensor.cuh"
#include "util.cuh"

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
            Dropout
        };

        class Layer
        {

        public:
            Tensor *n;

            Layer(std::vector<int> n_shape);
            Layer(FILE *file_ptr);
            ~Layer();

            virtual LayerType get_type() = 0;
            virtual std::vector<int> get_input_shape() = 0;
            virtual std::vector<int> get_output_shape() = 0;
            virtual void evaluate(Tensor *nxt_n, bool train_flg) = 0;
            virtual Tensor *derive(Tensor *dc) = 0;
            virtual void save(FILE *file_ptr);
        };

        class LearnableLayer : public Layer
        {
        public:
            Tensor *w;
            Tensor *b;
            Tensor *dw;
            Tensor *db;

            LearnableLayer(std::vector<int> n_shape);
            LearnableLayer(FILE *file_ptr);
            ~LearnableLayer();

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
            virtual std::vector<int> get_input_shape();
            virtual std::vector<int> get_output_shape();
            virtual void evaluate(Tensor *nxt_n, bool train_flg);
            virtual Tensor *derive(Tensor *dc);
            virtual void save(FILE *file_ptr);

            virtual void step(int batch_size, float learning_rate);
        };

        class ConvolutionalLayer : public LearnableLayer
        {
        public:
            ConvolutionalLayer(std::vector<int> n_shape,
                               int fltr_cnt, int w_row_cnt, int w_col_cnt,
                               InitializationFunction init_fn);
            ConvolutionalLayer(FILE *file_ptr);
            ~ConvolutionalLayer();

            virtual LayerType get_type();
            virtual std::vector<int> get_input_shape();
            virtual std::vector<int> get_output_shape();
            virtual void evaluate(Tensor *nxt_n, bool train_flg);
            virtual Tensor *derive(Tensor *dc);
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
            virtual std::vector<int> get_input_shape();
            virtual std::vector<int> get_output_shape();
            virtual void evaluate(Tensor *nxt_n, bool train_flg);
            virtual Tensor *derive(Tensor *dc);
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
            virtual std::vector<int> get_input_shape();
            virtual std::vector<int> get_output_shape();
            virtual void evaluate(Tensor *nxt_n, bool train_flg);
            virtual Tensor *derive(Tensor *dc);
            virtual void save(FILE *file_ptr);
        };
    }
}