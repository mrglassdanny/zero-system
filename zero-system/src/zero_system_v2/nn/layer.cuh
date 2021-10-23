#pragma once

#include "../core/tensor.cuh"
#include "util.cuh"

namespace zero_v2
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

            Layer();
            Layer(std::vector<int> n_shape);
            ~Layer();

            virtual LayerType get_type() = 0;
            virtual void evaluate(Tensor *nxt_n, bool train_flg);
            virtual Tensor *derive(Tensor *dc) = 0;
            virtual void load(FILE *file_ptr) = 0;
            virtual void save(FILE *file_ptr) = 0;
            virtual std::vector<int> get_input_shape() = 0;
            virtual std::vector<int> get_output_shape() = 0;
        };

        class LearnableLayer : public Layer
        {
        public:
            Tensor *w;
            Tensor *b;
            Tensor *dw;
            Tensor *db;

            LearnableLayer();
            LearnableLayer(std::vector<int> n_shape);
            ~LearnableLayer();

            virtual void step(int batch_size, float learning_rate) = 0;
        };

        class LinearLayer : public LearnableLayer
        {
        public:
            LinearLayer();
            LinearLayer(std::vector<int> n_shape, int nxt_n_cnt, InitializationFunction init_fn);
            ~LinearLayer();

            virtual LayerType get_type();
            virtual std::vector<int> get_input_shape();
            virtual std::vector<int> get_output_shape();
            virtual void evaluate(Tensor *nxt_n, bool train_flg);
            virtual Tensor *derive(Tensor *dc);
            virtual void load(FILE *file_ptr);
            virtual void save(FILE *file_ptr);

            virtual void step(int batch_size, float learning_rate);
        };

        class ConvolutionalLayer : public LearnableLayer
        {
        public:
            ConvolutionalLayer();
            ConvolutionalLayer(std::vector<int> n_shape,
                               int fltr_cnt, int w_row_cnt, int w_col_cnt,
                               InitializationFunction init_fn);
            ~ConvolutionalLayer();

            virtual LayerType get_type();
            virtual std::vector<int> get_input_shape();
            virtual std::vector<int> get_output_shape();
            virtual void evaluate(Tensor *nxt_n, bool train_flg);
            virtual Tensor *derive(Tensor *dc);
            virtual void load(FILE *file_ptr);
            virtual void save(FILE *file_ptr);

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
            virtual std::vector<int> get_input_shape();
            virtual std::vector<int> get_output_shape();
            virtual void evaluate(Tensor *nxt_n, bool train_flg);
            virtual Tensor *derive(Tensor *dc);
            virtual void load(FILE *file_ptr);
            virtual void save(FILE *file_ptr);
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
            virtual std::vector<int> get_input_shape();
            virtual std::vector<int> get_output_shape();
            virtual void evaluate(Tensor *nxt_n, bool train_flg);
            virtual Tensor *derive(Tensor *dc);
            virtual void load(FILE *file_ptr);
            virtual void save(FILE *file_ptr);
        };
    }
}