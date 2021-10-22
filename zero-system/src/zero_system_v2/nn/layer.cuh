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
            Activation,
            Dropout
        };

        class Layer
        {

        public:
            Tensor *n;

            Layer();
            ~Layer();

            virtual LayerType get_type() = 0;
            virtual void evaluate(Tensor *nxt_n, bool train_flg) = 0;
            virtual Tensor *derive(Tensor *dc) = 0;
            virtual void load(FILE *file_ptr) = 0;
            virtual void save(FILE *file_ptr) = 0;
        };

        class LearnableLayer : public Layer
        {
        public:
            Tensor *w;
            Tensor *b;
            Tensor *dw;
            Tensor *db;

            LearnableLayer();
            ~LearnableLayer();

            virtual void step(int batch_size, float learning_rate) = 0;
        };

        class LinearLayer : public LearnableLayer
        {
        public:
            LinearLayer();
            LinearLayer(int n_cnt, int nxt_n_cnt, InitializationFunction init_fn);
            ~LinearLayer();

            virtual LayerType get_type();
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
            ActivationLayer(int n_cnt, ActivationFunction activation_fn);
            ~ActivationLayer();

            virtual LayerType get_type();
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
            DropoutLayer(int n_cnt, float dropout_rate);
            ~DropoutLayer();

            virtual LayerType get_type();
            virtual void evaluate(Tensor *nxt_n, bool train_flg);
            virtual Tensor *derive(Tensor *dc);
            virtual void load(FILE *file_ptr);
            virtual void save(FILE *file_ptr);
        };
    }
}