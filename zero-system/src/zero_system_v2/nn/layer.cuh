#pragma once

#include "../core/tensor.cuh"
#include "util.cuh"

namespace zero_v2
{
    using namespace core;

    namespace nn
    {
        class Layer
        {

        public:
            Tensor *n;

            Layer();
            ~Layer();

            virtual void evaluate(Tensor *nxt_n) = 0;
            virtual void derive(Tensor *dc) = 0;
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

            virtual void evaluate(Tensor *nxt_n) = 0;
            virtual void derive(Tensor *dc) = 0;
            virtual void step(int batch_size, float learning_rate) = 0;
            virtual void load(FILE *file_ptr) = 0;
            virtual void save(FILE *file_ptr) = 0;
        };

        class LinearLayer : public LearnableLayer
        {
        public:
            LinearLayer(int n_cnt, int nxt_n_cnt, InitializationFunction init_fn);
            ~LinearLayer();

            virtual void evaluate(Tensor *nxt_n);
            virtual void derive(Tensor *dc);
            virtual void step(int batch_size, float learning_rate);
            virtual void load(FILE *file_ptr);
            virtual void save(FILE *file_ptr);
        };

        class ActivationLayer : public Layer
        {
        private:
            ActivationFunction activation_fn;

        public:
            ActivationLayer(int n_cnt, ActivationFunction activation_fn);
            ~ActivationLayer();

            virtual void evaluate(Tensor *nxt_n);
            virtual void derive(Tensor *dc);
        };
    }
}