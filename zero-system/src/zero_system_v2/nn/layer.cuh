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

        class LinearLayer : public Layer
        {
        private:
            Tensor *w;
            Tensor *b;
            Tensor *dw;
            Tensor *db;

        public:
            LinearLayer(int n_cnt, int nxt_n_cnt, WeightInitializationType wgt_init_typ);
            ~LinearLayer();

            virtual void evaluate(Tensor *nxt_n);
            virtual void derive(Tensor *dc);
        };

        class ActivationLayer : public Layer
        {
        private:
            ActivationType typ;

        public:
            ActivationLayer(int n_cnt, ActivationType typ);
            ~ActivationLayer();

            virtual void evaluate(Tensor *nxt_n);
            virtual void derive(Tensor *dc);
        };
    }
}