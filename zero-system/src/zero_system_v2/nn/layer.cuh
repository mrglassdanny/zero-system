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

            virtual void evaluate(Tensor *n) = 0;
            virtual void derive(Tensor *d) = 0;
        };

        class DenseLayer : public Layer
        {
        private:
            Tensor *w;
            Tensor *b;
            Tensor *dw;
            Tensor *db;

        public:
            DenseLayer();
            ~DenseLayer();

            virtual void evaluate(Tensor *n);
            virtual void derive(Tensor *dc);
        };

        class ActivationLayer : public Layer
        {
        private:
            ActivationType typ;

        public:
            ActivationLayer(ActivationType typ);
            ~ActivationLayer();

            virtual void evaluate(Tensor *n);
            virtual void derive(Tensor *dc);
        };
    }
}