#pragma once

#include "../core/tensor.cuh"

namespace zero_v2
{
    using namespace core;

    namespace nn
    {
        enum ActivationFunction
        {
            None,
            ReLU,
            Sigmoid,
            Tanh,
            Sine,
            Cosine
        };

        enum CostFunction
        {
            MSE,
            CrossEntropy
        };

        enum PoolingFunction
        {
            Average,
            Max,
            Global
        };

        enum InitializationFunction
        {
            He,
            Xavier,
            Zeros
        };

        enum OptimizationFunction
        {
            GradientDescent,
            Momentum,
            RMSProp,
            Adam
        };

        class Initializer
        {
        public:
            static void initialize(InitializationFunction init_fn, Tensor *out);
        };
    }
}