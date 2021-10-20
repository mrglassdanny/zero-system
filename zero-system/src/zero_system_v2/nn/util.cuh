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

        enum PoolingType
        {
            Average,
            Max,
            Global
        };

        enum WeightInitializationType
        {
            He,
            Xavier,
            Zeros
        };
    }
}