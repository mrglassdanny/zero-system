#pragma once

#include "../core/tensor.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        enum ActivationFunction
        {
            ReLU,
            LeakyReLU,
            AbsoluteValue,
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
            Max
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
            static void initialize(InitializationFunction init_fn, Tensor *tensor, int fan_in, int fan_out);
        };

        class Report
        {
        public:
            float cost;
            int correct_cnt;
            int total_cnt;

            void print();
            void update_correct_cnt(Tensor *n, Tensor *y);
        };

        class CSVUtils
        {
        public:
            static void write_csv_header(FILE *csv_file_ptr);
            static void write_to_csv(FILE *csv_file_ptr, int epoch, int iteration, Report rpt);
        };
    }
}