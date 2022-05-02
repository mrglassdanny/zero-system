#pragma once

#include "../core/mod.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        enum ActivationFunction
        {
            None,
            ReLU,
            LeakyReLU,
            AbsoluteValue,
            Sigmoid,
            Tanh,
            Sine,
            Cosine
        };

        enum PoolingFunction
        {
            Average,
            Max
        };

        enum CostFunction
        {
            MSE,
            CrossEntropy
        };

        enum InitializationFunction
        {
            He,
            Xavier,
            Zeros,
            Ones,
            Simple
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

        typedef void (*UpdateResultFn)(Tensor *, Tensor *, int *);

        class Report
        {
        public:
            float cost;
            int correct_cnt;
            int total_cnt;

            void print();
            void update(Tensor *n, Tensor *y, UpdateResultFn fn);
        };

        class CSVUtils
        {
        public:
            static void write_csv_header(FILE *csv_file_ptr);
            static void write_to_csv(FILE *csv_file_ptr, int epoch, int iteration, Report rpt);
        };
    }
}