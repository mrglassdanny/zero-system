#pragma once

#include "../core/mod.cuh"

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
            None,
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

        typedef void (*upd_rslt_fn)(Tensor *, Tensor *, int *);

        class Report
        {
        public:
            float cost;
            int correct_cnt;
            int total_cnt;

            void print();
            void update(Tensor *n, Tensor *y, upd_rslt_fn fn);
        };

        class CSVUtils
        {
        public:
            static void write_csv_header(FILE *csv_file_ptr);
            static void write_to_csv(FILE *csv_file_ptr, int epoch, int iteration, Report rpt);
        };
    }
}