#pragma once

#include <vector>

#include "../core/tensor.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        enum ActivationFunctionId
        {
            None,
            ReLU,
            Sigmoid,
            Tanh,
            Sine,
            Cosine
        };

        enum CostFunctionId
        {
            MSE,
            CrossEntropy
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

        void write_csv_header(FILE *csv_file_ptr);
        void write_to_csv(FILE *csv_file_ptr, int epoch, Report rpt);
    }
}