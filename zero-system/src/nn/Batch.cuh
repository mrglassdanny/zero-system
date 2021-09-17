#pragma once

#include <vector>

#include "../Tensor.cuh"

namespace nn
{
    class Batch
    {
    private:
        // Batch does NOT own Tensors!
        std::vector<Tensor *> xs;
        std::vector<Tensor *> ys;

    public:
        Batch();
        ~Batch();

        void add(Tensor *x, Tensor *y);

        int get_size();
        Tensor *get_x(int idx);
        Tensor *get_y(int idx);
    };
}
