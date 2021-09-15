#pragma once

#include <vector>

#include "Tensor.cuh"

class Batch
{
private:
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