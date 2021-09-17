#include "Batch.cuh"

using namespace nn;

Batch::Batch()
{
}

Batch::~Batch()
{
}

void Batch::add(Tensor *x, Tensor *y)
{
    this->xs.push_back(x);
    this->ys.push_back(y);
}

int Batch::get_size()
{
    return this->xs.size();
}

Tensor *Batch::get_x(int idx)
{
    return this->xs[idx];
}

Tensor *Batch::get_y(int idx)
{
    return this->ys[idx];
}