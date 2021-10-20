#include "model.cuh"

using namespace zero_v2::core;
using namespace zero_v2::nn;

Model::Model()
{
}

Model::~Model()
{
    for (Layer *lyr : this->layers)
    {
        delete lyr;
    }
}

void Model::add_layer(Layer *lyr)
{
    this->layers.push_back(lyr);
}

void Model::forward(Tensor *x)
{
}

void Model::backward(Tensor *y)
{
}