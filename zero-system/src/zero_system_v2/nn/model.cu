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
    int lst_lyr_idx = this->layers.size() - 1;

    Layer *fst_lyr = this->layers[0];
    Layer *lst_lyr = this->layers[lst_lyr_idx];

    for (int i = 0; i < lst_lyr_idx; i++)
    {
        Layer *lyr = this->layers[i];
        Layer *nxt_lyr = this->layers[i + 1];

        lyr->evaluate(nxt_lyr->n);
    }

    //lst_lyr->evaluate();
}

void Model::backward(Tensor *y)
{
}