
#include <iostream>

#include <zero_system/nn/model.cuh>

using namespace zero::core;
using namespace zero::nn;

int main(int argc, char **argv)
{

    Tensor *x = Tensor::from_csv("C:\\Users\\danie\\OneDrive\\Desktop\\palpck-x.csv");
    Tensor *y = Tensor::from_csv("C:\\Users\\danie\\OneDrive\\Desktop\\palpck-y.csv");

    InMemorySupervisor *sup = new InMemorySupervisor(0.90f, 0.10f, x->get_cnt(), x->get_shape()[0],
                                                     0, x->get_arr(), y->get_arr(), Device::Cpu);

    Model *model = new Model(CostFunction::MSE, 0.01f);

    model->add_layer(new LinearLayer(x->get_shape(), 18, InitializationFunction::Xavier));
    model->add_layer(new LinearLayer(model->get_output_shape(), 12, InitializationFunction::Xavier));
    model->add_layer(new LinearLayer(model->get_output_shape(), 6, InitializationFunction::Xavier));
    model->add_layer(new LinearLayer(model->get_output_shape(), y->get_shape()[0], InitializationFunction::Xavier));

    model->all(sup, 100, 1000, "C:\\Users\\danie\\OneDrive\\Desktop\\lm-zero-train.csv");

    model->save("C:\\Users\\danie\\OneDrive\\Desktop\\lm-zero.nn");

    delete model;
    delete sup;

    delete x;
    delete y;

    return 0;
}