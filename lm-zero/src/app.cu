
#include <iostream>

#include <zero_system/nn/model.cuh>

using namespace zero::core;
using namespace zero::nn;

int main(int argc, char **argv)
{

    Tensor *x = Tensor::from_csv("C:\\Users\\danie\\OneDrive\\Desktop\\palpck-x.csv");
    Tensor *y = Tensor::from_csv("C:\\Users\\danie\\OneDrive\\Desktop\\palpck-y.csv");

    InMemorySupervisor *sup = new InMemorySupervisor(0.90f, 0.10f, x->get_shape()[0], x->get_shape()[1],
                                                     0, x->get_arr(), y->get_arr(), Device::Cpu);

    Model *model = new Model(CostFunction::MSE, 0.001f);

    std::vector<int> x_shape{x->get_shape()[1]};

    model->add_layer(new LinearLayer(x_shape, 64, InitializationFunction::Xavier));
    model->add_layer(new LinearLayer(model->get_output_shape(), 32, InitializationFunction::Xavier));
    model->add_layer(new LinearLayer(model->get_output_shape(), 8, InitializationFunction::Xavier));
    model->add_layer(new LinearLayer(model->get_output_shape(), 1, InitializationFunction::Xavier));

    model->train_and_test(sup, 100, 5, "C:\\Users\\danie\\OneDrive\\Desktop\\lm-zero-train.csv");

    model->save("C:\\Users\\danie\\OneDrive\\Desktop\\lm-zero.nn");

    delete model;
    delete sup;

    delete x;
    delete y;

    return 0;
}