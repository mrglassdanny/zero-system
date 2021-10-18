
#include <iostream>

#include <zero_system/nn/nn.cuh>
#include <zero_system/nn/cnn.cuh>

using namespace zero::core;
using namespace zero::nn;

int main(int argc, char **argv)
{

    Tensor *x = Tensor::from_csv("C:\\Users\\danie\\OneDrive\\Desktop\\palpck-x.csv");
    Tensor *y = Tensor::from_csv("C:\\Users\\danie\\OneDrive\\Desktop\\palpck-y.csv");

    Supervisor *sup = new Supervisor(x->get_row_cnt(), x->get_col_cnt(), 0,
                                     x->get_arr(Cpu), y->get_arr(Cpu), Cpu);

    NN *nn = new NN(MSE, 0.01f);

    nn->add_layer(x->get_col_cnt());
    nn->add_layer(18, ReLU);
    nn->add_layer(12, ReLU);
    nn->add_layer(6, ReLU);
    nn->add_layer(y->get_col_cnt(), ReLU);

    nn->compile();

    nn->all(sup, 100, 1000, "C:\\Users\\danie\\OneDrive\\Desktop\\lm-zero-train.csv");

    nn->save("C:\\Users\\danie\\OneDrive\\Desktop\\lm-zero.nn");

    delete nn;
    delete sup;

    delete x;
    delete y;

    return 0;
}