
#include <iostream>

#include <zero_system/nn/model.cuh>
#include <zero_system/cluster/kmeans.cuh>

using namespace zero::core;
using namespace zero::nn;
using namespace zero::cluster;

int main(int argc, char **argv)
{
    srand(time(NULL));

    Tensor *xs = Tensor::fr_csv("data/locmst.csv");

    KMeans::save_best(xs, 30, 10, "temp/model.km");

    KMeans *km = new KMeans("temp/model.km");

    km->print();

    delete km;

    delete xs;

    return 0;
}