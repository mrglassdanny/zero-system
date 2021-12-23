
#include <iostream>

#include <zero_system/nn/model.cuh>
#include <zero_system/cluster/kmeans.cuh>

using namespace zero::core;
using namespace zero::nn;
using namespace zero::cluster;

int main(int argc, char **argv)
{
    srand(time(NULL));

    Tensor *xs = Tensor::fr_csv("data/locmst-encoded.csv");

    // KMeans::run_elbow_analysis(xs, 100, 200, 20, "temp/elbow-analysis.csv");

    printf("MIN COST: %f\n", KMeans::save_best(xs, 200, 1000, "temp/model.km"));

    KMeans *model = new KMeans("temp/model.km");

    Tensor *preds = model->predict(xs);

    preds->to_csv("temp/preds.csv");

    delete preds;
    delete model;

    delete xs;

    return 0;
}