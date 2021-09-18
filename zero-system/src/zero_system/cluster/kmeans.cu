#include "kmeans.cuh"

using namespace zero::core;
using namespace zero::cluster;

KMeans::KMeans(int cluster_cnt, int feature_cnt)
{
    this->cluster_cnt = cluster_cnt;
    this->feature_cnt = feature_cnt;

    this->clusters = new Tensor(cluster_cnt, feature_cnt, Gpu);
}

KMeans::KMeans(const KMeans &src)
{
    this->cluster_cnt = src.cluster_cnt;
    this->feature_cnt = src.feature_cnt;
    this->clusters = new Tensor(*src.clusters);
}

KMeans::KMeans(const char *path)
{
}

KMeans::~KMeans()
{
    delete this->clusters;
}

void KMeans::print()
{
}

void KMeans::dump_to_file(const char *path)
{
}

float KMeans::train(Tensor *x, int train_chk_freq)
{
}

Tensor *get_input_assignments(Tensor *x)
{
}