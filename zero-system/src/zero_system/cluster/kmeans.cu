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
    FILE *file_ptr = fopen(path, "rb");

    fread(&this->cluster_cnt, sizeof(int), 1, file_ptr);
    fread(&this->feature_cnt, sizeof(int), 1, file_ptr);

    int tot_cnt = (this->cluster_cnt * this->feature_cnt);
    float *cluster_buf = (float *)malloc(sizeof(float) * tot_cnt);
    fread(cluster_buf, sizeof(float), tot_cnt, file_ptr);
    this->clusters = new Tensor(this->cluster_cnt, this->feature_cnt, Gpu, cluster_buf);
    free(cluster_buf);

    fclose(file_ptr);
}

KMeans::~KMeans()
{
    delete this->clusters;
}

void KMeans::print()
{
    this->clusters->print();
}

void KMeans::dump(const char *path)
{
    FILE *file_ptr = fopen(path, "wb");

    fwrite(&this->cluster_cnt, sizeof(int), 1, file_ptr);
    fwrite(&this->feature_cnt, sizeof(int), 1, file_ptr);
    fwrite(this->clusters->get_arr(Cpu), sizeof(float), (this->clusters->get_row_cnt() * this->clusters->get_col_cnt()), file_ptr);

    fclose(file_ptr);
}

float KMeans::train(Tensor *x, int train_chk_freq)
{
    return 0.0f;
}

Tensor *get_input_assignments(Tensor *x)
{
    return nullptr;
}