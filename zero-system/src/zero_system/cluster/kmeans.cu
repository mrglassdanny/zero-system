#include "kmeans.cuh"

#define THREADS_PER_BLOCK 32

using namespace zero::core;
using namespace zero::cluster;

// Device functions:

float __device__ d_calc_cost(float a, float b)
{
    return pow(a - b, 2.0f);
}

int __device__ d_get_min(float *arr, int cnt)
{

    int min_idx = 0;
    float min_val = FLT_MAX;

    for (int i = 0; i < cnt; i++)
    {
        float cur_val = arr[i];
        if (cur_val < min_val)
        {
            min_idx = i;
            min_val = cur_val;
        }
    }

    return min_idx;
}

// Kernel functions:

void __global__ k_assign(float *x, float *assignments, float *clusters, int feature_cnt, int cluster_cnt, float *cost)
{
    extern __shared__ float temp[];
    memset(temp, 0, sizeof(float) * cluster_cnt);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < feature_cnt; i++)
    {
        for (int j = 0; j < cluster_cnt; j++)
        {
            float cost = d_calc_cost(x[tid * feature_cnt + i], clusters[j * feature_cnt + i]);
            temp[j] += cost;
        }
    }

    for (int j = 0; j < cluster_cnt; j++)
    {
        temp[j] = sqrt(temp[j]);
    }

    int idx = d_get_min(temp, cluster_cnt);

    assignments[tid] = idx;

    if (cost != nullptr)
    {
        *cost += temp[idx];
    }
}

void __global__ k_update(float *x, float *assignments, float *clusters, float *assignment_cnts, int feature_cnt, int cluster_cnt)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int cluster_idx = assignments[tid];
    assignment_cnts[cluster_idx]++;

    for (int i = 0; i < feature_cnt; i++)
    {
        atomicAdd(&clusters[cluster_idx * feature_cnt + i], x[tid * feature_cnt + i]);
    }
}

void __global__ k_div(float *clusters, float *assignment_cnts, int cluster_cnt, int feature_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int cluster_idx = tid / cluster_cnt;
    int feature_idx = tid % feature_cnt;

    clusters[cluster_idx * feature_cnt + feature_idx] /= assignment_cnts[cluster_idx];
}

// KMeans member functions:

void KMeans::set(Tensor *x)
{
    this->clusters->translate(Cpu);

    for (int cluster_idx = 0; cluster_idx < this->cluster_cnt; cluster_idx++)
    {
        int rand_row_idx = rand() % x->get_row_cnt();
        memcpy(this->clusters->get_slice(cluster_idx * this->feature_cnt, Cpu), x->get_slice(rand_row_idx * x->get_col_cnt(), Cpu), sizeof(float) * this->feature_cnt);
    }

    this->clusters->translate(Gpu);
}

void KMeans::reset()
{
    this->clusters->set_all(0.0f);
}

KMeans::KMeans(int cluster_cnt, int feature_cnt)
{
    this->cluster_cnt = cluster_cnt;
    this->feature_cnt = feature_cnt;

    this->clusters = new Tensor(cluster_cnt, feature_cnt, Cpu);
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

float KMeans::train(Tensor *x)
{
    this->set(x);

    Tensor *assignments = new Tensor(x->get_row_cnt(), 1, Gpu);

    int epoch = 1;

    float h_cost;
    float h_prv_cost = FLT_MAX;

    float *d_cost;
    cudaMalloc(&d_cost, sizeof(float));
    cudaMemset(d_cost, 0, sizeof(float));

    Tensor *assignment_cnts = new Tensor(this->cluster_cnt, 1, Gpu);

    while (true)
    {

        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((x->get_row_cnt() / threads_per_block) + 1);
            k_assign<<<num_blocks, threads_per_block, sizeof(float) * this->cluster_cnt>>>(x->get_arr(Gpu), assignments->get_arr(Gpu), this->clusters->get_arr(Gpu),
                                                                                           this->feature_cnt, this->cluster_cnt, d_cost);
        }

        cudaMemcpy(&h_cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost);

        h_cost /= x->get_row_cnt();

        if (h_prv_cost <= h_cost)
        {
            break;
        }

        h_prv_cost = h_cost;

        {
            {
                int threads_per_block(THREADS_PER_BLOCK);
                int num_blocks((x->get_row_cnt() / threads_per_block) + 1);
                k_update<<<num_blocks, threads_per_block>>>(x->get_arr(Gpu), assignments->get_arr(Gpu), this->clusters->get_arr(Gpu),
                                                            assignment_cnts->get_arr(Gpu), this->feature_cnt, this->cluster_cnt);
            }

            for (int i = 0; i < this->cluster_cnt; i++)
            {
                int threads_per_block(THREADS_PER_BLOCK);
                int num_blocks(((this->cluster_cnt * this->feature_cnt) / threads_per_block) + 1);
                k_div<<<num_blocks, threads_per_block>>>(this->clusters->get_arr(Gpu), assignment_cnts->get_arr(Gpu), cluster_cnt, feature_cnt);
            }
        }

        cudaMemset(d_cost, 0, sizeof(float));
        assignment_cnts->set_all(0.0f);

        epoch++;
    }

    cudaFree(d_cost);

    return h_cost;
}

Tensor *KMeans::predict(Tensor *x)
{
    Tensor *assignments = new Tensor(x->get_row_cnt(), 1, Gpu);

    {
        int threads_per_block(THREADS_PER_BLOCK);
        int num_blocks((x->get_row_cnt() / threads_per_block) + 1);
        k_assign<<<num_blocks, threads_per_block, sizeof(float) * this->cluster_cnt>>>(x->get_arr(Gpu), assignments->get_arr(Gpu), this->clusters->get_arr(Gpu),
                                                                                       this->feature_cnt, this->cluster_cnt, nullptr);
    }

    assignments->translate(Cpu);

    return assignments;
}

// KMeans static functions:

void KMeans::find_best(Tensor *x, int cluster_cnt, int iter_cnt, const char *path)
{
    KMeans *kmeans = new KMeans(cluster_cnt, x->get_col_cnt());
    KMeans *best_kmeans = new KMeans(cluster_cnt, x->get_col_cnt());

    float cost;
    float min_cost = FLT_MAX;

    for (int i = 0; i < iter_cnt; i++)
    {
        cost = kmeans->train(x);

        if (cost < min_cost)
        {
            min_cost = cost;

            best_kmeans->clusters->set_arr(kmeans->clusters->get_arr(Cpu));
        }

        kmeans->reset();
    }

    best_kmeans->dump(path);

    delete kmeans;
    delete best_kmeans;
}