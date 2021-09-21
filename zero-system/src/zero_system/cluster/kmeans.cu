#include "kmeans.cuh"

#define THREADS_PER_BLOCK 32
#define MAX_CLUSTER_CNT 128

using namespace zero::core;
using namespace zero::cluster;

// Device functions:

float __device__ d_get_cost(float x_val, float cluster_val)
{
    return ((x_val - cluster_val) * (x_val - cluster_val));
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

void __global__ k_reset(float *cluster_arr, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        cluster_arr[tid] = 0.0f;
    }
}

void __global__ k_assign(float *x_arr, float *assignment_arr, float *cluster_arr, float *cost, int feature_cnt, int cluster_cnt, int row_cnt)
{
    float temp[MAX_CLUSTER_CNT];
    memset(temp, 0, sizeof(float) * MAX_CLUSTER_CNT);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < row_cnt)
    {
        for (int feature_idx = 0; feature_idx < feature_cnt; feature_idx++)
        {
            for (int cluster_idx = 0; cluster_idx < cluster_cnt; cluster_idx++)
            {
                temp[cluster_idx] += d_get_cost(x_arr[tid * feature_cnt + feature_idx], cluster_arr[cluster_idx * feature_cnt + feature_idx]);
            }
        }

        for (int cluster_idx = 0; cluster_idx < cluster_cnt; cluster_idx++)
        {
            temp[cluster_idx] = sqrt(temp[cluster_idx]);
        }

        int min_cluster_idx = d_get_min(temp, cluster_cnt);

        assignment_arr[tid] = min_cluster_idx;

        if (cost != nullptr)
        {
            atomicAdd(cost, temp[min_cluster_idx]);
        }
    }
}

void __global__ k_update_part_1(float *x_arr, float *assignment_arr, float *cluster_arr, float *assignment_cnt_arr, int feature_cnt, int cluster_cnt, int row_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < row_cnt)
    {
        int cluster_idx = assignment_arr[tid];

        atomicAdd(&assignment_cnt_arr[cluster_idx], 1.0f);

        for (int feature_idx = 0; feature_idx < feature_cnt; feature_idx++)
        {
            atomicAdd(&cluster_arr[cluster_idx * feature_cnt + feature_idx], x_arr[tid * feature_cnt + feature_idx]);
        }
    }
}

void __global__ k_update_part_2(float *cluster_arr, float *assignment_cnt_arr, int cluster_cnt, int feature_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cluster_cnt * feature_cnt)
    {
        int cluster_idx = tid / feature_cnt;

        cluster_arr[tid] /= (assignment_cnt_arr[cluster_idx]);
    }
}

// KMeans member functions:

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

void KMeans::initialize_clusters(Tensor *x)
{
    this->clusters->translate(Cpu);

    std::vector<int> rand_nums;

    for (int i = 0; i < this->cluster_cnt; i++)
    {
        bool rand_num_already_added;
        int rand_num;

        do
        {
            rand_num_already_added = false;
            rand_num = rand() % x->get_row_cnt();

            for (int j = 0; j < rand_nums.size(); j++)
            {
                if (rand_nums[j] == rand_num)
                {
                    rand_num_already_added = true;
                    break;
                }
            }

        } while (rand_num_already_added);

        rand_nums.push_back(rand_num);
    }

    for (int cluster_idx = 0; cluster_idx < this->cluster_cnt; cluster_idx++)
    {
        int rand_row_idx = rand_nums[cluster_idx];
        memcpy(this->clusters->get_slice(cluster_idx * this->feature_cnt, Cpu), x->get_slice(rand_row_idx * x->get_col_cnt(), Cpu), sizeof(float) * this->feature_cnt);
    }

    this->clusters->translate(Gpu);
}

void KMeans::reset_clusters()
{
    this->clusters->set_all(0.0f);
}

float KMeans::train(Tensor *x)
{
    this->initialize_clusters(x);

    Tensor *assignments = new Tensor(x->get_row_cnt(), 1, Gpu);
    assignments->set_all(0.0f);

    Tensor *assignment_cnts = new Tensor(this->cluster_cnt, 1, Gpu);
    assignment_cnts->set_all(0.0f);

    int epoch = 1;

    float h_cost;
    float h_prv_cost = FLT_MAX;

    float *d_cost;
    cudaMalloc(&d_cost, sizeof(float));
    cudaMemset(d_cost, 0, sizeof(float));

    while (true)
    {
        // Assign xs to clusters:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((x->get_row_cnt() / threads_per_block) + 1);
            k_assign<<<num_blocks, threads_per_block>>>(x->get_arr(Gpu), assignments->get_arr(Gpu), this->clusters->get_arr(Gpu), d_cost, this->feature_cnt, this->cluster_cnt, x->get_row_cnt());
        }

        // Analyze cost:
        {
            cudaMemcpy(&h_cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost);

            h_cost /= x->get_row_cnt();

            if (h_prv_cost <= h_cost)
            {
                break;
            }

            h_prv_cost = h_cost;
        }

        // Reset clusters prior to update:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(((this->cluster_cnt * this->feature_cnt) / threads_per_block) + 1);
            k_reset<<<num_blocks, threads_per_block>>>(this->clusters->get_arr(Gpu), (this->cluster_cnt * this->feature_cnt));
        }

        // Update clusters:
        {

            // Update clusters part 1:
            {
                int threads_per_block(THREADS_PER_BLOCK);
        int num_blocks((x->get_row_cnt() / threads_per_block) + 1);
        k_update_part_1<<<num_blocks, threads_per_block>>>(x->get_arr(Gpu), assignments->get_arr(Gpu), this->clusters->get_arr(Gpu),
                                                           assignment_cnts->get_arr(Gpu), this->feature_cnt, this->cluster_cnt, x->get_row_cnt());
    }

    // Update clusters part 2:
    {
        int threads_per_block(THREADS_PER_BLOCK);
        int num_blocks(((this->cluster_cnt * this->feature_cnt) / threads_per_block) + 1);
        k_update_part_2<<<num_blocks, threads_per_block>>>(this->clusters->get_arr(Gpu), assignment_cnts->get_arr(Gpu), this->cluster_cnt, this->feature_cnt);
    }
}

// Reset for next epoch:
{
    cudaMemset(d_cost, 0, sizeof(float));
    assignment_cnts->set_all(0.0f);
}

epoch++;
}

    cudaFree(d_cost);

    delete assignments;
    delete assignment_cnts;

    return h_cost;
}

Tensor *KMeans::predict(Tensor *x)
{
    Tensor *assignments = new Tensor(x->get_row_cnt(), 1, Gpu);

    {
        int threads_per_block(THREADS_PER_BLOCK);
        int num_blocks((x->get_row_cnt() / threads_per_block) + 1);
        k_assign<<<num_blocks, threads_per_block, sizeof(float) * this->cluster_cnt>>>(x->get_arr(Gpu), assignments->get_arr(Gpu), this->clusters->get_arr(Gpu), nullptr,
                                                                                       this->feature_cnt, this->cluster_cnt, x->get_row_cnt());
    }

    assignments->translate(Cpu);

    return assignments;
}

// KMeans static functions:

void KMeans::dump_best(Tensor *x, int cluster_cnt, int iter_cnt, const char *path)
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

            printf("LOWEST COST: %f\n", min_cost);

            best_kmeans->clusters->set_arr(kmeans->clusters->get_arr(Cpu));
        }

        kmeans->reset_clusters();
    }

    best_kmeans->dump(path);

    delete kmeans;
    delete best_kmeans;
}