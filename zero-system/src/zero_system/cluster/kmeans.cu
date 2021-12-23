#include "kmeans.cuh"

#define THREADS_PER_BLOCK 32
#define MAX_CLUSTER_CNT 256

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

void __global__ k_reset_arr(float *arr, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        arr[tid] = 0.0f;
    }
}

void __global__ k_assign_to_clusters(float *xs_arr, float *cluster_assignments_arr, float *cluster_arr, float *cost, int feature_cnt, int cluster_cnt, int row_cnt)
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
                temp[cluster_idx] += d_get_cost(xs_arr[tid * feature_cnt + feature_idx], cluster_arr[cluster_idx * feature_cnt + feature_idx]);
            }
        }

        for (int cluster_idx = 0; cluster_idx < cluster_cnt; cluster_idx++)
        {
            temp[cluster_idx] = sqrt(temp[cluster_idx]);
        }

        int min_cluster_idx = d_get_min(temp, cluster_cnt);

        cluster_assignments_arr[tid] = min_cluster_idx;

        if (cost != nullptr)
        {
            atomicAdd(cost, temp[min_cluster_idx]);
        }
    }
}

void __global__ k_update_clusters_part_1(float *xs_arr, float *cluster_assignments_arr, float *cluster_arr, float *cluster_assignment_cnts_arr, int feature_cnt, int cluster_cnt, int row_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < row_cnt)
    {
        int cluster_idx = cluster_assignments_arr[tid];

        atomicAdd(&cluster_assignment_cnts_arr[cluster_idx], 1.0f);

        for (int feature_idx = 0; feature_idx < feature_cnt; feature_idx++)
        {
            atomicAdd(&cluster_arr[cluster_idx * feature_cnt + feature_idx], xs_arr[tid * feature_cnt + feature_idx]);
        }
    }
}

void __global__ k_update_clusters_part_2(float *cluster_arr, float *cluster_assignment_cnts_arr, int cluster_cnt, int feature_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cluster_cnt * feature_cnt)
    {
        int cluster_idx = tid / feature_cnt;

        cluster_arr[tid] /= (cluster_assignment_cnts_arr[cluster_idx]);
    }
}

// KMeans member functions:

KMeans::KMeans(int cluster_cnt, int feature_cnt)
{
    this->cluster_cnt = cluster_cnt;
    this->feature_cnt = feature_cnt;

    this->clusters = new Tensor(Device::Cpu, cluster_cnt, feature_cnt);
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
    this->clusters = new Tensor(Device::Cpu, this->cluster_cnt, this->feature_cnt);
    this->clusters->set_arr(cluster_buf);
    this->clusters->to(Device::Cuda);

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

void KMeans::save(const char *path)
{
    FILE *file_ptr = fopen(path, "wb");

    int cluster_row_cnt = this->clusters->get_shape()[0];
    int cluster_col_cnt = this->clusters->get_shape()[1];

    fwrite(&this->cluster_cnt, sizeof(int), 1, file_ptr);
    fwrite(&this->feature_cnt, sizeof(int), 1, file_ptr);
    fwrite(this->clusters->get_arr(Cpu), sizeof(float), (cluster_row_cnt * cluster_col_cnt), file_ptr);

    fclose(file_ptr);
}

void KMeans::initialize_clusters(Tensor *xs)
{
    this->clusters->to(Device::Cpu);

    std::vector<int> rand_nums;
    rand_nums.reserve(this->cluster_cnt);

    int x_row_cnt = xs->get_shape()[0];
    int x_col_cnt = xs->get_shape()[1];

    for (int i = 0; i < this->cluster_cnt; i++)
    {
        bool rand_num_already_added;
        int rand_num;

        do
        {
            rand_num_already_added = false;
            rand_num = rand() % x_row_cnt;

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
        cudaMemcpy(&this->clusters->get_arr()[cluster_idx * this->feature_cnt],
                   &xs->get_arr()[rand_row_idx * x_col_cnt], sizeof(float) * this->feature_cnt, cudaMemcpyDefault);
    }

    this->clusters->to(Device::Cuda);
    xs->to(Device::Cuda);
}

void KMeans::reset_clusters()
{
    int threads_per_block(THREADS_PER_BLOCK);
    int num_blocks(((this->cluster_cnt * this->feature_cnt) / threads_per_block) + 1);
    k_reset_arr<<<num_blocks, threads_per_block>>>(this->clusters->get_arr(), (this->cluster_cnt * this->feature_cnt));
}

float KMeans::train(Tensor *xs)
{
    this->initialize_clusters(xs);

    int x_row_cnt = xs->get_shape()[0];

    Tensor *cluster_assignments = new Tensor(Device::Cuda, x_row_cnt, 1);
    cluster_assignments->set_all(0.0f);

    Tensor *cluster_assignment_cnts = new Tensor(Device::Cuda, this->cluster_cnt, 1);
    cluster_assignment_cnts->set_all(0.0f);

    int epoch = 1;

    float h_cost_val;
    float h_prv_cost_val = FLT_MAX;

    float *d_cost_val;
    cudaMalloc(&d_cost_val, sizeof(float));
    cudaMemset(d_cost_val, 0, sizeof(float));

    while (true)
    {
        // Assign xs to clusters:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((x_row_cnt / threads_per_block) + 1);
            k_assign_to_clusters<<<num_blocks, threads_per_block>>>(xs->get_arr(), cluster_assignments->get_arr(), this->clusters->get_arr(), d_cost_val, this->feature_cnt, this->cluster_cnt, x_row_cnt);
        }

        // Analyze cost:
        {
            cudaMemcpy(&h_cost_val, d_cost_val, sizeof(float), cudaMemcpyDeviceToHost);

            h_cost_val /= x_row_cnt;

            if (h_prv_cost_val <= h_cost_val)
            {
                break;
            }

            h_prv_cost_val = h_cost_val;
        }

        // Reset clusters prior to update:
        this->reset_clusters();

        // Update clusters part 1:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((x_row_cnt / threads_per_block) + 1);
            k_update_clusters_part_1<<<num_blocks, threads_per_block>>>(xs->get_arr(), cluster_assignments->get_arr(), this->clusters->get_arr(),
                                                                        cluster_assignment_cnts->get_arr(), this->feature_cnt, this->cluster_cnt, x_row_cnt);
        }

        // Update clusters part 2:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(((this->cluster_cnt * this->feature_cnt) / threads_per_block) + 1);
            k_update_clusters_part_2<<<num_blocks, threads_per_block>>>(this->clusters->get_arr(), cluster_assignment_cnts->get_arr(), this->cluster_cnt, this->feature_cnt);
        }

        // Reset cost and assignment counts for next epoch:
        {
            cudaMemset(d_cost_val, 0, sizeof(float));

            cluster_assignment_cnts->reset();
        }

        epoch++;
    }

    cudaFree(d_cost_val);

    delete cluster_assignments;
    delete cluster_assignment_cnts;

    return h_cost_val;
}

Tensor *KMeans::predict(Tensor *xs)
{
    int x_row_cnt = xs->get_shape()[0];

    Tensor *cluster_assignments = new Tensor(Device::Cuda, x_row_cnt, 1);

    {
        int threads_per_block(THREADS_PER_BLOCK);
        int num_blocks((x_row_cnt / threads_per_block) + 1);
        k_assign_to_clusters<<<num_blocks, threads_per_block>>>(xs->get_arr(), cluster_assignments->get_arr(), this->clusters->get_arr(), nullptr,
                                                                this->feature_cnt, this->cluster_cnt, x_row_cnt);
    }

    cluster_assignments->to(Device::Cpu);

    return cluster_assignments;
}

// KMeans static functions:

float KMeans::save_best(Tensor *xs, int cluster_cnt, int iter_cnt, const char *path)
{
    int x_col_cnt = xs->get_shape()[1];

    KMeans *kmeans = new KMeans(cluster_cnt, x_col_cnt);
    KMeans *best_kmeans = new KMeans(cluster_cnt, x_col_cnt);

    float cost;
    float min_cost = FLT_MAX;

    for (int i = 0; i < iter_cnt; i++)
    {
        cost = kmeans->train(xs);

        if (cost < min_cost)
        {
            best_kmeans->clusters->copy(kmeans->clusters);

            min_cost = cost;
        }

        kmeans->reset_clusters();
    }

    best_kmeans->save(path);

    delete kmeans;
    delete best_kmeans;

    return min_cost;
}

void KMeans::run_elbow_analysis(Tensor *xs, int cluster_cnt_lower, int cluster_cnt_upper,
                                int iter_cnt, const char *csv_path)
{
    FILE *csv_file_ptr = fopen(csv_path, "w");
    fprintf(csv_file_ptr, "cluster_cnt,min_cost\n");

    int x_col_cnt = xs->get_shape()[1];

    for (int cluster_cnt = cluster_cnt_lower; cluster_cnt < cluster_cnt_upper; cluster_cnt++)
    {
        KMeans *kmeans = new KMeans(cluster_cnt, x_col_cnt);
        KMeans *best_kmeans = new KMeans(cluster_cnt, x_col_cnt);

        float cost;
        float min_cost = FLT_MAX;

        for (int i = 0; i < iter_cnt; i++)
        {
            cost = kmeans->train(xs);

            if (cost < min_cost)
            {
                best_kmeans->clusters->copy(kmeans->clusters);

                min_cost = cost;
            }

            kmeans->reset_clusters();
        }

        printf("CLUSTERS: %d\tCOST: %f\n", cluster_cnt, min_cost);

        fprintf(csv_file_ptr, "%d,%f\n", cluster_cnt, min_cost);

        delete kmeans;
        delete best_kmeans;
    }

    fclose(csv_file_ptr);
}