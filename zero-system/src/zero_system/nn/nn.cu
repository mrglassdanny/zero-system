#include "nn.cuh"

#define THREADS_PER_BLOCK 32

using namespace zero::nn;
using namespace zero::core;

// Device functions:

__device__ float d_relu(float val)
{
    return val > 0.0f ? val : 0.0f;
}

__device__ float d_derive_relu(float val)
{
    return val > 0.0f ? 1.0f : 0.0f;
}

__device__ float d_sigmoid(float val)
{
    return (1.0 / (1.0 + exp(-val)));
}

__device__ float d_derive_sigmoid(float val)
{
    return (val) * (1.0 - val);
}

__device__ float d_tanh(float val)
{
    return ((exp(val) - exp(-val)) / (exp(val) + exp(-val)));
}

__device__ float d_derive_tanh(float val)
{
    return (1 - (val * val));
}

__device__ float d_mse_cost(float n_val, float y_val)
{
    return ((n_val - y_val) * (n_val - y_val));
}

__device__ float d_derive_mse_cost(float n_val, float y_val)
{
    return 2.0f * (n_val - y_val);
}

__device__ float d_cross_entropy_cost(float n_val, float y_val)
{
    return (float)((y_val * log(n_val)) + ((1.0 - y_val) * log(1.0 - n_val)));
}

__device__ float d_derive_cross_entropy_cost(float n_val, float y_val)
{
    return (n_val - y_val);
}

// Kernel functions:

__global__ void k_set_arr(float *arr, int cnt, float val)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        arr[tid] = val;
    }
}

__global__ void k_dot(float *n_arr, float *w_arr, float *nxt_n_arr, int n_cnt, int nxt_n_cnt)
{
    __shared__ float temp[THREADS_PER_BLOCK];
    memset(temp, 0, THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int w_cnt = n_cnt * nxt_n_cnt;

    int n_idx = tid % n_cnt;
    int nxt_n_idx = tid / n_cnt;
    int w_idx = tid;

    if (w_idx < w_cnt)
    {
        temp[threadIdx.x] = n_arr[n_idx] * w_arr[w_idx];
    }

    __syncthreads();

    if (threadIdx.x == 0) // threadIdx MUST be 0 for below logic to work!
    {
        /*
        The goal here is to try to minimize atomic adds. If the neuron count is
        greater than or equal to the threads per block, a maximum of 2 atomic adds
        is necessary for this block. However, most of the time we can get away with just 1.

        If the threads per block is greater than the neuron count, we just play it safe
        and incur an atomic add for each thread in the block.
        */

        int lower_idx = tid / n_cnt;
        int upper_idx = ((tid + THREADS_PER_BLOCK) - 1) / n_cnt;

        if (n_cnt >= THREADS_PER_BLOCK)
        {
            if (lower_idx == upper_idx)
            {
                float sum = 0.0f;

#pragma unroll
                for (int i = 0; i < THREADS_PER_BLOCK; i++)
                {
                    sum += temp[i];
                }

                atomicAdd(&nxt_n_arr[lower_idx], sum);
            }
            else
            {
                float sums[2] = {0.0f, 0.0f};

#pragma unroll
                for (int i = 0; i < THREADS_PER_BLOCK; i++)
                {
                    if ((tid + i) / n_cnt == lower_idx)
                    {
                        sums[0] += temp[i];
                    }
                    else
                    {
                        sums[1] += temp[i];
                    }
                }

                atomicAdd(&nxt_n_arr[lower_idx], sums[0]);
                if (upper_idx < nxt_n_cnt)
                {
                    atomicAdd(&nxt_n_arr[upper_idx], sums[1]);
                }
            }
        }
        else
        {

#pragma unroll
            for (int i = 0; i < THREADS_PER_BLOCK; i++)
            {
                atomicAdd(&nxt_n_arr[(tid + i) / n_cnt], temp[i]);
            }
        }
    }
}

__global__ void k_add_bias(float *b_arr, float *nxt_n_arr, int nxt_n_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nxt_n_cnt)
    {
        nxt_n_arr[tid] += b_arr[tid];
    }
}

__global__ void k_activate(float *n_arr, int n_cnt, ActivationFunctionId activation_func_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (activation_func_id)
        {
        case ReLU:
            n_arr[tid] = d_relu(n_arr[tid]);
            break;
        case Sigmoid:
            n_arr[tid] = d_sigmoid(n_arr[tid]);
            break;
        case Tanh:
            n_arr[tid] = d_tanh(n_arr[tid]);
            break;
        default:
            // None
            break;
        }
    }
}

__global__ void k_cost(float *n_arr, float *y_arr, float *cost, int n_cnt, CostFunctionId cost_func_id)
{
    __shared__ float temp[THREADS_PER_BLOCK];
    memset(temp, 0, THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (cost_func_id)
        {
        case MSE:
            temp[threadIdx.x] = d_mse_cost(n_arr[tid], y_arr[tid]);
            break;
        case CrossEntropy:
            temp[threadIdx.x] = d_cross_entropy_cost(n_arr[tid], y_arr[tid]);
            break;
        default:
            break;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        float sum = 0.0f;

#pragma unroll
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            sum += temp[i];
        }

        atomicAdd(cost, sum);
    }
}

__global__ void k_derive_cost(float *n_arr, float *y_arr, float *agg_derivatives_arr, int n_cnt, CostFunctionId cost_func_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (cost_func_id)
        {
        case MSE:
            agg_derivatives_arr[tid] *= d_derive_mse_cost(n_arr[tid], y_arr[tid]);
            break;
        case CrossEntropy:
            agg_derivatives_arr[tid] *= d_derive_cross_entropy_cost(n_arr[tid], y_arr[tid]);
            break;
        default:
            break;
        }
    }
}

__global__ void k_derive_activation(float *n_arr, float *agg_derivatives_arr, int n_cnt, ActivationFunctionId activation_func_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (activation_func_id)
        {
        case ReLU:
            agg_derivatives_arr[tid] *= d_derive_relu(n_arr[tid]);
            break;
        case Sigmoid:
            agg_derivatives_arr[tid] *= d_derive_sigmoid(n_arr[tid]);
            break;
        case Tanh:
            agg_derivatives_arr[tid] *= d_derive_tanh(n_arr[tid]);
            break;
        default:
            // None
            break;
        }
    }
}

__global__ void k_derive_z_and_increment_weight_derivative(float *agg_derivatives_arr, float *n_arr, float *dw_arr, int n_cnt, int nxt_n_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int w_cnt = n_cnt * nxt_n_cnt;

    int nxt_n_idx = tid % nxt_n_cnt;
    int n_idx = tid / nxt_n_cnt;
    int w_idx = n_idx * nxt_n_cnt + nxt_n_idx;

    if (w_idx < w_cnt)
    {
        dw_arr[w_idx] += (agg_derivatives_arr[n_idx] * n_arr[nxt_n_idx]);
    }
}

__global__ void k_derive_z_and_increment_bias_derivative(float *agg_derivatives_arr, float *db_arr, int n_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        db_arr[tid] += (agg_derivatives_arr[tid]);
    }
}

__global__ void k_derive_z_and_aggregate_derivatives(float *agg_derivatives_arr, float *w_arr, float *nxt_agg_derivatives_arr, int n_cnt, int nxt_n_cnt)
{
    __shared__ float temp[THREADS_PER_BLOCK];
    memset(temp, 0, THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int w_cnt = nxt_n_cnt * n_cnt;

    // Transpose the weights "matrix".
    int n_idx = tid % n_cnt;
    int nxt_n_idx = tid / n_cnt;
    int w_idx = n_idx * nxt_n_cnt + nxt_n_idx;

    if (w_idx < w_cnt)
    {
        temp[threadIdx.x] = (agg_derivatives_arr[n_idx] * w_arr[w_idx]);
    }

    __syncthreads();

    if (threadIdx.x == 0) // threadIdx MUST be 0 for below logic to work!
    {
        /*
        The goal here is to try to minimize atomic adds. If the neuron count is
        greater than or equal to the threads per block, a maximum of 2 atomic adds
        is necessary for this block. However, most of the time we can get away with just 1.

        If the threads per block is greater than the neuron count, we just play it safe
        and incur an atomic add for each thread in the block.
        */

        int lower_idx = tid / n_cnt;
        int upper_idx = ((tid + THREADS_PER_BLOCK) - 1) / n_cnt;

        if (n_cnt >= THREADS_PER_BLOCK)
        {
            if (lower_idx == upper_idx)
            {
                float sum = 0.0f;

#pragma unroll
                for (int i = 0; i < THREADS_PER_BLOCK; i++)
                {
                    sum += temp[i];
                }
                atomicAdd(&nxt_agg_derivatives_arr[lower_idx], sum);
            }
            else
            {
                float sums[2] = {0.0f, 0.0f};

#pragma unroll
                for (int i = 0; i < THREADS_PER_BLOCK; i++)
                {
                    if ((tid + i) / n_cnt == lower_idx)
                    {
                        sums[0] += temp[i];
                    }
                    else
                    {
                        sums[1] += temp[i];
                    }
                }

                atomicAdd(&nxt_agg_derivatives_arr[lower_idx], sums[0]);
                if (upper_idx < nxt_n_cnt)
                {
                    atomicAdd(&nxt_agg_derivatives_arr[upper_idx], sums[1]);
                }
            }
        }
        else
        {

#pragma unroll
            for (int i = 0; i < THREADS_PER_BLOCK; i++)
            {
                atomicAdd(&nxt_agg_derivatives_arr[(tid + i) / n_cnt], temp[i]);
            }
        }
    }
}

__global__ void k_adjust_weight(float *w_arr, float *dw_arr, int batch_size, float learning_rate, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        w_arr[tid] -= ((dw_arr[tid] * learning_rate) / (float)batch_size);
        dw_arr[tid] = 0.0f;
    }
}

__global__ void k_adjust_bias(float *b_arr, float *db_arr, int batch_size, float learning_rate, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        b_arr[tid] -= ((db_arr[tid] * learning_rate) / (float)batch_size);
        db_arr[tid] = 0.0f;
    }
}

// Report member functions:
void Report::print()
{
    printf("COST: %f\tACCURACY: %f%%\n", this->cost, ((float)this->correct_cnt / (float)this->total_cnt) * 100.0f);
}

void Report::update_correct_cnt(Tensor *n, Tensor *y)
{
    int lst_lyr_n_cnt = n->get_col_cnt();

    if (lst_lyr_n_cnt > 1)
    {
        // One hot encoded:

        TensorTuple max_tup = n->get_max();
        if (y->get_idx(max_tup.idx) == 1.0f)
        {
            this->correct_cnt++;
        }
    }
    else
    {
        // Single value:

        float y_val = y->get_idx(0);
        float n_val = n->get_idx(0);

        float lower = y_val < n_val ? y_val : n_val;
        float upper = y_val < n_val ? n_val : y_val;

        float prcnt = 1.0f - (lower / upper);

        // 10% is our number.
        if (prcnt <= 0.10f)
        {
            this->correct_cnt++;
        }
    }
}

// NN static functions:

void NN::write_csv_header(FILE *csv_file_ptr)
{
    fprintf(csv_file_ptr, "epoch,cost,accuracy,correct_cnt,total_cnt\n");
}

void NN::write_to_csv(FILE *csv_file_ptr, int epoch, Report rpt)
{
    fprintf(csv_file_ptr, "%d,%f,%f,%d,%d\n", epoch, rpt.cost, ((float)rpt.correct_cnt / (float)rpt.total_cnt) * 100.0f, rpt.correct_cnt, rpt.total_cnt);
}

// NN member functions:

NN::NN(std::vector<int> lyr_cfg, ActivationFunctionId hidden_layer_activation_func_id,
       ActivationFunctionId output_layer_activation_func_id, CostFunctionId cost_func_id, float learning_rate)
{

    int lyr_cnt = lyr_cfg.size();

    // Leave input neurons NULL for now!
    this->neurons.push_back(nullptr);

    for (int lyr_idx = 0; lyr_idx < lyr_cnt - 1; lyr_idx++)
    {
        int n_cnt = lyr_cfg[lyr_idx];
        int nxt_n_cnt = lyr_cfg[lyr_idx + 1];

        Tensor *n = new Tensor(1, nxt_n_cnt, Gpu);
        n->set_all(0.0f);
        this->neurons.push_back(n);

        Tensor *w = new Tensor(nxt_n_cnt, n_cnt, Gpu);
        w->set_all_rand(1.0f / sqrt(n_cnt)); // Xavier initialization!
        this->weights.push_back(w);

        Tensor *b = new Tensor(nxt_n_cnt, 1, Gpu);
        b->set_all(0.0f);
        this->biases.push_back(b);

        Tensor *dw = new Tensor(nxt_n_cnt, n_cnt, Gpu);
        dw->set_all(0.0f);
        this->weight_derivatives.push_back(dw);

        Tensor *db = new Tensor(nxt_n_cnt, 1, Gpu);
        db->set_all(0.0f);
        this->bias_derivatives.push_back(db);
    }

    this->hidden_layer_activation_func_id = hidden_layer_activation_func_id;
    this->output_layer_activation_func_id = output_layer_activation_func_id;

    this->cost_func_id = cost_func_id;

    this->learning_rate = learning_rate;

    cudaMalloc(&this->d_cost, sizeof(float));
    cudaMemset(this->d_cost, 0, sizeof(float));
}

NN::NN(const char *path)
{
    FILE *file_ptr = fopen(path, "rb");

    int lyr_cnt = 0;
    fread(&lyr_cnt, sizeof(int), 1, file_ptr);

    std::vector<int> lyr_cfg;

    for (int i = 0; i < lyr_cnt; i++)
    {
        int n_cnt = 0;
        fread(&n_cnt, sizeof(int), 1, file_ptr);
        lyr_cfg.push_back(n_cnt);
    }

    fread(&this->hidden_layer_activation_func_id, sizeof(ActivationFunctionId), 1, file_ptr);
    fread(&this->output_layer_activation_func_id, sizeof(ActivationFunctionId), 1, file_ptr);
    fread(&this->cost_func_id, sizeof(CostFunctionId), 1, file_ptr);
    fread(&this->learning_rate, sizeof(float), 1, file_ptr);

    // Leave input neurons NULL for now!
    this->neurons.push_back(nullptr);

    for (int lyr_idx = 0; lyr_idx < lyr_cnt - 1; lyr_idx++)
    {
        int n_cnt = lyr_cfg[lyr_idx];
        int nxt_n_cnt = lyr_cfg[lyr_idx + 1];

        int w_cnt = n_cnt * nxt_n_cnt;
        int b_cnt = nxt_n_cnt;

        Tensor *n = new Tensor(1, nxt_n_cnt, Gpu);
        n->set_all(0.0f);
        this->neurons.push_back(n);

        float *w_buf = (float *)malloc(sizeof(float) * w_cnt);
        fread(w_buf, sizeof(float), w_cnt, file_ptr);
        Tensor *w = new Tensor(nxt_n_cnt, n_cnt, Gpu, w_buf);
        this->weights.push_back(w);
        free(w_buf);

        float *b_buf = (float *)malloc(sizeof(float) * b_cnt);
        fread(b_buf, sizeof(float), b_cnt, file_ptr);
        Tensor *b = new Tensor(nxt_n_cnt, 1, Gpu, b_buf);
        this->biases.push_back(b);
        free(b_buf);

        Tensor *dw = new Tensor(nxt_n_cnt, n_cnt, Gpu);
        dw->set_all(0.0f);
        this->weight_derivatives.push_back(dw);

        Tensor *db = new Tensor(nxt_n_cnt, 1, Gpu);
        db->set_all(0.0f);
        this->bias_derivatives.push_back(db);
    }

    fclose(file_ptr);

    cudaMalloc(&this->d_cost, sizeof(float));
    cudaMemset(this->d_cost, 0, sizeof(float));
}

NN::~NN()
{
    int lyr_cnt = this->neurons.size();

    // Do not free input neurons since we do not own the Tensor!
    this->neurons[0] = nullptr;

    for (int lyr_idx = 0; lyr_idx < lyr_cnt - 1; lyr_idx++)
    {
        delete this->neurons[lyr_idx + 1];
        delete this->weights[lyr_idx];
        delete this->biases[lyr_idx];
        delete this->weight_derivatives[lyr_idx];
        delete this->bias_derivatives[lyr_idx];
    }

    cudaFree(this->d_cost);
}

void NN::print()
{
    int lyr_cnt = this->neurons.size();

    for (int lyr_idx = 0; lyr_idx < lyr_cnt; lyr_idx++)
    {
        printf("\n\n==================== LAYER: %d ====================\n\n", lyr_idx + 1);

        printf("NEURONS:\n");
        if (this->neurons[lyr_idx] == nullptr)
        {
            printf("NULL\n");
        }
        else
        {
            this->neurons[lyr_idx]->print();
        }

        if (lyr_idx < lyr_cnt - 1)
        {
            printf("WEIGHTS:\n");
            this->weights[lyr_idx]->print();

            printf("BIASES:\n");
            this->biases[lyr_idx]->print();
        }
    }
}

void NN::dump(const char *path)
{

    FILE *file_ptr = fopen(path, "wb");

    int lyr_cnt = this->neurons.size();
    fwrite(&lyr_cnt, sizeof(int), 1, file_ptr);

    for (int lyr_idx = 0; lyr_idx < lyr_cnt; lyr_idx++)
    {
        int n_cnt = this->neurons[lyr_idx]->get_col_cnt();
        fwrite(&n_cnt, sizeof(int), 1, file_ptr);
    }

    fwrite(&this->hidden_layer_activation_func_id, sizeof(ActivationFunctionId), 1, file_ptr);
    fwrite(&this->output_layer_activation_func_id, sizeof(ActivationFunctionId), 1, file_ptr);
    fwrite(&this->cost_func_id, sizeof(CostFunctionId), 1, file_ptr);
    fwrite(&this->learning_rate, sizeof(float), 1, file_ptr);

    for (int lyr_idx = 0; lyr_idx < lyr_cnt - 1; lyr_idx++)
    {
        Tensor *w = this->weights[lyr_idx];
        fwrite(w->get_arr(Cpu), sizeof(float), (w->get_row_cnt() * w->get_col_cnt()), file_ptr);

        Tensor *b = this->biases[lyr_idx];
        fwrite(b->get_arr(Cpu), sizeof(float), (b->get_row_cnt()), file_ptr);
    }

    fclose(file_ptr);
}

void NN::set_learning_rate(float learning_rate)
{
    this->learning_rate = learning_rate;
}

void NN::feed_forward(Tensor *x)
{
    x->translate(Gpu);
    this->neurons[0] = x;

    int lyr_cnt = this->neurons.size();
    int lst_lyr_idx = lyr_cnt - 1;

    for (int lyr_idx = 0; lyr_idx < lst_lyr_idx; lyr_idx++)
    {
        Tensor *n = this->neurons[lyr_idx];
        Tensor *w = this->weights[lyr_idx];
        Tensor *b = this->biases[lyr_idx];
        Tensor *nxt_n = this->neurons[lyr_idx + 1];

        int n_cnt = w->get_col_cnt();
        int nxt_n_cnt = w->get_row_cnt();

        ActivationFunctionId activation_func_id = (lyr_idx == lst_lyr_idx - 1) ? this->output_layer_activation_func_id : this->hidden_layer_activation_func_id;

        // Need to reset neurons before we do anything:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((nxt_n_cnt / threads_per_block) + 1);
            k_set_arr<<<num_blocks, threads_per_block>>>(nxt_n->get_arr(Gpu), nxt_n_cnt, 0.0f);
        }

        // Dot product:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(((n_cnt * nxt_n_cnt) / threads_per_block) + 1);
            k_dot<<<num_blocks, threads_per_block>>>(n->get_arr(Gpu), w->get_arr(Gpu),
                                                     nxt_n->get_arr(Gpu), n_cnt, nxt_n_cnt);
        }

        // Add biases:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((nxt_n_cnt / threads_per_block) + 1);
            k_add_bias<<<num_blocks, threads_per_block>>>(b->get_arr(Gpu), nxt_n->get_arr(Gpu),
                                                          nxt_n_cnt);
        }

        // Activate:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((nxt_n_cnt / threads_per_block) + 1);
            k_activate<<<num_blocks, threads_per_block>>>(nxt_n->get_arr(Gpu), nxt_n_cnt, activation_func_id);
        }
    }
}

float NN::get_cost(Tensor *y)
{
    y->translate(Gpu);

    float h_cost = 0.0f;

    int lyr_cnt = this->neurons.size();
    int lst_lyr_idx = lyr_cnt - 1;

    Tensor *lst_lyr_n = this->neurons[lst_lyr_idx];

    int lst_lyr_n_cnt = lst_lyr_n->get_col_cnt();

    {
        int threads_per_block(THREADS_PER_BLOCK);
        int num_blocks((lst_lyr_n_cnt / threads_per_block) + 1);

        k_cost<<<num_blocks, threads_per_block>>>(lst_lyr_n->get_arr(Gpu), y->get_arr(Gpu),
                                                  this->d_cost, lst_lyr_n_cnt, this->cost_func_id);
    }

    cudaMemcpy(&h_cost, this->d_cost, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemset(this->d_cost, 0, sizeof(float));

    return h_cost;
}

void NN::back_propagate(Tensor *y)
{
    y->translate(Gpu);

    int lyr_cnt = this->neurons.size();
    int lst_lyr_idx = lyr_cnt - 1;
    int lst_lyr_n_cnt = this->neurons[lst_lyr_idx]->get_col_cnt();

    Tensor *agg_derivatives = new Tensor(1, lst_lyr_n_cnt, Gpu);
    agg_derivatives->set_all(1.0f);

    // Derive cost (activation):
    {
        int threads_per_block(THREADS_PER_BLOCK);
        int num_blocks((lst_lyr_n_cnt / threads_per_block) + 1);
        k_derive_cost<<<num_blocks, threads_per_block>>>(this->neurons[lst_lyr_idx]->get_arr(Gpu),
                                                         y->get_arr(Gpu), agg_derivatives->get_arr(Gpu), lst_lyr_n_cnt, this->cost_func_id);
    }

    for (int lyr_idx = lst_lyr_idx; lyr_idx > 0; lyr_idx--)
    {
        Tensor *n = this->neurons[lyr_idx];
        Tensor *nxt_n = this->neurons[lyr_idx - 1];
        Tensor *nxt_w = this->weights[lyr_idx - 1];
        Tensor *nxt_b = this->biases[lyr_idx - 1];
        Tensor *nxt_dw = this->weight_derivatives[lyr_idx - 1];
        Tensor *nxt_db = this->bias_derivatives[lyr_idx - 1];

        int n_cnt = nxt_w->get_row_cnt();
        int nxt_n_cnt = nxt_w->get_col_cnt();

        ActivationFunctionId activation_func_id = (lyr_idx == lst_lyr_idx) ? this->output_layer_activation_func_id : this->hidden_layer_activation_func_id;

        // Derive activation (z):
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((n_cnt / threads_per_block) + 1);
            k_derive_activation<<<num_blocks, threads_per_block>>>(n->get_arr(Gpu),
                                                                   agg_derivatives->get_arr(Gpu), n_cnt, activation_func_id);
        }

        // Derive z (weight):
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(((n_cnt * nxt_n_cnt) / threads_per_block) + 1);
            k_derive_z_and_increment_weight_derivative<<<num_blocks, threads_per_block>>>(agg_derivatives->get_arr(Gpu),
                                                                                          nxt_n->get_arr(Gpu),
                                                                                          nxt_dw->get_arr(Gpu),
                                                                                          n_cnt, nxt_n_cnt);
        }

        // Derive z (bias):
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((n_cnt / threads_per_block) + 1);
            k_derive_z_and_increment_bias_derivative<<<num_blocks, threads_per_block>>>(agg_derivatives->get_arr(Gpu), nxt_db->get_arr(Gpu), n_cnt);
        }

        // Derive z (activation) and aggregate derivatives:
        {
            if (lyr_idx > 1)
            {
                Tensor *nxt_agg_derivatives = new Tensor(1, nxt_n_cnt, Gpu);
                nxt_agg_derivatives->set_all(0.0f);

                {
                    int threads_per_block(THREADS_PER_BLOCK);
                    int num_blocks(((nxt_n_cnt * n_cnt) / threads_per_block) + 1);
                    k_derive_z_and_aggregate_derivatives<<<num_blocks, threads_per_block>>>(agg_derivatives->get_arr(Gpu), nxt_w->get_arr(Gpu),
                                                                                            nxt_agg_derivatives->get_arr(Gpu),
                                                                                            n_cnt, nxt_n_cnt);
                }

                delete agg_derivatives;
                agg_derivatives = nxt_agg_derivatives;
            }
        }
    }

    delete agg_derivatives;
}

void NN::optimize(int batch_size)
{

    int lyr_cnt = this->neurons.size();
    int lst_lyr_idx = lyr_cnt - 1;

    for (int lyr_idx = 0; lyr_idx < lst_lyr_idx; lyr_idx++)
    {
        Tensor *w = this->weights[lyr_idx];
        Tensor *b = this->biases[lyr_idx];
        Tensor *dw = this->weight_derivatives[lyr_idx];
        Tensor *db = this->bias_derivatives[lyr_idx];

        int n_cnt = w->get_col_cnt();
        int nxt_n_cnt = w->get_row_cnt();

        // Weights:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(((nxt_n_cnt * n_cnt) / threads_per_block) + 1);
            k_adjust_weight<<<num_blocks, threads_per_block>>>(w->get_arr(Gpu), dw->get_arr(Gpu), batch_size, this->learning_rate,
                                                               (nxt_n_cnt * n_cnt));
        }

        // Biases:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((nxt_n_cnt / threads_per_block) + 1);
            k_adjust_bias<<<num_blocks, threads_per_block>>>(b->get_arr(Gpu), db->get_arr(Gpu), batch_size, this->learning_rate, nxt_n_cnt);
        }
    }
}

void NN::check_gradient(Tensor *x, Tensor *y, bool print_flg)
{
    float agg_ana_grad = 0.0f;
    float agg_num_grad = 0.0f;
    float agg_grad_diff = 0.0f;

    float epsilon = 0.001f;

    // Analytical gradients:
    {
        this->feed_forward(x);
        this->back_propagate(y);
    }

    // Numerical gradients:
    {
        int lyr_cnt = this->neurons.size();
        int lst_lyr_idx = lyr_cnt - 1;

        for (int lyr_idx = 0; lyr_idx < lst_lyr_idx; lyr_idx++)
        {
            Tensor *w = this->weights[lyr_idx];
            Tensor *b = this->biases[lyr_idx];
            Tensor *dw = this->weight_derivatives[lyr_idx];
            Tensor *db = this->bias_derivatives[lyr_idx];

            // Weights:
            for (int w_idx = 0; w_idx < w->get_row_cnt() * w->get_col_cnt(); w_idx++)
            {
                float left_cost = 0.0;
                float right_cost = 0.0;

                float orig_w_val = w->get_idx(w_idx);

                float left_w_val = orig_w_val - epsilon;
                float right_w_val = orig_w_val + epsilon;

                float ana_grad = dw->get_idx(w_idx);

                // Left:
                w->set_idx(w_idx, left_w_val);
                {
                    this->feed_forward(x);
                    left_cost += this->get_cost(y);
                }

                // Right:
                w->set_idx(w_idx, right_w_val);
                {
                    this->feed_forward(x);
                    right_cost += this->get_cost(y);
                }

                float num_grad = (right_cost - left_cost) / (2.0f * epsilon);

                if (print_flg)
                {
                    printf("W: %d  %d\t%f : %f  (%f)\n", lyr_idx, w_idx, ana_grad, num_grad, fabs(ana_grad - num_grad));
                }

                agg_ana_grad += (ana_grad * ana_grad);
                agg_num_grad += (num_grad * num_grad);
                agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));

                w->set_idx(w_idx, orig_w_val);
            }

            // Biases:
            for (int b_idx = 0; b_idx < b->get_row_cnt(); b_idx++)
            {
                float left_cost = 0.0;
                float right_cost = 0.0;

                float orig_b_val = b->get_idx(b_idx);

                float left_b_val = orig_b_val - epsilon;
                float right_b_val = orig_b_val + epsilon;

                float ana_grad = db->get_idx(b_idx);

                // Left:
                b->set_idx(b_idx, left_b_val);
                {
                    this->feed_forward(x);
                    left_cost += this->get_cost(y);
                }

                // Right:
                b->set_idx(b_idx, right_b_val);
                {
                    this->feed_forward(x);
                    right_cost += this->get_cost(y);
                }

                float num_grad = (right_cost - left_cost) / (2.0f * epsilon);

                if (print_flg)
                {
                    printf("B: %d  %d\t%f : %f  (%f)\n", lyr_idx, b_idx, ana_grad, num_grad, fabs(ana_grad - num_grad));
                }

                agg_ana_grad += (ana_grad * ana_grad);
                agg_num_grad += (num_grad * num_grad);
                agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));

                b->set_idx(b_idx, orig_b_val);
            }
        }
    }

    if ((agg_grad_diff) == 0.0f && (agg_ana_grad + agg_num_grad) == 0.0f)
    {
        printf("GRADIENT CHECK RESULT: %f\n", 0.0f);
    }
    else
    {
        printf("GRADIENT CHECK RESULT: %f\n", (agg_grad_diff) / (agg_ana_grad + agg_num_grad));
    }
}

Report NN::train(Batch *batch)
{
    Report rpt;

    int batch_size = batch->get_size();

    rpt.correct_cnt = 0;
    rpt.total_cnt = batch_size;

    float cost = 0.0f;

    int lst_lyr_idx = this->neurons.size() - 1;

    for (int i = 0; i < batch_size; i++)
    {
        Tensor *x = batch->get_x(i);
        Tensor *y = batch->get_y(i);

        this->feed_forward(x);
        cost += this->get_cost(y);
        this->back_propagate(y);

        rpt.update_correct_cnt(this->neurons[lst_lyr_idx], y);

        // Translate back to CPU as to not overload GPU.
        x->translate(Cpu);
        y->translate(Cpu);
    }

    cost /= batch_size;

    rpt.cost = cost;

    this->optimize(batch_size);

    return rpt;
}

Report NN::validate(Batch *batch)
{
    Report rpt;

    int batch_size = batch->get_size();

    rpt.correct_cnt = 0;
    rpt.total_cnt = batch_size;

    float cost = 0.0f;

    int lst_lyr_idx = this->neurons.size() - 1;

    for (int i = 0; i < batch_size; i++)
    {
        Tensor *x = batch->get_x(i);
        Tensor *y = batch->get_y(i);

        this->feed_forward(x);
        cost += this->get_cost(y);

        rpt.update_correct_cnt(this->neurons[lst_lyr_idx], y);

        // Translate back to CPU as to not overload GPU.
        x->translate(Cpu);
        y->translate(Cpu);
    }

    cost /= batch_size;

    rpt.cost = cost;

    return rpt;
}

Report NN::test(Batch *batch)
{
    Report rpt;

    int batch_size = batch->get_size();

    rpt.correct_cnt = 0;
    rpt.total_cnt = batch_size;

    float cost = 0.0f;

    int lst_lyr_idx = this->neurons.size() - 1;

    for (int i = 0; i < batch_size; i++)
    {
        Tensor *x = batch->get_x(i);
        Tensor *y = batch->get_y(i);

        this->feed_forward(x);
        cost += this->get_cost(y);

        rpt.update_correct_cnt(this->neurons[lst_lyr_idx], y);

        // Translate back to CPU as to not overload GPU.
        x->translate(Cpu);
        y->translate(Cpu);
    }

    cost /= batch_size;

    rpt.cost = cost;

    return rpt;
}

// Trains, validates, and tests. Press 'q' to force quit.
void NN::all(Supervisor *supervisor, int train_batch_size, int validation_chk_freq, const char *csv_path)
{
    FILE *csv_file_ptr;

    if (csv_path != nullptr)
    {
        csv_file_ptr = fopen(csv_path, "w");
        NN::write_csv_header(csv_file_ptr);
    }

    Batch *validation_batch = supervisor->create_validation_batch();
    float prv_validation_cost = FLT_MAX;

    Batch *test_batch = supervisor->create_test_batch();

    unsigned long int epoch = 1;
    while (true)
    {
        Batch *train_batch = supervisor->create_train_batch(train_batch_size);
        Report train_rpt = this->train(train_batch);

        if (csv_path != nullptr)
        {
            NN::write_to_csv(csv_file_ptr, epoch, train_rpt);
        }

        delete train_batch;

        // Validate every x epochs.
        if (epoch % validation_chk_freq == 0)
        {
            Report validation_rpt = this->validate(validation_batch);
            printf("VALIDATION: ");
            validation_rpt.print();

            if (prv_validation_cost <= validation_rpt.cost)
            {
                break;
            }

            prv_validation_cost = validation_rpt.cost;
        }

        // Allow for manual override.
        {
            if (_kbhit())
            {
                if (_getch() == 'q')
                {
                    break;
                }
            }
        }

        epoch++;
    }

    Report test_rpt = this->test(test_batch);
    printf("TEST: ");
    test_rpt.print();

    delete validation_batch;
    delete test_batch;

    if (csv_path != nullptr)
    {
        fclose(csv_file_ptr);
    }
}

Tensor *NN::predict(Tensor *x)
{
    this->feed_forward(x);
    Tensor *pred = new Tensor(*this->neurons[this->neurons.size() - 1]);
    return pred;
}