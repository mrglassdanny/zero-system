#include "NN.cuh"

#define THREADS_PER_BLOCK 16

__device__ float d_relu(float z)
{
    return z > 0.0f ? z : 0.0f;
}

__device__ float d_derive_relu(float a)
{
    return a > 0.0f ? 1.0f : 0.0f;
}

__device__ float d_mse_cost(float p, float y)
{
    return pow((p - y), 2.0);
}

__device__ float d_derive_mse_cost(float p, float y)
{
    return 2.0f * (p - y);
}

__global__ void k_set_arr(float *arr, int cnt, float val)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        arr[tid] = val;
    }
}

__global__ void k_dot_all(float *neu_arr, float *wgt_arr, float *nxt_neu_arr, int neu_cnt, int nxt_neu_cnt)
{
    __shared__ float temp[THREADS_PER_BLOCK];
    memset(temp, 0, THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int tot_cnt = neu_cnt * nxt_neu_cnt;

    if (tid < tot_cnt)
    {
        temp[threadIdx.x] = neu_arr[tid % neu_cnt] * wgt_arr[tid];
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        // NOTE: this only works if we assume the threadIdx.x is 0!!!
        int a = tid / neu_cnt;
        int b = (tid + THREADS_PER_BLOCK - 1) / neu_cnt;

        if (neu_cnt >= THREADS_PER_BLOCK)
        {
            if (a == b)
            {
                float sum = 0.0f;

                for (int i = 0; i < THREADS_PER_BLOCK; i++)
                {
                    sum += temp[i];
                }
                atomicAdd(&nxt_neu_arr[tid / neu_cnt], sum);
            }
            else
            {
                float sums[2] = {0.0f, 0.0f};

                for (int i = 0; i < THREADS_PER_BLOCK; i++)
                {
                    if ((tid + i) / neu_cnt == a)
                    {
                        sums[0] += temp[i];
                    }
                    else
                    {
                        sums[1] += temp[i];
                    }
                }

                atomicAdd(&nxt_neu_arr[a], sums[0]);
                if (a + 1 < neu_cnt)
                {
                    atomicAdd(&nxt_neu_arr[a + 1], sums[1]);
                }
            }
        }
        else
        {
            for (int i = 0; i < THREADS_PER_BLOCK; i++)
            {
                atomicAdd(&nxt_neu_arr[(tid + i) / neu_cnt], temp[i]);
            }
        }
    }
}

__global__ void k_add_bias(float *bias_arr, float *nxt_neu_arr, int nxt_neu_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nxt_neu_cnt)
    {
        nxt_neu_arr[tid] += bias_arr[tid];
    }
}

__global__ void k_activate(float *neu_arr, int neu_cnt, ActivationFunctionId activation_func_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < neu_cnt)
    {
        switch (activation_func_id)
        {
        case ReLU:
            neu_arr[tid] = d_relu(neu_arr[tid]);
            break;
        case Sigmoid:
            neu_arr[tid] = d_relu(neu_arr[tid]);
            break;
        case Tanh:
            neu_arr[tid] = d_relu(neu_arr[tid]);
            break;
        default:
            // None
            break;
        }
    }
}

__global__ void k_cost(float *neu_arr, float *y_arr, float *atomic_cost, int neu_cnt)
{
    __shared__ float temp[THREADS_PER_BLOCK];
    memset(temp, 0, THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < neu_cnt)
    {
        temp[threadIdx.x] = d_mse_cost(neu_arr[tid], y_arr[tid]);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        float cost = 0.0f;
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            cost += temp[i];
        }

        atomicAdd(atomic_cost, cost);
    }
}

__global__ void k_derive_cost(float *neu_arr, float *y_arr, float *out_arr, int neu_cnt, CostFunctionId cost_func_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < neu_cnt)
    {
        switch (cost_func_id)
        {
        case MSE:
            out_arr[tid] = d_derive_mse_cost(neu_arr[tid], y_arr[tid]);
            break;
        case CrossEntropy:
            out_arr[tid] = d_derive_mse_cost(neu_arr[tid], y_arr[tid]);
            break;
        default:
            break;
        }
    }
}

__global__ void k_derive_activation(float *neu_arr, float *out_arr, int neu_cnt, ActivationFunctionId activation_func_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < neu_cnt)
    {
        switch (activation_func_id)
        {
        case ReLU:
            out_arr[tid] *= d_derive_relu(neu_arr[tid]);
            break;
        case Sigmoid:
            out_arr[tid] *= d_derive_relu(neu_arr[tid]);
            break;
        case Tanh:
            out_arr[tid] *= d_derive_relu(neu_arr[tid]);
            break;
        default:
            // None
            break;
        }
    }
}

__global__ void k_inc_weights(float *agg_arr, float *neu_arr, float *dw_arr, int row_cnt, int col_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int row_idx = tid / col_cnt;
    int col_idx = tid % col_cnt;
    int wgt_idx = row_idx * col_cnt + col_idx;

    if (tid < (row_cnt * col_cnt))
    {
        dw_arr[wgt_idx] += (agg_arr[row_idx] * neu_arr[col_idx]);
    }
}

__global__ void k_inc_biases(float *agg_arr, float *db_arr, int row_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int row_idx = tid;

    if (row_idx < row_cnt)
    {
        db_arr[row_idx] += (agg_arr[row_idx]);
    }
}

__global__ void k_agg_derivatives(float *wgt_arr, float *agg_arr, float *out_arr, int col_cnt, int row_cnt)
{
    __shared__ float temp[THREADS_PER_BLOCK];
    memset(temp, 0, THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int tot_cnt = col_cnt * row_cnt;

    int row_idx = tid / col_cnt;
    int col_idx = tid % col_cnt;
    int wgt_idx = row_idx * col_cnt + col_idx;

    if (tid < tot_cnt)
    {
        temp[threadIdx.x] = (agg_arr[row_idx] * wgt_arr[wgt_idx]);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            // NOTE: this only works if we assume the threadIdx.x is 0!!!
            int idx = (tid + i) % col_cnt;
            atomicAdd(&out_arr[idx], temp[i]);
        }
    }
}

NN::NN(std::vector<int> layer_config, ActivationFunctionId hidden_layer_activation_func_id,
       ActivationFunctionId output_layer_activation_func_id, CostFunctionId cost_func_id, float learning_rate)
{
    // Leave input neurons NULL for now!
    this->neurons.push_back(nullptr);
    for (int lyr_idx = 0; lyr_idx < layer_config.size() - 1; lyr_idx++)
    {
        this->neurons.push_back(new Tensor(1, layer_config[lyr_idx + 1], Gpu));
        this->weights.push_back(new Tensor(layer_config[lyr_idx + 1], layer_config[lyr_idx], Gpu));
        this->biases.push_back(new Tensor(layer_config[lyr_idx + 1], 1, Gpu));
        this->weight_derivatives.push_back(new Tensor(layer_config[lyr_idx + 1], layer_config[lyr_idx], Gpu));
        this->bias_derivatives.push_back(new Tensor(layer_config[lyr_idx + 1], 1, Gpu));
    }

    this->hidden_layer_activation_func_id = hidden_layer_activation_func_id;
    this->output_layer_activation_func_id = output_layer_activation_func_id;

    this->cost_func_id = cost_func_id;

    this->learning_rate = learning_rate;
}

NN::~NN()
{
}

void NN::feed_forward(Tensor *x)
{
    x->translate(Gpu);
    this->neurons[0] = x;

    int lst_lyr_idx = this->neurons.size() - 1;

    for (int lyr_idx = 0; lyr_idx < lst_lyr_idx; lyr_idx++)
    {
        int neu_cnt = this->weights[lyr_idx]->get_col_cnt();
        int nxt_neu_cnt = this->weights[lyr_idx]->get_row_cnt();

        ActivationFunctionId activation_func_id = (lyr_idx == lst_lyr_idx - 1) ? this->output_layer_activation_func_id : this->hidden_layer_activation_func_id;

        // Need to reset neurons before we do anything!
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(ceil((float)nxt_neu_cnt / (float)threads_per_block));
            k_set_arr<<<num_blocks, threads_per_block>>>(this->neurons[lyr_idx + 1]->get_arr(Gpu), nxt_neu_cnt, 0.0f);
        }

        // Dot product:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(ceil((float)(neu_cnt * nxt_neu_cnt) / (float)threads_per_block));
            k_dot_all<<<num_blocks, threads_per_block>>>(this->neurons[lyr_idx]->get_arr(Gpu), this->weights[lyr_idx]->get_arr(Gpu),
                                                         this->neurons[lyr_idx + 1]->get_arr(Gpu), neu_cnt, nxt_neu_cnt);
        }

        // Add biases:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(ceil((float)nxt_neu_cnt / (float)threads_per_block));
            k_add_bias<<<num_blocks, threads_per_block>>>(this->biases[lyr_idx]->get_arr(Gpu), this->neurons[lyr_idx + 1]->get_arr(Gpu),
                                                          nxt_neu_cnt);
        }

        // Activate:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(ceil((float)nxt_neu_cnt / (float)threads_per_block));
            k_activate<<<num_blocks, threads_per_block>>>(this->neurons[lyr_idx + 1]->get_arr(Gpu), nxt_neu_cnt, activation_func_id);
        }
    }
}

// float NN::get_cost(Tensor *y, int batch_size)
// {
//     Tensor *cost_tensor = Tensor_init(1, 1, 1);
//     Tensor_set_all(cost_tensor, 0.0f);

//     int cost_threads_per_block(THREADS_PER_BLOCK);
//     int cost_num_blocks(ceil((float)nn->ly_cfg[nn->ly_cnt - 1] / (float)cost_threads_per_block.x));

//     // Add async:
//     k_cost<<<cost_num_blocks, cost_threads_per_block>>>(nn->n[nn->ly_cnt - 1]->arr, y->arr,
//                                                         &cost_tensor->arr[0], nn->ly_cfg[nn->ly_cnt - 1]);

//     Tensor_to_cpu(cost_tensor);
//     float cost = Tensor_get_idx(cost_tensor, 0);
//     Tensor_free(cost_tensor);

//     // Divide sync:
//     cost /= (float)m;

//     return cost;
// }

void NN::back_propagate(Tensor *y, int batch_size)
{

    int lst_lyr_idx = this->neurons.size() - 1;
    int lst_lyr_neu_cnt = this->weights[lst_lyr_idx]->get_row_cnt();

    Tensor *agg = new Tensor(1, lst_lyr_neu_cnt, Gpu);
    agg->set_all(0.0f);

    // Cost:
    {
        int threads_per_block(THREADS_PER_BLOCK);
        int num_blocks(ceil((float)lst_lyr_neu_cnt / (float)threads_per_block));
        k_derive_cost<<<num_blocks, threads_per_block>>>(this->neurons[lst_lyr_idx]->get_arr(Gpu),
                                                         y->get_arr(Gpu), agg->get_arr(Gpu), lst_lyr_neu_cnt);
    }

    for (int lyr_idx = nn->ly_cnt - 1; lyr_idx > 0; lyr_idx--)
    {
        int neu_cnt = this->weights[lyr_idx]->get_row_cnt();
        int prv_neu_cnt = this->weights[lyr_idx]->get_col_cnt();

        Tensor *neu = this->neurons[lyr_idx];
        Tensor *prv_neu = this->neurons[lyr_idx - 1];
        Tensor *prv_dw = this->weight_derivatives[lyr_idx - 1];

        // Activations:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(ceil((float)neu_cnt / (float)threads_per_block));
            k_derive_activation<<<num_blocks, threads_per_block>>>(neu->get_arr(Gpu),
                                                                   agg->get_arr(Gpu), neu_cnt);
        }

        // Weights:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(ceil((float)neu_cnt * prv_neu_cnt / (float)threads_per_block));
            k_inc_weights<<<num_blocks, threads_per_block>>>(agg->get_arr(Gpu),
                                                             prv_neu->get_arr(Gpu),
                                                             prv_dw->get_arr(Gpu),
                                                             neu_cnt, prv_neu_cnt);
        }

        // Biases:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(ceil((float)neu_cnt / (float)threads_per_block));
            k_inc_biases<<<num_blocks, threads_per_block>>>(agg->arr, nn->db[lyr_idx - 1]->arr, neu_cnt);
        }

        // Aggregate:
        {
            if (lyr_idx > 1)
            {

                Tensor *temp = Tensor_init(1, prv_neu_cnt, 1);
                Tensor_set_all(temp, 0.0f);

                {
                    int threads_per_block(THREADS_PER_BLOCK);
                    int num_blocks(ceil((float)prv_neu_cnt * neu_cnt / (float)threads_per_block));
                    k_agg_derivatives<<<num_blocks, threads_per_block>>>(nn->w[lyr_idx - 1]->arr,
                                                                         agg->arr, temp->arr,
                                                                         prv_neu_cnt, neu_cnt);
                }

                Tensor_free(agg);
                agg = Tensor_copy(temp);
                Tensor_free(temp);
            }
        }
    }

    Tensor_free(agg);
}