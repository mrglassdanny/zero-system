#include "NN.cuh"

#define THREADS_PER_BLOCK 4

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
    return (1 - pow(val, 2.0));
}

__device__ float d_mse_cost(float n_val, float y_val)
{
    return pow((n_val - y_val), 2.0);
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

// TODO
__global__ void k_dot_all(float *n_arr, float *w_arr, float *nxt_n_arr, int n_cnt, int nxt_n_cnt)
{
    __shared__ float temp[THREADS_PER_BLOCK];
    memset(temp, 0, THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int w_cnt = n_cnt * nxt_n_cnt;

    if (tid < w_cnt)
    {
        temp[threadIdx.x] = n_arr[tid % n_cnt] * w_arr[tid];
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        // NOTE: this only works if we assume the threadIdx.x is 0!!!
        int lower_idx = tid / n_cnt;
        int upper_idx = (tid + THREADS_PER_BLOCK - 1) / n_cnt;

        if (n_cnt >= THREADS_PER_BLOCK)
        {
            if (lower_idx == upper_idx)
            {
                float sum = 0.0f;

                for (int i = 0; i < THREADS_PER_BLOCK; i++)
                {
                    sum += temp[i];
                }

                atomicAdd(&nxt_n_arr[lower_idx], sum);
            }
            else
            {
                float sums[2] = {0.0f, 0.0f};

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
                if (upper_idx < n_cnt)
                {
                    atomicAdd(&nxt_n_arr[upper_idx], sums[1]);
                }
            }
        }
        else
        {
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

        //temp[threadIdx.x] = d_mse_cost(n_arr[tid], y_arr[tid]);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        float sum = 0.0f;
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            sum += temp[i];
        }

        atomicAdd(cost, sum);
    }
}

__global__ void k_derive_cost(float *n_arr, float *y_arr, float *agg_arr, int n_cnt, CostFunctionId cost_func_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (cost_func_id)
        {
        case MSE:
            agg_arr[tid] *= d_derive_mse_cost(n_arr[tid], y_arr[tid]);
            break;
        case CrossEntropy:
            agg_arr[tid] *= d_derive_cross_entropy_cost(n_arr[tid], y_arr[tid]);
            break;
        default:
            break;
        }

        //agg_arr[tid] *= d_derive_mse_cost(n_arr[tid], y_arr[tid]);
    }
}

__global__ void k_derive_activation(float *n_arr, float *agg_arr, int n_cnt, ActivationFunctionId activation_func_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (activation_func_id)
        {
        case ReLU:
            agg_arr[tid] *= d_derive_relu(n_arr[tid]);
            break;
        case Sigmoid:
            agg_arr[tid] *= d_derive_sigmoid(n_arr[tid]);
            break;
        case Tanh:
            agg_arr[tid] *= d_derive_tanh(n_arr[tid]);
            break;
        default:
            // None
            break;
        }
        //agg_arr[tid] *= d_derive_relu(n_arr[tid]);
    }
}

__global__ void k_derive_weights_n_increment_derivatives(float *agg_arr, float *n_arr, float *dw_arr, int nxt_n_cnt, int n_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int nxt_n_idx = tid / n_cnt;
    int n_idx = tid % n_cnt;
    int w_idx = nxt_n_idx * n_cnt + n_idx;

    if (tid < (nxt_n_cnt * n_cnt))
    {
        dw_arr[w_idx] += (agg_arr[nxt_n_idx] * n_arr[n_idx]);
    }
}

__global__ void k_derive_biases_n_increment_derivatives(float *agg_arr, float *db_arr, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        db_arr[tid] += (agg_arr[tid]);
    }
}

// TODO
__global__ void k_aggregate_derivatives(float *w_arr, float *agg_arr, float *temp_agg_arr, int prv_n_cnt, int n_cnt)
{
    __shared__ float temp[THREADS_PER_BLOCK];
    memset(temp, 0, THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int w_cnt = prv_n_cnt * n_cnt;

    int n_idx = tid / prv_n_cnt;
    int prv_n_idx = tid % prv_n_cnt;
    int w_idx = n_idx * prv_n_cnt + prv_n_idx;

    if (tid < w_cnt)
    {
        temp[threadIdx.x] = (agg_arr[n_idx] * w_arr[w_idx]);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            // NOTE: this only works if we assume the threadIdx.x is 0!!!
            int idx = (tid + i) % prv_n_cnt;
            atomicAdd(&temp_agg_arr[idx], temp[i]);
        }
    }
}

__global__ void k_optimize_weights(float *w_arr, float *dw_arr, int batch_size, float learning_rate, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        w_arr[tid] -= ((dw_arr[tid] * learning_rate) / (float)batch_size);
        dw_arr[tid] = 0.0f;
    }
}

__global__ void k_optimize_biases(float *b_arr, float *db_arr, int batch_size, float learning_rate, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        b_arr[tid] -= ((db_arr[tid] * learning_rate) / (float)batch_size);
        db_arr[tid] = 0.0f;
    }
}

// Member functions:
NN::NN(std::vector<int> layer_config, ActivationFunctionId hidden_layer_activation_func_id,
       ActivationFunctionId output_layer_activation_func_id, CostFunctionId cost_func_id, float learning_rate)
{

    int lyr_cnt = layer_config.size();

    // Leave input neurons NULL for now!
    this->neurons.push_back(nullptr);

    for (int lyr_idx = 0; lyr_idx < lyr_cnt - 1; lyr_idx++)
    {
        int n_cnt = layer_config[lyr_idx];
        int nxt_n_cnt = layer_config[lyr_idx + 1];

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
            int num_blocks(ceil((float)nxt_n_cnt / (float)threads_per_block));
            k_set_arr<<<num_blocks, threads_per_block>>>(nxt_n->get_arr(Gpu), nxt_n_cnt, 0.0f);
        }

        // Dot product:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(ceil((float)(n_cnt * nxt_n_cnt) / (float)threads_per_block));
            k_dot_all<<<num_blocks, threads_per_block>>>(n->get_arr(Gpu), w->get_arr(Gpu),
                                                         nxt_n->get_arr(Gpu), n_cnt, nxt_n_cnt);
        }

        // Add biases:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(ceil((float)nxt_n_cnt / (float)threads_per_block));
            k_add_bias<<<num_blocks, threads_per_block>>>(b->get_arr(Gpu), nxt_n->get_arr(Gpu),
                                                          nxt_n_cnt);
        }

        // Activate:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(ceil((float)nxt_n_cnt / (float)threads_per_block));
            k_activate<<<num_blocks, threads_per_block>>>(nxt_n->get_arr(Gpu), nxt_n_cnt, activation_func_id);
        }
    }
}

float NN::get_cost(Tensor *y)
{
    y->translate(Gpu);

    float h_cost = 0.0f;
    float *d_cost;
    cudaMalloc(&d_cost, sizeof(float));
    cudaMemset(d_cost, 0, sizeof(float));

    int lyr_cnt = this->neurons.size();
    int lst_lyr_idx = lyr_cnt - 1;

    Tensor *lst_lyr_n = this->neurons[lst_lyr_idx];

    int lst_lyr_n_cnt = lst_lyr_n->get_col_cnt();

    {
        int threads_per_block(THREADS_PER_BLOCK);
        int num_blocks(ceil((float)lst_lyr_n_cnt / (float)threads_per_block));

        k_cost<<<num_blocks, threads_per_block>>>(lst_lyr_n->get_arr(Gpu), y->get_arr(Gpu),
                                                  d_cost, lst_lyr_n_cnt, this->cost_func_id);
    }

    cudaMemcpy(&h_cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_cost);

    return h_cost;
}

void NN::back_propagate(Tensor *y)
{
    y->translate(Gpu);

    int lyr_cnt = this->neurons.size();
    int lst_lyr_idx = lyr_cnt - 1;
    int lst_lyr_n_cnt = this->neurons[lst_lyr_idx]->get_col_cnt();

    Tensor *agg = new Tensor(1, lst_lyr_n_cnt, Gpu);
    agg->set_all(1.0f);

    // Cost:
    {
        int threads_per_block(THREADS_PER_BLOCK);
        int num_blocks(ceil((float)lst_lyr_n_cnt / (float)threads_per_block));
        k_derive_cost<<<num_blocks, threads_per_block>>>(this->neurons[lst_lyr_idx]->get_arr(Gpu),
                                                         y->get_arr(Gpu), agg->get_arr(Gpu), lst_lyr_n_cnt, this->cost_func_id);
    }

    for (int lyr_idx = lst_lyr_idx; lyr_idx > 0; lyr_idx--)
    {
        Tensor *n = this->neurons[lyr_idx];
        Tensor *prv_n = this->neurons[lyr_idx - 1];
        Tensor *prv_w = this->weights[lyr_idx - 1];
        Tensor *prv_b = this->biases[lyr_idx - 1];
        Tensor *prv_dw = this->weight_derivatives[lyr_idx - 1];
        Tensor *prv_db = this->bias_derivatives[lyr_idx - 1];

        int n_cnt = prv_w->get_row_cnt();
        int prv_n_cnt = prv_w->get_col_cnt();

        ActivationFunctionId activation_func_id = (lyr_idx == lst_lyr_idx) ? this->output_layer_activation_func_id : this->hidden_layer_activation_func_id;

        // Activations:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(ceil((float)n_cnt / (float)threads_per_block));
            k_derive_activation<<<num_blocks, threads_per_block>>>(n->get_arr(Gpu),
                                                                   agg->get_arr(Gpu), n_cnt, activation_func_id);
        }

        // Weights:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(ceil((float)n_cnt * prv_n_cnt / (float)threads_per_block));
            k_derive_weights_n_increment_derivatives<<<num_blocks, threads_per_block>>>(agg->get_arr(Gpu),
                                                                                        prv_n->get_arr(Gpu),
                                                                                        prv_dw->get_arr(Gpu),
                                                                                        n_cnt, prv_n_cnt);
        }

        // Biases:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(ceil((float)n_cnt / (float)threads_per_block));
            k_derive_biases_n_increment_derivatives<<<num_blocks, threads_per_block>>>(agg->get_arr(Gpu), prv_db->get_arr(Gpu), n_cnt);
        }

        // Aggregate:
        {
            if (lyr_idx > 1)
            {
                Tensor *temp_agg = new Tensor(1, prv_n_cnt, Gpu);
                temp_agg->set_all(0.0f);

                {
                    int threads_per_block(THREADS_PER_BLOCK);
                    int num_blocks(ceil((float)prv_n_cnt * n_cnt / (float)threads_per_block));
                    k_aggregate_derivatives<<<num_blocks, threads_per_block>>>(prv_w->get_arr(Gpu),
                                                                               agg->get_arr(Gpu), temp_agg->get_arr(Gpu),
                                                                               prv_n_cnt, n_cnt);
                }

                delete agg;
                agg = temp_agg;
            }
        }
    }

    delete agg;
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
            int num_blocks(ceil(((float)nxt_n_cnt * n_cnt) / (float)threads_per_block));
            k_optimize_weights<<<num_blocks, threads_per_block>>>(w->get_arr(Gpu), dw->get_arr(Gpu), batch_size, this->learning_rate,
                                                                  (nxt_n_cnt * n_cnt));
        }

        // Biases:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(ceil((float)nxt_n_cnt / (float)threads_per_block));
            k_optimize_biases<<<num_blocks, threads_per_block>>>(b->get_arr(Gpu), db->get_arr(Gpu), batch_size, this->learning_rate, nxt_n_cnt);
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

                agg_ana_grad += pow(ana_grad, 2.0f);
                agg_num_grad += pow(num_grad, 2.0f);
                agg_grad_diff += pow(ana_grad - num_grad, 2.0f);

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

                agg_ana_grad += pow(ana_grad, 2.0f);
                agg_num_grad += pow(num_grad, 2.0f);
                agg_grad_diff += pow(ana_grad - num_grad, 2.0f);

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

void NN::profile(Tensor *x, Tensor *y)
{
    int epoch_cnt = 100;
    int batch_size = 100;

    printf("START PERFORMANCE TEST\n");
    clock_t t;
    t = clock();

    for (int epoch = 0; epoch < epoch_cnt; epoch++)
    {
        for (int batch = 0; batch < batch_size; batch++)
        {
            this->feed_forward(x);
            //this->get_cost(y);
            this->back_propagate(y);
        }
        //this->optimize(batch_size);
    }

    t = clock() - t;
    double time_taken = ((double)t) / CLOCKS_PER_SEC;

    printf("END PERFORMANCE TEST\n");
    printf("Elapsed Seconds: %f\n\n", time_taken);
}

// TODO
Result NN::train(Batch *batch)
{
    Result result;

    int batch_size = batch->get_size();

    result.cor_cnt = 0;
    result.tot_cnt = batch_size;

    float cost = 0.0f;

    for (int i = 0; i < batch_size; i++)
    {
        Tensor *x = batch->get_x(i);

        // Depending on our layer configuration, we may need to adjust y.
        // TODO
        float y_val = batch->get_y(i)->get_idx(0);
        Tensor *y;
        bool own_y_flg = false;
        if (this->neurons[this->neurons.size() - 1]->get_col_cnt() > 1)
        {
            // One hot encode!
            own_y_flg = true;
            y = new Tensor(1, this->neurons[this->neurons.size() - 1]->get_col_cnt(), Cpu);
            y->set_all(0.0f);
            for (int j = 0; j < this->neurons[this->neurons.size() - 1]->get_col_cnt(); j++)
            {
                if ((int)y_val == j)
                {
                    y->set_idx(j, 1.0f);
                    break;
                }
            }
        }
        else
        {
            y = batch->get_y(i);
        }

        this->feed_forward(x);
        cost += this->get_cost(y);
        this->back_propagate(y);

        TensorTuple max_tup = this->neurons[this->neurons.size() - 1]->get_max();
        if (max_tup.idx == (int)y_val)
        {
            result.cor_cnt++;
        }

        if (own_y_flg)
        {
            delete y;
        }
    }

    cost /= batch_size;

    result.cost = cost;

    this->optimize(batch_size);

    return result;
}

Result NN::validate(Batch *batch)
{
    Result result;

    int batch_size = batch->get_size();

    float cost = 0.0f;

    for (int i = 0; i < batch_size; i++)
    {
        Tensor *x = batch->get_x(i);
        Tensor *y = batch->get_y(i);

        this->feed_forward(x);
        cost += this->get_cost(y);
    }

    cost /= batch_size;

    result.cost = cost;

    return result;
}

Result NN::test(Batch *batch)
{
    Result result;

    int batch_size = batch->get_size();

    float cost = 0.0f;

    for (int i = 0; i < batch_size; i++)
    {
        Tensor *x = batch->get_x(i);
        Tensor *y = batch->get_y(i);

        this->feed_forward(x);
        cost += this->get_cost(y);
    }

    cost /= batch_size;

    result.cost = cost;

    return result;
}
