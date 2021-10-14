#include "cnn.cuh"

#define THREADS_PER_BLOCK 32

using namespace zero::core;
using namespace zero::nn;

// Device functions:

__device__ float d_cnn_relu(float val)
{
    return val > 0.0f ? val : 0.0f;
}

__device__ float d_cnn_derive_relu(float val)
{
    return val > 0.0f ? 1.0f : 0.0f;
}

__device__ float d_cnn_sigmoid(float val)
{
    return (1.0 / (1.0 + exp(-val)));
}

__device__ float d_cnn_derive_sigmoid(float val)
{
    return (val) * (1.0 - val);
}

__device__ float d_cnn_tanh(float val)
{
    return ((exp(val) - exp(-val)) / (exp(val) + exp(-val)));
}

__device__ float d_cnn_derive_tanh(float val)
{
    return (1 - (val * val));
}

__device__ float d_cnn_sine(float val)
{
    return sin(val);
}

__device__ float d_cnn_derive_sine(float val)
{
    return cos(val);
}

__device__ float d_cnn_cosine(float val)
{
    return cos(val);
}

__device__ float d_cnn_derive_cosine(float val)
{
    return -sin(val);
}

__device__ float d_cnn_mse_cost(float n_val, float y_val)
{
    return ((n_val - y_val) * (n_val - y_val));
}

__device__ float d_cnn_derive_mse_cost(float n_val, float y_val)
{
    return 2.0f * (n_val - y_val);
}

__device__ float d_cnn_cross_entropy_cost(float n_val, float y_val)
{
    return (float)((y_val * log(n_val)) + ((1.0 - y_val) * log(1.0 - n_val)));
}

__device__ float d_cnn_derive_cross_entropy_cost(float n_val, float y_val)
{
    return (n_val - y_val);
}

// Kernel functions:

__global__ void k_cnn_set_arr(float *arr, int cnt, float val)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        arr[tid] = val;
    }
}

__global__ void k_cnn_convolution(float *n_arr, float *f_arr, float *b_arr, float *nxt_n_arr, int chan_cnt, int n_row_cnt, int n_col_cnt,
                                  int f_row_cnt, int f_col_cnt, int nxt_n_row_cnt, int nxt_n_col_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int nxt_n_idx = tid;

    if (nxt_n_idx < (nxt_n_row_cnt * nxt_n_col_cnt))
    {
        int nxt_n_row_idx = nxt_n_idx / nxt_n_col_cnt;
        int nxt_n_col_idx = nxt_n_idx % nxt_n_col_cnt;

        for (int chan_idx = 0; chan_idx < chan_cnt; chan_idx++)
        {
            for (int f_row_idx = 0; f_row_idx < f_row_cnt; f_row_idx++)
            {
                for (int f_col_idx = 0; f_col_idx < f_col_cnt; f_col_idx++)
                {
                    int n_local_row_idx = nxt_n_row_idx + f_row_idx;
                    int n_local_col_idx = nxt_n_col_idx + f_col_idx;

                    int f_local_rot_idx = (f_row_cnt * f_col_cnt) - (f_row_idx * f_col_cnt + f_col_idx);

                    float val = n_arr[(chan_idx * n_row_cnt * n_col_cnt) + (n_local_row_idx * n_col_cnt) + n_local_col_idx];
                    val *= f_arr[(chan_idx * f_row_cnt * f_col_cnt) + f_local_rot_idx];
                    val += b_arr[nxt_n_idx];
                    nxt_n_arr[nxt_n_idx] += val;
                }
            }
        }
    }
}

__global__ void k_cnn_activate(float *n_arr, int n_cnt, ActivationFunctionId activation_func_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (activation_func_id)
        {
        case ReLU:
            n_arr[tid] = d_cnn_relu(n_arr[tid]);
            break;
        case Sigmoid:
            n_arr[tid] = d_cnn_sigmoid(n_arr[tid]);
            break;
        case Tanh:
            n_arr[tid] = d_cnn_tanh(n_arr[tid]);
            break;
        case Sine:
            n_arr[tid] = d_cnn_sine(n_arr[tid]);
            break;
        case Cosine:
            n_arr[tid] = d_cnn_cosine(n_arr[tid]);
            break;
        default:
            // None
            break;
        }
    }
}

__global__ void k_cnn_derive_activation(float *n_arr, float *agg_derivatives_arr, int n_cnt, ActivationFunctionId activation_func_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (activation_func_id)
        {
        case ReLU:
            agg_derivatives_arr[tid] *= d_cnn_derive_relu(n_arr[tid]);
            break;
        case Sigmoid:
            agg_derivatives_arr[tid] *= d_cnn_derive_sigmoid(n_arr[tid]);
            break;
        case Tanh:
            agg_derivatives_arr[tid] *= d_cnn_derive_tanh(n_arr[tid]);
            break;
        case Sine:
            agg_derivatives_arr[tid] *= d_cnn_derive_sine(n_arr[tid]);
            break;
        case Cosine:
            agg_derivatives_arr[tid] *= d_cnn_derive_cosine(n_arr[tid]);
            break;
        default:
            // None
            break;
        }
    }
}

__global__ void k_cnn_derive_z_and_increment_filter_derivative(float *agg_derivatives_arr, float *n_arr, float *df_arr, int chan_cnt, int n_row_cnt, int n_col_cnt, int f_row_cnt, int f_col_cnt, int prv_n_row_cnt, int prv_n_col_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int f_idx = tid;
    int f_local_cnt = (f_row_cnt * f_col_cnt);
    int chan_idx = f_idx / f_local_cnt;
    int f_local_idx = f_idx - (chan_idx * f_local_cnt);
    int f_local_rot_idx = f_local_cnt - f_local_idx;
    int f_local_row_idx = f_local_idx / f_col_cnt;
    int f_local_col_idx = f_local_idx % f_col_cnt;
    int f_rot_idx = (chan_idx * f_local_cnt) + f_local_rot_idx;

    if (f_idx < (chan_cnt * f_row_cnt * f_col_cnt))
    {
        float val = 0.0f;

        for (int i = 0; i < prv_n_row_cnt; i++)
        {
            int row_idx = (i + f_local_row_idx);

            for (int j = 0; j < prv_n_col_cnt; j++)
            {
                int col_idx = (j + f_local_col_idx);

                val += (agg_derivatives_arr[(i * prv_n_col_cnt + j)] * n_arr[(chan_idx * n_row_cnt * n_col_cnt) + (row_idx * n_col_cnt) + col_idx]);
            }
        }

        df_arr[f_rot_idx] += val;
    }
}

__global__ void k_cnn_derive_z_and_increment_bias_derivative(float *agg_derivatives_arr, float *db_arr, int n_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        db_arr[tid] += (agg_derivatives_arr[tid]);
    }
}

__global__ void k_cnn_derive_z_and_aggregate_derivatives(float *agg_derivatives_arr, float *f_arr, float *nxt_agg_derivatives_arr,
                                                         int n_row_cnt, int n_col_cnt, int nxt_f_row_cnt, int nxt_f_col_cnt, int nxt_n_row_cnt, int nxt_n_col_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int nxt_n_idx = tid;

    //nxt_agg_derivatives_arr[nxt_n_idx] += (agg_derivatives_arr[n_idx] * f_arr[w_idx]);
}

__global__ void
k_cnn_adjust_filter(float *f_arr, float *df_arr, int batch_size, float learning_rate, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        f_arr[tid] -= ((df_arr[tid] * learning_rate) / (float)batch_size);
        df_arr[tid] = 0.0f;
    }
}

__global__ void k_cnn_adjust_bias(float *b_arr, float *db_arr, int batch_size, float learning_rate, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        b_arr[tid] -= ((db_arr[tid] * learning_rate) / (float)batch_size);
        db_arr[tid] = 0.0f;
    }
}

// CNNLayerConfiguration member functions:

CNNLayerConfiguration::CNNLayerConfiguration()
{
}

CNNLayerConfiguration::CNNLayerConfiguration(int channel_cnt, int neuron_row_cnt, int neuron_col_cnt,
                                             int filter_cnt, int filter_row_cnt, int filter_col_cnt,
                                             ActivationFunctionId activation_func_id)
{
    this->channel_cnt = channel_cnt;
    this->neuron_row_cnt = neuron_row_cnt;
    this->neuron_col_cnt = neuron_col_cnt;
    this->filter_cnt = filter_cnt;
    this->filter_row_cnt = filter_row_cnt;
    this->filter_col_cnt = filter_col_cnt;
    this->activation_func_id = activation_func_id;
}

CNNLayerConfiguration::~CNNLayerConfiguration()
{
}

// CNN member functions:

CNN::CNN(CostFunctionId cost_func_id, float learning_rate)
{
    this->learning_rate = learning_rate;

    this->nn = new NN(cost_func_id, learning_rate);
}

CNN::~CNN()
{
    int lyr_cnt = this->layer_configurations.size();
    int lst_lyr_idx = lyr_cnt - 1;

    for (int lyr_idx = 0; lyr_idx < lst_lyr_idx; lyr_idx++)
    {
        CNNLayerConfiguration *lyr_cfg = &this->layer_configurations[lyr_idx];

        delete this->neurons[lyr_idx];

        for (int filter_idx = 0; filter_idx < lyr_cfg->filter_cnt; filter_idx++)
        {
            delete this->filters[lyr_idx][filter_idx];
            delete this->biases[lyr_idx][filter_idx];
            delete this->filter_derivatives[lyr_idx][filter_idx];
            delete this->bias_derivatives[lyr_idx][filter_idx];
        }
    }

    // Dont forget about the output layer!
    {
        delete this->neurons[lst_lyr_idx];
    }

    delete this->nn;
}

void CNN::add_layer(ActivationFunctionId activation_func_id)
{
    this->add_layer(1, 0, 0, activation_func_id);
}

void CNN::add_layer(int filter_cnt, int filter_row_cnt, int filter_col_cnt,
                    ActivationFunctionId activation_func_id)
{
    int lyr_cnt = this->layer_configurations.size();
    CNNLayerConfiguration *prv_lyr_cfg = &this->layer_configurations[lyr_cnt - 1];

    int chan_cnt = prv_lyr_cfg->filter_cnt;
    int n_row_cnt = prv_lyr_cfg->neuron_row_cnt - prv_lyr_cfg->filter_row_cnt + 1;
    int n_col_cnt = prv_lyr_cfg->neuron_col_cnt - prv_lyr_cfg->filter_col_cnt + 1;

    this->add_layer(chan_cnt, n_row_cnt, n_col_cnt,
                    filter_cnt, filter_row_cnt, filter_col_cnt, activation_func_id);
}

void CNN::add_layer(int channel_cnt, int neuron_row_cnt, int neuron_col_cnt,
                    int filter_cnt, int filter_row_cnt, int filter_col_cnt)
{
    this->add_layer(channel_cnt, neuron_row_cnt, neuron_col_cnt,
                    filter_cnt, filter_row_cnt, filter_col_cnt, None);
}

void CNN::add_layer(int channel_cnt, int neuron_row_cnt, int neuron_col_cnt,
                    int filter_cnt, int filter_row_cnt, int filter_col_cnt,
                    ActivationFunctionId activation_func_id)
{
    this->layer_configurations.push_back(CNNLayerConfiguration(channel_cnt, neuron_row_cnt, neuron_col_cnt,
                                                               filter_cnt, filter_row_cnt, filter_col_cnt, activation_func_id));
}

NN *CNN::fully_connected()
{
    return this->nn;
}

void CNN::compile()
{
    int lyr_cnt = this->layer_configurations.size();
    int lst_lyr_idx = lyr_cnt - 1;

    for (int lyr_idx = 0; lyr_idx < lst_lyr_idx; lyr_idx++)
    {
        CNNLayerConfiguration *lyr_cfg = &this->layer_configurations[lyr_idx];
        CNNLayerConfiguration *nxt_lyr_cfg = &this->layer_configurations[lyr_idx + 1];

        Tensor *n = new Tensor(lyr_cfg->channel_cnt * lyr_cfg->neuron_row_cnt, lyr_cfg->neuron_col_cnt, Gpu);
        n->set_all(0.0f);
        this->neurons.push_back(n);

        this->filters.push_back(std::vector<Tensor *>());
        this->biases.push_back(std::vector<Tensor *>());
        this->filter_derivatives.push_back(std::vector<Tensor *>());
        this->bias_derivatives.push_back(std::vector<Tensor *>());

        for (int filter_idx = 0; filter_idx < lyr_cfg->filter_cnt; filter_idx++)
        {
            Tensor *f = new Tensor(lyr_cfg->channel_cnt * lyr_cfg->filter_row_cnt, lyr_cfg->filter_col_cnt, Gpu);
            f->set_all_rand_normal_distribution(0.0f, sqrt(2.0f / (lyr_cfg->neuron_row_cnt * lyr_cfg->neuron_col_cnt)));
            this->filters[lyr_idx].push_back(f);

            Tensor *b = new Tensor(nxt_lyr_cfg->neuron_row_cnt, nxt_lyr_cfg->neuron_col_cnt, Gpu);
            b->set_all(0.0f);
            this->biases[lyr_idx].push_back(b);

            Tensor *df = new Tensor(lyr_cfg->channel_cnt * lyr_cfg->filter_row_cnt, lyr_cfg->filter_col_cnt, Gpu);
            df->set_all(0.0f);
            this->filter_derivatives[lyr_idx].push_back(df);

            Tensor *db = new Tensor(nxt_lyr_cfg->neuron_row_cnt, nxt_lyr_cfg->neuron_col_cnt, Gpu);
            db->set_all(0.0f);
            this->bias_derivatives[lyr_idx].push_back(db);
        }
    }

    // Dont forget about the output layer!
    {
        CNNLayerConfiguration *lyr_cfg = &this->layer_configurations[lst_lyr_idx];

        Tensor *n = new Tensor(lyr_cfg->filter_cnt * lyr_cfg->neuron_row_cnt, lyr_cfg->neuron_col_cnt, Gpu);
        n->set_all(0.0f);
        this->neurons.push_back(n);
    }

    // Add input layer of fully connected nn:
    {
        CNNLayerConfiguration *lyr_cfg = &this->layer_configurations[lst_lyr_idx];

        this->nn->add_layer(lyr_cfg->filter_cnt * lyr_cfg->neuron_row_cnt * lyr_cfg->neuron_col_cnt);
    }
}

void CNN::feed_forward(Tensor *x, bool train_flg)
{
    x->translate(Gpu);

    // Need to set input neurons before we do anything.
    this->neurons[0]->set_arr(x->get_arr(Gpu), Gpu);

    int lyr_cnt = this->layer_configurations.size();
    int lst_lyr_idx = lyr_cnt - 1;

    for (int lyr_idx = 0; lyr_idx < lst_lyr_idx; lyr_idx++)
    {
        CNNLayerConfiguration *lyr_cfg = &this->layer_configurations[lyr_idx];
        CNNLayerConfiguration *nxt_lyr_cfg = &this->layer_configurations[lyr_idx + 1];

        int n_cnt = (lyr_cfg->channel_cnt * lyr_cfg->neuron_row_cnt * lyr_cfg->neuron_col_cnt);
        int nxt_n_cnt = (lyr_cfg->filter_cnt * nxt_lyr_cfg->neuron_row_cnt * nxt_lyr_cfg->neuron_col_cnt);

        Tensor *n = this->neurons[lyr_idx];
        Tensor *nxt_n = this->neurons[lyr_idx + 1];

        // Need to reset next layer neurons before we do anything:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((nxt_n_cnt / threads_per_block) + 1);
            k_cnn_set_arr<<<num_blocks, threads_per_block>>>(nxt_n->get_arr(Gpu), nxt_n_cnt, 0.0f);
        }

        // Convolution:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(((nxt_n_cnt / lyr_cfg->filter_cnt) / threads_per_block) + 1);

            for (int filter_idx = 0; filter_idx < lyr_cfg->filter_cnt; filter_idx++)
            {
                Tensor *f = this->filters[lyr_idx][filter_idx];
                Tensor *b = this->biases[lyr_idx][filter_idx];

                k_cnn_convolution<<<num_blocks, threads_per_block>>>(n->get_arr(Gpu), f->get_arr(Gpu), b->get_arr(Gpu), nxt_n->get_slice(filter_idx * nxt_lyr_cfg->neuron_row_cnt * nxt_lyr_cfg->neuron_col_cnt, Gpu),
                                                                     lyr_cfg->channel_cnt, lyr_cfg->neuron_row_cnt, lyr_cfg->neuron_col_cnt, lyr_cfg->filter_row_cnt, lyr_cfg->filter_col_cnt,
                                                                     nxt_lyr_cfg->neuron_row_cnt, nxt_lyr_cfg->neuron_col_cnt);
            }
        }

        // Activate:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((nxt_n_cnt / threads_per_block) + 1);
            k_cnn_activate<<<num_blocks, threads_per_block>>>(nxt_n->get_arr(Gpu), nxt_n_cnt, nxt_lyr_cfg->activation_func_id);
        }
    }

    // Fully connected:
    {
        this->nn->feed_forward(this->neurons[lst_lyr_idx], train_flg);
    }
}

float CNN::get_cost(Tensor *y)
{
    y->translate(Gpu);

    return this->nn->get_cost(y);
}

void CNN::back_propagate(Tensor *y)
{
    y->translate(Gpu);

    Tensor *agg_derivatives;

    // Fully connected:
    {
        agg_derivatives = this->nn->back_propagate(y, true);
    }

    int lyr_cnt = this->layer_configurations.size();
    int lst_lyr_idx = lyr_cnt - 1;

    for (int lyr_idx = lst_lyr_idx; lyr_idx > 0; lyr_idx--)
    {
        CNNLayerConfiguration *lyr_cfg = &this->layer_configurations[lyr_idx];
        CNNLayerConfiguration *nxt_lyr_cfg = &this->layer_configurations[lyr_idx - 1];

        int n_cnt = lyr_cfg->channel_cnt * lyr_cfg->neuron_row_cnt * lyr_cfg->neuron_col_cnt;
        int nxt_n_cnt = nxt_lyr_cfg->channel_cnt * nxt_lyr_cfg->neuron_row_cnt * nxt_lyr_cfg->neuron_col_cnt;

        Tensor *n = this->neurons[lyr_idx];
        Tensor *nxt_n = this->neurons[lyr_idx - 1];

        // Derive activation (with respect to z):
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks((n_cnt / threads_per_block) + 1);
            k_cnn_derive_activation<<<num_blocks, threads_per_block>>>(n->get_arr(Gpu),
                                                                       agg_derivatives->get_arr(Gpu), n_cnt, lyr_cfg->activation_func_id);
        }

        // Derive z (with respect to filter):
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(((nxt_lyr_cfg->channel_cnt * nxt_lyr_cfg->filter_row_cnt * nxt_lyr_cfg->filter_col_cnt) / threads_per_block) + 1);

            for (int filter_idx = 0; filter_idx < lyr_cfg->filter_cnt; filter_idx++)
            {
                Tensor *nxt_df = this->filter_derivatives[lyr_idx - 1][filter_idx];

                k_cnn_derive_z_and_increment_filter_derivative<<<num_blocks, threads_per_block>>>(agg_derivatives->get_slice(filter_idx * lyr_cfg->neuron_row_cnt * lyr_cfg->neuron_col_cnt, Gpu),
                                                                                                  nxt_n->get_arr(Gpu),
                                                                                                  nxt_df->get_arr(Gpu),
                                                                                                  nxt_lyr_cfg->channel_cnt, nxt_lyr_cfg->neuron_row_cnt, nxt_lyr_cfg->neuron_col_cnt,
                                                                                                  nxt_lyr_cfg->filter_row_cnt, nxt_lyr_cfg->filter_col_cnt, lyr_cfg->neuron_row_cnt, lyr_cfg->neuron_col_cnt);
            }
        }

        // Derive z (with respect to bias):
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(((n_cnt / lyr_cfg->filter_cnt) / threads_per_block) + 1);

            for (int filter_idx = 0; filter_idx < lyr_cfg->filter_cnt; filter_idx++)
            {
                Tensor *nxt_db = this->bias_derivatives[lyr_idx - 1][filter_idx];

                k_cnn_derive_z_and_increment_bias_derivative<<<num_blocks, threads_per_block>>>(agg_derivatives->get_arr(Gpu),
                                                                                                nxt_db->get_arr(Gpu),
                                                                                                n_cnt);
            }
        }

        // Derive z (with respect to activation) and aggregate derivatives:
        {
            if (lyr_idx > 1)
            {
                Tensor *nxt_agg_derivatives = new Tensor(1, nxt_n_cnt, Gpu);
                nxt_agg_derivatives->set_all(0.0f);

                {
                    int threads_per_block(THREADS_PER_BLOCK);
                    int num_blocks((nxt_n_cnt / threads_per_block) + 1);

                    for (int filter_idx = 0; filter_idx < lyr_cfg->filter_cnt; filter_idx++)
                    {
                        Tensor *nxt_f = this->filters[lyr_idx - 1][filter_idx];

                        k_cnn_derive_z_and_aggregate_derivatives<<<num_blocks, threads_per_block>>>(agg_derivatives->get_arr(Gpu), nxt_f->get_arr(Gpu),
                                                                                                    nxt_agg_derivatives->get_arr(Gpu),
                                                                                                    lyr_cfg->neuron_row_cnt, lyr_cfg->neuron_col_cnt,
                                                                                                    nxt_lyr_cfg->filter_row_cnt, nxt_lyr_cfg->filter_col_cnt,
                                                                                                    nxt_lyr_cfg->neuron_row_cnt, nxt_lyr_cfg->neuron_col_cnt);
                    }
                }

                delete agg_derivatives;
                agg_derivatives = nxt_agg_derivatives;
            }
        }
    }

    delete agg_derivatives;
}

void CNN::optimize(int batch_size)
{

    int lyr_cnt = this->layer_configurations.size();
    int lst_lyr_idx = lyr_cnt - 1;

    for (int lyr_idx = 0; lyr_idx < lst_lyr_idx; lyr_idx++)
    {
        CNNLayerConfiguration *lyr_cfg = &this->layer_configurations[lyr_idx];
        CNNLayerConfiguration *nxt_lyr_cfg = &this->layer_configurations[lyr_idx + 1];

        // Weights:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(((lyr_cfg->channel_cnt * lyr_cfg->filter_row_cnt * lyr_cfg->filter_col_cnt) / threads_per_block) + 1);

            for (int filter_idx = 0; filter_idx < lyr_cfg->filter_cnt; filter_idx++)
            {
                Tensor *f = this->filters[lyr_idx][filter_idx];
                Tensor *df = this->filter_derivatives[lyr_idx][filter_idx];

                k_cnn_adjust_filter<<<num_blocks, threads_per_block>>>(f->get_arr(Gpu), df->get_arr(Gpu), batch_size, this->learning_rate,
                                                                       (lyr_cfg->channel_cnt * lyr_cfg->filter_row_cnt * lyr_cfg->filter_col_cnt));
            }
        }

        // Biases:
        {
            int threads_per_block(THREADS_PER_BLOCK);
            int num_blocks(((nxt_lyr_cfg->neuron_row_cnt * nxt_lyr_cfg->neuron_col_cnt) / threads_per_block) + 1);

            for (int filter_idx = 0; filter_idx < lyr_cfg->filter_cnt; filter_idx++)
            {
                Tensor *b = this->biases[lyr_idx][filter_idx];
                Tensor *db = this->bias_derivatives[lyr_idx][filter_idx];

                k_cnn_adjust_bias<<<num_blocks, threads_per_block>>>(b->get_arr(Gpu), db->get_arr(Gpu), batch_size, this->learning_rate,
                                                                     (nxt_lyr_cfg->neuron_row_cnt * nxt_lyr_cfg->neuron_col_cnt));
            }
        }
    }

    // Fully connected:
    {
        this->nn->optimize(batch_size);
    }
}

void CNN::check_gradient(Tensor *x, Tensor *y, bool print_flg)
{
    float agg_ana_grad = 0.0f;
    float agg_num_grad = 0.0f;
    float agg_grad_diff = 0.0f;

    float epsilon = 0.001f;

    // Analytical gradients:
    {
        this->feed_forward(x, true);
        this->back_propagate(y);
    }

    // Numerical gradients:
    {
        int lyr_cnt = this->layer_configurations.size();
        int lst_lyr_idx = lyr_cnt - 1;

        for (int lyr_idx = 0; lyr_idx < lst_lyr_idx; lyr_idx++)
        {
            CNNLayerConfiguration *lyr_cfg = &this->layer_configurations[lyr_idx];
            CNNLayerConfiguration *nxt_lyr_cfg = &this->layer_configurations[lyr_idx + 1];

            for (int filter_idx = 0; filter_idx < lyr_cfg->filter_cnt; filter_idx++)
            {
                Tensor *w = this->filters[lyr_idx][filter_idx];
                Tensor *dw = this->filter_derivatives[lyr_idx][filter_idx];
                Tensor *b = this->biases[lyr_idx][filter_idx];
                Tensor *db = this->bias_derivatives[lyr_idx][filter_idx];

                // Filters:
                for (int w_idx = 0; w_idx < (lyr_cfg->channel_cnt * lyr_cfg->filter_row_cnt * lyr_cfg->filter_col_cnt); w_idx++)
                {
                    float left_cost = 0.0;
                    float right_cost = 0.0;

                    float orig_w_val = w->get_val(w_idx);

                    float left_w_val = orig_w_val - epsilon;
                    float right_w_val = orig_w_val + epsilon;

                    float ana_grad = dw->get_val(w_idx);

                    // Left:
                    w->set_val(w_idx, left_w_val);
                    {
                        this->feed_forward(x, true);
                        left_cost += this->get_cost(y);
                    }

                    // Right:
                    w->set_val(w_idx, right_w_val);
                    {
                        this->feed_forward(x, true);
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

                    w->set_val(w_idx, orig_w_val);
                }

                // Biases:
                for (int b_idx = 0; b_idx < nxt_lyr_cfg->neuron_row_cnt * nxt_lyr_cfg->neuron_col_cnt; b_idx++)
                {
                    float left_cost = 0.0;
                    float right_cost = 0.0;

                    float orig_b_val = b->get_val(b_idx);

                    float left_b_val = orig_b_val - epsilon;
                    float right_b_val = orig_b_val + epsilon;

                    float ana_grad = db->get_val(b_idx);

                    // Left:
                    b->set_val(b_idx, left_b_val);
                    {
                        this->feed_forward(x, true);
                        left_cost += this->get_cost(y);
                    }

                    // Right:
                    b->set_val(b_idx, right_b_val);
                    {
                        this->feed_forward(x, true);
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

                    b->set_val(b_idx, orig_b_val);
                }
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