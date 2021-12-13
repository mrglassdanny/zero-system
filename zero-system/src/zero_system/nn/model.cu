#include "model.cuh"

using namespace zero::core;
using namespace zero::nn;

// Device functions:

__device__ float d_mse_cost(float n_val, float y_val)
{
    return ((n_val - y_val) * (n_val - y_val));
}

__device__ float d_derive_mse_cost(float n_val, float y_val)
{
    return 2.0f * (n_val - y_val);
}

__device__ float d_softmax(float val, float *arr, int cnt)
{
    float e_sum_val = 0.0f;

    for (int i = 0; i < cnt; i++)
    {
        e_sum_val += exp(arr[i]);
    }

    return exp(val) / e_sum_val;
}

__device__ float d_cross_entropy_cost(float n_val, float y_val, float *n_arr, int n_cnt)
{
    float np_val = d_softmax(n_val, n_arr, n_cnt);
    return -(y_val * log(np_val));
}

__device__ float d_derive_cross_entropy_cost(float n_val, float y_val, float *n_arr, int n_cnt)
{
    float np_val = d_softmax(n_val, n_arr, n_cnt);
    return np_val - y_val;
}

// Kernel functions:

__global__ void k_cost(float *n_arr, float *y_arr, float *cost_val, int n_cnt, CostFunction cost_fn)
{
    __shared__ float temp[CUDA_THREADS_PER_BLOCK];
    memset(temp, 0, CUDA_THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (cost_fn)
        {
        case MSE:
            temp[threadIdx.x] = d_mse_cost(n_arr[tid], y_arr[tid]);
            break;
        case CrossEntropy:
            temp[threadIdx.x] = d_cross_entropy_cost(n_arr[tid], y_arr[tid], n_arr, n_cnt);
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
        for (int i = 0; i < CUDA_THREADS_PER_BLOCK; i++)
        {
            sum += temp[i];
        }

        atomicAdd(cost_val, sum);
    }
}

__global__ void k_derive_cost(float *n_arr, float *y_arr, float *dc_arr, int n_cnt, CostFunction cost_fn)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (cost_fn)
        {
        case MSE:
            dc_arr[tid] *= d_derive_mse_cost(n_arr[tid], y_arr[tid]);
            break;
        case CrossEntropy:
            dc_arr[tid] *= d_derive_cross_entropy_cost(n_arr[tid], y_arr[tid], n_arr, n_cnt);
            break;
        default:
            break;
        }
    }
}

// Model functions:

Model::Model(CostFunction cost_fn, float learning_rate)
{
    this->cost_fn = cost_fn;
    this->learning_rate = learning_rate;

    cudaMalloc(&this->d_cost_val, sizeof(float));
    cudaMemset(this->d_cost_val, 0, sizeof(float));
}

Model::Model(const char *path)
{
    FILE *file_ptr = fopen(path, "rb");

    fread(&this->cost_fn, sizeof(CostFunction), 1, file_ptr);
    fread(&this->learning_rate, sizeof(float), 1, file_ptr);

    int lyr_cnt = 0;
    fread(&lyr_cnt, sizeof(int), 1, file_ptr);

    for (int i = 0; i < lyr_cnt; i++)
    {
        LayerType lyr_typ;
        fread(&lyr_typ, sizeof(LayerType), 1, file_ptr);

        Layer *lyr = nullptr;

        switch (lyr_typ)
        {
        case LayerType::Linear:
            lyr = new LinearLayer(file_ptr);
            break;
        case LayerType::Convolutional:
            lyr = new ConvolutionalLayer(file_ptr);
            break;
        case LayerType::Activation:
            lyr = new ActivationLayer(file_ptr);
            break;
        case LayerType::Dropout:
            lyr = new DropoutLayer(file_ptr);
            break;
        case LayerType::Pooling:
            lyr = new PoolingLayer(file_ptr);
            break;
        default:
            break;
        }

        this->add_layer(lyr);
    }

    cudaMalloc(&this->d_cost_val, sizeof(float));
    cudaMemset(this->d_cost_val, 0, sizeof(float));

    fclose(file_ptr);
}

Model::~Model()
{
    for (Layer *lyr : this->layers)
    {
        delete lyr;
    }

    cudaFree(this->d_cost_val);
}

void Model::save(const char *path)
{
    FILE *file_ptr = fopen(path, "wb");

    fwrite(&this->cost_fn, sizeof(CostFunction), 1, file_ptr);
    fwrite(&this->learning_rate, sizeof(float), 1, file_ptr);

    int lyr_cnt = this->layers.size();
    fwrite(&lyr_cnt, sizeof(int), 1, file_ptr);

    for (Layer *lyr : this->layers)
    {
        LayerType lyr_typ = lyr->get_type();
        fwrite(&lyr_typ, sizeof(LayerType), 1, file_ptr);

        lyr->save(file_ptr);
    }

    fclose(file_ptr);
}

void Model::add_layer(Layer *lyr)
{
    this->layers.push_back(lyr);
}

void Model::linear(int nxt_n_cnt)
{
    this->linear(this->get_output_shape(), nxt_n_cnt, InitializationFunction::Xavier);
}

void Model::linear(int nxt_n_cnt, InitializationFunction init_fn)
{
    this->linear(this->get_output_shape(), nxt_n_cnt, init_fn);
}

void Model::linear(std::vector<int> n_shape, int nxt_n_cnt)
{
    this->linear(n_shape, nxt_n_cnt, InitializationFunction::Xavier);
}

void Model::linear(std::vector<int> n_shape, int nxt_n_cnt, InitializationFunction init_fn)
{
    this->add_layer(new LinearLayer(n_shape, nxt_n_cnt, init_fn));
}

void Model::convolutional(int fltr_cnt, int w_row_cnt, int w_col_cnt)
{
    this->convolutional(this->get_output_shape(), fltr_cnt, w_row_cnt, w_col_cnt, InitializationFunction::Xavier);
}

void Model::convolutional(int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn)
{
    this->convolutional(this->get_output_shape(), fltr_cnt, w_row_cnt, w_col_cnt, init_fn);
}

void Model::convolutional(std::vector<int> n_shape, int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn)
{
    this->convolutional(n_shape, fltr_cnt, w_row_cnt, w_col_cnt, InitializationFunction::Xavier);
}

void Model::convolutional(std::vector<int> n_shape, int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn)
{
    this->add_layer(new ConvolutionalLayer(n_shape, fltr_cnt, w_row_cnt, w_col_cnt, init_fn));
}

void Model::activation(ActivationFunction activation_fn)
{
    this->activation(this->get_output_shape(), activation_fn);
}

void Model::activation(std::vector<int> n_shape, ActivationFunction activation_fn)
{
    this->add_layer(new ActivationLayer(n_shape, activation_fn));
}

void Model::dropout(float dropout_rate)
{
    this->dropout(this->get_output_shape(), dropout_rate);
}

void Model::dropout(std::vector<int> n_shape, float dropout_rate)
{
    this->add_layer(new DropoutLayer(n_shape, dropout_rate));
}

void Model::pooling(PoolingFunction pool_fn)
{
    this->pooling(this->get_output_shape(), pool_fn);
}

void Model::pooling(std::vector<int> n_shape, PoolingFunction pool_fn)
{
    this->add_layer(new PoolingLayer(n_shape, pool_fn));
}

std::vector<int> Model::get_input_shape()
{
    return this->layers[0]->get_input_shape();
}

std::vector<int> Model::get_output_shape()
{
    return this->layers[this->layers.size() - 1]->get_output_shape();
}

void Model::set_learning_rate(float learning_rate)
{
    this->learning_rate = learning_rate;
}

Tensor *Model::forward(Tensor *x, bool train_flg)
{
    // Convert to Cuda.
    x->to(Device::Cuda);

    int lst_lyr_idx = this->layers.size() - 1;

    Layer *frst_lyr = this->layers[0];
    Layer *lst_lyr = this->layers[lst_lyr_idx];

    frst_lyr->set_neurons(x);

    for (int i = 0; i < lst_lyr_idx; i++)
    {
        Layer *lyr = this->layers[i];
        Layer *nxt_lyr = this->layers[i + 1];

        lyr->forward(nxt_lyr->get_neurons(), train_flg);
    }

    Tensor *pred = new Tensor(Device::Cuda, lst_lyr->get_output_shape());
    lst_lyr->forward(pred, train_flg);

    return pred;
}

float Model::cost(Tensor *pred, Tensor *y)
{
    // Convert to Cuda.
    y->to(Device::Cuda);

    float h_cost_val = 0.0f;

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (pred->get_cnt() / threads_per_block) + 1;
        k_cost<<<num_blocks, threads_per_block>>>(pred->get_arr(), y->get_arr(),
                                                  this->d_cost_val, pred->get_cnt(), this->cost_fn);
    }

    cudaMemcpy(&h_cost_val, this->d_cost_val, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemset(this->d_cost_val, 0, sizeof(float));

    return h_cost_val;
}

void Model::backward(Tensor *pred, Tensor *y)
{
    // Convert to Cuda.
    y->to(Device::Cuda);

    Tensor *dc = new Tensor(Device::Cuda, pred->get_shape());
    dc->set_all(1.0f);

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (pred->get_cnt() / threads_per_block) + 1;
        k_derive_cost<<<num_blocks, threads_per_block>>>(pred->get_arr(),
                                                         y->get_arr(), dc->get_arr(), pred->get_cnt(), this->cost_fn);
    }

    int lst_lyr_idx = this->layers.size() - 1;

    for (int i = lst_lyr_idx; i >= 0; i--)
    {
        Layer *lyr = this->layers[i];
        dc = lyr->backward(dc);
    }

    delete dc;
}

void Model::step(int batch_size)
{
    for (Layer *lyr : this->layers)
    {
        if (LearnableLayer *lrn_lyr = dynamic_cast<LearnableLayer *>(lyr))
        {
            lrn_lyr->step(batch_size, this->learning_rate);
        }
    }
}

void Model::gradient_check(Tensor *x, Tensor *y, bool print_flg)
{
    x->to(Device::Cuda);
    y->to(Device::Cuda);

    float agg_ana_grad = 0.0f;
    float agg_num_grad = 0.0f;
    float agg_grad_diff = 0.0f;

    // Analytical gradients:
    {
        Tensor *pred = this->forward(x, true);
        this->cost(pred, y);
        this->backward(pred, y);
        delete pred;
    }

    // Numerical gradients:
    {
        int lyr_idx = 0;

        for (Layer *lyr : this->layers)
        {
            lyr_idx++;

            if (LearnableLayer *lrn_lyr = dynamic_cast<LearnableLayer *>(lyr))
            {
                Tensor *w = lrn_lyr->get_weights();
                Tensor *dw = lrn_lyr->get_weight_derivatives();
                Tensor *b = lrn_lyr->get_biases();
                Tensor *db = lrn_lyr->get_bias_derivatives();

                for (int i = 0; i < w->get_cnt(); i++)
                {
                    float left_cost = 0.0f;
                    float right_cost = 0.0f;

                    float orig_w_val = w->get_val(i);

                    float left_w_val = orig_w_val - EPSILON;
                    float right_w_val = orig_w_val + EPSILON;

                    float ana_grad = dw->get_val(i);

                    w->set_val(i, left_w_val);
                    {
                        Tensor *pred = this->forward(x, true);
                        left_cost = this->cost(pred, y);
                        delete pred;
                    }

                    w->set_val(i, right_w_val);
                    {
                        Tensor *pred = this->forward(x, true);
                        right_cost = this->cost(pred, y);
                        delete pred;
                    }

                    float num_grad = (right_cost - left_cost) / (2.0f * EPSILON);

                    if (print_flg)
                    {
                        printf("W: %d  %d\t%f : %f  (%f)\n", lyr_idx, i, ana_grad, num_grad, fabs(ana_grad - num_grad));
                    }

                    agg_ana_grad += (ana_grad * ana_grad);
                    agg_num_grad += (num_grad * num_grad);
                    agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));

                    w->set_val(i, orig_w_val);
                }

                for (int i = 0; i < b->get_cnt(); i++)
                {
                    float left_cost = 0.0f;
                    float right_cost = 0.0f;

                    float orig_b_val = b->get_val(i);

                    float left_b_val = orig_b_val - EPSILON;
                    float right_b_val = orig_b_val + EPSILON;

                    float ana_grad = db->get_val(i);

                    b->set_val(i, left_b_val);
                    {
                        Tensor *pred = this->forward(x, true);
                        left_cost = this->cost(pred, y);
                        delete pred;
                    }

                    b->set_val(i, right_b_val);
                    {
                        Tensor *pred = this->forward(x, true);
                        right_cost = this->cost(pred, y);
                        delete pred;
                    }

                    float num_grad = (right_cost - left_cost) / (2.0f * EPSILON);

                    if (print_flg)
                    {
                        printf("B: %d  %d\t%f : %f  (%f)\n", lyr_idx, i, ana_grad, num_grad, fabs(ana_grad - num_grad));
                    }

                    agg_ana_grad += (ana_grad * ana_grad);
                    agg_num_grad += (num_grad * num_grad);
                    agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));

                    b->set_val(i, orig_b_val);
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

Report Model::train(Batch *batch)
{
    Report rpt;

    int batch_size = batch->get_size();

    rpt.correct_cnt = 0;
    rpt.total_cnt = batch_size;

    float cost = 0.0f;

    for (int i = 0; i < batch_size; i++)
    {
        Tensor *x = batch->get_x(i);
        Tensor *y = batch->get_y(i);

        Tensor *pred = this->forward(x, true);
        cost += this->cost(pred, y);
        this->backward(pred, y);

        rpt.update_correct_cnt(pred, y);

        delete pred;

        // Convert back to CPU as to not overload GPU.
        x->to(Device::Cpu);
        y->to(Device::Cpu);
    }

    // Get mean cost.
    cost /= batch_size;

    rpt.cost = cost;

    this->step(batch_size);

    return rpt;
}

Report Model::test(Batch *batch)
{
    Report rpt;

    int batch_size = batch->get_size();

    rpt.correct_cnt = 0;
    rpt.total_cnt = batch_size;

    float cost = 0.0f;

    for (int i = 0; i < batch_size; i++)
    {
        Tensor *x = batch->get_x(i);
        Tensor *y = batch->get_y(i);

        Tensor *pred = this->forward(x, false);
        cost += this->cost(pred, y);

        rpt.update_correct_cnt(pred, y);

        delete pred;

        // Convert back to CPU as to not overload GPU.
        x->to(Device::Cpu);
        y->to(Device::Cpu);
    }

    // Get mean cost.
    cost /= batch_size;

    rpt.cost = cost;

    return rpt;
}

// Trains and tests. Press 'q' to force quit.
void Model::fit(Supervisor *supervisor, int batch_size, int target_epoch, const char *csv_path)
{
    FILE *csv_file_ptr;

    if (csv_path != nullptr)
    {
        csv_file_ptr = fopen(csv_path, "w");
        CSVUtils::write_csv_header(csv_file_ptr);
    }

    unsigned long int epoch = 0;
    unsigned long int iteration = 0;

    while (true)
    {
        Batch *train_batch = supervisor->create_batch(batch_size);
        Report train_rpt = this->train(train_batch);

        if (csv_path != nullptr)
        {
            CSVUtils::write_to_csv(csv_file_ptr, epoch, iteration, train_rpt);
        }
        else
        {
            if (iteration % 100 == 0)
            {
                printf("TRAIN\t\t");
                train_rpt.print();
            }
        }

        delete train_batch;

        // Quit if we hit target epoch count.
        if (epoch >= target_epoch)
        {
            break;
        }

        // Allow for manual override.
        {
            if (_kbhit())
            {
                if (_getch() == 'q')
                {
                    printf("Quitting...\n");
                    break;
                }
            }
        }

        iteration++;
        epoch = ((iteration * batch_size) / supervisor->get_cnt());
    }

    if (csv_path != nullptr)
    {
        fclose(csv_file_ptr);
    }
}

Tensor *Model::predict(Tensor *x)
{
    return this->forward(x, false);
}