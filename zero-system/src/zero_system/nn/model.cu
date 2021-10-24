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

__device__ float d_cross_entropy_cost(float n_val, float y_val)
{
    return (float)((y_val * log(n_val)) + ((1.0 - y_val) * log(1.0 - n_val)));
}

__device__ float d_derive_cross_entropy_cost(float n_val, float y_val)
{
    return (n_val - y_val);
}

// Kernel functions:

__global__ void k_cost(float *n_arr, float *y_arr, float *cost, int n_cnt, CostFunction cost_fn)
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
        for (int i = 0; i < CUDA_THREADS_PER_BLOCK; i++)
        {
            sum += temp[i];
        }

        atomicAdd(cost, sum);
    }
}

__global__ void k_derive_cost(float *n_arr, float *y_arr, float *agg_derivatives_arr, int n_cnt, CostFunction cost_fn)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_cnt)
    {
        switch (cost_fn)
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

// Model functions:

Model::Model(CostFunction cost_fn, float learning_rate)
{
    this->cost_fn = cost_fn;
    this->learning_rate = learning_rate;

    cudaMalloc(&this->d_cost, sizeof(float));
    cudaMemset(this->d_cost, 0, sizeof(float));
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
        default:
            break;
        }

        this->add_layer(lyr);
    }

    cudaMalloc(&this->d_cost, sizeof(float));
    cudaMemset(this->d_cost, 0, sizeof(float));

    fclose(file_ptr);
}

Model::~Model()
{
    for (Layer *lyr : this->layers)
    {
        delete lyr;
    }

    cudaFree(this->d_cost);
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

std::vector<int> Model::get_input_shape()
{
    return this->layers[0]->get_input_shape();
}

std::vector<int> Model::get_output_shape()
{
    return this->layers[this->layers.size() - 1]->get_output_shape();
}

Tensor *Model::forward(Tensor *x, bool train_flg)
{
    int lst_lyr_idx = this->layers.size() - 1;

    Layer *frst_lyr = this->layers[0];
    Layer *lst_lyr = this->layers[lst_lyr_idx];

    frst_lyr->n->set_arr(x->get_arr());

    for (int i = 0; i < lst_lyr_idx; i++)
    {
        Layer *lyr = this->layers[i];
        Layer *nxt_lyr = this->layers[i + 1];

        lyr->evaluate(nxt_lyr->n, train_flg);
    }

    Tensor *pred = new Tensor(Device::Cuda, lst_lyr->n->get_shape());
    lst_lyr->evaluate(pred, train_flg);

    return pred;
}

float Model::cost(Tensor *pred, Tensor *y)
{
    float h_cost = 0.0f;

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (pred->get_cnt() / threads_per_block) + 1;

        k_cost<<<num_blocks, threads_per_block>>>(pred->get_arr(), y->get_arr(),
                                                  this->d_cost, pred->get_cnt(), this->cost_fn);
    }

    cudaMemcpy(&h_cost, this->d_cost, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemset(this->d_cost, 0, sizeof(float));

    return h_cost;
}

void Model::backward(Tensor *pred, Tensor *y)
{
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
        dc = lyr->derive(dc);
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
    float agg_ana_grad = 0.0f;
    float agg_num_grad = 0.0f;
    float agg_grad_diff = 0.0f;

    float epsilon = 0.001f;

    // Analytical gradients:
    {
        Tensor *pred = this->forward(x, true);
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
                for (int i = 0; i < lrn_lyr->w->get_cnt(); i++)
                {
                    float left_cost = 0.0f;
                    float right_cost = 0.0f;

                    float orig_w_val = lrn_lyr->w->get_val(i);

                    float left_w_val = orig_w_val - epsilon;
                    float right_w_val = orig_w_val + epsilon;

                    float ana_grad = lrn_lyr->dw->get_val(i);

                    lrn_lyr->w->set_val(i, left_w_val);
                    {
                        Tensor *pred = this->forward(x, true);
                        left_cost = this->cost(pred, y);
                        delete pred;
                    }

                    lrn_lyr->w->set_val(i, right_w_val);
                    {
                        Tensor *pred = this->forward(x, true);
                        right_cost = this->cost(pred, y);
                        delete pred;
                    }

                    float num_grad = (right_cost - left_cost) / (2.0f * epsilon);

                    if (print_flg)
                    {
                        printf("W: %d  %d\t%f : %f  (%f)\n", lyr_idx, i, ana_grad, num_grad, fabs(ana_grad - num_grad));
                    }

                    agg_ana_grad += (ana_grad * ana_grad);
                    agg_num_grad += (num_grad * num_grad);
                    agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));

                    lrn_lyr->w->set_val(i, orig_w_val);
                }

                for (int i = 0; i < lrn_lyr->b->get_cnt(); i++)
                {
                    float left_cost = 0.0f;
                    float right_cost = 0.0f;

                    float orig_b_val = lrn_lyr->b->get_val(i);

                    float left_b_val = orig_b_val - epsilon;
                    float right_b_val = orig_b_val + epsilon;

                    float ana_grad = lrn_lyr->db->get_val(i);

                    lrn_lyr->b->set_val(i, left_b_val);
                    {
                        Tensor *pred = this->forward(x, true);
                        left_cost = this->cost(pred, y);
                        delete pred;
                    }

                    lrn_lyr->b->set_val(i, right_b_val);
                    {
                        Tensor *pred = this->forward(x, true);
                        right_cost = this->cost(pred, y);
                        delete pred;
                    }

                    float num_grad = (right_cost - left_cost) / (2.0f * epsilon);

                    if (print_flg)
                    {
                        printf("B: %d  %d\t%f : %f  (%f)\n", lyr_idx, i, ana_grad, num_grad, fabs(ana_grad - num_grad));
                    }

                    agg_ana_grad += (ana_grad * ana_grad);
                    agg_num_grad += (num_grad * num_grad);
                    agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));

                    lrn_lyr->b->set_val(i, orig_b_val);
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

        // Convert to Cuda for model.
        x->to(Device::Cuda);
        y->to(Device::Cuda);

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

        // Convert to Cuda for model.
        x->to(Device::Cuda);
        y->to(Device::Cuda);

        Tensor *pred = this->forward(x, true);
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
void Model::train_and_test(Supervisor *supervisor, int train_batch_size, const char *csv_path)
{
    FILE *csv_file_ptr;

    if (csv_path != nullptr)
    {
        csv_file_ptr = fopen(csv_path, "w");
        CSVUtils::write_csv_header(csv_file_ptr);
    }

    Batch *test_batch = supervisor->create_test_batch();

    unsigned long int epoch = 1;
    while (true)
    {
        Batch *train_batch = supervisor->create_train_batch(train_batch_size);
        Report train_rpt = this->train(train_batch);

        if (csv_path != nullptr)
        {
            CSVUtils::write_to_csv(csv_file_ptr, epoch, train_rpt);
        }
        else
        {
            if (epoch % 100 == 0)
            {
                printf("TRAIN\t\t");
                train_rpt.print();
            }
        }

        delete train_batch;

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
    printf("TEST\t\t");
    test_rpt.print();

    delete test_batch;

    if (csv_path != nullptr)
    {
        fclose(csv_file_ptr);
    }
}

// Trains, validates, and tests. Press 'q' to force quit.
void Model::all(Supervisor *supervisor, int train_batch_size, int validation_chk_freq, const char *csv_path)
{
    FILE *csv_file_ptr;

    if (csv_path != nullptr)
    {
        csv_file_ptr = fopen(csv_path, "w");
        CSVUtils::write_csv_header(csv_file_ptr);
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
            CSVUtils::write_to_csv(csv_file_ptr, epoch, train_rpt);
        }

        delete train_batch;

        // Validate every x epochs.
        if (epoch % validation_chk_freq == 0)
        {
            Report validation_rpt = this->test(validation_batch);
            printf("VALIDATION\t");
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
    printf("TEST\t\t");
    test_rpt.print();

    delete validation_batch;
    delete test_batch;

    if (csv_path != nullptr)
    {
        fclose(csv_file_ptr);
    }
}

Tensor *Model::predict(Tensor *x)
{
    return this->forward(x, false);
}