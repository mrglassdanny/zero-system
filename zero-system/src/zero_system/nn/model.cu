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

Model::Model()
{
    this->cost_fn = CostFunction::MSE;
    this->learning_rate = 0.001f;
}

Model::Model(CostFunction cost_fn, float learning_rate)
{
    this->cost_fn = cost_fn;
    this->learning_rate = learning_rate;
}

Model::~Model()
{
    for (Layer *lyr : this->layers)
    {
        delete lyr;
    }
}

void Model::load(FILE *file_ptr)
{
    fread(&this->cost_fn, sizeof(CostFunction), 1, file_ptr);
    fread(&this->learning_rate, sizeof(float), 1, file_ptr);

    int lyr_cnt = 0;
    fread(&lyr_cnt, sizeof(int), 1, file_ptr);

    for (int lyr_idx = 0; lyr_idx < lyr_cnt; lyr_idx++)
    {
        LayerType lyr_typ;
        fread(&lyr_typ, sizeof(LayerType), 1, file_ptr);

        Layer *lyr = nullptr;

        switch (lyr_typ)
        {
        case LayerType::Linear:
            lyr = new LinearLayer();
            break;
        case LayerType::Convolutional:
            lyr = new ConvolutionalLayer();
            break;
        case LayerType::Activation:
            lyr = new ActivationLayer();
            break;
        case LayerType::Dropout:
            lyr = new DropoutLayer();
            break;
        case LayerType::Pooling:
            lyr = new PoolingLayer();
            break;
        default:
            break;
        }

        lyr->load(file_ptr);

        this->add_layer(lyr);
    }
}

void Model::load(const char *path)
{
    FILE *file_ptr = fopen(path, "rb");

    Model::load(file_ptr);

    fclose(file_ptr);
}

void Model::save(FILE *file_ptr)
{
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
}

void Model::save(const char *path)
{
    FILE *file_ptr = fopen(path, "wb");

    Model::save(file_ptr);

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

void Model::linear(int n_cnt, int nxt_n_cnt)
{
    std::vector<int> n_shape{n_cnt};
    this->linear(n_shape, nxt_n_cnt, InitializationFunction::Xavier);
}

void Model::linear(int n_cnt, int nxt_n_cnt, InitializationFunction init_fn)
{
    std::vector<int> n_shape{n_cnt};
    this->linear(n_shape, nxt_n_cnt, init_fn);
}

void Model::linear(std::vector<int> n_shape, int nxt_n_cnt, InitializationFunction init_fn)
{
    this->add_layer(new LinearLayer(n_shape, nxt_n_cnt, init_fn));
}

void Model::activation(ActivationFunction activation_fn)
{
    this->add_layer(new ActivationLayer(this->get_output_shape(), activation_fn));
}

void Model::dropout(float dropout_rate)
{
    this->add_layer(new DropoutLayer(this->get_output_shape(), dropout_rate));
}

void Model::aggregation(AggregationFunction agg_fn, int grp_cnt)
{
    this->aggregation(this->get_output_shape(), agg_fn, grp_cnt);
}

void Model::aggregation(int n_cnt, AggregationFunction agg_fn, int grp_cnt)
{
    std::vector<int> n_shape{n_cnt};
    this->aggregation(n_shape, agg_fn, grp_cnt);
}

void Model::aggregation(std::vector<int> n_shape, AggregationFunction agg_fn, int grp_cnt)
{
    this->add_layer(new AggregationLayer(n_shape, agg_fn, grp_cnt));
}

std::vector<int> Model::get_input_shape()
{
    return this->layers[0]->get_input_shape();
}

std::vector<int> Model::get_output_shape()
{
    return this->layers[this->layers.size() - 1]->get_output_shape();
}

std::vector<Layer *> Model::get_layers()
{
    return this->layers;
}

void Model::set_learning_rate(float learning_rate)
{
    this->learning_rate = learning_rate;
}

Tensor *Model::forward(Tensor *x, bool train_flg)
{
    x->to_device(Device::Cuda);

    int lst_lyr_idx = this->layers.size() - 1;

    Layer *frst_lyr = this->layers[0];
    Layer *lst_lyr = this->layers[lst_lyr_idx];

    frst_lyr->set_neurons(x);

    for (int lyr_idx = 0; lyr_idx < lst_lyr_idx; lyr_idx++)
    {
        Layer *lyr = this->layers[lyr_idx];
        Layer *nxt_lyr = this->layers[lyr_idx + 1];

        lyr->forward(nxt_lyr->get_neurons(), train_flg);
    }

    Tensor *pred = new Tensor(Device::Cuda, lst_lyr->get_output_shape());
    lst_lyr->forward(pred, train_flg);

    return pred;
}

float Model::cost(Tensor *pred, Tensor *y)
{
    y->to_device(Device::Cuda);

    float *d_cost_val;
    float h_cost_val = 0.0f;

    cudaMalloc(&d_cost_val, sizeof(float));
    cudaMemset(d_cost_val, 0, sizeof(float));

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (pred->get_cnt() / threads_per_block) + 1;
        k_cost<<<num_blocks, threads_per_block>>>(pred->get_arr(), y->get_arr(),
                                                  d_cost_val, pred->get_cnt(), this->cost_fn);
    }

    cudaMemcpy(&h_cost_val, d_cost_val, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_cost_val);

    return h_cost_val;
}

Tensor *Model::backward(Tensor *pred, Tensor *y)
{
    y->to_device(Device::Cuda);

    Tensor *dc = new Tensor(Device::Cuda, pred->get_shape());
    dc->set_all(1.0f);

    {
        int threads_per_block = CUDA_THREADS_PER_BLOCK;
        int num_blocks = (pred->get_cnt() / threads_per_block) + 1;
        k_derive_cost<<<num_blocks, threads_per_block>>>(pred->get_arr(),
                                                         y->get_arr(), dc->get_arr(), pred->get_cnt(), this->cost_fn);
    }

    int lst_lyr_idx = this->layers.size() - 1;

    for (int lyr_idx = lst_lyr_idx; lyr_idx >= 0; lyr_idx--)
    {
        Layer *lyr = this->layers[lyr_idx];
        dc = lyr->backward(dc);
    }

    return dc;
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

void Model::check_grad(Tensor *x, Tensor *y, bool print_flg)
{
    x->to_device(Device::Cuda);
    y->to_device(Device::Cuda);

    float agg_ana_grad = 0.0f;
    float agg_num_grad = 0.0f;
    float agg_grad_diff = 0.0f;

    // Analytical gradients:
    {
        Tensor *pred = this->forward(x, true);
        this->cost(pred, y);
        delete this->backward(pred, y);
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

                for (int w_idx = 0; w_idx < w->get_cnt(); w_idx++)
                {
                    float left_cost = 0.0f;
                    float right_cost = 0.0f;

                    float orig_w_val = w->get_val(w_idx);

                    float left_w_val = orig_w_val - EPSILON;
                    float right_w_val = orig_w_val + EPSILON;

                    float ana_grad = dw->get_val(w_idx);

                    w->set_val(w_idx, left_w_val);
                    {
                        Tensor *pred = this->forward(x, true);
                        left_cost = this->cost(pred, y);
                        delete pred;
                    }

                    w->set_val(w_idx, right_w_val);
                    {
                        Tensor *pred = this->forward(x, true);
                        right_cost = this->cost(pred, y);
                        delete pred;
                    }

                    float num_grad = (right_cost - left_cost) / (2.0f * EPSILON);

                    if (print_flg)
                    {
                        printf("W: %d  %d\t%f : %f  (%f)\n", lyr_idx, w_idx, ana_grad, num_grad, fabs(ana_grad - num_grad));
                    }

                    agg_ana_grad += (ana_grad * ana_grad);
                    agg_num_grad += (num_grad * num_grad);
                    agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));

                    w->set_val(w_idx, orig_w_val);
                }

                for (int b_idx = 0; b_idx < b->get_cnt(); b_idx++)
                {
                    float left_cost = 0.0f;
                    float right_cost = 0.0f;

                    float orig_b_val = b->get_val(b_idx);

                    float left_b_val = orig_b_val - EPSILON;
                    float right_b_val = orig_b_val + EPSILON;

                    float ana_grad = db->get_val(b_idx);

                    b->set_val(b_idx, left_b_val);
                    {
                        Tensor *pred = this->forward(x, true);
                        left_cost = this->cost(pred, y);
                        delete pred;
                    }

                    b->set_val(b_idx, right_b_val);
                    {
                        Tensor *pred = this->forward(x, true);
                        right_cost = this->cost(pred, y);
                        delete pred;
                    }

                    float num_grad = (right_cost - left_cost) / (2.0f * EPSILON);

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

Report Model::train(Batch *batch, UpdateResultFn fn)
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
        delete this->backward(pred, y);

        rpt.update(pred, y, fn);

        delete pred;

        // Convert back to CPU as to not overload GPU.
        x->to_device(Device::Cpu);
        y->to_device(Device::Cpu);
    }

    // Get mean cost.
    cost /= batch_size;

    rpt.cost = cost;

    this->step(batch_size);

    return rpt;
}

Report Model::test(Batch *batch, UpdateResultFn fn)
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

        rpt.update(pred, y, fn);

        delete pred;

        // Convert back to CPU as to not overload GPU.
        x->to_device(Device::Cpu);
        y->to_device(Device::Cpu);
    }

    // Get mean cost.
    cost /= batch_size;

    rpt.cost = cost;

    return rpt;
}

void Model::fit(Batch *batch, UpdateResultFn fn)
{
    unsigned long int epoch = 0;
    int batch_size = batch->get_size();

    while (true)
    {
        Report train_rpt = this->train(batch, fn);

        printf("EPOCH: %d\t", epoch);
        train_rpt.print();

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

        epoch++;
    }
}

void Model::fit(Supervisor *supervisor, int batch_size, int target_epoch, const char *csv_path, UpdateResultFn fn)
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
        Report train_rpt = this->train(train_batch, fn);

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

// ConvNet functions:

ConvNet::ConvNet()
    : Model()
{
}

ConvNet::ConvNet(CostFunction cost_fn, float learning_rate)
    : Model(cost_fn, learning_rate)
{
}

ConvNet::~ConvNet()
{
}

void ConvNet::convolutional(int fltr_cnt, int w_row_cnt, int w_col_cnt)
{
    this->convolutional(this->get_output_shape(), fltr_cnt, w_row_cnt, w_col_cnt, InitializationFunction::Xavier);
}

void ConvNet::convolutional(int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn)
{
    this->convolutional(this->get_output_shape(), fltr_cnt, w_row_cnt, w_col_cnt, init_fn);
}

void ConvNet::convolutional(std::vector<int> n_shape, int fltr_cnt, int w_row_cnt, int w_col_cnt)
{
    this->convolutional(n_shape, fltr_cnt, w_row_cnt, w_col_cnt, InitializationFunction::Xavier);
}

void ConvNet::convolutional(std::vector<int> n_shape, int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn)
{
    this->add_layer(new ConvolutionalLayer(n_shape, fltr_cnt, w_row_cnt, w_col_cnt, init_fn));
}

void ConvNet::pooling(PoolingFunction pool_fn)
{
    this->add_layer(new PoolingLayer(this->get_output_shape(), pool_fn));
}

// Embedding functions:

Embedding::Embedding()
    : Model()
{
    this->cost_fn = CostFunction::None;
}

Embedding::Embedding(CostFunction cost_fn, float learning_rate)
    : Model(cost_fn, learning_rate)
{
    this->cost_fn = CostFunction::None;
}

Embedding::~Embedding()
{
}

Tensor *Embedding::embedding_backward(Tensor *dc, int embd_x_offset)
{
    int lst_lyr_idx = this->layers.size() - 1;

    // Need to adjust derivatives tensor since embedding only influences a handful of next layer neurons:
    {
        Tensor *embg_dc = new Tensor(dc->get_device(), this->get_output_shape());

        cudaMemcpy(embg_dc->get_arr(), &dc->get_arr()[embd_x_offset], sizeof(float) * (embg_dc->get_cnt()), cudaMemcpyDefault);

        delete dc;
        dc = embg_dc;
    }

    for (int lyr_idx = lst_lyr_idx; lyr_idx >= 0; lyr_idx--)
    {
        Layer *lyr = this->layers[lyr_idx];
        dc = lyr->backward(dc);
    }

    return dc;
}

// EmbeddableModel functions:

EmbeddedModel::EmbeddedModel()
    : Model()
{
}

EmbeddedModel::EmbeddedModel(CostFunction cost_fn, float learning_rate)
    : Model(cost_fn, learning_rate)
{
}

EmbeddedModel::~EmbeddedModel()
{
    // Let caller handle Embedding cleanup!
}

void EmbeddedModel::load(FILE *file_ptr)
{
    Model::load(file_ptr);

    int embg_cnt = 0;

    fread(&embg_cnt, sizeof(int), 1, file_ptr);

    for (int i = 0; i < embg_cnt; i++)
    {
        Range embg_range;
        fread(&embg_range, sizeof(Range), 1, file_ptr);

        // Since Embeddings are saved to their own files, we will let the caller worry about loading them in accordance with ranges.
        this->embed(nullptr, embg_range);
    }
}

void EmbeddedModel::load(const char *path)
{
    FILE *file_ptr = fopen(path, "rb");

    EmbeddedModel::load(file_ptr);

    fclose(file_ptr);
}

void EmbeddedModel::save(FILE *file_ptr)
{
    Model::save(file_ptr);

    int embg_cnt = this->embgs.size();

    fwrite(&embg_cnt, sizeof(int), 1, file_ptr);

    for (int embg_idx = 0; embg_idx < embg_cnt; embg_idx++)
    {
        Range embg_range = this->embg_ranges[embg_idx];
        fwrite(&embg_range, sizeof(Range), 1, file_ptr);

        // We will let the caller save Embeddings as they please!
    }
}

void EmbeddedModel::save(const char *path)
{
    FILE *file_ptr = fopen(path, "wb");

    EmbeddedModel::save(file_ptr);

    fclose(file_ptr);
}

std::vector<int> EmbeddedModel::calc_embedded_input_shape(std::vector<int> n_shape)
{
    return this->calc_embedded_input_shape(Tensor::get_cnt(n_shape));
}

std::vector<int> EmbeddedModel::calc_embedded_input_shape(int n_cnt)
{
    int embd_n_cnt = n_cnt;

    for (Embedding *embg : this->embgs)
    {
        embd_n_cnt += Tensor::get_cnt(embg->get_output_shape());

        // Make sure we subtract old dims.
        embd_n_cnt -= Tensor::get_cnt(embg->get_input_shape());
    }

    std::vector<int> embd_n_shape{embd_n_cnt};
    return embd_n_shape;
}

void EmbeddedModel::add_embedding(Embedding *embg, Range embg_range)
{
    if (embg != nullptr)
    {
        embg->set_learning_rate(this->learning_rate);
        this->embgs.push_back(embg);
    }

    this->embg_ranges.push_back(embg_range);
}

void EmbeddedModel::embed(Embedding *embg)
{
    // We are assuming that caller is pushing Embedding into the right spot given range positions.
    this->embgs.push_back(embg);
}

void EmbeddedModel::embed(Embedding *embg, Range embg_range)
{
    this->add_embedding(embg, embg_range);
}

Tensor *EmbeddedModel::forward(Tensor *x, bool train_flg)
{
    x->to_device(Device::Cuda);

    // We need to create an embedded x tensor to match our updated shape and values due to embeddings:
    Tensor *embd_x = new Tensor(x->get_device(), this->get_input_shape());
    cudaMemcpy(embd_x->get_arr(), x->get_arr(), sizeof(float) * x->get_cnt(), cudaMemcpyDefault);

    if (this->embgs.size() > 0)
    {
        // First we need to shift all non-embedding inputs in accordance to new embedded x layout:
        {
            int embd_x_offset = this->embg_ranges[0].beg_idx;

            int lst_x_idx = x->get_cnt() - 1;

            for (int embg_idx = 0; embg_idx < this->embgs.size() - 1; embg_idx++)
            {
                Embedding *embg = this->embgs[embg_idx];
                Embedding *nxt_embg = this->embgs[embg_idx + 1];

                Range embg_range = this->embg_ranges[embg_idx];
                Range nxt_embg_range = this->embg_ranges[embg_idx + 1];

                embd_x_offset += Tensor::get_cnt(embg->get_output_shape());

                int non_embg_range_len = ((nxt_embg_range.beg_idx - 1) - embg_range.end_idx);

                if (non_embg_range_len > 0)
                {
                    cudaMemcpy(&embd_x->get_arr()[embd_x_offset], &x->get_arr()[embg_range.beg_idx + 1], sizeof(float) * non_embg_range_len, cudaMemcpyDefault);
                    embd_x_offset += non_embg_range_len;
                }
            }

            Embedding *lst_embg = this->embgs[this->embgs.size() - 1];
            Range lst_embg_range = this->embg_ranges[this->embgs.size() - 1];

            embd_x_offset += Tensor::get_cnt(lst_embg->get_output_shape());

            int non_embg_range_len = (lst_x_idx - lst_embg_range.end_idx);

            if (non_embg_range_len > 0)
            {
                cudaMemcpy(&embd_x->get_arr()[embd_x_offset], &x->get_arr()[lst_embg_range.end_idx + 1], sizeof(float) * non_embg_range_len, cudaMemcpyDefault);
            }
        }

        // Now we can evaluate embeddings and stick the predictions in their correct spots:
        {
            int embg_output_shape_cnt = 0;

            int embd_x_offset = 0;

            for (int embg_idx = 0; embg_idx < this->embgs.size(); embg_idx++)
            {
                Embedding *embg = this->embgs[embg_idx];
                Range embg_range = this->embg_ranges[embg_idx];

                embg_output_shape_cnt = Tensor::get_cnt(embg->get_output_shape());

                Tensor *embg_x = new Tensor(x->get_device(), embg->get_input_shape());

                cudaMemcpy(embg_x->get_arr(), &x->get_arr()[embg_range.beg_idx], sizeof(float) * (embg_range.end_idx - embg_range.beg_idx), cudaMemcpyDefault);

                Tensor *embg_pred = embg->forward(embg_x, train_flg);

                cudaMemcpy(&embd_x->get_arr()[embg_range.beg_idx + embd_x_offset], embg_pred->get_arr(), sizeof(float) * embg_output_shape_cnt, cudaMemcpyDefault);

                embd_x_offset += (embg_output_shape_cnt - Tensor::get_cnt(embg->get_input_shape()));

                delete embg_x;
                delete embg_pred;
            }
        }
    }

    Tensor *pred = Model::forward(embd_x, train_flg);
    delete embd_x;
    return pred;
}

Tensor *EmbeddedModel::backward(Tensor *pred, Tensor *y)
{
    Tensor *dc = Model::backward(pred, y);

    int embd_x_offset = 0;

    for (int embg_idx = 0; embg_idx < this->embgs.size(); embg_idx++)
    {
        Embedding *embg = this->embgs[embg_idx];
        Range embg_range = this->embg_ranges[embg_idx];

        Tensor *cpy_dc = new Tensor(*dc);

        delete embg->embedding_backward(cpy_dc, embg_range.beg_idx + embd_x_offset);

        embd_x_offset += (Tensor::get_cnt(embg->get_output_shape()) - Tensor::get_cnt(embg->get_input_shape()));
    }

    return dc;
}

void EmbeddedModel::step(int batch_size)
{
    for (Embedding *embg : this->embgs)
    {
        embg->step(batch_size);
    }

    Model::step(batch_size);
}

void EmbeddedModel::check_grad(Tensor *x, Tensor *y, bool print_flg)
{
    x->to_device(Device::Cuda);
    y->to_device(Device::Cuda);

    float agg_ana_grad = 0.0f;
    float agg_num_grad = 0.0f;
    float agg_grad_diff = 0.0f;

    // Analytical gradients:
    {
        Tensor *pred = this->forward(x, true);
        this->cost(pred, y);
        delete this->backward(pred, y);
        delete pred;
    }

    // Numerical gradients:
    {

        // Embeddings:
        {
            int embg_idx = 0;
            for (Embedding *embg : this->embgs)
            {
                embg_idx++;

                int embg_lyr_idx = 0;
                for (Layer *embg_lyr : embg->get_layers())
                {
                    embg_lyr_idx++;

                    if (LearnableLayer *embg_lrn_lyr = dynamic_cast<LearnableLayer *>(embg_lyr))
                    {
                        Tensor *w = embg_lrn_lyr->get_weights();
                        Tensor *dw = embg_lrn_lyr->get_weight_derivatives();
                        Tensor *b = embg_lrn_lyr->get_biases();
                        Tensor *db = embg_lrn_lyr->get_bias_derivatives();

                        for (int w_idx = 0; w_idx < w->get_cnt(); w_idx++)
                        {
                            float left_cost = 0.0f;
                            float right_cost = 0.0f;

                            float orig_w_val = w->get_val(w_idx);

                            float left_w_val = orig_w_val - EPSILON;
                            float right_w_val = orig_w_val + EPSILON;

                            float ana_grad = dw->get_val(w_idx);

                            w->set_val(w_idx, left_w_val);
                            {
                                Tensor *pred = this->forward(x, true);
                                left_cost = this->cost(pred, y);
                                delete pred;
                            }

                            w->set_val(w_idx, right_w_val);
                            {
                                Tensor *pred = this->forward(x, true);
                                right_cost = this->cost(pred, y);
                                delete pred;
                            }

                            float num_grad = (right_cost - left_cost) / (2.0f * EPSILON);

                            if (print_flg)
                            {
                                printf("E: %d W: %d  %d\t%f : %f  (%f)\n", embg_idx, embg_lyr_idx, w_idx, ana_grad, num_grad, fabs(ana_grad - num_grad));
                            }

                            agg_ana_grad += (ana_grad * ana_grad);
                            agg_num_grad += (num_grad * num_grad);
                            agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));

                            w->set_val(w_idx, orig_w_val);
                        }

                        for (int b_idx = 0; b_idx < b->get_cnt(); b_idx++)
                        {
                            float left_cost = 0.0f;
                            float right_cost = 0.0f;

                            float orig_b_val = b->get_val(b_idx);

                            float left_b_val = orig_b_val - EPSILON;
                            float right_b_val = orig_b_val + EPSILON;

                            float ana_grad = db->get_val(b_idx);

                            b->set_val(b_idx, left_b_val);
                            {
                                Tensor *pred = this->forward(x, true);
                                left_cost = this->cost(pred, y);
                                delete pred;
                            }

                            b->set_val(b_idx, right_b_val);
                            {
                                Tensor *pred = this->forward(x, true);
                                right_cost = this->cost(pred, y);
                                delete pred;
                            }

                            float num_grad = (right_cost - left_cost) / (2.0f * EPSILON);

                            if (print_flg)
                            {
                                printf("E: %d B: %d  %d\t%f : %f  (%f)\n", embg_idx, embg_lyr_idx, b_idx, ana_grad, num_grad, fabs(ana_grad - num_grad));
                            }

                            agg_ana_grad += (ana_grad * ana_grad);
                            agg_num_grad += (num_grad * num_grad);
                            agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));

                            b->set_val(b_idx, orig_b_val);
                        }
                    }
                }
            }
        }

        // Layers:
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

                    for (int w_idx = 0; w_idx < w->get_cnt(); w_idx++)
                    {
                        float left_cost = 0.0f;
                        float right_cost = 0.0f;

                        float orig_w_val = w->get_val(w_idx);

                        float left_w_val = orig_w_val - EPSILON;
                        float right_w_val = orig_w_val + EPSILON;

                        float ana_grad = dw->get_val(w_idx);

                        w->set_val(w_idx, left_w_val);
                        {
                            Tensor *pred = this->forward(x, true);
                            left_cost = this->cost(pred, y);
                            delete pred;
                        }

                        w->set_val(w_idx, right_w_val);
                        {
                            Tensor *pred = this->forward(x, true);
                            right_cost = this->cost(pred, y);
                            delete pred;
                        }

                        float num_grad = (right_cost - left_cost) / (2.0f * EPSILON);

                        if (print_flg)
                        {
                            printf("W: %d  %d\t%f : %f  (%f)\n", lyr_idx, w_idx, ana_grad, num_grad, fabs(ana_grad - num_grad));
                        }

                        agg_ana_grad += (ana_grad * ana_grad);
                        agg_num_grad += (num_grad * num_grad);
                        agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));

                        w->set_val(w_idx, orig_w_val);
                    }

                    for (int b_idx = 0; b_idx < b->get_cnt(); b_idx++)
                    {
                        float left_cost = 0.0f;
                        float right_cost = 0.0f;

                        float orig_b_val = b->get_val(b_idx);

                        float left_b_val = orig_b_val - EPSILON;
                        float right_b_val = orig_b_val + EPSILON;

                        float ana_grad = db->get_val(b_idx);

                        b->set_val(b_idx, left_b_val);
                        {
                            Tensor *pred = this->forward(x, true);
                            left_cost = this->cost(pred, y);
                            delete pred;
                        }

                        b->set_val(b_idx, right_b_val);
                        {
                            Tensor *pred = this->forward(x, true);
                            right_cost = this->cost(pred, y);
                            delete pred;
                        }

                        float num_grad = (right_cost - left_cost) / (2.0f * EPSILON);

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
