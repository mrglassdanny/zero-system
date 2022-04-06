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

Model::Model(CostFunction cost_fn)
{
    this->cost_fn = cost_fn;
    this->learning_rate = 0.001f;
}

Model::Model(float learning_rate)
{
    this->cost_fn = CostFunction::MSE;
    this->learning_rate = learning_rate;
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

    // Let caller handle children!
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

        Layer *lyr = NULL;

        switch (lyr_typ)
        {
        case LayerType::Dense:
            lyr = new DenseLayer();
            break;
        case LayerType::Convolutional:
            lyr = new ConvolutionalLayer();
            break;
        case LayerType::Embedding:
            lyr = new EmbeddingLayer();
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
        case LayerType::Custom:
            lyr = new CustomLayer();
            break;
        default:
            break;
        }

        lyr->load(file_ptr);
        this->add_layer(lyr);
    }

    int child_cnt = 0;
    fread(&child_cnt, sizeof(int), 1, file_ptr);

    for (int i = 0; i < child_cnt; i++)
    {
        Range child_range;
        fread(&child_range, sizeof(Range), 1, file_ptr);

        // Since children are saved to their own files, we will let the caller worry about loading them in accordance with ranges.
        this->child(NULL, child_range);
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

    int child_cnt = this->children.size();
    fwrite(&child_cnt, sizeof(int), 1, file_ptr);

    for (int child_idx = 0; child_idx < child_cnt; child_idx++)
    {
        Range child_range = this->child_ranges[child_idx];
        fwrite(&child_range, sizeof(Range), 1, file_ptr);

        // We will let the caller save children as they please!
    }
}

void Model::save(const char *path)
{
    FILE *file_ptr = fopen(path, "wb");

    Model::save(file_ptr);

    fclose(file_ptr);
}

void Model::copy(Model *src)
{
    this->cost_fn = src->cost_fn;
    this->learning_rate = src->learning_rate;

    for (Layer *src_lyr : src->layers)
    {
        Layer *lyr;

        switch (src_lyr->get_type())
        {
        case LayerType::Dense:
            lyr = new DenseLayer();
            lyr->copy((DenseLayer *)src_lyr);
            break;
        case LayerType::Convolutional:
            lyr = new ConvolutionalLayer();
            lyr->copy((ConvolutionalLayer *)src_lyr);
            break;
        case LayerType::Activation:
            lyr = new ActivationLayer();
            lyr->copy((ActivationLayer *)src_lyr);
            break;
        case LayerType::Dropout:
            lyr = new DropoutLayer();
            lyr->copy((DropoutLayer *)src_lyr);
            break;
        case LayerType::Pooling:
            lyr = new PoolingLayer();
            lyr->copy((PoolingLayer *)src_lyr);
            break;
        case LayerType::Custom:
            lyr = new CustomLayer();
            lyr->copy((CustomLayer *)src_lyr);
            break;
        default:
            break;
        }

        this->add_layer(lyr);
    }

    for (int i = 0; i < src->children.size(); i++)
    {
        Model *src_child = src->children[i];
        Range src_child_range = src->child_ranges[i];

        Model *child = new Model();
        child->copy(src_child);

        this->child(child, src_child_range);
    }
}

void Model::add_layer(Layer *lyr)
{
    this->layers.push_back(lyr);
}

void Model::add_child(Model *child, Range child_range)
{
    if (child != NULL)
    {
        child->set_learning_rate(this->learning_rate);
        this->children.push_back(child);
    }

    this->child_ranges.push_back(child_range);
}

void Model::dense(int nxt_n_cnt)
{
    this->dense(this->get_output_shape(), nxt_n_cnt, InitializationFunction::Xavier);
}

void Model::dense(int nxt_n_cnt, InitializationFunction init_fn)
{
    this->dense(this->get_output_shape(), nxt_n_cnt, init_fn);
}

void Model::dense(std::vector<int> n_shape, int nxt_n_cnt)
{
    this->dense(n_shape, nxt_n_cnt, InitializationFunction::Xavier);
}

void Model::dense(int n_cnt, int nxt_n_cnt)
{
    std::vector<int> n_shape{n_cnt};
    this->dense(n_shape, nxt_n_cnt, InitializationFunction::Xavier);
}

void Model::dense(int n_cnt, int nxt_n_cnt, InitializationFunction init_fn)
{
    std::vector<int> n_shape{n_cnt};
    this->dense(n_shape, nxt_n_cnt, init_fn);
}

void Model::dense(std::vector<int> n_shape, int nxt_n_cnt, InitializationFunction init_fn)
{
    this->add_layer(new DenseLayer(n_shape, nxt_n_cnt, init_fn));
}

void Model::convolutional(int fltr_cnt, int w_row_cnt, int w_col_cnt)
{
    this->convolutional(this->get_output_shape(), fltr_cnt, w_row_cnt, w_col_cnt, InitializationFunction::Xavier);
}

void Model::convolutional(int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn)
{
    this->convolutional(this->get_output_shape(), fltr_cnt, w_row_cnt, w_col_cnt, init_fn);
}

void Model::convolutional(std::vector<int> n_shape, int fltr_cnt, int w_row_cnt, int w_col_cnt)
{
    this->convolutional(n_shape, fltr_cnt, w_row_cnt, w_col_cnt, InitializationFunction::Xavier);
}

void Model::convolutional(std::vector<int> n_shape, int fltr_cnt, int w_row_cnt, int w_col_cnt, InitializationFunction init_fn)
{
    this->add_layer(new ConvolutionalLayer(n_shape, fltr_cnt, w_row_cnt, w_col_cnt, init_fn));
}

void Model::embedding(int embg_cnt, int embg_dim_cnt)
{
    this->embedding(embg_cnt, embg_dim_cnt, InitializationFunction::He);
}

void Model::embedding(int embg_cnt, int embg_dim_cnt, InitializationFunction init_fn)
{
    this->add_layer(new EmbeddingLayer(embg_cnt, embg_dim_cnt, init_fn));
}

void Model::activation(ActivationFunction activation_fn)
{
    this->activation(this->get_output_shape(), activation_fn);
}

void Model::activation(int n_cnt, ActivationFunction activation_fn)
{
    std::vector<int> n_shape{n_cnt};
    this->activation(n_shape, activation_fn);
}

void Model::activation(std::vector<int> n_shape, ActivationFunction activation_fn)
{
    this->add_layer(new ActivationLayer(n_shape, activation_fn));
}

void Model::dropout(float dropout_rate)
{
    this->add_layer(new DropoutLayer(this->get_output_shape(), dropout_rate));
}

void Model::pooling(PoolingFunction pool_fn)
{
    this->add_layer(new PoolingLayer(this->get_output_shape(), pool_fn));
}

void Model::custom(std::vector<int> (*get_output_shape_fn)(),
                   void (*forward_fn)(Tensor *n, Tensor *nxt_n, bool train_flg),
                   Tensor *(*backward_fn)(Tensor *n, Tensor *dc))
{
    this->custom(this->get_output_shape(), get_output_shape_fn, forward_fn, backward_fn);
}

void Model::custom(int n_cnt,
                   std::vector<int> (*get_output_shape_fn)(),
                   void (*forward_fn)(Tensor *n, Tensor *nxt_n, bool train_flg),
                   Tensor *(*backward_fn)(Tensor *n, Tensor *dc))
{
    std::vector<int> n_shape{n_cnt};
    this->custom(n_cnt, get_output_shape_fn, forward_fn, backward_fn);
}

void Model::custom(std::vector<int> n_shape,
                   std::vector<int> (*get_output_shape_fn)(),
                   void (*forward_fn)(Tensor *n, Tensor *nxt_n, bool train_flg),
                   Tensor *(*backward_fn)(Tensor *n, Tensor *dc))
{
    this->add_layer(new CustomLayer(n_shape, get_output_shape_fn, forward_fn, backward_fn));
}

void Model::child(Model *child)
{
    // We are assuming that caller is pushing child into the right spot given range positions.
    this->children.push_back(child);
}

void Model::child(Model *child, Range child_range)
{
    this->add_child(child, child_range);
}

std::vector<int> Model::get_input_shape()
{
    if (this->children.size() == 0)
    {
        return this->layers[0]->get_input_shape();
    }
    else
    {
        int n_cnt = Tensor::get_cnt(this->layers[0]->get_input_shape());

        for (Model *child : this->children)
        {
            n_cnt -= Tensor::get_cnt(child->get_output_shape());
            n_cnt += Tensor::get_cnt(child->get_input_shape());
        }

        std::vector<int> n_shape{n_cnt};
        return n_shape;
    }
}

std::vector<int> Model::get_output_shape()
{
    return this->layers[this->layers.size() - 1]->get_output_shape();
}

std::vector<int> Model::get_adjusted_input_shape()
{
    return this->layers[0]->get_input_shape();
}

std::vector<int> Model::calc_adjusted_input_shape(std::vector<int> n_shape)
{
    return this->calc_adjusted_input_shape(Tensor::get_cnt(n_shape));
}

std::vector<int> Model::calc_adjusted_input_shape(int n_cnt)
{
    int adj_n_cnt = n_cnt;

    for (Model *child : this->children)
    {
        adj_n_cnt += Tensor::get_cnt(child->get_output_shape());
        adj_n_cnt -= Tensor::get_cnt(child->get_input_shape());
    }

    std::vector<int> adj_n_shape{adj_n_cnt};
    return adj_n_shape;
}

std::vector<Layer *> Model::get_layers()
{
    return this->layers;
}

std::vector<Model *> Model::get_children()
{
    return this->children;
}

std::vector<Range> Model::get_child_ranges()
{
    return this->child_ranges;
}

void Model::set_learning_rate(float learning_rate)
{
    this->learning_rate = learning_rate;
}

void Model::share_parameters(Model *other_model)
{
    for (int lyr_idx = 0; lyr_idx < this->get_layers().size(); lyr_idx++)
    {
        Layer *lyr = this->get_layers()[lyr_idx];

        if (LearnableLayer *lrn_lyr = dynamic_cast<LearnableLayer *>(lyr))
        {
            LearnableLayer *other_lrn_lyr = dynamic_cast<LearnableLayer *>(other_model->get_layers()[lyr_idx]);

            lrn_lyr->set_weights(other_lrn_lyr->get_weights());
            lrn_lyr->set_weight_derivatives(other_lrn_lyr->get_weight_derivatives());
            lrn_lyr->set_biases(other_lrn_lyr->get_biases());
            lrn_lyr->set_bias_derivatives(other_lrn_lyr->get_bias_derivatives());
        }
    }
}

Tensor *Model::forward(Tensor *x, bool train_flg)
{
    x->to_device(Device::Cuda);

    int lst_lyr_idx = this->layers.size() - 1;

    Layer *frst_lyr = this->layers[0];
    Layer *lst_lyr = this->layers[lst_lyr_idx];

    if (this->children.size() > 0)
    {
        // We need to create an adjusted x tensor to match our updated shape and values due to children:
        Tensor *adj_x = new Tensor(x->get_device(), this->get_adjusted_input_shape());
        cudaMemcpy(adj_x->get_arr(), x->get_arr(), sizeof(float) * this->child_ranges[0].beg_idx, cudaMemcpyDefault);

        // First we need to shift all non-child inputs in accordance to new adjusted x layout:
        {
            int adj_x_offset = this->child_ranges[0].beg_idx;

            int lst_x_idx = x->get_cnt() - 1;

            for (int child_idx = 0; child_idx < this->children.size() - 1; child_idx++)
            {
                Model *child = this->children[child_idx];
                Model *nxt_child = this->children[child_idx + 1];

                Range child_range = this->child_ranges[child_idx];
                Range nxt_child_range = this->child_ranges[child_idx + 1];

                adj_x_offset += Tensor::get_cnt(child->get_output_shape());

                int non_child_range_len = ((nxt_child_range.beg_idx - 1) - child_range.end_idx);

                if (non_child_range_len > 0)
                {
                    cudaMemcpy(&adj_x->get_arr()[adj_x_offset], &x->get_arr()[child_range.end_idx + 1], sizeof(float) * non_child_range_len, cudaMemcpyDefault);
                    adj_x_offset += non_child_range_len;
                }
            }

            {
                Model *lst_child = this->children[this->children.size() - 1];
                Range lst_child_range = this->child_ranges[this->children.size() - 1];

                adj_x_offset += Tensor::get_cnt(lst_child->get_output_shape());

                int non_child_range_len = (lst_x_idx - lst_child_range.end_idx);

                if (non_child_range_len > 0 && adj_x_offset + non_child_range_len <= adj_x->get_cnt())
                {
                    cudaMemcpy(&adj_x->get_arr()[adj_x_offset], &x->get_arr()[lst_child_range.end_idx + 1], sizeof(float) * non_child_range_len, cudaMemcpyDefault);
                }
            }
        }

        // Now we can evaluate children and stick the predictions in their correct spots:
        {
            int child_output_shape_cnt = 0;

            int adj_x_offset = 0;

            for (int child_idx = 0; child_idx < this->children.size(); child_idx++)
            {
                Model *child = this->children[child_idx];
                Range child_range = this->child_ranges[child_idx];

                child_output_shape_cnt = Tensor::get_cnt(child->get_output_shape());

                Tensor *child_x = new Tensor(x->get_device(), child->get_input_shape());

                cudaMemcpy(child_x->get_arr(), &x->get_arr()[child_range.beg_idx], sizeof(float) * (child_range.end_idx - child_range.beg_idx + 1), cudaMemcpyDefault);

                Tensor *child_pred = child->forward(child_x, train_flg);

                cudaMemcpy(&adj_x->get_arr()[child_range.beg_idx + adj_x_offset], child_pred->get_arr(), sizeof(float) * child_output_shape_cnt, cudaMemcpyDefault);

                adj_x_offset += (child_output_shape_cnt - Tensor::get_cnt(child->get_input_shape()));

                delete child_x;
                delete child_pred;
            }
        }

        frst_lyr->set_neurons(adj_x);
        delete adj_x;
    }
    else
    {
        frst_lyr->set_neurons(x);
    }

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

    if (this->children.size() > 0)
    {
        int adj_x_offset = this->child_ranges[0].beg_idx;

        for (int child_idx = 0; child_idx < this->children.size() - 1; child_idx++)
        {
            Model *child = this->children[child_idx];
            Model *nxt_child = this->children[child_idx + 1];

            Range child_range = this->child_ranges[child_idx];
            Range nxt_child_range = this->child_ranges[child_idx + 1];

            Tensor *cpy_dc = new Tensor(*dc);

            delete child->child_backward(cpy_dc, adj_x_offset);

            int non_child_range_len = ((nxt_child_range.beg_idx - 1) - child_range.end_idx);
            adj_x_offset += (Tensor::get_cnt(child->get_output_shape()) + non_child_range_len);
        }

        {
            Model *lst_child = this->children[this->children.size() - 1];
            Range lst_child_range = this->child_ranges[this->children.size() - 1];

            Tensor *cpy_dc = new Tensor(*dc);

            delete lst_child->child_backward(cpy_dc, adj_x_offset);
        }
    }

    return dc;
}

Tensor *Model::child_backward(Tensor *dc, int adj_x_offset)
{
    int lst_lyr_idx = this->layers.size() - 1;

    // Need to adjust derivatives tensor since child only influences a handful of next layer neurons:
    {
        Tensor *child_dc = new Tensor(dc->get_device(), this->get_output_shape());

        cudaMemcpy(child_dc->get_arr(), &dc->get_arr()[adj_x_offset], sizeof(float) * (child_dc->get_cnt()), cudaMemcpyDefault);

        delete dc;
        dc = child_dc;
    }

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

    for (Model *child : this->children)
    {
        child->step(batch_size);
    }
}

void Model::grad_check(Tensor *x, Tensor *y, bool print_flg)
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

        // Children:
        {
            int child_idx = 0;
            for (Model *child : this->children)
            {
                child_idx++;
                child->child_grad_check(this, x, y, &agg_ana_grad, &agg_num_grad, &agg_grad_diff,
                                        child_idx, print_flg);
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

void Model::child_grad_check(Model *parent, Tensor *x, Tensor *y,
                             float *agg_ana_grad, float *agg_num_grad, float *agg_grad_diff,
                             int child_idx, bool print_flg)
{
    int child_lyr_idx = 0;
    for (Layer *child_lyr : this->get_layers())
    {
        child_lyr_idx++;

        if (LearnableLayer *child_lrn_lyr = dynamic_cast<LearnableLayer *>(child_lyr))
        {
            Tensor *w = child_lrn_lyr->get_weights();
            Tensor *dw = child_lrn_lyr->get_weight_derivatives();
            Tensor *b = child_lrn_lyr->get_biases();
            Tensor *db = child_lrn_lyr->get_bias_derivatives();

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
                    Tensor *pred = parent->forward(x, true);
                    left_cost = parent->cost(pred, y);
                    delete pred;
                }

                w->set_val(w_idx, right_w_val);
                {
                    Tensor *pred = parent->forward(x, true);
                    right_cost = parent->cost(pred, y);
                    delete pred;
                }

                float num_grad = (right_cost - left_cost) / (2.0f * EPSILON);

                if (print_flg)
                {
                    printf("E: %d W: %d  %d\t%f : %f  (%f)\n", child_idx, child_lyr_idx, w_idx, ana_grad, num_grad, fabs(ana_grad - num_grad));
                }

                *agg_ana_grad += (ana_grad * ana_grad);
                *agg_num_grad += (num_grad * num_grad);
                *agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));

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
                    Tensor *pred = parent->forward(x, true);
                    left_cost = parent->cost(pred, y);
                    delete pred;
                }

                b->set_val(b_idx, right_b_val);
                {
                    Tensor *pred = parent->forward(x, true);
                    right_cost = parent->cost(pred, y);
                    delete pred;
                }

                float num_grad = (right_cost - left_cost) / (2.0f * EPSILON);

                if (print_flg)
                {
                    printf("E: %d B: %d  %d\t%f : %f  (%f)\n", child_idx, child_lyr_idx, b_idx, ana_grad, num_grad, fabs(ana_grad - num_grad));
                }

                *agg_ana_grad += (ana_grad * ana_grad);
                *agg_num_grad += (num_grad * num_grad);
                *agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));

                b->set_val(b_idx, orig_b_val);
            }
        }
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

    if (csv_path != NULL)
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

        if (csv_path != NULL)
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

    if (csv_path != NULL)
    {
        fclose(csv_file_ptr);
    }
}

Tensor *Model::predict(Tensor *x)
{
    return this->forward(x, false);
}

// Model static functions: