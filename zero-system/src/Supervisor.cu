#include "Supervisor.cuh"

Supervisor::Supervisor()
{
}

Supervisor::Supervisor(int row_cnt, int col_cnt, int output_col_cnt, float *x_arr, float *y_arr, TensorType typ)
{
    this->add_all(row_cnt, col_cnt, output_col_cnt, x_arr, y_arr, typ);
}

Supervisor::~Supervisor()
{
    this->clear();
}

void Supervisor::add(int col_cnt, int output_col_cnt, float *x_arr, float y_val, TensorType typ)
{
    Tensor *x = new Tensor(1, col_cnt, typ, x_arr);

    Tensor *y;
    // Single value or one hot encoded?
    if (output_col_cnt > 1)
    {
        // One hot encode!
        y = Tensor::one_hot_encode(1, output_col_cnt, typ, &y_val);
    }
    else
    {
        // Single value.
        y = new Tensor(1, 1, typ, &y_val);
    }

    this->xs.push_back(x);
    this->ys.push_back(y);
}

void Supervisor::add_all(int row_cnt, int col_cnt, int output_col_cnt, float *x_arr, float *y_arr, TensorType typ)
{
    this->xs.reserve(row_cnt);
    this->ys.reserve(row_cnt);

    for (int i = 0; i < row_cnt; i++)
    {
        this->add(col_cnt, output_col_cnt, &x_arr[i * col_cnt], y_arr[i], typ);
    }
}

void Supervisor::clear()
{
    for (int i = 0; i < this->xs.size(); i++)
    {
        delete this->xs[i];
        delete this->ys[i];
    }

    this->xs.clear();
    this->ys.clear();
}

Batch *Supervisor::create_batch(int batch_size, int lower, int upper)
{
    if (this->xs.size() == 0)
    {
        return nullptr;
    }

    Batch *batch = new Batch();

    for (int i = 0; i < batch_size; i++)
    {
        int idx = (rand() % (upper - lower)) + lower;
        batch->add(this->xs[idx], this->ys[idx]);
    }

    return batch;
}

Batch *Supervisor::create_train_batch(int batch_size)
{
    return this->create_batch(batch_size, 0, (int)floor(this->xs.size() * 0.70f));
}

Batch *Supervisor::create_validation_batch(int batch_size)
{
    return this->create_batch(batch_size, (int)floor(this->xs.size() * 0.70f), (int)floor(this->xs.size() * 0.85f));
}

Batch *Supervisor::create_test_batch(int batch_size)
{
    return this->create_batch(batch_size, (int)floor(this->xs.size() * 0.85f), this->xs.size());
}