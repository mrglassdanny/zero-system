#include "Supervisor.cuh"

Supervisor::Supervisor()
{
}

Supervisor::Supervisor(int row_cnt, int col_cnt, float *x_arr, float *y_arr)
{
    for (int i = 0; i < row_cnt; i++)
    {
        Tensor *x = new Tensor(1, col_cnt, Cpu, &x_arr[i * col_cnt]);
        Tensor *y = new Tensor(1, 1, Cpu, &y_arr[i]);

        this->xs.push_back(x);
        this->ys.push_back(y);
    }
}

Supervisor::~Supervisor()
{
    for (int i = 0; i < this->xs.size(); i++)
    {
        delete this->xs[i];
        delete this->ys[i];
    }
}

void Supervisor::add(int col_cnt, float *x_arr, float y_val)
{
    Tensor *x = new Tensor(1, col_cnt, Cpu, x_arr);
    Tensor *y = new Tensor(1, 1, Cpu, &y_val);

    this->xs.push_back(x);
    this->ys.push_back(y);
}

Batch *Supervisor::create_batch(int batch_size, int lower, int upper)
{
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