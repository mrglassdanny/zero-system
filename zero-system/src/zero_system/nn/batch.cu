#include "batch.cuh"

using namespace zero::nn;
using namespace zero::core;

Record::Record(Tensor *x, Tensor *y)
{
    this->x = x;
    this->y = y;
}

Record::~Record()
{
    delete this->x;
    delete this->y;
}

Batch::Batch(int batch_size)
{
    this->owns_records_flg = false;
    this->records.reserve(batch_size);
}

Batch::Batch(bool owns_records_flg, int batch_size)
{
    this->owns_records_flg = owns_records_flg;
    this->records.reserve(batch_size);
}

Batch::~Batch()
{
    if (this->owns_records_flg)
    {
        for (int i = 0; i < this->records.size(); i++)
        {
            delete this->records[i];
        }
    }
}

void Batch::add(Record *record)
{
    this->records.push_back(record);
}

int Batch::get_size()
{
    return this->records.size();
}

Tensor *Batch::get_x(int idx)
{
    return this->records[idx]->x;
}

Tensor *Batch::get_y(int idx)
{
    return this->records[idx]->y;
}

Supervisor::Supervisor()
{
}

Supervisor::Supervisor(int row_cnt, int col_cnt, int one_hot_cnt, float *x_arr, float *y_arr, float train_pct, float test_pct, Device device)
{
    this->add_all(row_cnt, col_cnt, one_hot_cnt, x_arr, y_arr, device);

    if (train_pct + test_pct > 1.0f || train_pct > 1.0f || test_pct > 1.0f)
    {
        this->train_pct = 0.60f;
        this->validation_pct = 0.20f;
        this->test_pct = 0.20f;
    }
    else
    {
        this->train_pct = train_pct;
        this->validation_pct = 1.0f - (train_pct + test_pct);
        this->test_pct = test_pct;
    }
}

Supervisor::~Supervisor()
{
    for (int i = 0; i < this->records.size(); i++)
    {
        delete this->records[i];
    }

    this->clear();
}

void Supervisor::add(int col_cnt, int one_hot_cnt, float *x_arr, float y_val, Device device)
{
    Tensor *x = new Tensor(device, 1, col_cnt);
    x->set_arr(x_arr);

    Tensor *y;
    // Single value or one hot encoded?
    if (one_hot_cnt > 1)
    {
        // One hot encode!
        y = Tensor::one_hot_encode(device, 1, one_hot_cnt, &y_val);
    }
    else
    {
        // Single value.
        y = new Tensor(device, 1, 1);
        y->set_arr(&y_val);
    }

    this->records.push_back(new Record(x, y));
}

void Supervisor::add_all(int row_cnt, int col_cnt, int one_hot_cnt, float *x_arr, float *y_arr, Device device)
{
    this->records.reserve(row_cnt);

    for (int i = 0; i < row_cnt; i++)
    {
        this->add(col_cnt, one_hot_cnt, &x_arr[i * col_cnt], y_arr[i], device);
    }
}

void Supervisor::clear()
{
    this->records.clear();
}

int Supervisor::get_cnt()
{
    return this->records.size();
}

std::vector<int> Supervisor::get_x_shape()
{
    return this->records[0]->x->get_shape();
}

std::vector<int> Supervisor::get_y_shape()
{
    return this->records[0]->y->get_shape();
}

void Supervisor::shuffle()
{
    std::random_shuffle(this->records.begin(), this->records.end());
}

// Creates batch with all data
Batch *Supervisor::create_batch()
{
    int cnt = this->get_cnt();

    if (cnt == 0)
    {
        return nullptr;
    }

    Batch *batch = new Batch(cnt);

    for (int i = 0; i < cnt; i++)
    {
        batch->add(this->records[i]);
    }

    return batch;
}

Batch *Supervisor::create_batch(int lower, int upper)
{
    if (this->get_cnt() == 0)
    {
        return nullptr;
    }

    Batch *batch = new Batch(upper - lower);

    for (int i = lower; i < upper; i++)
    {
        batch->add(this->records[i]);
    }

    return batch;
}

// Creates batch of specified size and bounds with random records.
Batch *Supervisor::create_batch(int batch_size, int lower, int upper)
{
    if (this->get_cnt() == 0)
    {
        return nullptr;
    }

    Batch *batch = new Batch(batch_size);

    for (int i = 0; i < batch_size; i++)
    {
        int rand_idx = (rand() % (upper - lower)) + lower;
        batch->add(this->records[rand_idx]);
    }

    return batch;
}

Batch *Supervisor::create_train_batch()
{
    return this->create_batch(0, (int)floor(this->records.size() * this->train_pct));
}

Batch *Supervisor::create_train_batch(int batch_size)
{
    return this->create_batch(batch_size, 0, (int)floor(this->records.size() * this->train_pct));
}

Batch *Supervisor::create_validation_batch()
{
    return this->create_batch((int)floor(this->records.size() * this->train_pct), (int)floor(this->records.size() *
                                                                                             (this->train_pct + this->validation_pct)));
}

Batch *Supervisor::create_test_batch()
{
    return this->create_batch((int)floor(this->records.size() * (this->train_pct + this->validation_pct)), this->records.size());
}