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

Batch::Batch()
{
}

Batch::Batch(int batch_size)
{
    this->records.reserve(batch_size);
}

Batch::~Batch()
{
    for (int i = 0; i < this->records.size(); i++)
    {
        delete this->records[i];
    }
}

void Batch::add(Tensor *x, Tensor *y)
{
    this->records.push_back(new Record(x, y));
}

void Batch::add(Record *record)
{
    this->records.push_back(record);
}

void Batch::add_all(Tensor *xs, Tensor *ys)
{
    int record_cnt = xs->get_shape()[0];
    int x_record_size = xs->get_cnt() / record_cnt;
    int y_record_size = ys->get_cnt() / record_cnt;

    std::vector<int> x_shape;
    std::vector<int> y_shape;

    for (int xs_shape_idx = 1; xs_shape_idx < xs->get_dim_cnt(); xs_shape_idx++)
    {
        x_shape.push_back(xs->get_shape()[xs_shape_idx]);
    }

    for (int ys_shape_idx = 1; ys_shape_idx < ys->get_dim_cnt(); ys_shape_idx++)
    {
        y_shape.push_back(ys->get_shape()[ys_shape_idx]);
    }

    for (int record_idx = 0; record_idx < record_cnt; record_idx++)
    {
        float *x_arr = &xs->get_arr()[record_idx * x_record_size];
        float *y_arr = &ys->get_arr()[record_idx * y_record_size];

        Tensor *x = new Tensor(xs->get_device(), x_shape);
        Tensor *y = new Tensor(ys->get_device(), y_shape);

        x->set_arr(x_arr);
        y->set_arr(y_arr);

        this->add(x, y);
    }
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

std::vector<int> Batch::get_x_shape()
{
    return this->get_x(0)->get_shape();
}

std::vector<int> Batch::get_y_shape()
{
    return this->get_y(0)->get_shape();
}

Supervisor::Supervisor(const char *x_path, const char *y_path, std::vector<int> x_shape, int y_one_hot_cnt)
{
    this->x_file_ptr = fopen(x_path, "rb");
    this->y_file_ptr = fopen(y_path, "rb");

    this->x_file_size = FileUtils::get_file_size(x_path);
    this->y_file_size = FileUtils::get_file_size(y_path);

    this->x_shape = x_shape;
    this->y_one_hot_cnt = y_one_hot_cnt;
}

Supervisor::~Supervisor()
{
    fclose(this->x_file_ptr);
    fclose(this->y_file_ptr);
}

int Supervisor::get_cnt()
{
    return this->x_file_size / (sizeof(float) * Tensor::get_cnt(this->x_shape));
}

std::vector<int> Supervisor::get_x_shape()
{
    return this->x_shape;
}

std::vector<int> Supervisor::get_y_shape()
{
    std::vector<int> y_shape;
    if (this->y_one_hot_cnt > 1)
    {
        y_shape.push_back(this->y_one_hot_cnt);
    }
    else
    {
        y_shape.push_back(1);
    }

    return y_shape;
}

Batch *Supervisor::create_batch(int cnt, int lower, int upper, bool rand_flg)
{
    Batch *batch = new Batch(cnt);

    int x_shape_cnt = Tensor::get_cnt(this->x_shape);

    int x_record_size = sizeof(float) * x_shape_cnt;
    int y_record_size = sizeof(float);

    float *x_buf = (float *)malloc(x_record_size);
    float *y_buf = (float *)malloc(y_record_size);

    int x_lower_offset = lower * x_record_size;
    int y_lower_offset = lower * y_record_size;

    fseek(this->x_file_ptr, x_lower_offset, SEEK_SET);
    fseek(this->y_file_ptr, y_lower_offset, SEEK_SET);

    for (int i = 0; i < cnt; i++)
    {
        if (rand_flg)
        {
            int rand_idx = rand() % (upper - lower) + lower;

            fseek(this->x_file_ptr, rand_idx * x_record_size + x_lower_offset, SEEK_SET);
            fseek(this->y_file_ptr, rand_idx * y_record_size + y_lower_offset, SEEK_SET);
        }

        fread(x_buf, x_record_size, 1, this->x_file_ptr);
        Tensor *x = new Tensor(Device::Cpu, this->x_shape);
        x->set_arr(x_buf);

        fread(y_buf, y_record_size, 1, this->y_file_ptr);
        Tensor *y;
        if (this->y_one_hot_cnt > 1)
        {
            y = Tensor::one_hot_encode(Device::Cpu, 1, this->y_one_hot_cnt, y_buf);
        }
        else
        {
            y = new Tensor(Device::Cpu, 1);
            y->set_arr(y_buf);
        }

        batch->add(new Record(x, y));
    }

    free(x_buf);
    free(y_buf);

    return batch;
}

Batch *Supervisor::create_batch()
{
    int cnt = this->get_cnt();

    if (cnt == 0)
    {
        return nullptr;
    }

    Batch *batch = this->create_batch(cnt, 0, this->get_cnt(), false);

    return batch;
}

Batch *Supervisor::create_batch(int batch_size)
{
    return this->create_batch(batch_size, 0, this->get_cnt());
}

Batch *Supervisor::create_batch(int lower, int upper)
{
    if (this->get_cnt() == 0)
    {
        return nullptr;
    }

    Batch *batch = this->create_batch(upper - lower, lower, upper, false);

    return batch;
}

Batch *Supervisor::create_batch(int batch_size, int lower, int upper)
{
    if (this->get_cnt() == 0)
    {
        return nullptr;
    }

    Batch *batch = this->create_batch(batch_size, lower, upper, true);

    return batch;
}
