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

Supervisor::Supervisor(float train_pct, float test_pct)
{
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
}

float Supervisor::get_train_pct()
{
    return this->train_pct;
}

float Supervisor::get_validation_pct()
{
    return this->validation_pct;
}

float Supervisor::get_test_pct()
{
    return this->test_pct;
}

InMemorySupervisor::InMemorySupervisor(float train_pct, float test_pct, int row_cnt, int col_cnt, int one_hot_cnt, float *x_arr, float *y_arr, Device device)
    : Supervisor(train_pct, test_pct)
{
    this->add_all(row_cnt, col_cnt, one_hot_cnt, x_arr, y_arr, device);

    // Go ahead and shuffle!
    std::random_shuffle(this->records.begin(), this->records.end());
}

InMemorySupervisor::~InMemorySupervisor()
{
    for (int i = 0; i < this->records.size(); i++)
    {
        delete this->records[i];
    }

    this->clear();
}

void InMemorySupervisor::add(int col_cnt, int one_hot_cnt, float *x_arr, float y_val, Device device)
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

void InMemorySupervisor::add_all(int row_cnt, int col_cnt, int one_hot_cnt, float *x_arr, float *y_arr, Device device)
{
    this->records.reserve(row_cnt);

    for (int i = 0; i < row_cnt; i++)
    {
        this->add(col_cnt, one_hot_cnt, &x_arr[i * col_cnt], y_arr[i], device);
    }
}

void InMemorySupervisor::clear()
{
    this->records.clear();
}

int InMemorySupervisor::get_cnt()
{
    return this->records.size();
}

std::vector<int> InMemorySupervisor::get_x_shape()
{
    return this->records[0]->x->get_shape();
}

std::vector<int> InMemorySupervisor::get_y_shape()
{
    return this->records[0]->y->get_shape();
}

// Creates batch with all data
Batch *InMemorySupervisor::create_batch()
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

Batch *InMemorySupervisor::create_batch(int lower, int upper)
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
Batch *InMemorySupervisor::create_batch(int batch_size, int lower, int upper)
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

Batch *InMemorySupervisor::create_train_batch()
{
    return this->create_batch(0, (int)floor(this->records.size() * this->train_pct));
}

Batch *InMemorySupervisor::create_train_batch(int batch_size)
{
    return this->create_batch(batch_size, 0, (int)floor(this->records.size() * this->train_pct));
}

Batch *InMemorySupervisor::create_validation_batch()
{
    return this->create_batch((int)floor(this->records.size() * this->train_pct), (int)floor(this->records.size() *
                                                                                             (this->train_pct + this->validation_pct)));
}

Batch *InMemorySupervisor::create_test_batch()
{
    return this->create_batch((int)floor(this->records.size() * (this->train_pct + this->validation_pct)), this->records.size());
}

OnDiskSupervisor::OnDiskSupervisor(float train_pct, float test_pct, const char *x_path, const char *y_path, std::vector<int> x_shape, int y_one_hot_cnt)
    : Supervisor(train_pct, test_pct)
{
    this->x_file_ptr = fopen(x_path, "rb");
    this->y_file_ptr = fopen(y_path, "rb");

    this->x_file_size = FileUtils::get_file_size(x_path);
    this->y_file_size = FileUtils::get_file_size(y_path);

    this->x_shape = x_shape;
    this->y_one_hot_cnt = y_one_hot_cnt;
}

OnDiskSupervisor::~OnDiskSupervisor()
{
    fclose(this->x_file_ptr);
    fclose(this->y_file_ptr);
}

int OnDiskSupervisor::get_cnt()
{
    return this->x_file_size / (sizeof(float) * Tensor::get_cnt(this->x_shape));
}

std::vector<int> OnDiskSupervisor::get_x_shape()
{
    return this->x_shape;
}

std::vector<int> OnDiskSupervisor::get_y_shape()
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

Batch *OnDiskSupervisor::create_batch(int cnt, int lower, int upper, bool rand_flg)
{
    Batch *batch = new Batch(true, cnt);

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

        fread(x_buf, sizeof(float), x_shape_cnt, this->x_file_ptr);
        Tensor *x = new Tensor(Device::Cpu, this->x_shape);
        x->set_arr(x_buf);

        fread(y_buf, sizeof(float), 1, this->y_file_ptr);
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

// Creates batch with all data
Batch *OnDiskSupervisor::create_batch()
{
    int cnt = this->get_cnt();

    if (cnt == 0)
    {
        return nullptr;
    }

    Batch *batch = this->create_batch(cnt, 0, this->get_cnt(), false);

    return batch;
}

Batch *OnDiskSupervisor::create_batch(int lower, int upper)
{
    if (this->get_cnt() == 0)
    {
        return nullptr;
    }

    Batch *batch = this->create_batch(upper - lower, lower, upper, false);

    return batch;
}

// Creates batch of specified size and bounds with random records.
Batch *OnDiskSupervisor::create_batch(int batch_size, int lower, int upper)
{
    if (this->get_cnt() == 0)
    {
        return nullptr;
    }

    Batch *batch = this->create_batch(batch_size, lower, upper, true);

    return batch;
}

Batch *OnDiskSupervisor::create_train_batch()
{
    return this->create_batch(0, (int)floor(this->get_cnt() * this->train_pct));
}

Batch *OnDiskSupervisor::create_train_batch(int batch_size)
{
    return this->create_batch(batch_size, 0, (int)floor(this->get_cnt() * this->train_pct));
}

Batch *OnDiskSupervisor::create_validation_batch()
{
    int cnt = this->get_cnt();

    return this->create_batch((int)floor(cnt * this->train_pct), (int)floor(cnt *
                                                                            (this->train_pct + this->validation_pct)));
}

Batch *OnDiskSupervisor::create_test_batch()
{
    int cnt = this->get_cnt();

    return this->create_batch((int)floor(cnt * (this->train_pct + this->validation_pct)), cnt);
}