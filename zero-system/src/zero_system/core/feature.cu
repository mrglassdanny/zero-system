#include "feature.cuh"

using namespace zero::core;

Column::Column()
{
    memset(this->name, 0, sizeof(this->name));
    this->numeric = true;
    this->data = NULL;
}

Column::Column(const char *name)
{
    memset(this->name, 0, sizeof(this->name));
    strcpy(this->name, name);
    this->numeric = true;
    this->data = NULL;
}

Column::Column(const char *name, bool numeric)
{
    memset(this->name, 0, sizeof(this->name));
    strcpy(this->name, name);
    this->numeric = numeric;
    this->data = NULL;
}

Column::Column(const char *name, bool numeric, int row_cnt)
{
    memset(this->name, 0, sizeof(this->name));
    strcpy(this->name, name);
    this->numeric = numeric;
    this->data = NULL;

    this->alloc_data(row_cnt);
}

Column::Column(Column &src)
{
    memset(this->name, 0, sizeof(this->name));
    this->copy(&src);
}

Column::Column(const char *name, Column &src)
{
    memset(this->name, 0, sizeof(this->name));
    strcpy(this->name, name);

    this->copy(&src);
}

Column::~Column()
{
    if (this->data != NULL)
    {
        free(this->data);
    }
}

void Column::copy(Column *src)
{
    if (strlen(this->name) == 0)
    {
        strcpy(this->name, src->name);
    }

    this->numeric = src->numeric;
    this->data = NULL;

    this->alloc_data(src->row_cnt);
    memcpy(this->data, src->data, this->get_data_size());
}

void Column::print()
{
    printf("%s (%s)\n", this->name, this->numeric ? "NUMERIC" : "NON NUMERIC");

    if (this->row_cnt > 0 && this->data != NULL)
    {
        if (this->numeric)
        {
            for (int row_idx = 0; row_idx < this->row_cnt; row_idx++)
            {
                printf("\t%f\n", this->get_numeric_val(row_idx));
            }
        }
        else
        {
            for (int row_idx = 0; row_idx < this->row_cnt; row_idx++)
            {
                printf("\t%s\n", this->get_non_numeric_val(row_idx));
            }
        }
    }
}

size_t Column::get_data_size()
{
    if (this->numeric)
    {
        return sizeof(float) * this->row_cnt;
    }
    else
    {
        return sizeof(char) * COLUMN_NON_NUMERIC_MAX_LEN * this->row_cnt;
    }
}

void Column::alloc_data(int row_cnt)
{
    if (this->data != NULL)
    {
        free(this->data);
        this->data = NULL;
    }

    this->row_cnt = row_cnt;

    size_t size = this->get_data_size();

    this->data = malloc(size);
    memset(this->data, 0, size);
}

float Column::get_numeric_val(int idx)
{
    if (!this->numeric)
    {
        return 0.0f;
    }

    float *col_data = (float *)this->data;

    return col_data[idx];
}

char *Column::get_non_numeric_val(int idx)
{
    if (this->numeric)
    {
        return NULL;
    }

    char *col_data = (char *)this->data;

    return &col_data[idx * COLUMN_NON_NUMERIC_MAX_LEN];
}

void Column::set_val(int idx, float val)
{
    if (this->numeric)
    {
        float *col_data = (float *)this->data;
        col_data[idx] = val;
    }
}

void Column::set_val(int idx, const char *val)
{
    if (!this->numeric)
    {
        char *col_data = (char *)this->data;
        memcpy(&col_data[idx * COLUMN_NON_NUMERIC_MAX_LEN], val,
               sizeof(char) * COLUMN_NON_NUMERIC_MAX_LEN);
    }
}

float Column::get_cnt()
{
    return (float)this->row_cnt;
}

float Column::get_sum()
{
    float sum = 0.0f;

    if (!this->numeric)
    {
        return sum;
    }

    for (int row_idx = 0; row_idx < this->row_cnt; row_idx++)
    {
        sum += this->get_numeric_val(row_idx);
    }

    return sum;
}

float Column::get_mean()
{
    return this->get_sum() / this->get_cnt();
}

float Column::get_stddev()
{
    float stddev = 0.0f;

    if (!this->numeric)
    {
        return stddev;
    }

    float mean = this->get_mean();

    for (int row_idx = 0; row_idx < this->row_cnt; row_idx++)
    {
        float sq_diff = (this->get_numeric_val(row_idx) - mean);
        stddev += (sq_diff * sq_diff);
    }

    stddev /= this->get_cnt();

    return sqrt(stddev);
}

float Column::get_min()
{
    float min = FLT_MAX;

    if (!this->numeric)
    {
        return min;
    }

    float cur;

    for (int i = 0; i < this->row_cnt; i++)
    {
        cur = this->get_numeric_val(i);

        if (cur < min)
        {
            min = cur;
        }
    }

    return min;
}

float Column::get_abs_min()
{
    float min = FLT_MAX;

    if (!this->numeric)
    {
        return min;
    }

    float cur;

    for (int i = 0; i < this->row_cnt; i++)
    {
        cur = abs(this->get_numeric_val(i));

        if (cur < min)
        {
            min = cur;
        }
    }

    return min;
}

float Column::get_max()
{
    float max = -FLT_MAX;

    if (!this->numeric)
    {
        return max;
    }

    float cur;

    for (int i = 0; i < this->row_cnt; i++)
    {
        cur = this->get_numeric_val(i);

        if (cur > max)
        {
            max = cur;
        }
    }

    return max;
}

float Column::get_abs_max()
{
    float max = -FLT_MAX;

    if (!this->numeric)
    {
        return max;
    }

    float cur;

    for (int i = 0; i < this->row_cnt; i++)
    {
        cur = abs(this->get_numeric_val(i));

        if (cur > max)
        {
            max = cur;
        }
    }

    return max;
}

void Column::scale_down()
{
    if (this->numeric)
    {
        float max = this->get_abs_max();

        int factor = 1;
        while (true)
        {
            if ((max / (factor * 1.0f)) <= 1.0f)
            {
                break;
            }
            else
            {
                factor *= 10;
            }
        }

        float flt_factor = factor * 1.0f;

        for (int i = 0; i < this->row_cnt; i++)
        {
            this->set_val(i, this->get_numeric_val(i) / flt_factor);
        }
    }
}

void Column::scale_gaussian(float factor)
{
    if (!this->numeric)
    {
        return;
    }

    float mean = this->get_mean();
    float stddev = this->get_stddev();

    for (int row_idx = 0; row_idx < this->row_cnt; row_idx++)
    {
        this->set_val(row_idx, (((this->get_numeric_val(row_idx) - mean) / stddev) * factor));
    }
}

void Column::add(Column *col)
{
    if (!this->numeric || !col->numeric)
    {
        return;
    }

    for (int i = 0; i < this->row_cnt; i++)
    {
        this->set_val(i, this->get_numeric_val(i) + col->get_numeric_val(i));
    }
}

void Column::subtract(Column *col)
{
    if (!this->numeric || !col->numeric)
    {
        return;
    }

    for (int i = 0; i < this->row_cnt; i++)
    {
        this->set_val(i, this->get_numeric_val(i) - col->get_numeric_val(i));
    }
}

void Column::subtract_abs(Column *col)
{
    if (!this->numeric || !col->numeric)
    {
        return;
    }

    for (int i = 0; i < this->row_cnt; i++)
    {
        this->set_val(i, abs(this->get_numeric_val(i) - col->get_numeric_val(i)));
    }
}

std::map<std::string, int> *Column::to_ordinal_map()
{
    std::map<std::string, int> *ordinal_map = new std::map<std::string, int>();

    int ordinal_id = 1;

    for (int row_idx = 0; row_idx < this->row_cnt; row_idx++)
    {
        std::string text = std::string(this->get_non_numeric_val(row_idx));

        if (ordinal_map->find(text) == ordinal_map->end())
        {
            (*ordinal_map)[text] = ordinal_id++;
        }
    }

    return ordinal_map;
}

Column *Column::encode_ordinal()
{
    Column *ordinal_col = new Column(this->name, true, this->row_cnt);

    std::map<std::string, int> ordinal_map;

    int ordinal_id = 1;

    for (int row_idx = 0; row_idx < this->row_cnt; row_idx++)
    {
        std::string text = std::string(this->get_non_numeric_val(row_idx));

        if (ordinal_map.find(text) != ordinal_map.end())
        {
            std::map<std::string, int>::iterator iter = ordinal_map.find(text);
            ordinal_col->set_val(row_idx, iter->second);
        }
        else
        {
            ordinal_col->set_val(row_idx, ordinal_id);
            ordinal_map[text] = ordinal_id++;
        }
    }

    return ordinal_col;
}

Column *Column::encode_ordinal(std::map<std::string, int> *ordinal_map)
{
    Column *ordinal_col = new Column(this->name, true, this->row_cnt);

    for (int row_idx = 0; row_idx < this->row_cnt; row_idx++)
    {
        std::string text = std::string(this->get_non_numeric_val(row_idx));

        if (ordinal_map->find(text) != ordinal_map->end())
        {
            std::map<std::string, int>::iterator iter = ordinal_map->find(text);
            ordinal_col->set_val(row_idx, iter->second);
        }
    }

    return ordinal_col;
}

std::vector<Column *> Column::encode_onehot()
{
    std::vector<Column *> onehot_cols;

    Column *ordinal_col = this->encode_ordinal();

    for (int ordinal_idx = 0; ordinal_idx < (int)ordinal_col->get_max(); ordinal_idx++)
    {
        Column *onehot_col = new Column(this->name, true, this->row_cnt);

        for (int row_idx = 0; row_idx < row_cnt; row_idx++)
        {
            // NOTE: subtract 1 since ordinal encoding starts at 1!
            if ((ordinal_col->get_numeric_val(row_idx) - 1) == ordinal_idx)
            {
                onehot_col->set_val(row_idx, 1.0f);
            }
            else
            {
                onehot_col->set_val(row_idx, 0.0f);
            }
        }

        onehot_cols.push_back(onehot_col);
    }

    delete ordinal_col;

    return onehot_cols;
}

std::vector<Column *> Column::encode_onehot(std::map<std::string, int> *ordinal_map)
{
    std::vector<Column *> onehot_cols;

    Column *ordinal_col = this->encode_ordinal(ordinal_map);

    for (int ordinal_idx = 0; ordinal_idx < (int)ordinal_col->get_max(); ordinal_idx++)
    {
        Column *onehot_col = new Column(this->name, true, this->row_cnt);

        for (int row_idx = 0; row_idx < row_cnt; row_idx++)
        {
            // NOTE: subtract 1 since ordinal encoding starts at 1!
            if ((ordinal_col->get_numeric_val(row_idx) - 1) == ordinal_idx)
            {
                onehot_col->set_val(row_idx, 1.0f);
            }
            else
            {
                onehot_col->set_val(row_idx, 0.0f);
            }
        }

        onehot_cols.push_back(onehot_col);
    }

    delete ordinal_col;

    return onehot_cols;
}

std::vector<Column *> Column::encode_custom(int encoded_col_cnt, CustomEncodeFn encode_fn)
{
    std::vector<Column *> encoded_cols;
    for (int col_idx = 0; col_idx < encoded_col_cnt; col_idx++)
    {
        encoded_cols.push_back(new Column(this->name, true, this->row_cnt));
    }

    for (int row_idx = 0; row_idx < row_cnt; row_idx++)
    {
        std::vector<float> vals = encode_fn(this->get_non_numeric_val(row_idx), encoded_col_cnt);

        for (int col_idx = 0; col_idx < encoded_col_cnt; col_idx++)
        {
            encoded_cols[col_idx]->set_val(row_idx, vals[col_idx]);
        }
    }

    return encoded_cols;
}

// Column static functions:

Tensor *Column::to_tensor(Column *col)
{
    int row_cnt = col->row_cnt;

    Tensor *tensor = new Tensor(Device::Cpu, row_cnt);

    for (int row_idx = 0; row_idx < row_cnt; row_idx++)
    {
        tensor->set_val(row_idx, col->get_numeric_val(row_idx));
    }

    return tensor;
}

Table::Table() {}

Table::~Table()
{
    this->clear();
}

void Table::print()
{
    printf("FEATURE COUNT: %d\n", this->cols.size());
    for (int i = 0; i < this->cols.size(); i++)
    {
        Column *col = this->cols[i];

        printf("\tidx: %d\tname: %s (%s)\n", i, col->name, col->numeric ? "NUMERIC" : "NON NUMERIC");
    }

    printf("ROW COUNT: %d\n", this->get_row_cnt());
}

int Table::get_row_cnt()
{
    if (this->cols.size() == 0)
    {
        return 0;
    }
    else
    {
        return this->cols[0]->row_cnt;
    }
}

int Table::get_column_cnt()
{
    return this->cols.size();
}

void Table::add_column(Column *col)
{
    this->cols.push_back(col);
}

void Table::add_column(Column *col, int idx)
{
    this->cols.insert(this->cols.begin() + idx, col);
}

void Table::add_column(Column *col, const char *col_name)
{
    this->cols.insert(this->cols.begin() + this->get_last_column_idx(col_name) + 1, col);
}

int Table::get_column_idx(const char *col_name)
{
    int col_idx = COLUMN_INVALID_IDX;

    for (int _col_idx = 0; _col_idx < this->cols.size(); _col_idx++)
    {
        if (strcmp(this->cols[_col_idx]->name, col_name) == 0)
        {
            col_idx = _col_idx;
            break;
        }
    }

    return col_idx;
}

int Table::get_last_column_idx(const char *col_name)
{
    int col_idx = COLUMN_INVALID_IDX;

    for (int _col_idx = this->cols.size() - 1; _col_idx >= 0; _col_idx--)
    {
        if (strcmp(this->cols[_col_idx]->name, col_name) == 0)
        {
            col_idx = _col_idx;
            break;
        }
    }

    return col_idx;
}

Range Table::get_column_range(const char *col_name)
{
    Range range;

    range.beg_idx = this->get_column_idx(col_name);
    range.end_idx = this->get_last_column_idx(col_name);

    return range;
}

Column *Table::get_column(int col_idx)
{
    return this->cols[col_idx];
}

Column *Table::get_column(const char *col_name)
{
    return this->get_column(this->get_column_idx(col_name));
}

Column *Table::remove_column(int col_idx)
{
    Column *col = this->get_column(col_idx);
    this->cols.erase(this->cols.begin() + col_idx);
    return col;
}

Column *Table::remove_column(const char *col_name)
{
    return this->remove_column(this->get_column_idx(col_name));
}

void Table::clear()
{
    for (int col_idx = 0; col_idx < this->cols.size(); col_idx++)
    {
        delete this->cols[col_idx];
    }

    this->cols.clear();
}

Table *Table::split(int split_idx)
{
    Table *tbl = new Table();

    int split_cnt = this->cols.size() - split_idx;

    for (int i = 0; i < split_cnt; i++)
    {
        tbl->add_column(this->remove_column(split_idx));
    }

    return tbl;
}

Table *Table::split(const char *split_name)
{
    int split_idx = this->get_column_idx(split_name);
    return this->split(split_idx);
}

void Table::scale_down()
{
    for (int i = 0; i < this->cols.size(); i++)
    {
        this->cols[i]->scale_down();
    }
}

void Table::encode_ordinal(const char *col_name)
{
    int col_idx = this->get_column_idx(col_name);
    Column *col = this->get_column(col_idx);

    if (col == NULL)
    {
        return;
    }

    Column *ordinal_col = col->encode_ordinal();

    if (ordinal_col == NULL)
    {
        return;
    }

    this->cols.insert(this->cols.begin() + col_idx + 1, ordinal_col);
    this->cols.erase(this->cols.begin() + col_idx);

    delete col;
}

void Table::encode_ordinal(const char *col_name, std::map<std::string, int> *ordinal_map)
{
    int col_idx = this->get_column_idx(col_name);
    Column *col = this->get_column(col_idx);

    if (col == NULL)
    {
        return;
    }

    Column *ordinal_col = col->encode_ordinal(ordinal_map);

    if (ordinal_col == NULL)
    {
        return;
    }

    this->cols.insert(this->cols.begin() + col_idx + 1, ordinal_col);
    this->cols.erase(this->cols.begin() + col_idx);

    delete col;
}

void Table::encode_onehot(const char *col_name)
{
    int col_idx = this->get_column_idx(col_name);
    Column *col = this->get_column(col_idx);

    if (col == NULL)
    {
        return;
    }

    std::vector<Column *> onehot_cols = col->encode_onehot();

    if (onehot_cols.size() == 0)
    {
        return;
    }

    this->cols.insert(this->cols.begin() + col_idx + 1, onehot_cols.begin(), onehot_cols.end());
    this->cols.erase(this->cols.begin() + col_idx);

    delete col;
}

void Table::encode_onehot(const char *col_name, std::map<std::string, int> *ordinal_map)
{
    int col_idx = this->get_column_idx(col_name);
    Column *col = this->get_column(col_idx);

    if (col == NULL)
    {
        return;
    }

    std::vector<Column *> onehot_cols = col->encode_onehot(ordinal_map);

    if (onehot_cols.size() == 0)
    {
        return;
    }

    this->cols.insert(this->cols.begin() + col_idx + 1, onehot_cols.begin(), onehot_cols.end());
    this->cols.erase(this->cols.begin() + col_idx);

    delete col;
}

void Table::encode_custom(const char *col_name, int encoded_col_cnt, CustomEncodeFn encode_fn)
{
    int col_idx = this->get_column_idx(col_name);
    Column *col = this->get_column(col_idx);

    if (col == NULL)
    {
        return;
    }

    std::vector<Column *> encoded_cols = col->encode_custom(encoded_col_cnt, encode_fn);

    if (encoded_cols.size() == 0)
    {
        return;
    }

    this->cols.insert(this->cols.begin() + col_idx + 1, encoded_cols.begin(), encoded_cols.end());
    this->cols.erase(this->cols.begin() + col_idx);

    delete col;
}

// Table static functions:

Table *Table::fr_csv(const char *path)
{
    Table *tbl = new Table();

    FILE *csv_file = fopen(path, "rb");

    long file_size = FileUtils::get_file_size(path);

    char *csv_buf = (char *)malloc(file_size + sizeof(char));
    int csv_buf_idx = 0;

    memset(csv_buf, 0, file_size + sizeof(char));

    fread(csv_buf, sizeof(char), file_size, csv_file);

    fclose(csv_file);

    StackBuffer buf;

    int col_cnt = 0;
    int col_idx = 0;

    int row_cnt = 0;
    int row_idx = 0;

    // Get col names:
    {
        while (csv_buf[csv_buf_idx] != '\n')
        {
            if (csv_buf[csv_buf_idx] == ',')
            {
                tbl->cols.push_back(new Column(buf.get()));

                col_cnt++;

                buf.clear();
            }
            else
            {
                if (csv_buf[csv_buf_idx] != '"' && csv_buf[csv_buf_idx] != '\r')
                {
                    buf.append(csv_buf[csv_buf_idx]);
                }
            }

            csv_buf_idx++;
        }

        // Make sure to process last col.
        tbl->cols.push_back(new Column(buf.get()));

        col_cnt++;

        buf.clear();

        // csv_buf_idx is currently indexing a newline.
        csv_buf_idx++;
    }

    // Get row count:
    {
        int lst_row_idx = 0;
        for (int i = csv_buf_idx; i < file_size; i++)
        {
            if (csv_buf[i] == '\n')
            {
                row_cnt++;
                lst_row_idx = i;
            }
        }

        // If file does not end in newline, add to the row count.
        if (lst_row_idx < file_size - 1)
        {
            row_cnt++;
        }
    }

    // Process col data:
    {
        int _csv_buf_idx = csv_buf_idx;

        // Determine data types:
        {
            buf.clear();

            for (; csv_buf_idx < file_size; csv_buf_idx++)
            {
                while (csv_buf[csv_buf_idx] != ',' && csv_buf[csv_buf_idx] != '\n' && csv_buf_idx < file_size)
                {
                    if (csv_buf[csv_buf_idx] != '"' && csv_buf[csv_buf_idx] != '\r')
                    {
                        buf.append(csv_buf[csv_buf_idx]);
                    }

                    csv_buf_idx++;
                }

                if (csv_buf[csv_buf_idx] == ',')
                {
                    if (tbl->cols[col_idx]->numeric)
                    {
                        if (!buf.is_numeric())
                        {
                            tbl->cols[col_idx]->numeric = false;
                        }
                    }

                    buf.clear();
                    col_idx++;
                }
                else if (csv_buf[csv_buf_idx] == '\n')
                {
                    if (tbl->cols[col_idx]->numeric)
                    {
                        if (!buf.is_numeric())
                        {
                            tbl->cols[col_idx]->numeric = false;
                        }
                    }

                    buf.clear();
                    row_idx++;
                    col_idx = 0;
                }
            }

            // Make sure to grab the last bit before we finish up!
            if (!buf.is_empty())
            {
                if (tbl->cols[col_idx]->numeric)
                {
                    if (!buf.is_numeric())
                    {
                        tbl->cols[col_idx]->numeric = false;
                    }
                }
            }
        }

        // Allocate memory for col data:
        for (int col_idx = 0; col_idx < tbl->cols.size(); col_idx++)
        {
            Column *col = tbl->cols[col_idx];
            col->alloc_data(row_cnt);
        }

        // Set col data:
        {
            // Reset:
            csv_buf_idx = _csv_buf_idx;
            col_idx = 0;
            row_idx = 0;
            buf.clear();

            for (; csv_buf_idx < file_size; csv_buf_idx++)
            {
                while (csv_buf[csv_buf_idx] != ',' && csv_buf[csv_buf_idx] != '\n' && csv_buf_idx < file_size)
                {
                    if (csv_buf[csv_buf_idx] != '"' && csv_buf[csv_buf_idx] != '\r')
                    {
                        buf.append(csv_buf[csv_buf_idx]);
                    }

                    csv_buf_idx++;
                }

                if (csv_buf[csv_buf_idx] == ',')
                {
                    Column *col = tbl->cols[col_idx];

                    if (col->numeric)
                    {
                        col->set_val(row_idx, (float)atof(buf.get()));
                    }
                    else
                    {
                        col->set_val(row_idx, buf.get());
                    }

                    buf.clear();
                    col_idx++;
                }
                else if (csv_buf[csv_buf_idx] == '\n')
                {
                    Column *col = tbl->cols[col_idx];

                    if (col->numeric)
                    {
                        col->set_val(row_idx, (float)atof(buf.get()));
                    }
                    else
                    {
                        col->set_val(row_idx, buf.get());
                    }

                    buf.clear();
                    row_idx++;
                    col_idx = 0;
                }
            }

            // Make sure to grab the last bit before we finish up!
            if (!buf.is_empty())
            {
                Column *col = tbl->cols[col_idx];

                if (col->numeric)
                {
                    col->set_val(row_idx, (float)atof(buf.get()));
                }
                else
                {
                    col->set_val(row_idx, buf.get());
                }
            }
        }
    }

    free(csv_buf);

    return tbl;
}

void Table::to_csv(const char *path, Table *tbl)
{
    FILE *csv_file = fopen(path, "w");

    // Headers first:
    for (int col_idx = 0; col_idx < tbl->cols.size(); col_idx++)
    {
        Column *col = tbl->cols[col_idx];

        if (col_idx == tbl->cols.size() - 1)
        {
            fprintf(csv_file, "%s\n", col->name);
        }
        else
        {
            fprintf(csv_file, "%s,", col->name);
        }
    }

    // Now data:
    for (int row_idx = 0; row_idx < tbl->get_row_cnt(); row_idx++)
    {
        for (int col_idx = 0; col_idx < tbl->cols.size(); col_idx++)
        {
            Column *col = tbl->cols[col_idx];

            if (col_idx == tbl->cols.size() - 1)
            {
                if (col->numeric)
                {
                    fprintf(csv_file, "%f\n", col->get_numeric_val(row_idx));
                }
                else
                {
                    fprintf(csv_file, "%s\n", col->get_non_numeric_val(row_idx));
                }
            }
            else
            {
                if (col->numeric)
                {
                    fprintf(csv_file, "%f,", col->get_numeric_val(row_idx));
                }
                else
                {
                    fprintf(csv_file, "%s,", col->get_non_numeric_val(row_idx));
                }
            }
        }
    }

    fclose(csv_file);
}

Tensor *Table::to_tensor(Table *tbl)
{
    int row_cnt = tbl->get_row_cnt();
    int col_cnt = tbl->get_column_cnt();

    Tensor *tensor = new Tensor(Device::Cpu, row_cnt, col_cnt);

    for (int col_idx = 0; col_idx < col_cnt; col_idx++)
    {
        Column *col = tbl->get_column(col_idx);

        for (int row_idx = 0; row_idx < row_cnt; row_idx++)
        {
            tensor->set_val(row_idx * col_cnt + col_idx, col->get_numeric_val(row_idx));
        }
    }

    return tensor;
}

Table *Table::fr_ordinal_map(std::map<std::string, int> *ordinal_map, const char *key_col_name, const char *val_col_name)
{
    Table *tbl = new Table();
    Column *key_col = new Column(key_col_name, false, ordinal_map->size());
    Column *val_col = new Column(val_col_name, true, ordinal_map->size());

    std::map<std::string, int>::iterator iter;
    int row_idx;

    for (iter = ordinal_map->begin(), row_idx = 0; iter != ordinal_map->end(); iter++, row_idx++)
    {
        key_col->set_val(row_idx, iter->first.data());
        val_col->set_val(row_idx, iter->second);
    }

    tbl->add_column(key_col);
    tbl->add_column(val_col);

    return tbl;
}