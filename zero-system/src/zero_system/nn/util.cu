#include "util.cuh"

using namespace zero::core;
using namespace zero::nn;

// Report member functions:

void Report::print()
{
    printf("COST: %f\tACCURACY: %f%%\n", this->cost, ((float)this->correct_cnt / (float)this->total_cnt) * 100.0f);
}

void Report::update_correct_cnt(Tensor *n, Tensor *y)
{
    int lst_lyr_n_cnt = n->get_col_cnt();

    if (lst_lyr_n_cnt > 1)
    {
        // One hot encoded:

        TensorTuple max_tup = n->get_max();
        if (y->get_val(max_tup.idx) == 1.0f)
        {
            this->correct_cnt++;
        }
    }
    else
    {
        // Single value:

        float y_val = y->get_val(0);
        float n_val = n->get_val(0);

        float lower = y_val < n_val ? y_val : n_val;
        float upper = y_val < n_val ? n_val : y_val;

        float prcnt = 1.0f - (lower / upper);

        // 10% is our number.
        if (prcnt <= 0.10f)
        {
            this->correct_cnt++;
        }
    }
}

void CSVUtils::write_csv_header(FILE *csv_file_ptr)
{
    fprintf(csv_file_ptr, "epoch,cost,accuracy,correct_cnt,total_cnt\n");
}

void CSVUtils::write_to_csv(FILE *csv_file_ptr, int epoch, Report rpt)
{
    fprintf(csv_file_ptr, "%d,%f,%f,%d,%d\n", epoch, rpt.cost, ((float)rpt.correct_cnt / (float)rpt.total_cnt) * 100.0f, rpt.correct_cnt, rpt.total_cnt);
}
