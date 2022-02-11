#pragma once

#include "core_util.cuh"

#define COLUMN_NAME_MAX_LEN 64
#define COLUMN_NON_NUMERIC_MAX_LEN 64

#define COLUMN_INVALID_IDX -1

namespace zero
{
    namespace core
    {

        class Column
        {
        private:
            void *data;

        public:
            char name[COLUMN_NAME_MAX_LEN];
            bool numeric;
            int row_cnt;

            Column();
            Column(const char *name);
            Column(const char *name, bool numeric);
            Column(const char *name, bool numeric, int row_cnt);
            ~Column();

            Column *copy(const char *name);

            void print();

            size_t get_data_size();

            void alloc_data(int row_cnt);

            float get_numeric_val(int idx);
            char *get_non_numeric_val(int idx);

            void set_val(int idx, float val);
            void set_val(int idx, const char *val);

            float get_cnt();
            float get_sum();

            float get_mean();
            float get_stddev();

            float get_min();
            float get_abs_min();
            float get_max();
            float get_abs_max();

            void scale_down();
            void scale_gaussian(float factor);

            Column *encode_ordinal();
            std::vector<Column *> encode_onehot();
        };

        class Table
        {
        private:
            std::vector<Column *> cols;

        public:
            Table();
            ~Table();

            void print();

            int get_row_cnt();

            void add_column(Column *col);

            int get_column_idx(const char *col_name);
            Column *get_column(int col_idx);
            Column *get_column(const char *col_name);

            Column *remove_column(int col_idx);
            Column *remove_column(const char *col_name);

            Table *split(int col_idx);
            Table *split(const char *col_name);

            void scale_down();

            void encode_ordinal(const char *col_name);
            void encode_onehot(const char *col_name);

            static Table *fr_csv(const char *csv_file_name);
            static void to_csv(const char *csv_file_name, Table *tbl);
        };
    }
}
