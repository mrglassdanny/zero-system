#pragma once

#include "tensor.cuh"
#include "core_util.cuh"

#define COLUMN_NAME_MAX_LEN 64
#define COLUMN_NON_NUMERIC_MAX_LEN 64

#define COLUMN_INVALID_IDX -1

namespace zero
{
    namespace core
    {

        typedef std::vector<float> (*CustomEncodeFn)(const char *val, int encoded_col_cnt);

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
            Column(Column &src);
            Column(const char *name, Column &src);
            ~Column();

            void copy(Column *src);

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

            void add(Column *col);
            void subtract(Column *col);
            void subtract_abs(Column *col);

            std::map<std::string, int> *to_ordinal_map();

            Column *encode_ordinal();
            Column *encode_ordinal(std::map<std::string, int> *ordinal_map);
            std::vector<Column *> encode_onehot();
            std::vector<Column *> encode_onehot(std::map<std::string, int> *ordinal_map);
            std::vector<Column *> encode_custom(int encoded_col_cnt, CustomEncodeFn encode_fn);

            static Tensor *to_tensor(Column *col);
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
            int get_column_cnt();

            void add_column(Column *col);
            void add_column(Column *col, int idx);
            void add_column(Column *col, const char *col_name);

            int get_column_idx(const char *col_name);
            int get_last_column_idx(const char *col_name);

            Range get_column_range(const char *col_name);

            Column *get_column(int col_idx);
            Column *get_column(const char *col_name);

            Column *remove_column(int col_idx);
            Column *remove_column(const char *col_name);

            void clear();

            Table *split(int col_idx);
            Table *split(const char *col_name);

            void scale_down();

            void encode_ordinal(const char *col_name);
            void encode_ordinal(const char *col_name, std::map<std::string, int> *ordinal_map);
            void encode_onehot(const char *col_name);
            void encode_onehot(const char *col_name, std::map<std::string, int> *ordinal_map);
            void encode_custom(const char *col_name, int encoded_col_cnt, CustomEncodeFn encode_fn);

            static Table *fr_csv(const char *path);
            static void to_csv(const char *path, Table *tbl);

            static Tensor *to_tensor(Table *tbl);

            static Table *fr_ordinal_map(std::map<std::string, int> *ordinal_map, const char *key_col_name, const char *val_col_name);
        };
    }
}
