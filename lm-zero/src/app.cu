
#include <zero_system/mod.cuh>

#define ACTCODTYP_EMBG_DIM_CNT 4
#define LOC_EMBG_DIM_CNT 12

std::vector<int> get_output_shape()
{
    std::vector<int> n_shape{ACTCODTYP_EMBG_DIM_CNT + LOC_EMBG_DIM_CNT};
    return n_shape;
}

std::vector<int> get_output_shape2()
{
    std::vector<int> n_shape{2};
    return n_shape;
}

std::vector<int> get_output_shape3()
{
    std::vector<int> n_shape{ACTCODTYP_EMBG_DIM_CNT + 1};
    return n_shape;
}

void forward(Tensor *n, Tensor *nxt_n, bool train_flg)
{
    for (int i = 0; i < ACTCODTYP_EMBG_DIM_CNT; i++)
    {
        nxt_n->set_val(i, n->get_val(i));
    }

    int src_loc_beg_idx = ACTCODTYP_EMBG_DIM_CNT;
    int dst_loc_beg_idx = src_loc_beg_idx + LOC_EMBG_DIM_CNT;

    for (int i = 0; i < LOC_EMBG_DIM_CNT; i++)
    {
        float loc_diff = n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i);
        nxt_n->set_val(i + ACTCODTYP_EMBG_DIM_CNT, loc_diff);
    }
}

void forward2(Tensor *n, Tensor *nxt_n, bool train_flg)
{
    float a = n->get_val(0);
    float t = 0.0f;

    int src_loc_beg_idx = 1;
    int dst_loc_beg_idx = src_loc_beg_idx + LOC_EMBG_DIM_CNT;

    for (int i = 0; i < LOC_EMBG_DIM_CNT; i++)
    {
        float loc_diff = n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i);
        t += (loc_diff * loc_diff);
    }

    t = sqrt(t);

    nxt_n->set_val(0, a);
    nxt_n->set_val(1, t);
}

void forward3(Tensor *n, Tensor *nxt_n, bool train_flg)
{
    for (int i = 0; i < ACTCODTYP_EMBG_DIM_CNT; i++)
    {
        nxt_n->set_val(i, n->get_val(i));
    }

    int src_loc_beg_idx = ACTCODTYP_EMBG_DIM_CNT;
    int dst_loc_beg_idx = src_loc_beg_idx + LOC_EMBG_DIM_CNT;

    float t = 0.0f;

    for (int i = 0; i < LOC_EMBG_DIM_CNT; i++)
    {
        float loc_diff = n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i);
        t += (loc_diff * loc_diff);
    }

    t = sqrt(t);

    nxt_n->set_val(ACTCODTYP_EMBG_DIM_CNT, t);
}

Tensor *backward(Tensor *n, Tensor *dc)
{
    Tensor *nxt_dc = new Tensor(n->get_device(), n->get_shape());

    for (int i = 0; i < ACTCODTYP_EMBG_DIM_CNT; i++)
    {
        nxt_dc->set_val(i, dc->get_val(i));
    }

    int src_loc_beg_idx = ACTCODTYP_EMBG_DIM_CNT;
    int dst_loc_beg_idx = src_loc_beg_idx + LOC_EMBG_DIM_CNT;

    for (int i = 0; i < LOC_EMBG_DIM_CNT; i++)
    {
        nxt_dc->set_val(src_loc_beg_idx + i, dc->get_val(i + ACTCODTYP_EMBG_DIM_CNT) * 1.0f);
        nxt_dc->set_val(dst_loc_beg_idx + i, dc->get_val(i + ACTCODTYP_EMBG_DIM_CNT) * -1.0f);
    }

    return nxt_dc;
}

Tensor *backward2(Tensor *n, Tensor *dc)
{
    Tensor *nxt_dc = new Tensor(n->get_device(), n->get_shape());

    nxt_dc->set_val(0, dc->get_val(0) * 1.0f);

    int src_loc_beg_idx = 1;
    int dst_loc_beg_idx = src_loc_beg_idx + LOC_EMBG_DIM_CNT;

    float t = 0.0f;

    for (int i = 0; i < LOC_EMBG_DIM_CNT; i++)
    {
        float loc_diff = n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i);
        t += (loc_diff * loc_diff);
    }

    t = sqrt(t);

    for (int i = 0; i < LOC_EMBG_DIM_CNT; i++)
    {
        float dc_val = dc->get_val(1);

        if (t == 0.0f)
        {
            nxt_dc->set_val(src_loc_beg_idx + i, 0.0f);
            nxt_dc->set_val(dst_loc_beg_idx + i, 0.0f);
        }
        else
        {
            float dv = 1.0f / (2.0f * t);

            nxt_dc->set_val(src_loc_beg_idx + i, dc_val * dv * (2.0f * (n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i))));
            nxt_dc->set_val(dst_loc_beg_idx + i, dc_val * dv * (-2.0f * (n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i))));
        }
    }

    return nxt_dc;
}

Tensor *backward3(Tensor *n, Tensor *dc)
{
    Tensor *nxt_dc = new Tensor(n->get_device(), n->get_shape());

    for (int i = 0; i < ACTCODTYP_EMBG_DIM_CNT; i++)
    {
        nxt_dc->set_val(i, dc->get_val(i));
    }

    int src_loc_beg_idx = ACTCODTYP_EMBG_DIM_CNT;
    int dst_loc_beg_idx = src_loc_beg_idx + LOC_EMBG_DIM_CNT;

    float t = 0.0f;

    for (int i = 0; i < LOC_EMBG_DIM_CNT; i++)
    {
        float loc_diff = n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i);
        t += (loc_diff * loc_diff);
    }

    t = sqrt(t);

    for (int i = 0; i < LOC_EMBG_DIM_CNT; i++)
    {
        float dc_val = dc->get_val(ACTCODTYP_EMBG_DIM_CNT);

        if (t == 0.0f)
        {
            nxt_dc->set_val(src_loc_beg_idx + i, 0.0f);
            nxt_dc->set_val(dst_loc_beg_idx + i, 0.0f);
        }
        else
        {
            float dv = 1.0f / (2.0f * t);

            nxt_dc->set_val(src_loc_beg_idx + i, dc_val * dv * (2.0f * (n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i))));
            nxt_dc->set_val(dst_loc_beg_idx + i, dc_val * dv * (-2.0f * (n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i))));
        }
    }

    return nxt_dc;
}

void upd_rslt_fn(Tensor *p, Tensor *y, int *cnt)
{
    float y_val = y->get_val(0);
    float p_val = p->get_val(0);

    float lower = y_val < p_val ? y_val : p_val;
    float upper = y_val < p_val ? p_val : y_val;

    float prcnt = 1.0f - (lower / upper);

    if (prcnt <= 0.20f)
    {
        (*cnt)++;
    }
}

std::vector<float> loc_encode_fn(const char *loc_name, int dim_cnt)
{
    char delims[] = {'-'};
    char numerics[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

    StackBuffer buf;

    std::vector<float> parsed_loc;

    int loc_name_len = strlen(loc_name);

    bool delim_flg = false;
    bool numeric_flg = false;
    bool alpha_flg = false;

    for (int i = 0; i < loc_name_len; i++)
    {
        char c = loc_name[i];

        for (int j = 0; j < sizeof(delims); j++)
        {
            if (c == delims[j])
            {
                if (buf.get_size() > 0)
                {
                    parsed_loc.push_back(atof(buf.get()));
                }

                buf.clear();

                delim_flg = true;
                numeric_flg = false;
                alpha_flg = false;

                break;
            }
        }

        if (!delim_flg)
        {
            numeric_flg = false;

            for (int j = 0; j < sizeof(numerics); j++)
            {
                if (c == numerics[j])
                {
                    numeric_flg = true;

                    if (alpha_flg)
                    {
                        if (buf.get_size() > 0)
                        {
                            parsed_loc.push_back(atof(buf.get()));
                        }

                        buf.clear();
                        buf.append(c);
                    }
                    else
                    {
                        buf.append(c);
                    }

                    alpha_flg = false;

                    break;
                }
            }
        }

        if (!delim_flg && !numeric_flg)
        {
            if (alpha_flg)
            {
                float buf_num = 0.0f;

                if (buf.get_size() > 0)
                {
                    buf_num = atof(buf.get());
                }

                buf.clear();
                buf_num += (int)c;
                buf.append(buf_num);
            }
            else
            {
                if (buf.get_size() > 0)
                {
                    parsed_loc.push_back(atof(buf.get()));
                }

                buf.clear();
                buf.append((int)c);
            }

            alpha_flg = true;
        }
    }

    if (buf.get_size() > 0)
    {
        parsed_loc.push_back(atof(buf.get()));
    }

    if (parsed_loc.size() < dim_cnt)
    {
        for (int i = parsed_loc.size(); i < dim_cnt; i++)
        {
            parsed_loc.push_back(0.0f);
        }
    }

    return parsed_loc;
}

int main(int argc, char **argv)
{
    ZERO();

    // Data setup:

    Table *xs_tbl = Table::fr_csv("data/palmov_data-test.csv");
    Table *ys_tbl = xs_tbl->split("elapsed_secs");

    Table *actcodtyps_tbl = Table::fr_csv("data/actcodtyps.csv");
    std::map<std::string, int> *actcodtyp_map = actcodtyps_tbl->get_column(0)->to_ordinal_map();

    Table *locs_tbl = Table::fr_csv("data/locs.csv");
    std::map<std::string, int> *loc_map = locs_tbl->get_column(0)->to_ordinal_map();

    Column *actcod_col = new Column(*xs_tbl->get_column("actcod"));
    Column *typ_col = new Column(*xs_tbl->get_column("typ"));
    Column *fr_loc_col = new Column(*xs_tbl->get_column("fr_loc"));
    Column *to_loc_col = new Column(*xs_tbl->get_column("to_loc"));

    delete xs_tbl->remove_column("actcod");
    delete xs_tbl->remove_column("typ");
    // delete xs_tbl->remove_column("fr_loc");
    // delete xs_tbl->remove_column("to_loc");
    delete xs_tbl->remove_column("fx");
    delete xs_tbl->remove_column("fy");
    delete xs_tbl->remove_column("tx");
    delete xs_tbl->remove_column("ty");
    delete xs_tbl->remove_column("trvl");

    xs_tbl->encode_ordinal("actcodtyp", actcodtyp_map);
    // xs_tbl->encode_onehot("actcodtyp", actcodtyp_map);
    xs_tbl->encode_ordinal("fr_loc", loc_map);
    xs_tbl->encode_ordinal("to_loc", loc_map);

    // xs_tbl->get_column("trvl")->scale_down();
    ys_tbl->scale_down();

    Supervisor *sup;
    {
        Tensor *xs = Table::to_tensor(xs_tbl);
        Tensor *ys = Table::to_tensor(ys_tbl);

        Tensor::to_file("temp/xs.tr", xs);
        Tensor::to_file("temp/ys.tr", ys);

        std::vector<int> x_shape{xs->get_shape()[1]};
        sup = new Supervisor("temp/xs.tr", "temp/ys.tr", x_shape, 0);

        delete xs;
        delete ys;
    }

    // ===================================================================================================

    // Models:

    Model *lm = new Model(0.1f);

    // Model 1: 25%
    {
        // Model *actcodtyp_model = new Model();
        // actcodtyp_model->embedding((int)actcodtyps_tbl->get_column(0)->row_cnt, ACTCODTYP_EMBG_DIM_CNT);

        // lm->child(actcodtyp_model, xs_tbl->get_column_range("actcodtyp"));

        // lm->dense(lm->calc_adjusted_input_shape(xs_tbl->get_column_cnt()), 128);
        // lm->activation(ReLU);
        // lm->dense(32);
        // lm->activation(ReLU);
        // lm->dense(1);
        // lm->activation(ReLU);
    }

    // Model 2: 26%
    {
        // Model *actcodtyp_model = new Model();
        // actcodtyp_model->embedding((int)actcodtyps_tbl->get_column(0)->row_cnt, ACTCODTYP_EMBG_DIM_CNT);

        // Model *src_loc_model = new Model();
        // src_loc_model->embedding((int)locs_tbl->get_column(0)->row_cnt, LOC_EMBG_DIM_CNT);

        // Model *dst_loc_model = new Model();
        // dst_loc_model->copy(src_loc_model);
        // dst_loc_model->share_parameters(src_loc_model);

        // lm->child(actcodtyp_model, xs_tbl->get_column_range("actcodtyp"));
        // lm->child(src_loc_model, xs_tbl->get_column_range("fr_loc"));
        // lm->child(dst_loc_model, xs_tbl->get_column_range("to_loc"));

        // lm->dense(lm->calc_adjusted_input_shape(xs_tbl->get_column_cnt()), 256);
        // lm->activation(ReLU);
        // lm->dense(64);
        // lm->activation(ReLU);
        // lm->dense(1);
        // lm->activation(ReLU);
    }

    // Model 3: 26%
    {
        // Model *actcodtyp_model = new Model();
        // actcodtyp_model->embedding((int)actcodtyps_tbl->get_column(0)->row_cnt, ACTCODTYP_EMBG_DIM_CNT);
        // actcodtyp_model->dense(64);
        // actcodtyp_model->activation(Tanh);
        // actcodtyp_model->dense(1);

        // Model *src_loc_model = new Model();
        // src_loc_model->embedding((int)locs_tbl->get_column(0)->row_cnt, LOC_EMBG_DIM_CNT);

        // Model *dst_loc_model = new Model();
        // dst_loc_model->copy(src_loc_model);
        // dst_loc_model->share_parameters(src_loc_model);

        // lm->child(actcodtyp_model, xs_tbl->get_column_range("actcodtyp"));
        // lm->child(src_loc_model, xs_tbl->get_column_range("fr_loc"));
        // lm->child(dst_loc_model, xs_tbl->get_column_range("to_loc"));

        // lm->custom(lm->calc_adjusted_input_shape(xs_tbl->get_column_cnt()),
        //            get_output_shape2, forward2, backward2);
        // lm->activation(Tanh);
        // lm->dense(32);
        // lm->activation(Tanh);
        // lm->dense(8);
        // lm->activation(Tanh);
        // lm->dense(1);
    }

    // Model 4: 32%
    {
        Model *actcodtyp_model = new Model();
        actcodtyp_model->embedding((int)actcodtyps_tbl->get_column(0)->row_cnt, ACTCODTYP_EMBG_DIM_CNT);

        Model *src_loc_model = new Model();
        src_loc_model->embedding((int)locs_tbl->get_column(0)->row_cnt, LOC_EMBG_DIM_CNT);

        Model *dst_loc_model = new Model();
        dst_loc_model->copy(src_loc_model);
        dst_loc_model->share_parameters(src_loc_model);

        lm->child(actcodtyp_model, xs_tbl->get_column_range("actcodtyp"));
        lm->child(src_loc_model, xs_tbl->get_column_range("fr_loc"));
        lm->child(dst_loc_model, xs_tbl->get_column_range("to_loc"));

        lm->custom(lm->calc_adjusted_input_shape(xs_tbl->get_column_cnt()),
                   get_output_shape3, forward3, backward3);
        lm->activation(Tanh);
        lm->dense(32);
        lm->activation(Tanh);
        lm->dense(8);
        lm->activation(Tanh);
        lm->dense(1);
    }

    // ===================================================================================================

    // Fit:
    {
        lm->fit(sup, 25, 4, "temp/train.csv", upd_rslt_fn);

        Batch *test_batch = sup->create_batch();
        lm->test(test_batch, upd_rslt_fn).print();
        delete test_batch;
    }

    // ===================================================================================================

    // Test:
    {
        Column *y_col = ys_tbl->get_column("elapsed_secs");
        Column *pred_col = new Column("pred", true, xs_tbl->get_row_cnt());

        xs_tbl->clear();

        xs_tbl->add_column(actcod_col);
        xs_tbl->add_column(typ_col);
        xs_tbl->add_column(fr_loc_col);
        xs_tbl->add_column(to_loc_col);
        xs_tbl->add_column(y_col);
        xs_tbl->add_column(pred_col);

        Batch *test_batch = sup->create_batch();

        for (int i = 0; i < test_batch->get_size(); i++)
        {
            Tensor *pred = lm->forward(test_batch->get_x(i), false);
            pred_col->set_val(i, pred->get_val(0));
            delete pred;
        }

        delete test_batch;

        Table::to_csv("temp/preds.csv", xs_tbl);
    }

    // ===================================================================================================

    // Grad Check:
    {
        // Batch *grad_check_batch = sup->create_batch();
        // lm->grad_check(grad_check_batch->get_x(1), grad_check_batch->get_y(1), false);
        // delete grad_check_batch;
    }

    return 0;
}