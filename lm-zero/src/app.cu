
#include <zero_system/mod.cuh>

#define LOC_MODEL_OUTPUT_N_CNT 8

std::vector<int> get_output_shape()
{
    std::vector<int> n_shape{1};
    return n_shape;
}

void forward(Tensor *n, Tensor *nxt_n, bool train_flg)
{
    // t = aq + c + v

    float a = n->get_val(0);
    float c = n->get_val(1);
    float q = n->get_val(2);
    float v = 0.0f;

    int src_loc_beg_idx = 3;
    int dst_loc_beg_idx = src_loc_beg_idx + LOC_MODEL_OUTPUT_N_CNT;

    for (int i = 0; i < LOC_MODEL_OUTPUT_N_CNT; i++)
    {
        float loc_diff = n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i);
        v += (loc_diff * loc_diff);
    }

    v = sqrt(v);

    float t = a * q + c + v;

    nxt_n->set_val(0, t);
}

void forward2(Tensor *n, Tensor *nxt_n, bool train_flg)
{
    // t = a + v

    float a = n->get_val(0);
    float v = 0.0f;

    int src_loc_beg_idx = 1;
    int dst_loc_beg_idx = src_loc_beg_idx + LOC_MODEL_OUTPUT_N_CNT;

    for (int i = 0; i < LOC_MODEL_OUTPUT_N_CNT; i++)
    {
        float loc_diff = n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i);
        v += (loc_diff * loc_diff);
    }

    v = sqrt(v);

    float t = a + v;

    nxt_n->set_val(0, t);
}

Tensor *backward(Tensor *n, Tensor *dc)
{
    // t = aq + c + v

    Tensor *nxt_dc = new Tensor(n->get_device(), n->get_shape());

    float dc_val = dc->get_val(0);

    nxt_dc->set_val(0, dc_val * n->get_val(2));
    nxt_dc->set_val(1, dc_val * 1.0f);
    nxt_dc->set_val(2, dc_val * n->get_val(0));

    int src_loc_beg_idx = 3;
    int dst_loc_beg_idx = src_loc_beg_idx + LOC_MODEL_OUTPUT_N_CNT;

    float v = 0.0f;

    for (int i = 0; i < LOC_MODEL_OUTPUT_N_CNT; i++)
    {
        float loc_diff = n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i);
        v += (loc_diff * loc_diff);
    }

    v = sqrt(v);

    for (int i = 0; i < LOC_MODEL_OUTPUT_N_CNT; i++)
    {
        if (v == 0.0f)
        {
            nxt_dc->set_val(src_loc_beg_idx + i, 0.0f);
            nxt_dc->set_val(dst_loc_beg_idx + i, 0.0f);
        }
        else
        {
            float dv = 1.0f / (2.0f * v);

            nxt_dc->set_val(src_loc_beg_idx + i, dc_val * dv * (2.0f * (n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i))));
            nxt_dc->set_val(dst_loc_beg_idx + i, dc_val * dv * (-2.0f * (n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i))));
        }
    }

    return nxt_dc;
}

Tensor *backward2(Tensor *n, Tensor *dc)
{
    // t = a + v

    Tensor *nxt_dc = new Tensor(n->get_device(), n->get_shape());

    float dc_val = dc->get_val(0);

    nxt_dc->set_val(0, dc_val * 1.0f);

    int src_loc_beg_idx = 1;
    int dst_loc_beg_idx = src_loc_beg_idx + LOC_MODEL_OUTPUT_N_CNT;

    float v = 0.0f;

    for (int i = 0; i < LOC_MODEL_OUTPUT_N_CNT; i++)
    {
        float loc_diff = n->get_val(src_loc_beg_idx + i) - n->get_val(dst_loc_beg_idx + i);
        v += (loc_diff * loc_diff);
    }

    v = sqrt(v);

    for (int i = 0; i < LOC_MODEL_OUTPUT_N_CNT; i++)
    {
        if (v == 0.0f)
        {
            nxt_dc->set_val(src_loc_beg_idx + i, 0.0f);
            nxt_dc->set_val(dst_loc_beg_idx + i, 0.0f);
        }
        else
        {
            float dv = 1.0f / (2.0f * v);

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

void fit(Table *xs_tbl, Table *ys_tbl, Supervisor *sup)
{
    Model *lm = new Model(0.1f);

    Model *variable_act_model = new Model();
    variable_act_model->dense(xs_tbl->get_last_column_idx("typ") - xs_tbl->get_column_idx("actcod") + 1, 32);
    variable_act_model->activation(Tanh);
    variable_act_model->dense(8);
    variable_act_model->activation(Tanh);
    variable_act_model->dense(1);
    variable_act_model->activation(Tanh);

    // Model *constant_act_model = new Model();
    // constant_act_model->copy(variable_act_model);

    Model *src_loc_model = new Model();
    src_loc_model->dense(3, 32);
    src_loc_model->activation(Tanh);
    src_loc_model->dense(16);
    src_loc_model->activation(Tanh);
    src_loc_model->dense(LOC_MODEL_OUTPUT_N_CNT);
    src_loc_model->activation(Tanh);

    Model *dst_loc_model = new Model();
    dst_loc_model->copy(src_loc_model);
    dst_loc_model->share_parameters(src_loc_model);

    lm->child(variable_act_model, Range{xs_tbl->get_column_idx("actcod"), xs_tbl->get_last_column_idx("typ")});
    // lm->child(constant_act_model, Range{xs_tbl->get_column_idx("constant_actcod"), xs_tbl->get_last_column_idx("constant_typ")});
    lm->child(src_loc_model, xs_tbl->get_column_range("fr_loc"));
    lm->child(dst_loc_model, xs_tbl->get_column_range("to_loc"));

    lm->custom(lm->calc_adjusted_input_shape(xs_tbl->get_column_cnt()),
               get_output_shape, forward2, backward2);
    lm->activation(Tanh);

    lm->fit(sup, 25, 30, "temp/train.csv", upd_rslt_fn);

    Batch *test_batch = sup->create_batch();
    lm->test(test_batch, upd_rslt_fn).print();
    delete test_batch;

    lm->save("temp/lm.model");
    variable_act_model->save("temp/vact.model");
    // constant_act_model->save("temp/cact.model");
    src_loc_model->save("temp/loc.model");

    delete lm;
    delete variable_act_model;
    // delete constant_act_model;
    delete src_loc_model;
    delete dst_loc_model;
}

// Using this fn will cause memory leak for child models!
Model *load_lm()
{
    Model *lm = new Model();
    lm->load("temp/lm.model");

    Model *variable_act_model = new Model();
    variable_act_model->load("temp/vact.model");

    // Model *constant_act_model = new Model();
    // constant_act_model->load("temp/cact.model");

    Model *src_loc_model = new Model();
    src_loc_model->load("temp/loc.model");

    Model *dst_loc_model = new Model();
    dst_loc_model->load("temp/loc.model");
    dst_loc_model->share_parameters(src_loc_model);

    lm->child(variable_act_model);
    // lm->child(constant_act_model);
    lm->child(src_loc_model);
    lm->child(dst_loc_model);

    ((CustomLayer *)lm->get_layers()[0])->set_callbacks(get_output_shape, forward2, backward2);

    return lm;
}

void test(Supervisor *sup, Column *pred_col)
{
    Model *lm = load_lm();

    Batch *test_batch = sup->create_batch();
    lm->test(test_batch, upd_rslt_fn).print();

    for (int i = 0; i < test_batch->get_size(); i++)
    {
        Tensor *pred = lm->forward(test_batch->get_x(i), false);
        pred_col->set_val(i, pred->get_val(0));
        delete pred;
    }

    delete test_batch;

    delete lm;
}

void grad_check(Table *xs_tbl, Table *ys_tbl, Supervisor *sup)
{
    Model *lm = load_lm();

    Batch *grad_check_batch = sup->create_batch();

    lm->grad_check(grad_check_batch->get_x(1), grad_check_batch->get_y(1), true);

    delete grad_check_batch;

    delete lm;
}

int main(int argc, char **argv)
{
    ZERO();

    // Data setup:

    Table *xs_tbl = Table::fr_csv("data/palmov-test.csv");
    Table *ys_tbl = xs_tbl->split("elapsed_secs");

    delete xs_tbl->remove_column("cas_per_lyr");
    delete xs_tbl->remove_column("lyr_per_pal");
    delete xs_tbl->remove_column("cas_len");
    delete xs_tbl->remove_column("cas_wid");
    delete xs_tbl->remove_column("cas_hgt");
    delete xs_tbl->remove_column("cas_wgt");
    delete xs_tbl->remove_column("cas_qty");

    delete xs_tbl->remove_column("pal_qty");

    // Column *constant_actcod_col = new Column("constant_actcod", *xs_tbl->get_column("actcod"));
    // Column *constant_typ_col = new Column("constant_typ", *xs_tbl->get_column("typ"));

    // xs_tbl->add_column(constant_actcod_col, "typ");
    // xs_tbl->add_column(constant_typ_col, "constant_actcod");

    Column *actcod_col = new Column(*xs_tbl->get_column("actcod"));
    Column *typ_col = new Column(*xs_tbl->get_column("typ"));
    Column *fr_loc_col = new Column(*xs_tbl->get_column("fr_loc"));
    Column *to_loc_col = new Column(*xs_tbl->get_column("to_loc"));

    xs_tbl->encode_onehot("actcod");
    xs_tbl->encode_onehot("typ");
    // xs_tbl->encode_onehot("constant_actcod");
    // xs_tbl->encode_onehot("constant_typ");
    xs_tbl->encode_custom("fr_loc", 3, loc_encode_fn);
    xs_tbl->encode_custom("to_loc", 3, loc_encode_fn);

    xs_tbl->scale_down();
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

    // Fit:
    {
        fit(xs_tbl, ys_tbl, sup);
    }

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

        test(sup, pred_col);

        Table::to_csv("temp/preds.csv", xs_tbl);
    }

    // Grad Check:
    {
        // grad_check(xs_tbl, ys_tbl, sup);
    }

    // Cleanup:

    delete sup;

    return 0;
}