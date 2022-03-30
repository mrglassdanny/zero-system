
#include <zero_system/mod.cuh>

std::vector<int> get_output_shape()
{
    std::vector<int> n_shape{1};
    return n_shape;
}

void forward(Tensor *n, Tensor *nxt_n, bool train_flg)
{
    // t = aq + v
    // a: 0
    // q: 1
    // v: 2 - 3

    float a = n->get_val(0);
    float q = n->get_val(1);
    float v = n->get_val(2) - n->get_val(3);

    float t = a * q + v;

    nxt_n->set_val(0, t);
}

Tensor *backward(Tensor *n, Tensor *dc)
{
    // t = aq + v

    Tensor *nxt_dc = new Tensor(n->get_device(), n->get_shape());

    nxt_dc->set_val(0, dc->get_val(0) * n->get_val(1));
    nxt_dc->set_val(1, dc->get_val(0) * n->get_val(0));
    nxt_dc->set_val(2, dc->get_val(0) * 1.0f);
    nxt_dc->set_val(3, dc->get_val(0) * -1.0f);

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
    Model *lm = new Model(MSE, 0.001f);

    Model *variable_actcod_embg = new Model();
    variable_actcod_embg->linear(xs_tbl->get_last_column_idx("typ") - xs_tbl->get_column_idx("actcod") + 1, 256);
    variable_actcod_embg->activation(ReLU);
    variable_actcod_embg->linear(64);
    variable_actcod_embg->activation(ReLU);
    variable_actcod_embg->linear(16);
    variable_actcod_embg->activation(ReLU);
    variable_actcod_embg->linear(1);
    variable_actcod_embg->activation(ReLU);

    Model *src_loc_embg = new Model();
    src_loc_embg->linear(xs_tbl->get_last_column_idx("fr_loc") - xs_tbl->get_column_idx("fr_loc") + 1, 128);
    src_loc_embg->activation(ReLU);
    src_loc_embg->linear(32);
    src_loc_embg->activation(ReLU);
    src_loc_embg->linear(8);
    src_loc_embg->activation(ReLU);
    src_loc_embg->aggregation();
    // src_loc_embg->linear(1);
    // src_loc_embg->activation(ReLU);

    Model *dst_loc_embg = new Model();
    dst_loc_embg->linear(xs_tbl->get_last_column_idx("to_loc") - xs_tbl->get_column_idx("to_loc") + 1, 128);
    dst_loc_embg->activation(ReLU);
    dst_loc_embg->linear(32);
    dst_loc_embg->activation(ReLU);
    dst_loc_embg->linear(8);
    dst_loc_embg->activation(ReLU);
    dst_loc_embg->aggregation();
    // dst_loc_embg->linear(1);
    // dst_loc_embg->activation(ReLU);
    dst_loc_embg->share_parameters(src_loc_embg);

    lm->embed(variable_actcod_embg, Range{xs_tbl->get_column_idx("actcod"), xs_tbl->get_last_column_idx("typ")});
    lm->embed(src_loc_embg, xs_tbl->get_column_range("fr_loc"));
    lm->embed(dst_loc_embg, xs_tbl->get_column_range("to_loc"));

    lm->custom(Model::calc_embedded_input_shape(lm, xs_tbl->get_column_cnt()),
               get_output_shape, forward, backward);
    lm->activation(ReLU);

    lm->fit(sup, 200, 15, "temp/train.csv", upd_rslt_fn);

    Batch *test_batch = sup->create_batch(1000);
    lm->test(test_batch, upd_rslt_fn).print();
    delete test_batch;

    lm->save("temp/lm.nn");
    variable_actcod_embg->save("temp/vact.em");
    src_loc_embg->save("temp/loc.em");

    delete lm;
    delete variable_actcod_embg;
    delete src_loc_embg;
    delete dst_loc_embg;
}

void test(Supervisor *sup, Column *pred_col)
{
    Model *lm = new Model();
    lm->load("temp/lm.nn");

    Model *variable_actcod_embg = new Model();
    variable_actcod_embg->load("temp/vact.em");

    Model *src_loc_embg = new Model();
    src_loc_embg->load("temp/loc.em");

    Model *dst_loc_embg = new Model();
    dst_loc_embg->load("temp/loc.em");
    dst_loc_embg->share_parameters(src_loc_embg);

    lm->embed(variable_actcod_embg);
    lm->embed(src_loc_embg);
    lm->embed(dst_loc_embg);

    ((CustomLayer *)lm->get_layers()[0])->set_callbacks(get_output_shape, forward, backward);

    Batch *test_batch = sup->create_batch(1000);
    lm->test(test_batch, upd_rslt_fn).print();

    for (int i = 0; i < test_batch->get_size(); i++)
    {
        Tensor *pred = lm->forward(test_batch->get_x(i), false);
        pred_col->set_val(i, pred->get_val(0));
        delete pred;
    }

    delete test_batch;

    delete lm;
    delete variable_actcod_embg;
    delete src_loc_embg;
    delete dst_loc_embg;
}

void grad_check(Table *xs_tbl, Table *ys_tbl, Supervisor *sup)
{
    Model *lm = new Model();
    lm->load("temp/lm.nn");

    Model *variable_actcod_embg = new Model();
    variable_actcod_embg->load("temp/vact.em");

    Model *src_loc_embg = new Model();
    src_loc_embg->load("temp/loc.em");

    Model *dst_loc_embg = new Model();
    dst_loc_embg->load("temp/loc.em");
    dst_loc_embg->share_parameters(src_loc_embg);

    lm->embed(variable_actcod_embg);
    lm->embed(src_loc_embg);
    lm->embed(dst_loc_embg);

    ((CustomLayer *)lm->get_layers()[0])->set_callbacks(get_output_shape, forward, backward);

    Batch *grad_check_batch = sup->create_batch();

    lm->grad_check(grad_check_batch->get_x(1), grad_check_batch->get_y(1), true);

    delete grad_check_batch;

    delete lm;
    delete variable_actcod_embg;
    delete src_loc_embg;
    delete dst_loc_embg;
}

int main(int argc, char **argv)
{
    ZERO();

    // Data setup:

    Table *xs_tbl = Table::fr_csv("data/palmov.csv");
    Table *ys_tbl = xs_tbl->split("elapsed_secs");

    delete xs_tbl->remove_column("cas_per_lyr");
    delete xs_tbl->remove_column("lyr_per_pal");
    delete xs_tbl->remove_column("cas_len");
    delete xs_tbl->remove_column("cas_wid");
    delete xs_tbl->remove_column("cas_hgt");
    delete xs_tbl->remove_column("cas_wgt");
    delete xs_tbl->remove_column("cas_qty");

    Column *actcod_col = xs_tbl->get_column("actcod")->copy();
    Column *fr_loc_col = xs_tbl->get_column("fr_loc")->copy();
    Column *to_loc_col = xs_tbl->get_column("to_loc")->copy();

    xs_tbl->encode_onehot("actcod");
    xs_tbl->encode_onehot("typ");
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

        xs_tbl->add_column(actcod_col);
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