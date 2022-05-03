
#include <zero_system/mod.cuh>

#define ACTCOD_TYP_EMBG_DIM_CNT 3
#define LOC_EMBG_DIM_CNT 12

#define METRIC_PCT 0.20f

int adj_x_col_cnt = 0;
int fr_loc_idx = 0;

std::vector<int> get_output_shape_diff()
{
    std::vector<int> n_shape{adj_x_col_cnt - LOC_EMBG_DIM_CNT};
    return n_shape;
}

void forward_diff(Tensor *n, Tensor *nxt_n, bool train_flg)
{
    for (int i = 0; i < fr_loc_idx; i++)
    {
        nxt_n->set_val(i, n->get_val(i));
    }

    for (int i = fr_loc_idx + (LOC_EMBG_DIM_CNT * 2); i < adj_x_col_cnt; i++)
    {
        nxt_n->set_val(i - LOC_EMBG_DIM_CNT, n->get_val(i));
    }

    int to_loc_idx = fr_loc_idx + LOC_EMBG_DIM_CNT;

    for (int i = 0; i < LOC_EMBG_DIM_CNT; i++)
    {
        float loc_diff = n->get_val(fr_loc_idx + i) - n->get_val(to_loc_idx + i);
        nxt_n->set_val(i + fr_loc_idx, loc_diff);
    }
}

Tensor *backward_diff(Tensor *n, Tensor *dc)
{
    Tensor *nxt_dc = new Tensor(n->get_device(), n->get_shape());

    for (int i = 0; i < fr_loc_idx; i++)
    {
        nxt_dc->set_val(i, dc->get_val(i));
    }

    for (int i = fr_loc_idx + (LOC_EMBG_DIM_CNT * 2); i < adj_x_col_cnt; i++)
    {
        nxt_dc->set_val(i, dc->get_val(i - LOC_EMBG_DIM_CNT));
    }

    int to_loc_idx = fr_loc_idx + LOC_EMBG_DIM_CNT;

    for (int i = 0; i < LOC_EMBG_DIM_CNT; i++)
    {
        nxt_dc->set_val(fr_loc_idx + i, dc->get_val(i + fr_loc_idx) * 1.0f);
        nxt_dc->set_val(to_loc_idx + i, dc->get_val(i + fr_loc_idx) * -1.0f);
    }

    return nxt_dc;
}

std::vector<int> get_output_shape_dot()
{
    std::vector<int> n_shape{adj_x_col_cnt - (LOC_EMBG_DIM_CNT * 2) + 1};
    return n_shape;
}

void forward_dot(Tensor *n, Tensor *nxt_n, bool train_flg)
{
    for (int i = 0; i < fr_loc_idx; i++)
    {
        nxt_n->set_val(i, n->get_val(i));
    }

    for (int i = fr_loc_idx + (LOC_EMBG_DIM_CNT * 2); i < adj_x_col_cnt; i++)
    {
        nxt_n->set_val(i - (LOC_EMBG_DIM_CNT * 2) + 1, n->get_val(i));
    }

    int to_loc_idx = fr_loc_idx + LOC_EMBG_DIM_CNT;

    float dot = 0.0f;

    for (int i = 0; i < LOC_EMBG_DIM_CNT; i++)
    {
        dot += n->get_val(fr_loc_idx + i) * n->get_val(to_loc_idx + i);
    }

    nxt_n->set_val(fr_loc_idx, dot);
}

Tensor *backward_dot(Tensor *n, Tensor *dc)
{
    Tensor *nxt_dc = new Tensor(n->get_device(), n->get_shape());

    for (int i = 0; i < fr_loc_idx; i++)
    {
        nxt_dc->set_val(i, dc->get_val(i));
    }

    for (int i = fr_loc_idx + (LOC_EMBG_DIM_CNT * 2); i < adj_x_col_cnt; i++)
    {
        nxt_dc->set_val(i, dc->get_val(i - (LOC_EMBG_DIM_CNT * 2) + 1));
    }

    int to_loc_idx = fr_loc_idx + LOC_EMBG_DIM_CNT;

    float val = dc->get_val(fr_loc_idx);

    for (int i = 0; i < LOC_EMBG_DIM_CNT; i++)
    {
        nxt_dc->set_val(fr_loc_idx + i, val * (n->get_val(to_loc_idx + i)));
        nxt_dc->set_val(to_loc_idx + i, val * (n->get_val(fr_loc_idx + i)));
    }

    return nxt_dc;
}

void upd_rslt_pct(Tensor *p, Tensor *y, int *cnt)
{
    float y_val = y->get_val(0);
    float p_val = p->get_val(0);

    float lower = y_val < p_val ? y_val : p_val;
    float upper = y_val < p_val ? p_val : y_val;

    float prcnt = 1.0f - (lower / upper);

    if (prcnt <= METRIC_PCT)
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

    Table *xs_tbl = Table::fr_csv("data/tasks.csv");
    Table *ys_tbl = xs_tbl->split("task_time");

    delete xs_tbl->remove_column("usr_id");
    delete xs_tbl->remove_column("cstnum");
    delete xs_tbl->remove_column("supnum");
    delete xs_tbl->remove_column("actcod");
    delete xs_tbl->remove_column("typ");
    delete xs_tbl->remove_column("begdte");
    delete xs_tbl->remove_column("enddte");
    delete xs_tbl->remove_column("devcod");
    delete xs_tbl->remove_column("cas_cnt");
    delete xs_tbl->remove_column("pal_cnt");
    delete xs_tbl->remove_column("cub");
    delete xs_tbl->remove_column("wgt");
    delete xs_tbl->remove_column("actcod_typ");

    Table *locs_tbl = Table::fr_csv("data/locs.csv");
    std::map<std::string, int> *loc_map = locs_tbl->get_column(0)->to_ordinal_map();

    // xs_tbl->encode_ordinal("actcod_typ");
    xs_tbl->encode_ordinal("fr_loc", loc_map);
    xs_tbl->encode_ordinal("to_loc", loc_map);

    // xs_tbl->get_column("cas_cnt")->scale_down();
    // xs_tbl->get_column("pal_cnt")->scale_down();
    // xs_tbl->get_column("cub")->scale_down();
    // xs_tbl->get_column("wgt")->scale_down();

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

    // Model(s):

    Model *lm = new Model(0.01f);
    // Model *actcod_typ_m = new Model();
    Model *loc_m = new Model();

    // Model 1:
    {
        // Model *loc_m_cpy = new Model();

        // // int actcod_typ_max = xs_tbl->get_column("actcod_typ")->get_max();
        // // actcod_typ_m->embedding(actcod_typ_max, ACTCOD_TYP_EMBG_DIM_CNT);

        // int loc_max = loc_map->size();
        // loc_m->embedding(loc_max, LOC_EMBG_DIM_CNT);

        // loc_m_cpy->copy(loc_m);
        // loc_m_cpy->share_parameters(loc_m);

        // // lm->child(actcod_typ_m, xs_tbl->get_column_range("actcod_typ"));
        // lm->child(loc_m, xs_tbl->get_column_range("fr_loc"));
        // lm->child(loc_m_cpy, xs_tbl->get_column_range("to_loc"));

        // adj_x_col_cnt = lm->calc_adjusted_input_shape(xs_tbl->get_column_cnt())[0];
        // fr_loc_idx = xs_tbl->get_column_idx("fr_loc");

        // lm->custom(lm->calc_adjusted_input_shape(xs_tbl->get_column_cnt()), get_output_shape_diff, forward_diff, backward_diff);
        // // lm->custom(lm->calc_adjusted_input_shape(xs_tbl->get_column_cnt()), get_output_shape_dot, forward_dot, backward_dot);
        // lm->activation(Tanh);
        // lm->dense(64);
        // lm->activation(Tanh);
        // lm->dense(16);
        // lm->activation(Tanh);
        // lm->dense(1);
    }

    // ===================================================================================================

    // Fit:
    {
        // lm->fit(sup, 200, 5, "temp/train.csv", upd_rslt_pct);

        // Batch *test_batch = sup->create_batch(1000);
        // lm->test(test_batch, upd_rslt_pct).print();
        // delete test_batch;
    }

    // ===================================================================================================

    // Predict:
    {
        // Column *pred_col = new Column("pred", true, ys_tbl->get_row_cnt());

        // ys_tbl->add_column(pred_col);

        // Batch *test_batch = sup->create_batch();

        // for (int i = 0; i < test_batch->get_size(); i++)
        // {
        //     Tensor *pred = lm->forward(test_batch->get_x(i), false);
        //     pred_col->set_val(i, pred->get_val(0));
        //     delete pred;
        // }

        // delete test_batch;

        // Table::to_csv("temp/preds.csv", ys_tbl);
    }

    // ===================================================================================================

    // Grad Check:
    {
        // Batch *grad_check_batch = sup->create_batch();
        // lm->grad_check(grad_check_batch->get_x(1), grad_check_batch->get_y(1), true);
        // delete grad_check_batch;
    }

    // lm->save("temp/lm.m");
    // // actcod_typ_m->save("temp/actcod_typ.m");
    // loc_m->save("temp/loc.m");

    loc_m->load("temp/loc.m");

    int idx = loc_map->at("STG007");
    Tensor *x = new Tensor(Device::Cpu, 1);
    x->set_val(0, idx);

    Tensor *t = loc_m->forward(x, false);
    t->print();

    idx = loc_map->at("STG008");
    x = new Tensor(Device::Cpu, 1);
    x->set_val(0, idx);

    t = loc_m->forward(x, false);
    t->print();

    return 0;
}