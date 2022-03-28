
#include <zero_system/mod.cuh>

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

void test_loc_embedding(const char *src_loc_name, const char *dst_loc_name)
{
    std::vector<float> src_loc_tokens = loc_encode_fn(src_loc_name, 3);
    std::vector<float> dst_loc_tokens = loc_encode_fn(dst_loc_name, 3);

    Tensor *src_loc = new Tensor(Device::Cpu, 3);
    for (int i = 0; i < src_loc_tokens.size(); i++)
    {
        src_loc->set_val(i, src_loc_tokens[i]);
    }
    Tensor *dst_loc = new Tensor(Device::Cpu, 3);
    for (int i = 0; i < src_loc_tokens.size(); i++)
    {
        dst_loc->set_val(i, dst_loc_tokens[i]);
    }

    Embedding *loc_embg = new Embedding();
    loc_embg->load("temp/loc_embg.em");

    src_loc->scale_down();
    dst_loc->scale_down();

    printf("SRCLOC (%s): ", src_loc_name);
    src_loc->print();
    printf("DSTLOC (%s): ", dst_loc_name);
    dst_loc->print();

    Tensor *src_loc_p = loc_embg->forward(src_loc, false);
    Tensor *dst_loc_p = loc_embg->forward(dst_loc, false);

    printf("SRCLOC-PRED: ");
    src_loc_p->print();
    printf("DSTLOC-PRED: ");
    dst_loc_p->print();

    printf("DIFF:        ");
    src_loc_p->subtract(dst_loc_p);
    src_loc_p->print();

    printf("\n\n");

    delete src_loc_p;
    delete dst_loc_p;

    delete src_loc;
    delete dst_loc;

    delete loc_embg;
}

void fit(Table *xs_tbl, Table *ys_tbl, Supervisor *sup)
{
    EmbeddableModel *embl_model = new EmbeddableModel(MSE, 0.01f);

    int loc_embg_output_n_cnt = 9;

    Embedding *loc_embg = new Embedding();
    loc_embg->linear(3, 128);
    loc_embg->activation(LeakyReLU);
    loc_embg->linear(16);
    loc_embg->activation(LeakyReLU);
    loc_embg->linear(loc_embg_output_n_cnt);
    loc_embg->activation(LeakyReLU);

    Embedding *_loc_embg = new Embedding();
    _loc_embg->linear(3, 128);
    _loc_embg->activation(LeakyReLU);
    _loc_embg->linear(16);
    _loc_embg->activation(LeakyReLU);
    _loc_embg->linear(loc_embg_output_n_cnt);
    _loc_embg->activation(LeakyReLU);
    _loc_embg->share_parameters(loc_embg);

    EmbeddableModel *agg_embl_model = new EmbeddableModel();
    agg_embl_model->aggregation(loc_embg_output_n_cnt * 2, Subtract);

    agg_embl_model->embed(loc_embg, Range{0, 2});
    agg_embl_model->embed(_loc_embg, Range{3, 5});
    embl_model->embed(agg_embl_model, Range{xs_tbl->get_column_idx("fr_loc"), xs_tbl->get_last_column_idx("to_loc")});

    embl_model->linear(embl_model->calc_embedded_input_shape(sup->get_x_shape()), 256);
    embl_model->activation(LeakyReLU);
    embl_model->linear(256);
    embl_model->activation(LeakyReLU);
    embl_model->linear(16);
    embl_model->activation(LeakyReLU);
    embl_model->linear(1);

    embl_model->fit(sup, 256, 25, "temp/train.csv", upd_rslt_fn);

    Batch *test_batch = sup->create_batch();
    embl_model->test(test_batch, upd_rslt_fn).print();
    delete test_batch;

    embl_model->save("temp/embl.em");
    agg_embl_model->save("temp/agg_embl.em");
    loc_embg->save("temp/loc_embg.em");

    delete embl_model;
    delete agg_embl_model;
    delete loc_embg;
    delete _loc_embg;
}

void test(Supervisor *sup, Column *pred_col)
{
    EmbeddableModel *embl_model = new EmbeddableModel();
    embl_model->load("temp/embl.em");

    EmbeddableModel *agg_embl_model = new EmbeddableModel();
    agg_embl_model->load("temp/agg_embl.em");

    Embedding *loc_embg = new Embedding();
    loc_embg->load("temp/loc_embg.em");

    Embedding *_loc_embg = new Embedding();
    _loc_embg->load("temp/loc_embg.em");
    _loc_embg->share_parameters(loc_embg);

    agg_embl_model->embed(loc_embg);
    agg_embl_model->embed(_loc_embg);
    embl_model->embed(agg_embl_model);

    Batch *test_batch = sup->create_batch();
    embl_model->test(test_batch, upd_rslt_fn).print();

    for (int i = 0; i < test_batch->get_size(); i++)
    {
        Tensor *pred = embl_model->forward(test_batch->get_x(i), false);
        pred_col->set_val(i, pred->get_val(0));
        delete pred;
    }

    delete test_batch;

    delete embl_model;
    delete agg_embl_model;
    delete loc_embg;
    delete _loc_embg;
}

void grad_check(Supervisor *sup)
{
    EmbeddableModel *embl_model = new EmbeddableModel();
    embl_model->load("temp/embl.em");

    EmbeddableModel *agg_embl_model = new EmbeddableModel();
    agg_embl_model->load("temp/agg_embl.em");

    Embedding *loc_embg = new Embedding();
    loc_embg->load("temp/loc_embg.em");

    Embedding *_loc_embg = new Embedding();
    _loc_embg->load("temp/loc_embg.em");
    _loc_embg->share_parameters(loc_embg);

    agg_embl_model->embed(loc_embg);
    agg_embl_model->embed(_loc_embg);
    embl_model->embed(agg_embl_model);

    Batch *grad_chk_batch = sup->create_batch();
    Tensor *x = grad_chk_batch->get_x(0);
    Tensor *y = grad_chk_batch->get_y(0);

    embl_model->grad_check(x, y, true);

    delete grad_chk_batch;

    delete embl_model;
    delete agg_embl_model;
    delete loc_embg;
    delete _loc_embg;
}

int main(int argc, char **argv)
{
    ZERO();

    // Data setup:

    Table *xs_tbl = Table::fr_csv("data/palmov-test.csv");
    Table *ys_tbl = xs_tbl->split("elapsed_secs");

    delete xs_tbl->remove_column("cas_qty");
    delete xs_tbl->remove_column("pal_qty");
    delete xs_tbl->remove_column("cas_len");
    delete xs_tbl->remove_column("cas_wid");
    delete xs_tbl->remove_column("cas_hgt");
    delete xs_tbl->remove_column("cas_wgt");
    delete xs_tbl->remove_column("cas_per_lyr");
    delete xs_tbl->remove_column("lyr_per_pal");

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

    // Location embedding test:
    {
        // test_loc_embedding("STG383", "U278A");
        // test_loc_embedding("STG383", "STG383");
    }

    // Grad Check:
    {
        // grad_check(sup);
    }

    // Cleanup:

    delete sup;

    return 0;
}