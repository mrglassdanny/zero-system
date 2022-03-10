
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

void fit(Table *xs_tbl, Table *ys_tbl, Supervisor *sup)
{
    EmbeddedModel *embd_model = new EmbeddedModel(MSE, 0.001f);

    Embedding *loc_embg = new Embedding();
    loc_embg->linear(3, 12);
    loc_embg->activation(ReLU);

    Embedding *_loc_embg = new Embedding();
    _loc_embg->linear(3, 12);
    _loc_embg->activation(ReLU);
    _loc_embg->use_parameters(loc_embg);

    EmbeddedModel *agg_embd_model = new EmbeddedModel();
    agg_embd_model->aggregation(24, Subtract);

    agg_embd_model->embed(loc_embg, Range{0, 2});
    agg_embd_model->embed(_loc_embg, Range{3, 5});
    embd_model->embed(agg_embd_model, Range{xs_tbl->get_column_idx("fr_loc"), xs_tbl->get_last_column_idx("to_loc")});

    embd_model->linear(embd_model->calc_embedded_input_shape(sup->get_x_shape()), 256);
    embd_model->activation(ReLU);
    embd_model->linear(64);
    embd_model->activation(ReLU);
    embd_model->linear(1);

    embd_model->fit(sup, 50, 200, "temp/train.csv", upd_rslt_fn);

    Batch *test_batch = sup->create_batch();
    embd_model->test(test_batch, upd_rslt_fn).print();
    delete test_batch;

    embd_model->save("temp/embd.em");
    agg_embd_model->save("temp/agg_embd.em");
    loc_embg->save("temp/loc_embg.em");

    delete embd_model;
    delete agg_embd_model;
    delete loc_embg;
    delete _loc_embg;
}

void test(Supervisor *sup, Column *pred_col)
{
    EmbeddedModel *embd_model = new EmbeddedModel();
    embd_model->load("temp/embd.em");

    EmbeddedModel *agg_embd_model = new EmbeddedModel();
    agg_embd_model->load("temp/agg_embd.em");

    Embedding *loc_embg = new Embedding();
    loc_embg->load("temp/loc_embg.em");

    Embedding *_loc_embg = new Embedding();
    _loc_embg->linear(3, 12);
    _loc_embg->activation(ReLU);
    _loc_embg->use_parameters(loc_embg);

    agg_embd_model->embed(loc_embg);
    agg_embd_model->embed(_loc_embg);
    embd_model->embed(agg_embd_model);

    Batch *test_batch = sup->create_batch();
    embd_model->test(test_batch, upd_rslt_fn).print();

    for (int i = 0; i < test_batch->get_size(); i++)
    {
        Tensor *pred = embd_model->forward(test_batch->get_x(i), false);
        pred_col->set_val(i, pred->get_val(0));
        delete pred;
    }

    delete test_batch;

    delete embd_model;
    delete agg_embd_model;
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
        // Embedding *loc_embg = new Embedding();
        // loc_embg->load("temp/loc.emdg");

        // {
        //     // From loc:
        //     Table *fr_loc_tbl = new Table();
        //     Range fr_loc_range = xs_tbl->get_column_range("fr_loc");
        //     for (int i = fr_loc_range.beg_idx; i <= fr_loc_range.end_idx; i++)
        //     {
        //         fr_loc_tbl->add_column(xs_tbl->get_column(i)->copy());
        //     }

        //     Tensor *fr_loc_tensor = Table::to_tensor(fr_loc_tbl);

        //     Batch *fr_loc_batch = new Batch();
        //     fr_loc_batch->add_all(fr_loc_tensor, fr_loc_tensor);

        //     // To loc:
        //     Table *to_loc_tbl = new Table();
        //     Range to_loc_range = xs_tbl->get_column_range("to_loc");
        //     for (int i = to_loc_range.beg_idx; i <= to_loc_range.end_idx; i++)
        //     {
        //         to_loc_tbl->add_column(xs_tbl->get_column(i)->copy());
        //     }

        //     Tensor *to_loc_tensor = Table::to_tensor(to_loc_tbl);

        //     Batch *to_loc_batch = new Batch();
        //     to_loc_batch->add_all(to_loc_tensor, to_loc_tensor);

        //     // Test embedding predictions:
        //     for (int i = 0; i < fr_loc_batch->get_size(); i++)
        //     {
        //         Tensor *_fx = fr_loc_batch->get_x(i);
        //         Tensor *_fp = loc_embg->forward(_fx, false);
        //         _fx->print();
        //         _fp->print();

        //         Tensor *_tx = to_loc_batch->get_x(i);
        //         Tensor *_tp = loc_embg->forward(_tx, false);
        //         _tx->print();
        //         _tp->print();

        //         _fp->sub_abs(_tp);
        //         _fp->print();

        //         printf("\n=====================================================\n\n");

        //         delete _fp;
        //         delete _tp;
        //     }

        //     // Cleanup:
        //     delete fr_loc_batch;
        //     delete fr_loc_tensor;
        //     delete fr_loc_tbl;

        //     delete to_loc_batch;
        //     delete to_loc_tensor;
        //     delete to_loc_tbl;
        // }

        // delete loc_embg;
    }

    // Grad Check:
    {
        // Batch *grad_chk_batch = sup->create_batch();
        // Tensor *x = grad_chk_batch->get_x(0);
        // Tensor *y = grad_chk_batch->get_y(0);

        // EmbeddedModel *embd_model = new EmbeddedModel(MSE, 0.001f);

        // Embedding *loc_embg = new Embedding();
        // loc_embg->linear(3, 12);

        // Embedding *_loc_embg = new Embedding();
        // _loc_embg->linear(3, 12);
        // _loc_embg->use_parameters(loc_embg);

        // EmbeddedModel *agg_embd_model = new EmbeddedModel();
        // agg_embd_model->aggregation(24, Subtract);

        // agg_embd_model->embed(loc_embg, Range{0, 2});
        // agg_embd_model->embed(_loc_embg, Range{3, 5});
        // embd_model->embed(agg_embd_model, Range{xs_tbl->get_column_idx("fr_loc"), xs_tbl->get_last_column_idx("to_loc")});

        // embd_model->linear(embd_model->calc_embedded_input_shape(sup->get_x_shape()), 32);
        // embd_model->activation(Sigmoid);
        // embd_model->linear(12);
        // embd_model->activation(Sigmoid);
        // embd_model->linear(1);

        // embd_model->grad_check(x, y, true);

        // delete embd_model;
        // delete grad_chk_batch;
        // delete agg_embd_model;
        // delete loc_embg;
    }

    // Cleanup:

    delete sup;

    return 0;
}