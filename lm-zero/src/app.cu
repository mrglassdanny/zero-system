
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

void fit_1(Table *xs_tbl, Table *ys_tbl, Supervisor *sup, const char *path)
{
    int x_actcod_idx = xs_tbl->get_column_idx("actcod");
    int x_fr_loc_beg_idx = xs_tbl->get_column_idx("fr_loc_token_1");
    int x_fr_loc_end_idx = xs_tbl->get_column_idx("fr_loc_token_3");
    int x_to_loc_beg_idx = xs_tbl->get_column_idx("to_loc_token_1");
    int x_to_loc_end_idx = xs_tbl->get_column_idx("to_loc_token_3");

    EmbeddedModel *embd_m = new EmbeddedModel(MSE, 0.01f);

    Embedding *actcod_embg = new Embedding(x_actcod_idx);
    actcod_embg->linear(1, 8);
    actcod_embg->activation(ReLU);
    embd_m->embed(actcod_embg);

    Embedding *fr_loc_embg = new Embedding(x_fr_loc_beg_idx, x_fr_loc_end_idx);
    fr_loc_embg->linear(3, 16);
    fr_loc_embg->activation(ReLU);
    embd_m->embed(fr_loc_embg);

    Embedding *to_loc_embg = new Embedding(x_to_loc_beg_idx, x_to_loc_end_idx);
    to_loc_embg->linear(3, 16);
    to_loc_embg->activation(ReLU);
    embd_m->embed(to_loc_embg);

    embd_m->linear(embd_m->get_embedded_input_shape(sup->get_x_shape()), 256);
    embd_m->activation(ReLU);
    embd_m->linear(64);
    embd_m->activation(ReLU);
    embd_m->linear(16);
    embd_m->activation(ReLU);
    embd_m->linear(1);

    // Fit:

    embd_m->fit(sup, 100, 10, "temp/train-1.csv", upd_rslt_fn);

    Batch *test_batch = sup->create_batch();
    embd_m->test(test_batch, upd_rslt_fn).print();

    delete test_batch;

    embd_m->save(path);

    delete embd_m;
}

void fit_2(Table *xs_tbl, Table *ys_tbl, Supervisor *sup, const char *path)
{
    EmbeddedModel *embd_m = new EmbeddedModel(MSE, 0.01f);

    embd_m->linear(embd_m->get_embedded_input_shape(sup->get_x_shape()), 256);
    embd_m->activation(ReLU);
    embd_m->linear(64);
    embd_m->activation(ReLU);
    embd_m->linear(16);
    embd_m->activation(ReLU);
    embd_m->linear(1);

    // Fit:

    embd_m->fit(sup, 100, 10, "temp/train-2.csv", upd_rslt_fn);

    Batch *test_batch = sup->create_batch();
    embd_m->test(test_batch, upd_rslt_fn).print();

    delete test_batch;

    embd_m->save(path);

    delete embd_m;
}

void test(Supervisor *sup, Column *pred_col, const char *path)
{
    EmbeddedModel *embd_m = new EmbeddedModel();
    embd_m->load(path);

    Batch *test_batch = sup->create_batch();
    embd_m->test(test_batch, upd_rslt_fn).print();

    for (int i = 0; i < test_batch->get_size(); i++)
    {
        Tensor *pred = embd_m->forward(test_batch->get_x(i), false);
        pred_col->set_val(i, pred->get_val(0));
        delete pred;
    }

    delete test_batch;

    delete embd_m;
}

int main(int argc, char **argv)
{
    ZERO();

    // Data setup:

    Table *xs_tbl = Table::fr_csv("data/palpck-w-locs.csv");
    Table *ys_tbl = xs_tbl->split("elapsed_secs");

    Column *fr_loc_col = xs_tbl->remove_column("fr_loc");
    Column *to_loc_col = xs_tbl->remove_column("to_loc");

    delete xs_tbl->remove_column("cas_qty");
    delete xs_tbl->remove_column("cas_len");
    delete xs_tbl->remove_column("cas_wid");
    delete xs_tbl->remove_column("cas_hgt");
    delete xs_tbl->remove_column("cas_wgt");
    delete xs_tbl->remove_column("cas_per_lyr");
    delete xs_tbl->remove_column("lyr_per_pal");

    Column *fr_loc_token_1_col = xs_tbl->get_column("fr_loc_token_1");
    Column *fr_loc_token_2_col = xs_tbl->get_column("fr_loc_token_2");
    Column *fr_loc_token_3_col = xs_tbl->get_column("fr_loc_token_3");

    fr_loc_token_1_col->sub_abs(xs_tbl->get_column("to_loc_token_1"));
    delete xs_tbl->remove_column("to_loc_token_1");
    fr_loc_token_2_col->sub_abs(xs_tbl->get_column("to_loc_token_2"));
    delete xs_tbl->remove_column("to_loc_token_2");
    fr_loc_token_3_col->sub_abs(xs_tbl->get_column("to_loc_token_3"));
    delete xs_tbl->remove_column("to_loc_token_3");

    xs_tbl->encode_onehot("actcod");
    xs_tbl->encode_onehot("typ");

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
        // fit_1(xs_tbl, ys_tbl, sup, "temp/embd_m_1.em");
        fit_2(xs_tbl, ys_tbl, sup, "temp/embd_m_2.em");
    }

    // Test:
    // {
    //     Column *y_col = ys_tbl->get_column("elapsed_secs");
    //     Column *pred_col = new Column("pred", true, xs_tbl->get_row_cnt());

    //     xs_tbl->add_column(fr_loc_col);
    //     xs_tbl->add_column(to_loc_col);
    //     xs_tbl->add_column(y_col);
    //     xs_tbl->add_column(pred_col);

    //     test(sup, pred_col, "temp/embd_m_2.em");

    //     Table::to_csv("temp/preds-2.csv", xs_tbl);
    // }

    // Cleanup:

    delete sup;

    return 0;
}