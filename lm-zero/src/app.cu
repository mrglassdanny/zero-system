
#include <zero_system/mod.cuh>

void locmst_cluster_test()
{
    Tensor *xs = Tensor::fr_csv("data/locmst-encoded.csv");

    // KMeans::run_elbow_analysis(xs, (int)(xs->get_shape()[0] * 0.05f), (int)(xs->get_shape()[0] * 0.15f), 10, "temp/elbow-analysis.csv");

    printf("MIN COST: %f\n", KMeans::save_best(xs, (int)(xs->get_shape()[0] * 0.10f), 256, "temp/model.km"));

    KMeans *model = new KMeans("temp/model.km");

    Tensor *preds = model->predict(xs);

    preds->to_csv("temp/preds.csv");

    delete preds;
    delete model;

    delete xs;
}

int main(int argc, char **argv)
{
    ZERO();

    // Data setup:

    Table *xs_tbl = Table::fr_csv("data/deeplm_data-test.csv");
    Table *ys_tbl = xs_tbl->split("elapsed_secs");

    xs_tbl->scale_down();
    ys_tbl->scale_down();

    xs_tbl->encode_ordinal("actcod");
    xs_tbl->encode_onehot("typ");

    int x_actcod_idx = xs_tbl->get_column_idx("actcod");
    int x_fr_loc_beg_idx = xs_tbl->get_column_idx("fr_loc_token_1");
    int x_fr_loc_end_idx = xs_tbl->get_column_idx("fr_loc_token_3");
    int x_to_loc_beg_idx = xs_tbl->get_column_idx("to_loc_token_1");
    int x_to_loc_end_idx = xs_tbl->get_column_idx("to_loc_token_3");

    Tensor *xs = Table::to_tensor(xs_tbl);
    Tensor *ys = Table::to_tensor(ys_tbl);

    delete xs_tbl;
    delete ys_tbl;

    Batch *batch = new Batch();
    batch->add_all(xs, ys);

    delete xs;
    delete ys;

    // Model setup:

    EmbeddedModel *m = new EmbeddedModel(MSE, 0.1f);

    Embedding *actcod_embg = new Embedding(x_actcod_idx);
    actcod_embg->linear(1, 64);
    actcod_embg->activation(ReLU);
    actcod_embg->linear(16);
    actcod_embg->activation(ReLU);
    m->embed(actcod_embg);

    Embedding *fr_loc_embg = new Embedding(x_fr_loc_beg_idx, x_fr_loc_end_idx);
    fr_loc_embg->linear(3, 128);
    fr_loc_embg->activation(ReLU);
    fr_loc_embg->linear(32);
    fr_loc_embg->activation(ReLU);
    m->embed(fr_loc_embg);

    Embedding *to_loc_embg = new Embedding(x_to_loc_beg_idx, x_to_loc_end_idx);
    to_loc_embg->linear(3, 128);
    to_loc_embg->activation(ReLU);
    to_loc_embg->linear(32);
    to_loc_embg->activation(ReLU);
    m->embed(to_loc_embg);

    m->linear(m->get_embedded_input_shape(batch->get_x_shape()), 256);
    m->activation(ReLU);

    m->linear(128);
    m->activation(ReLU);

    m->linear(32);
    m->activation(ReLU);

    m->linear(Tensor::get_cnt(batch->get_y_shape()));

    // Fit:

    m->fit(batch);

    // Cleanup:

    delete m;
    delete batch;

    return 0;
}