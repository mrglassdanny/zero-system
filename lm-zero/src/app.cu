
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

    Table *xs_tbl = Table::fr_csv("data/deeplm_data-test.csv");
    Table *ys_tbl = xs_tbl->split("elapsed_secs");

    xs_tbl->encode_ordinal("actcod");
    xs_tbl->encode_onehot("typ");

    Tensor *xs = Table::to_tensor(xs_tbl);
    Tensor *ys = Table::to_tensor(ys_tbl);

    int x_actcod_emb_idx = xs_tbl->get_column_idx("actcod");

    delete xs_tbl;
    delete ys_tbl;

    EmbeddableModel *emb_model = new EmbeddableModel();

    Embedding *actcod_emb = new Embedding(x_actcod_emb_idx);
    emb_model->embed(actcod_emb);

    emb_model->linear(xs->get_shape()[1], 64);
    emb_model->activation(ReLU);

    emb_model->linear(32);
    emb_model->activation(ReLU);

    emb_model->linear(1);
    emb_model->activation(ReLU);

    delete emb_model;

    delete xs;
    delete ys;

    return 0;
}