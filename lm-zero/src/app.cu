
#include <zero_system/mod.cuh>

int main(int argc, char **argv)
{
    ZERO();

    // Data setup:

    Table *xs_tbl = Table::fr_csv("data/test.csv");
    Table *ys_tbl = xs_tbl->split("elapsed_secs");

    // delete xs_tbl->remove_column("cas_qty");
    // delete xs_tbl->remove_column("cas_len");
    // delete xs_tbl->remove_column("cas_wid");
    // delete xs_tbl->remove_column("cas_hgt");
    // delete xs_tbl->remove_column("cas_wgt");
    // delete xs_tbl->remove_column("pal_qty");

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

    Tensor::to_file("temp/xs.tr", xs);
    Tensor::to_file("temp/ys.tr", ys);

    std::vector<int> x_shape{xs->get_shape()[1]};
    Supervisor *sup = new Supervisor("temp/xs.tr", "temp/ys.tr", x_shape, 0);

    delete xs;
    delete ys;

    // Model setup:

    EmbeddedModel *embd_m = new EmbeddedModel(MSE, 0.001f);

    Embedding *actcod_embg = new Embedding(x_actcod_idx);
    actcod_embg->linear(1, 16);
    actcod_embg->activation(Sigmoid);
    embd_m->embed(actcod_embg);

    Embedding *fr_loc_embg = new Embedding(x_fr_loc_beg_idx, x_fr_loc_end_idx);
    fr_loc_embg->linear(3, 16);
    fr_loc_embg->activation(Sigmoid);
    embd_m->embed(fr_loc_embg);

    Embedding *to_loc_embg = new Embedding(x_to_loc_beg_idx, x_to_loc_end_idx);
    to_loc_embg->linear(3, 16);
    to_loc_embg->activation(Sigmoid);
    embd_m->embed(to_loc_embg);

    embd_m->linear(embd_m->get_embedded_input_shape(sup->get_x_shape()), 16);
    embd_m->activation(Sigmoid);
    embd_m->linear(1);

    // Fit:

    Batch *grad_batch = sup->create_batch();

    embd_m->check_grad(grad_batch->get_x(0), grad_batch->get_y(0), true);

    // embd_m->fit(sup, 100, 10, "temp/train.csv", NULL);

    // Batch *test_batch = sup->create_batch(100);
    // embd_m->test(test_batch, NULL).print();

    // for (int i = 0; i < test_batch->get_size(); i++)
    // {
    //     Tensor *pred = embd_m->forward(test_batch->get_x(i), false);
    //     test_batch->get_x(i)->print();
    //     test_batch->get_y(i)->print();
    //     pred->print();

    //     delete pred;
    // }

    // delete test_batch;

    // Cleanup:

    delete embd_m;
    delete sup;

    return 0;
}