
#include <zero_system/mod.cuh>

class LMModel : public Model
{
private:
    /*
        t = v/s + aq + C

        t: task time
        v: travel distance
        s: speed
        a: variable activity time
        q: quantity
        C: constant activity time
    */

public:
    LMModel()
        : Model()
    {
    }

    LMModel(CostFunction cost_fn, float learning_rate)
        : Model(cost_fn, learning_rate)
    {
    }

    ~LMModel()
    {
    }

    virtual Tensor *forward(Tensor *x, bool train_flg)
    {
        // Location embeddings:
    }

    virtual Tensor *backward(Tensor *pred, Tensor *y)
    {
    }
};

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

void hmmmm(Table *xs_tbl, Table *ys_tbl, Supervisor *sup)
{
    /*
            t = v/s + aq + C

            t: task time (sec)
            v: travel distance (?)
            s: speed (?)
            a: variable activity time (sec)
            q: quantity (units)
            C: constant activity time (sec)
    */

    Embedding *loc_embg = new Embedding();
    loc_embg->linear(3, 32);
    loc_embg->activation(Sigmoid);
    loc_embg->linear(8);
    loc_embg->activation(Sigmoid);
    loc_embg->aggregation();

    Embedding *_loc_embg = new Embedding();
    _loc_embg->linear(3, 32);
    _loc_embg->activation(Sigmoid);
    _loc_embg->linear(8);
    _loc_embg->activation(Sigmoid);
    _loc_embg->aggregation();
    _loc_embg->share_parameters(loc_embg);

    float v = 0.0f;

    Batch *b = sup->create_batch();

    delete loc_embg;
    delete _loc_embg;
}

void fit(Table *xs_tbl, Table *ys_tbl, Supervisor *sup)
{
    LMModel *lm = new LMModel(MSE, 0.001f);

    Embedding *actcod_embg = new Embedding();
    actcod_embg->linear(xs_tbl->get_last_column_idx("actcod") - xs_tbl->get_column_idx("actcod") + 1, 3);
    actcod_embg->activation(Sigmoid);
    lm->embed(actcod_embg, xs_tbl->get_column_range("actcod"));

    lm->linear(Model::calc_embedded_input_shape(lm, xs_tbl->get_column_cnt()), 32);
    lm->activation(Sigmoid);
    lm->linear(1);

    lm->fit(sup, 128, 25, "temp/train.csv", upd_rslt_fn);

    delete actcod_embg;
    delete lm;
}

void test(Supervisor *sup, Column *pred_col)
{
}

void grad_check(Table *xs_tbl, Table *ys_tbl, Supervisor *sup)
{
    LMModel *lm = new LMModel(MSE, 0.001f);

    Embedding *actcod_embg = new Embedding();
    actcod_embg->linear(xs_tbl->get_last_column_idx("actcod") - xs_tbl->get_column_idx("actcod") + 1, 3);
    actcod_embg->activation(Sigmoid);
    lm->embed(actcod_embg, xs_tbl->get_column_range("actcod"));

    lm->linear(Model::calc_embedded_input_shape(lm, xs_tbl->get_column_cnt()), 32);
    lm->activation(Sigmoid);
    lm->linear(1);

    Batch *b = sup->create_batch();

    lm->grad_check(b->get_x(0), b->get_y(0), true);

    delete b;

    delete actcod_embg;
    delete lm;
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
        // fit(xs_tbl, ys_tbl, sup);
    }

    // Test:
    {
        // Column *y_col = ys_tbl->get_column("elapsed_secs");
        // Column *pred_col = new Column("pred", true, xs_tbl->get_row_cnt());

        // xs_tbl->add_column(actcod_col);
        // xs_tbl->add_column(fr_loc_col);
        // xs_tbl->add_column(to_loc_col);
        // xs_tbl->add_column(y_col);
        // xs_tbl->add_column(pred_col);

        // test(sup, pred_col);

        // Table::to_csv("temp/preds.csv", xs_tbl);
    }

    // Location embedding test:
    {
        // test_loc_embedding("STG383", "U278A");
        // test_loc_embedding("STG383", "STG383");
    }

    // Grad Check:
    {
        grad_check(xs_tbl, ys_tbl, sup);
    }

    // Cleanup:

    delete sup;

    return 0;
}