#include <iostream>

#include <zero_system/nn/model.cuh>

using namespace zero::core;
using namespace zero::nn;

#define IMAGE_ROW_CNT 28
#define IMAGE_COL_CNT 28

Supervisor *get_mnist_train_supervisor()
{
    int img_area = IMAGE_ROW_CNT * IMAGE_COL_CNT;

    int img_cnt = 60000;

    FILE *img_file = fopen("data\\train-images.idx3-ubyte", "rb");
    FILE *lbl_file = fopen("data\\train-labels.idx1-ubyte", "rb");

    fseek(img_file, sizeof(int) * 4, 0);
    unsigned char *img_buf = (unsigned char *)malloc((sizeof(unsigned char) * img_area * img_cnt));
    fread(img_buf, 1, (sizeof(unsigned char) * img_area * img_cnt), img_file);

    fseek(lbl_file, sizeof(int) * 2, 0);
    unsigned char *lbl_buf = (unsigned char *)malloc(sizeof(unsigned char) * img_cnt);
    fread(lbl_buf, 1, (sizeof(unsigned char) * img_cnt), lbl_file);

    fclose(img_file);
    fclose(lbl_file);

    float *img_flt_buf = (float *)malloc(sizeof(float) * (img_area * img_cnt));
    for (int i = 0; i < (img_area * img_cnt); i++)
    {
        img_flt_buf[i] = ((float)img_buf[i] / (255.0));
    }

    float *lbl_flt_buf = (float *)malloc(sizeof(float) * (img_cnt));
    for (int i = 0; i < (img_cnt); i++)
    {
        lbl_flt_buf[i] = ((float)lbl_buf[i]);
    }

    free(img_buf);
    free(lbl_buf);

    FILE *img_dump_file = fopen("temp\\train-images", "wb");
    FILE *lbl_dump_file = fopen("temp\\train-labels", "wb");

    fwrite(img_flt_buf, sizeof(float), img_area * img_cnt, img_dump_file);
    fwrite(lbl_flt_buf, sizeof(float), img_cnt, lbl_dump_file);

    fclose(img_dump_file);
    fclose(lbl_dump_file);

    free(lbl_flt_buf);
    free(img_flt_buf);

    std::vector<int> x_shape{1, IMAGE_ROW_CNT, IMAGE_COL_CNT};
    return new Supervisor("temp\\train-images", "temp\\train-labels", x_shape, 10);
}

Supervisor *get_mnist_test_supervisor()
{
    int img_area = IMAGE_ROW_CNT * IMAGE_COL_CNT;

    int img_cnt = 10000;

    FILE *img_file = fopen("data\\t10k-images.idx3-ubyte", "rb");
    FILE *lbl_file = fopen("data\\t10k-labels.idx1-ubyte", "rb");

    fseek(img_file, sizeof(int) * 4, 0);
    unsigned char *img_buf = (unsigned char *)malloc((sizeof(unsigned char) * img_area * img_cnt));
    fread(img_buf, 1, (sizeof(unsigned char) * img_area * img_cnt), img_file);

    fseek(lbl_file, sizeof(int) * 2, 0);
    unsigned char *lbl_buf = (unsigned char *)malloc(sizeof(unsigned char) * img_cnt);
    fread(lbl_buf, 1, (sizeof(unsigned char) * img_cnt), lbl_file);

    fclose(img_file);
    fclose(lbl_file);

    float *img_flt_buf = (float *)malloc(sizeof(float) * (img_area * img_cnt));
    for (int i = 0; i < (img_area * img_cnt); i++)
    {
        img_flt_buf[i] = ((float)img_buf[i] / (255.0));
    }

    float *lbl_flt_buf = (float *)malloc(sizeof(float) * (img_cnt));
    for (int i = 0; i < (img_cnt); i++)
    {
        lbl_flt_buf[i] = ((float)lbl_buf[i]);
    }

    free(img_buf);
    free(lbl_buf);

    FILE *img_dump_file = fopen("temp\\test-images", "wb");
    FILE *lbl_dump_file = fopen("temp\\test-labels", "wb");

    fwrite(img_flt_buf, sizeof(float), img_area * img_cnt, img_dump_file);
    fwrite(lbl_flt_buf, sizeof(float), img_cnt, lbl_dump_file);

    fclose(img_dump_file);
    fclose(lbl_dump_file);

    free(lbl_flt_buf);
    free(img_flt_buf);

    std::vector<int> x_shape{1, IMAGE_ROW_CNT, IMAGE_COL_CNT};
    return new Supervisor("temp\\test-images", "temp\\test-labels", x_shape, 10);
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    Supervisor *train_sup = get_mnist_train_supervisor();
    Supervisor *test_sup = get_mnist_test_supervisor();

    Model *model = new Model(CostFunction::CrossEntropy, 0.01f);

    model->convolutional(train_sup->get_x_shape(), 64, 3, 3);
    model->activation(ActivationFunction::ReLU);

    model->convolutional(64, 3, 3);
    model->activation(ActivationFunction::ReLU);

    model->pooling(PoolingFunction::Max);

    model->linear(1024);
    model->activation(ActivationFunction::ReLU);

    model->linear(1024);
    model->activation(ActivationFunction::ReLU);

    model->linear(256);
    model->activation(ActivationFunction::ReLU);

    model->linear(Tensor::get_cnt(train_sup->get_y_shape()));
    model->activation(ActivationFunction::ReLU);

    model->fit(train_sup, 64, 5, "temp\\mnist-train.csv");

    Batch *test_batch = test_sup->create_batch();

    model->test(test_batch).print();

    delete test_batch;

    model->save("temp\\mnist.nn");

    delete model;

    delete train_sup;
    delete test_sup;

    return 0;
}