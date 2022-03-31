#include <iostream>

#include <zero_system/mod.cuh>

#define IMAGE_ROW_CNT 28
#define IMAGE_COL_CNT 28

Supervisor *get_mnist_train_supervisor()
{
    int img_area = IMAGE_ROW_CNT * IMAGE_COL_CNT;

    int img_cnt = 60000;

    FILE *img_file = fopen("data/train-images.idx3-ubyte", "rb");
    FILE *lbl_file = fopen("data/train-labels.idx1-ubyte", "rb");

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

    FILE *img_dump_file = fopen("temp/train-images", "wb");
    FILE *lbl_dump_file = fopen("temp/train-labels", "wb");

    fwrite(img_flt_buf, sizeof(float), img_area * img_cnt, img_dump_file);
    fwrite(lbl_flt_buf, sizeof(float), img_cnt, lbl_dump_file);

    fclose(img_dump_file);
    fclose(lbl_dump_file);

    free(lbl_flt_buf);
    free(img_flt_buf);

    std::vector<int> x_shape{1, IMAGE_ROW_CNT, IMAGE_COL_CNT};
    return new Supervisor("temp/train-images", "temp/train-labels", x_shape, 10);
}

Supervisor *get_mnist_test_supervisor()
{
    int img_area = IMAGE_ROW_CNT * IMAGE_COL_CNT;

    int img_cnt = 10000;

    FILE *img_file = fopen("data/t10k-images.idx3-ubyte", "rb");
    FILE *lbl_file = fopen("data/t10k-labels.idx1-ubyte", "rb");

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

    FILE *img_dump_file = fopen("temp/test-images", "wb");
    FILE *lbl_dump_file = fopen("temp/test-labels", "wb");

    fwrite(img_flt_buf, sizeof(float), img_area * img_cnt, img_dump_file);
    fwrite(lbl_flt_buf, sizeof(float), img_cnt, lbl_dump_file);

    fclose(img_dump_file);
    fclose(lbl_dump_file);

    free(lbl_flt_buf);
    free(img_flt_buf);

    std::vector<int> x_shape{1, IMAGE_ROW_CNT, IMAGE_COL_CNT};
    return new Supervisor("temp/test-images", "temp/test-labels", x_shape, 10);
}

int main(int argc, char **argv)
{
    ZERO();

    Supervisor *train_sup = get_mnist_train_supervisor();
    Supervisor *test_sup = get_mnist_test_supervisor();

    Model *conv = new Model(CrossEntropy, 0.1f);

    conv->convolutional(train_sup->get_x_shape(), 16, 3, 3);
    conv->activation(ReLU);
    conv->pooling(Max);

    conv->convolutional(train_sup->get_x_shape(), 16, 3, 3);
    conv->activation(ReLU);
    conv->pooling(Max);

    conv->dense(128);
    conv->activation(ReLU);

    conv->dense(128);
    conv->activation(ReLU);

    conv->dense(32);
    conv->activation(ReLU);

    conv->dense(Tensor::get_cnt(train_sup->get_y_shape()));
    conv->activation(ReLU);

    conv->fit(train_sup, 64, 20, "temp/mnist-train.csv", NULL);

    Batch *test_batch = test_sup->create_batch();

    conv->test(test_batch, NULL).print();

    delete test_batch;

    conv->save("temp/mnist.nn");

    delete conv;

    // ============

    // Model *conv = new Model();
    // conv->load("temp/mnist.nn");
    // Batch *test_batch = test_sup->create_batch();
    // conv->test(test_batch, NULL).print();
    // delete test_batch;

    delete conv;

    delete train_sup;
    delete test_sup;

    return 0;
}