#include <iostream>

#include <zero_system/nn/nn.cuh>
#include <zero_system/nn/cnn.cuh>

using namespace zero::core;
using namespace zero::nn;

#define IMAGE_ROW_CNT 28
#define IMAGE_COL_CNT 28

Supervisor *init_mnist_supervisor()
{

    int img_rows = 28;
    int img_cols = 28;
    int img_area = img_rows * img_cols;

    int img_cnt = 60000;

    FILE *img_file = fopen("C:\\Users\\d0g0825\\ML-Data\\mnist_digits\\train-images.idx3-ubyte", "rb");
    FILE *lbl_file = fopen("C:\\Users\\d0g0825\\ML-Data\\mnist_digits\\train-labels.idx1-ubyte", "rb");

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

    Supervisor *sup = new Supervisor(img_cnt, 784, 10, img_flt_buf, lbl_flt_buf, Cpu);

    free(lbl_flt_buf);
    free(img_flt_buf);

    return sup;
}

void mnist_nn()
{
    srand(time(NULL));

    Supervisor *sup = init_mnist_supervisor();

    NN *nn = new NN(MSE, 0.1f);

    nn->add_layer(784);
    nn->add_layer(2048, ReLU, 0.3f);
    nn->add_layer(1024, ReLU, 0.25f);
    nn->add_layer(512, ReLU, 0.2f);
    nn->add_layer(10, Sigmoid);

    nn->compile();

    nn->train_and_test(sup, 100, "C:\\Users\\d0g0825\\Desktop\\temp\\nn\\mnist-train-dr.csv");

    nn->save("C:\\Users\\d0g0825\\Desktop\\temp\\nn\\mnist-dr.nn");

    //NN *nn = new NN("C:\\Users\\d0g0825\\Desktop\\temp\\nn\\mnist-dr.nn");

    //nn->train_and_test(sup, 100, "C:\\Users\\d0g0825\\Desktop\\temp\\nn\\mnist-train-dr.csv");

    //nn->save("C:\\Users\\d0g0825\\Desktop\\temp\\nn\\mnist-dr.nn");

    //nn->test(sup->create_test_batch()).print();

    delete nn;

    delete sup;
}

void mnist_cnn()
{
    srand(time(NULL));

    Supervisor *sup = init_mnist_supervisor();

    CNN *cnn = new CNN(MSE, 0.1f);

    cnn->input_layer(1, IMAGE_ROW_CNT, IMAGE_COL_CNT, 64, 3, 3, None);
    cnn->add_layer(64, 3, 3, ReLU);

    cnn->flatten(ReLU);

    cnn->fully_connected()->add_layer(512, ReLU);
    cnn->fully_connected()->add_layer(10, ReLU);

    cnn->compile();

    cnn->train_and_test(sup, 100, "C:\\Users\\d0g0825\\Desktop\\cnn-mnist-train.csv");

    cnn->save("C:\\Users\\d0g0825\\Desktop\\mnist.cnn");

    delete cnn;

    delete sup;
}

int main(int argc, char **argv)
{
    //mnist_nn();

    mnist_cnn();

    return 0;
}