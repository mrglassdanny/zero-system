#include <iostream>

#include <zero_system/nn/nn.cuh>

using namespace zero::core;
using namespace zero::nn;

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

void mnist_test()
{
    srand(time(NULL));

    Supervisor *sup = init_mnist_supervisor();

    std::vector<int> layer_config = {784, 2048, 1024, 512, 256, 128, 64, 32, 10};
    NN *nn = new NN(layer_config, ReLU, ReLU, MSE, Xavier, 0.05f);

    nn->all(sup, 1000, 1000, "C:\\Users\\d0g0825\\Desktop\\mnist-train-x.csv");
    //nn->all(sup, 1000, 1000, "C:\\Users\\d0g0825\\Desktop\\mnist-train-h.csv");

    nn->dump("C:\\Users\\d0g0825\\Desktop\\cuda-mnist-xavier.nn");
    //nn->dump("C:\\Users\\d0g0825\\Desktop\\cuda-mnist-he.nn");

    //NN *nn = new NN("C:\\Users\\d0g0825\\Desktop\\temp\\cuda-zero-system-bests\\cuda-mnist.nn");

    //nn->test(sup->create_test_batch()).print();

    delete nn;

    delete sup;
}

int main(int argc, char **argv)
{
    mnist_test();

    return 0;
}