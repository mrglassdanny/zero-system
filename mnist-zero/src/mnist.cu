#include <iostream>

#include <zero_system/nn/nn.cuh>
#include <zero_system_v2/nn/model.cuh>

using namespace zero_v2::core;
using namespace zero_v2::nn;

#define IMAGE_ROW_CNT 28
#define IMAGE_COL_CNT 28

zero::nn::Supervisor *init_mnist_supervisor()
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

    zero::nn::Supervisor *sup = new Supervisor(img_cnt, 784, 10, img_flt_buf, lbl_flt_buf, Cpu);

    free(lbl_flt_buf);
    free(img_flt_buf);

    return sup;
}

void mnist_v2()
{
    zero::nn::Supervisor *sup = init_mnist_supervisor();
}

int main(int argc, char **argv)
{
    mnist_v2();

    return 0;
}