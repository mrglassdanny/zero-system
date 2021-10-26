#include <iostream>

#include <zero_system/nn/model.cuh>

using namespace zero::core;
using namespace zero::nn;

#define IMAGE_ROW_CNT 28
#define IMAGE_COL_CNT 28

Supervisor *init_mnist_supervisor()
{
    int img_area = IMAGE_ROW_CNT * IMAGE_COL_CNT;

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

    Supervisor *sup = new Supervisor(img_cnt, 784, 10, img_flt_buf, lbl_flt_buf, Device::Cpu);

    free(lbl_flt_buf);
    free(img_flt_buf);

    return sup;
}

int main(int argc, char **argv)
{
    Supervisor *sup = init_mnist_supervisor();

    Model *model = new Model(CostFunction::CrossEntropy, 0.1f);

    std::vector<int> n_shape{1, IMAGE_ROW_CNT, IMAGE_COL_CNT};

    model->add_layer(new ConvolutionalLayer(n_shape, 32, 3, 3, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 32, 3, 3, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new PoolingLayer(model->get_output_shape(), PoolingFunction::Max));

    model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 32, 3, 3, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 32, 3, 3, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new PoolingLayer(model->get_output_shape(), PoolingFunction::Max));

    model->add_layer(new LinearLayer(model->get_output_shape(), 256, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new LinearLayer(model->get_output_shape(), 96, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new LinearLayer(model->get_output_shape(), 32, InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->add_layer(new LinearLayer(model->get_output_shape(), Tensor::get_cnt(sup->get_y_shape()), InitializationFunction::Xavier));
    model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    model->train_and_test(sup, 60, 30, "C:\\Users\\d0g0825\\Desktop\\temp\\nn\\mnist.csv");

    model->save("C:\\Users\\d0g0825\\Desktop\\temp\\nn\\mnist.nn");

    // Model *model = new Model("C:\\Users\\d0g0825\\Desktop\\temp\\nn\\mnist.nn");

    // Batch *b = sup->create_test_batch();

    // model->test(b).print();

    delete model;

    return 0;
}