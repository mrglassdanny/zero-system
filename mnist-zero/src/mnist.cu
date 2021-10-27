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

    Supervisor *sup = new Supervisor(img_cnt, 784, 10, img_flt_buf, lbl_flt_buf, 1.0f, 0.0f, Device::Cpu);

    free(lbl_flt_buf);
    free(img_flt_buf);

    return sup;
}

Supervisor *get_mnist_test_supervisor()
{
    int img_area = IMAGE_ROW_CNT * IMAGE_COL_CNT;

    int img_cnt = 10000;

    FILE *img_file = fopen("C:\\Users\\d0g0825\\ML-Data\\mnist_digits\\t10k-images.idx3-ubyte", "rb");
    FILE *lbl_file = fopen("C:\\Users\\d0g0825\\ML-Data\\mnist_digits\\t10k-labels.idx1-ubyte", "rb");

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

    Supervisor *sup = new Supervisor(img_cnt, 784, 10, img_flt_buf, lbl_flt_buf, 0.0f, 1.0f, Device::Cpu);

    free(lbl_flt_buf);
    free(img_flt_buf);

    return sup;
}

void train_mnist(Model *model, Supervisor *train_sup, int train_batch_size, int target_epoch, const char *csv_path)
{
    FILE *csv_file_ptr;

    if (csv_path != nullptr)
    {
        csv_file_ptr = fopen(csv_path, "w");
        CSVUtils::write_csv_header(csv_file_ptr);
    }

    int train_total_size = train_sup->get_cnt();
    unsigned long int epoch = 0;
    unsigned long int iteration = 0;

    while (true)
    {
        Batch *train_batch = train_sup->create_train_batch(train_batch_size);
        Report train_rpt = model->train(train_batch);

        if (csv_path != nullptr)
        {
            CSVUtils::write_to_csv(csv_file_ptr, epoch, iteration, train_rpt);
        }
        else
        {
            if (iteration % 100 == 0)
            {
                printf("TRAIN\t\t");
                train_rpt.print();
            }
        }

        delete train_batch;

        // Quit if we hit target epoch count.
        if (epoch == target_epoch)
        {
            break;
        }

        // Allow for manual override.
        {
            if (_kbhit())
            {
                if (_getch() == 'q')
                {
                    break;
                }
            }
        }

        iteration++;
        epoch = ((iteration * train_batch_size) / train_total_size);
    }

    if (csv_path != nullptr)
    {
        fclose(csv_file_ptr);
    }
}

void test_mnist(Model *model, Supervisor *test_sup, Supervisor *train_sup)
{
    Batch *test_batch = test_sup->create_batch();
    Batch *train_batch = train_sup->create_batch();

    Report test_rpt = model->test(test_batch);
    Report train_rpt = model->test(train_batch);

    printf("TEST\t\t");
    test_rpt.print();

    printf("TRAIN\t\t");
    train_rpt.print();

    delete test_batch;
    delete train_batch;
}

int main(int argc, char **argv)
{
    Supervisor *train_sup = get_mnist_train_supervisor();
    Supervisor *test_sup = get_mnist_test_supervisor();

    // TRAIN NEW =======================================================================================

    // Model *model = new Model(CostFunction::CrossEntropy, 0.1f);

    // std::vector<int> n_shape{1, IMAGE_ROW_CNT, IMAGE_COL_CNT};

    // model->add_layer(new ConvolutionalLayer(n_shape, 64, 3, 3, InitializationFunction::Xavier));
    // model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    // model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 64, 3, 3, InitializationFunction::Xavier));
    // model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    // model->add_layer(new PoolingLayer(model->get_output_shape(), PoolingFunction::Max));

    // model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 64, 3, 3, InitializationFunction::Xavier));
    // model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    // model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 64, 3, 3, InitializationFunction::Xavier));
    // model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    // model->add_layer(new PoolingLayer(model->get_output_shape(), PoolingFunction::Max));

    // model->add_layer(new LinearLayer(model->get_output_shape(), 512, InitializationFunction::Xavier));
    // model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    // model->add_layer(new LinearLayer(model->get_output_shape(), 128, InitializationFunction::Xavier));
    // model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    // model->add_layer(new LinearLayer(model->get_output_shape(), 32, InitializationFunction::Xavier));
    // model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    // model->add_layer(new LinearLayer(model->get_output_shape(), Tensor::get_cnt(train_sup->get_y_shape()), InitializationFunction::Xavier));
    // model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::ReLU));

    // train_mnist(model, train_sup, 60, 30, "C:\\Users\\d0g0825\\Desktop\\temp\\mnist\\mnist.csv");

    // model->save("C:\\Users\\d0g0825\\Desktop\\temp\\mnist\\mnist.nn");

    // TRAIN EXISTING =======================================================================================

    // Model *model = new Model("C:\\Users\\d0g0825\\Desktop\\temp\\mnist\\mnist.nn");

    // train_mnist(model, train_sup, 60, 5, "C:\\Users\\d0g0825\\Desktop\\temp\\mnist\\mnist.csv");

    // model->save("C:\\Users\\d0g0825\\Desktop\\temp\\mnist\\mnist.nn");

    // TEST EXISTING =======================================================================================

    Model *model = new Model("C:\\Users\\d0g0825\\Desktop\\temp\\mnist\\mnist.nn");

    test_mnist(model, test_sup, train_sup);

    // =====================================================================================================

    delete model;
    delete train_sup;
    delete test_sup;

    return 0;
}