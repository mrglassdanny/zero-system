#include <iostream>

#include "Supervisor.cuh"
#include "NN.cuh"

Supervisor *init_mnist_supervisor()
{

	int img_rows = 28;
	int img_cols = 28;
	int img_area = img_rows * img_cols;

	int img_cnt = 10000;

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

	std::vector<int> layer_config = {784, 1024, 512, 10};
	NN *nn = new NN(layer_config, ReLU, ReLU, MSE, 0.01f);

	nn->all(sup, 100, 1000, "C:\\Users\\d0g0825\\Desktop\\mnist-train.csv", "C:\\Users\\d0g0825\\Desktop\\mnist-validation.csv");

	delete nn;

	delete sup;
}

void misc_test()
{
	srand(time(NULL));

	int x_col_cnt = 1024;
	int y_col_cnt = 1;

	Tensor *x = new Tensor(1, x_col_cnt, Gpu);
	x->set_all(0.5f);

	Tensor *y = new Tensor(1, y_col_cnt, Gpu);
	y->set_all(1.0f);

	std::vector<int> layer_config = {x_col_cnt, 1024, 1024, 512, 512, 256, y_col_cnt};
	NN *nn = new NN(layer_config, None, None, MSE, 0.001f);

	for (int i = 0; i < 5; i++)
		nn->profile(x, y);

	//nn->check_gradient(x, y, true);

	delete nn;

	delete x;
	delete y;
}

int main(int argc, char **argv)
{
	//misc_test();
	mnist_test();
	return 0;
}