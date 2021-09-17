#include <iostream>

#include "Supervisor.cuh"
#include "NN.cuh"

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

	std::vector<int> layer_config = {784, 128, 128, 64, 10};
	NN *nn = new NN(layer_config, ReLU, ReLU, MSE, 0.01f);

	nn->all(sup, 10, 100, "C:\\Users\\d0g0825\\Desktop\\mnist-train.csv", "C:\\Users\\d0g0825\\Desktop\\mnist-validation.csv");

	nn->dump_to_file("C:\\Users\\d0g0825\\Desktop\\cuda-mnist.nn");

	delete nn;

	delete sup;
}

void misc_test()
{
	srand(time(NULL));

	int x_col_cnt = 24;
	int y_col_cnt = 2;

	Tensor *x = new Tensor(1, x_col_cnt, Gpu);
	x->set_all(0.5f);

	Tensor *y = new Tensor(1, y_col_cnt, Gpu);
	y->set_all(0.0f);
	y->set_idx(1, 1.0f);

	std::vector<int> layer_config = {x_col_cnt, 12, 8, y_col_cnt};
	NN *nn = new NN(layer_config, ReLU, ReLU, MSE, 0.01f);

	//nn->profile(x, y);

	nn->check_gradient(x, y, false);

	nn->dump_to_file("C:\\Users\\d0g0825\\Desktop\\test.nn");

	delete nn;

	delete x;
	delete y;
}

void misc_test_2()
{
	NN *nn = new NN("C:\\Users\\d0g0825\\Desktop\\cuda-mnist.nn");

	Supervisor *sup = init_mnist_supervisor();

	ProgressReport rpt = nn->test(sup->create_test_batch());
	rpt.print();

	delete sup;

	delete nn;
}

void misc_test_3()
{
	// Tensor *t1 = new Tensor(3, 3, Cpu);
	// t1->set_all(11.0f);
	// t1->print();

	// Tensor *t2 = new Tensor(*t1);
	// t2->print();

	// t2->dump_to_csv("C:\\Users\\d0g0825\\Desktop\\tensor-test.csv");

	Tensor *t3 = Tensor::from_csv("C:\\Users\\d0g0825\\Desktop\\tensor-test.csv");
	t3->print();

	//delete t1;
	//delete t2;
	delete t3;
}

int main(int argc, char **argv)
{

	//mnist_test();
	//misc_test();
	//misc_test_2();
	misc_test_3();

	return 0;
}