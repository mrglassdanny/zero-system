#include <iostream>

#include <zero_system/nn/nn.cuh>

using namespace zero::core;
using namespace zero::nn;

void misc_test()
{
	srand(time(NULL));

	int x_col_cnt = 48;
	int y_col_cnt = 2;

	Tensor *x = new Tensor(1, x_col_cnt, Gpu);
	x->set_all(0.5f);

	Tensor *y = new Tensor(1, y_col_cnt, Gpu);
	y->set_all(0.0f);
	y->set_idx(1, 1.0f);

	std::vector<int> layer_config = {x_col_cnt, 128, 44, 8, y_col_cnt};
	NN *nn = new NN(layer_config, ReLU, ReLU, MSE, 0.01f);

	//nn->check_performance(x, y);

	nn->check_gradient(x, y, true);

	//nn->dump("C:\\Users\\d0g0825\\Desktop\\test.nn");

	delete nn;

	delete x;
	delete y;
}

int main(int argc, char **argv)
{
	misc_test();

	return 0;
}