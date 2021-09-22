#include <iostream>

#include <zero_system/nn/nn.cuh>
#include <zero_system/cluster/kmeans.cuh>

using namespace zero::core;
using namespace zero::nn;
using namespace zero::cluster;

void nn_test()
{
	int x_col_cnt = 56;
	int y_col_cnt = 4;

	Tensor *x = new Tensor(1, x_col_cnt, Gpu);
	x->set_all(0.5f);

	Tensor *y = new Tensor(1, y_col_cnt, Gpu);
	y->set_all(0.0f);
	y->set_idx(1, 1.0f);

	std::vector<int> layer_config = {x_col_cnt, 44, 32, 16, y_col_cnt};
	NN *nn = new NN(layer_config, None, None, MSE, Xavier, 0.01f);

	nn->check_gradient(x, y, true);

	//nn->dump("C:\\Users\\d0g0825\\Desktop\\test.nn");

	delete nn;

	delete x;
	delete y;
}

void kmeans_test()
{
	Tensor *x = Tensor::from_csv("C:\\Users\\d0g0825\\Desktop\\temp\\data.csv");

	KMeans::dump_best(x, 3, 10000, "C:\\Users\\d0g0825\\Desktop\\model.km");

	delete x;
}

void kmeans_test_2()
{
	KMeans *km = new KMeans("C:\\Users\\d0g0825\\Desktop\\model.km");

	km->print();

	Tensor *x = Tensor::from_csv("C:\\Users\\d0g0825\\Desktop\\temp\\data.csv");

	delete km;
	delete x;
}

int main(int argc, char **argv)
{
	srand(time(NULL));

	// kmeans_test();

	// kmeans_test_2();

	nn_test();

	return 0;
}