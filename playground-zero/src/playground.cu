#include <iostream>

#include <zero_system/nn/nn.cuh>
#include <zero_system/cluster/kmeans.cuh>

using namespace zero::core;
using namespace zero::nn;
using namespace zero::cluster;

void nn_test()
{
	int x_col_cnt = 384;
	int y_col_cnt = 1;

	Tensor *x = new Tensor(1, x_col_cnt, Gpu);
	x->set_all(0.5f);

	Tensor *y = new Tensor(1, y_col_cnt, Gpu);
	y->set_all(0.0f);
	y->set_idx(1, 1.0f);

	std::vector<int> layer_config = {x_col_cnt, 512, 256, y_col_cnt};
	NN *nn = new NN(layer_config, None, None, MSE, 0.01f);

	nn->check_gradient(x, y, false);

	delete nn;

	delete x;
	delete y;
}

void kmeans_test()
{
	Tensor *x = Tensor::from_csv("C:\\Users\\d0g0825\\Desktop\\temp\\kmeans\\data.csv");

	KMeans::dump_best(x, 3, 10000, "C:\\Users\\d0g0825\\Desktop\\temp\\kmeans\\model.km");

	KMeans *km = new KMeans("C:\\Users\\d0g0825\\Desktop\\temp\\kmeans\\model.km");

	km->print();

	delete km;

	delete x;
}

int main(int argc, char **argv)
{
	srand(time(NULL));

	//nn_test();

	//kmeans_test();

	Tensor *t1 = new Tensor(5, 5, Cpu);
	Tensor *t2 = new Tensor(5, 5, Cpu);

	t1->set_all_rand_normal_distribution(0.0f, sqrt(2.0f / 2048));
	t2->set_all_rand(2.0f / sqrt(2048));

	t1->print();
	t2->print();

	delete t1;
	delete t2;

	return 0;
}