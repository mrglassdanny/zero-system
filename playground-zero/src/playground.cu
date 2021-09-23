#include <iostream>

#include <zero_system/nn/nn.cuh>
#include <zero_system/cluster/kmeans.cuh>

using namespace zero::core;
using namespace zero::nn;
using namespace zero::cluster;

void nn_test()
{
	int x_col_cnt = 16;
	int y_col_cnt = 2;

	Tensor *x = new Tensor(1, x_col_cnt, Gpu);
	x->set_all(0.5f);

	Tensor *y = new Tensor(1, y_col_cnt, Gpu);
	y->set_all(0.0f);
	y->set_idx(1, 1.0f);

	std::vector<int> layer_config = {x_col_cnt, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, y_col_cnt};
	NN *nn = new NN(layer_config, None, None, MSE, Xavier, 0.01f);

	nn->check_gradient(x, y, true);

	delete nn;

	delete x;
	delete y;
}

void nn_perf_test()
{

	int x_col_cnt = 2048;
	int y_col_cnt = 2;

	Tensor *x = new Tensor(1, x_col_cnt, Gpu);
	x->set_all(0.5f);

	Tensor *y = new Tensor(1, y_col_cnt, Gpu);
	y->set_all(0.0f);
	y->set_idx(0, 1.0f);

	std::vector<int> lyr_cfg;
	lyr_cfg.push_back(512);
	for (int i = 0; i < 4; i++)
	{
		lyr_cfg.push_back(x_col_cnt);
	}
	lyr_cfg.push_back(1024);
	lyr_cfg.push_back(64);
	lyr_cfg.push_back(y_col_cnt);

	NN *nn = new NN(lyr_cfg, None, None, MSE, Xavier, 0.01f);

	for (int i = 0; i < 5; i++)
	{
		clock_t t;
		t = clock();

		for (int j = 0; j < 10; j++)
		{
			nn->feed_forward(x);
			nn->back_propagate(y);
		}

		t = clock() - t;
		double time_taken = ((double)t) / CLOCKS_PER_SEC;

		printf("Elapsed Seconds: %f\n\n", time_taken);
	}

	delete nn;
	delete x;
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

	nn_perf_test();

	//kmeans_test();

	return 0;
}