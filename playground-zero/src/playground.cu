#include <iostream>

#include <zero_system/nn/nn.cuh>
#include <zero_system/cluster/kmeans.cuh>

using namespace zero::core;
using namespace zero::nn;
using namespace zero::cluster;

void nn_test()
{
	int x_col_cnt = 64;
	int y_col_cnt = 4;

	Tensor *x = new Tensor(1, x_col_cnt, Gpu);
	x->set_all(0.5f);

	Tensor *y = new Tensor(1, y_col_cnt, Gpu);
	y->set_all(0.0f);
	y->set_idx(0, 1.0f);

	std::vector<LayerConfiguration> layer_configs = {LayerConfiguration(x_col_cnt, None, 0.0f), LayerConfiguration(56, Tanh, 0.0f), LayerConfiguration(36, None, 0.0f),
													 LayerConfiguration(24, Sigmoid, 0.0f), LayerConfiguration(12, Tanh, 0.0f), LayerConfiguration(y_col_cnt, Sigmoid, 0.0f)};
	NN *nn = new NN(layer_configs, MSE, 0.01f);

	nn->check_gradient(x, y, true);

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

void nn_performance_test()
{
	int epoch_cnt = 100;
	int batch_size = 100;

	int x_col_cnt = 832 * 2;
	int y_col_cnt = 1;

	std::vector<LayerConfiguration> layer_configs = {LayerConfiguration(x_col_cnt, None, 0.0f), LayerConfiguration(56, Tanh, 0.0f), LayerConfiguration(y_col_cnt, Sigmoid, 0.0f)};

	// -----------------------------------------------------------------

	Tensor *x = new Tensor(1, x_col_cnt, Gpu);
	x->set_all(0.5f);

	Tensor *y = new Tensor(1, y_col_cnt, Gpu);
	y->set_all(0.0f);
	y->set_idx(0, 1.0f);

	NN *nn = new NN(layer_configs, MSE, 0.01f);

	printf("Starting Performance Test...\n");
	clock_t t;
	t = clock();

	for (int i = 0; i < epoch_cnt; i++)
	{
		for (int j = 0; j < batch_size; j++)
		{
			nn->feed_forward(x);
			nn->back_propagate(y);
		}
	}

	t = clock() - t;
	double time_taken = ((double)t) / CLOCKS_PER_SEC;

	printf("Performance Test Complete!\n");
	printf("Elapsed Seconds: %f\n\n", time_taken);

	delete nn;

	delete x;
	delete y;
}

int main(int argc, char **argv)
{
	srand(time(NULL));

	nn_test();

	//kmeans_test();

	//nn_performance_test();

	return 0;
}