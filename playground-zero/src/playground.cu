#include <iostream>

#include <zero_system/nn/nn.cuh>
#include <zero_system/nn/cnn.cuh>
#include <zero_system/cluster/kmeans.cuh>

using namespace zero::core;
using namespace zero::nn;
using namespace zero::cluster;

void nn_test()
{
	int x_col_cnt = 126;
	int y_col_cnt = 4;

	Tensor *x = new Tensor(1, x_col_cnt, Gpu);
	x->set_all(0.5f);

	Tensor *y = new Tensor(1, y_col_cnt, Gpu);
	y->set_all(0.0f);
	y->set_val(0, 1.0f);

	NN *nn = new NN(MSE, 0.01f);

	nn->add_layer(x_col_cnt, 0.2f);
	nn->add_layer(90, None, 0.5f);
	nn->add_layer(28, None, 0.35f);
	nn->add_layer(y_col_cnt, Tanh);

	nn->compile();

	nn->check_gradient(x, y, true);

	delete nn;

	delete x;
	delete y;
}

void nn_performance_test()
{
	int epoch_cnt = 100;
	int batch_size = 100;

	int x_col_cnt = 832 * 2;
	int y_col_cnt = 1;

	NN *nn = new NN(MSE, 0.01f);
	nn->add_layer(x_col_cnt);
	nn->add_layer(416, Sigmoid);
	nn->add_layer(y_col_cnt, Sigmoid);
	nn->compile();

	// -----------------------------------------------------------------

	Tensor *x = new Tensor(1, x_col_cnt, Gpu);
	x->set_all(0.5f);

	Tensor *y = new Tensor(1, y_col_cnt, Gpu);
	y->set_all(0.0f);
	y->set_val(0, 1.0f);

	printf("Starting Performance Test...\n");
	clock_t t;
	t = clock();

	for (int i = 0; i < epoch_cnt; i++)
	{
		for (int j = 0; j < batch_size; j++)
		{
			nn->feed_forward(x, false);
			nn->back_propagate(y, false);
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

void kmeans_test()
{
	Tensor *x = Tensor::from_csv("C:\\Users\\d0g0825\\Desktop\\temp\\kmeans\\data.csv");

	KMeans::dump_best(x, 3, 10000, "C:\\Users\\d0g0825\\Desktop\\temp\\kmeans\\model.km");

	KMeans *km = new KMeans("C:\\Users\\d0g0825\\Desktop\\temp\\kmeans\\model.km");

	km->print();

	delete km;

	delete x;
}

void cnn_test()
{
	int x_col_cnt = 2 * 16 * 16;
	int y_col_cnt = 4;

	Tensor *x = new Tensor(1, x_col_cnt, Gpu);
	x->set_all_rand(1.0f);

	Tensor *y = new Tensor(1, y_col_cnt, Gpu);
	y->set_all(0.0f);
	y->set_val(0, 1.0f);

	CNN *cnn = new CNN(MSE, 0.001f);
	cnn->add_layer(2, 16, 16, 4, 4, 4, Sigmoid);
	cnn->add_layer(3, 2, 2, None);
	cnn->add_layer(None);
	cnn->compile();

	cnn->fully_connected()->add_layer(24, Sigmoid);
	cnn->fully_connected()->add_layer(y_col_cnt, Sigmoid);
	cnn->fully_connected()->compile();

	cnn->check_gradient(x, y, true);

	delete cnn;

	delete x;
	delete y;
}

int main(int argc, char **argv)
{
	srand(time(NULL));

	//nn_test();

	//nn_performance_test();

	//kmeans_test();

	cnn_test();

	return 0;
}