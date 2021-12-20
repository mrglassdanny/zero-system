#include <iostream>

#include <zero_system/core/tensor.cuh>
#include <zero_system/nn/layer.cuh>
#include <zero_system/nn/model.cuh>
#include <zero_system/cluster/kmeans.cuh>

using namespace zero::core;
using namespace zero::nn;
using namespace zero::cluster;

void nn_gradient_test()
{
	Model *model = new Model(CostFunction::CrossEntropy, 0.001f);

	Tensor *x = new Tensor(Device::Cuda, 3, 16, 16);
	x->set_all_rand(0.0f, 1.0f);

	Tensor *y = new Tensor(Device::Cuda, 8);
	y->set_val(2, 1.0f);

	model->convolutional(x->get_shape(), 8, 5, 5);
	model->activation(ActivationFunction::Sigmoid);

	model->pooling(PoolingFunction::Max);

	model->convolutional(4, 3, 3);
	model->activation(ActivationFunction::Sigmoid);

	model->pooling(PoolingFunction::Average);

	model->linear(64);
	model->activation(ActivationFunction::Tanh);

	model->linear(40);
	model->activation(ActivationFunction::Tanh);

	model->linear(Tensor::get_cnt(y->get_shape()));
	model->activation(ActivationFunction::Tanh);

	model->gradient_check(x, y, true);

	delete x;
	delete y;

	delete model;
}

void nn_performance_test()
{
	int batch_size = 64;
	std::vector<int> x_shape{28, 28};
	std::vector<int> y_shape{1};
	Batch *batch = new Batch(batch_size);

	for (int i = 0; i < batch_size; i++)
	{
		Tensor *x = new Tensor(Device::Cuda, x_shape);
		x->set_all_rand(0.0f, 1.0f);

		Tensor *y = new Tensor(Device::Cuda, y_shape);
		y->set_all(1.0f);

		batch->add(new Record(x, y));
	}

	Model *model = new Model(CostFunction::MSE, 0.001f);

	model->linear(x_shape, 2048);
	model->activation(ActivationFunction::Sigmoid);

	model->linear(2048);
	model->activation(ActivationFunction::Sigmoid);

	model->linear(1024);
	model->activation(ActivationFunction::Sigmoid);

	model->linear(Tensor::get_cnt(y_shape));
	model->activation(ActivationFunction::Sigmoid);

	printf("SYSTEM ZERO: PERFORMANCE TEST INITIATED...\n");
	clock_t t;
	t = clock();

	for (int e = 0; e < 10; e++)
	{
		for (int i = 0; i < batch_size; i++)
		{
			Tensor *pred = model->forward(batch->get_x(i), true);
			model->backward(pred, batch->get_y(i));
			delete pred;
		}
		model->step(batch_size);
	}

	t = clock() - t;
	double time_taken = ((double)t) / CLOCKS_PER_SEC;

	printf("SYSTEM ZERO: PERFORMANCE TEST COMPLETE\n");
	printf("SYSTEM ZERO: ELAPSED SECONDS: %f\n\n", time_taken);

	delete model;
	delete batch;
}

void nn_approx_test()
{
	Batch *batch = new Batch();

	{
		Tensor *xs = Tensor::from_csv("data/nn-approx-xs.csv");
		Tensor *ys = Tensor::from_csv("data/nn-approx-ys.csv");

		batch->add_all(xs, ys);

		delete xs;
		delete ys;
	}

	Model *model = new Model(MSE, 0.000001f);

	model->linear(1, 2048);
	//model->activation(Cosine);
	model->linear(2048);
	//model->activation(Cosine);
	model->linear(2048);
	//model->activation(Cosine);
	model->linear(1);

	model->fit(batch);

	for (int i = 0; i < batch->get_size(); i++)
	{
		Tensor *pred = model->predict(batch->get_x(i));
		printf("%f\n", pred->get_val(0));
		delete pred;
	}

	delete model;

	delete batch;
}

void kmeans_test()
{
	Tensor *x = Tensor::from_csv("data\\kmeans-data.csv");

	KMeans::save_best(x, 3, 1000, "temp\\model.km");

	KMeans *km = new KMeans("temp\\model.km");

	km->print();

	delete km;

	delete x;
}

int main(int argc, char **argv)
{
	srand(time(NULL));

	nn_approx_test();

	return 0;
}