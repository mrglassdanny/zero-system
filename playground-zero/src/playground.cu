#include <iostream>

#include <zero_system/mod.cuh>

void nn_gradient_test()
{
	ConvNet *conv = new ConvNet(CostFunction::CrossEntropy, 0.001f);

	Tensor *x = new Tensor(Device::Cuda, 3, 16, 16);
	x->set_all_rand(0.0f, 1.0f);

	Tensor *y = new Tensor(Device::Cuda, 8);
	y->set_val(2, 1.0f);

	conv->convolutional(x->get_shape(), 8, 5, 5);
	conv->activation(ActivationFunction::Sigmoid);

	conv->pooling(PoolingFunction::Max);

	conv->convolutional(4, 3, 3);
	conv->activation(ActivationFunction::Sigmoid);

	conv->pooling(PoolingFunction::Average);

	conv->linear(64);
	conv->activation(ActivationFunction::Tanh);

	conv->linear(40);
	conv->activation(ActivationFunction::Tanh);

	conv->linear(Tensor::get_cnt(y->get_shape()));
	conv->activation(ActivationFunction::Tanh);

	conv->grad_check(x, y, true);

	delete x;
	delete y;

	delete conv;
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
			delete model->backward(pred, batch->get_y(i));
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
		Tensor *xs = Tensor::fr_csv("data/nn-approx-xs.csv");
		Tensor *ys = Tensor::fr_csv("data/nn-approx-ys.csv");

		batch->add_all(xs, ys);

		delete xs;
		delete ys;
	}

	Model *model = new Model(MSE, 0.001f);

	model->linear(3, 128);
	model->activation(Sigmoid);
	model->linear(128);
	model->activation(Sigmoid);
	model->linear(1);

	model->fit(batch, NULL);

	Tensor *preds = new Tensor(Device::Cpu, batch->get_size());

	for (int i = 0; i < batch->get_size(); i++)
	{
		Tensor *pred = model->predict(batch->get_x(i));
		preds->set_val(i, pred->get_val(0));
		delete pred;
	}

	Tensor::to_csv("temp/nn-approx-preds.csv", preds);

	delete preds;

	delete model;

	delete batch;
}

void kmeans_test()
{
	Tensor *xs = Tensor::fr_csv("data/kmeans-data.csv");

	KMeans::save_best(xs, 3, 1000, "temp/model.km");

	KMeans *km = new KMeans();
	km->load("temp/model.km");

	km->print();

	delete km;

	delete xs;
}

int main(int argc, char **argv)
{
	ZERO();

	nn_gradient_test();

	return 0;
}