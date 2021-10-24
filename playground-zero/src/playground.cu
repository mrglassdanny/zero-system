#include <iostream>

#include <zero_system/core/tensor.cuh>
#include <zero_system/nn/layer.cuh>
#include <zero_system/nn/model.cuh>
#include <zero_system/cluster/kmeans.cuh>

using namespace zero::core;
using namespace zero::nn;
using namespace zero::cluster;

void nn_test()
{

	Model *model = new Model(CostFunction::MSE, 0.001f);

	Tensor *x = new Tensor(Device::Cuda, 3, 8, 8);
	x->set_all_rand(0.0f, 1.0f);

	Tensor *y = new Tensor(Device::Cuda, 8);
	y->set_val(2, 1.0f);

	model->add_layer(new ConvolutionalLayer(x->get_shape(), 6, 2, 2, InitializationFunction::He));
	model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Sigmoid));

	model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 12, 2, 2, InitializationFunction::He));
	model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Sigmoid));

	model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 4, 3, 3, InitializationFunction::He));
	model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Sigmoid));

	model->add_layer(new LinearLayer(model->get_output_shape(), Tensor::get_cnt(y->get_shape()), InitializationFunction::He));
	model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Sigmoid));

	model->gradient_check(x, y, true);

	delete x;
	delete y;

	delete model;
}

void kmeans_test()
{
	Tensor *x = Tensor::from_csv("C:\\Users\\d0g0825\\Desktop\\temp\\kmeans\\data.csv");

	KMeans::save_best(x, 3, 1000, "C:\\Users\\d0g0825\\Desktop\\temp\\kmeans\\model.km");

	KMeans *km = new KMeans("C:\\Users\\d0g0825\\Desktop\\temp\\kmeans\\model.km");

	km->print();

	delete km;

	delete x;
}

int main(int argc, char **argv)
{
	srand(time(NULL));

	nn_test();

	kmeans_test();

	return 0;
}