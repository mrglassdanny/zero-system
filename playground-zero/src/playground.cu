#include <iostream>

#include <zero_system_v2/core/tensor.cuh>
#include <zero_system_v2/nn/layer.cuh>
#include <zero_system_v2/nn/model.cuh>

using namespace zero_v2::core;
using namespace zero_v2::nn;

void v2_test()
{

	Model *model = new Model(CostFunction::MSE, 0.001f);

	Tensor *x = new Tensor(Device::Cuda, 3, 8, 8);
	x->set_all_rand(0.0f, 1.0f);
	Tensor *y = new Tensor(Device::Cuda, 8);
	y->set_val(2, 1.0f);

	model->add_layer(new ConvolutionalLayer(x->get_shape(), 6, 2, 2, InitializationFunction::He));
	model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::None));

	model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 12, 2, 2, InitializationFunction::He));
	model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::None));

	model->add_layer(new ConvolutionalLayer(model->get_output_shape(), 4, 3, 3, InitializationFunction::He));
	model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::None));

	model->add_layer(new LinearLayer(model->get_output_shape(), 8, InitializationFunction::He));
	model->add_layer(new ActivationLayer(model->get_output_shape(), ActivationFunction::Sigmoid));

	model->gradient_check(x, y, true);

	delete x;
	delete y;

	delete model;
}

int main(int argc, char **argv)
{
	srand(time(NULL));

	//nn_test();

	//nn_performance_test();

	//kmeans_test();

	//cnn_test();

	v2_test();

	return 0;
}