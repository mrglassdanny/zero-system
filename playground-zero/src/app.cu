#include <iostream>

#include "Tensor.cuh"
#include "NN.cuh"

int main(int argc, char **argv)
{

	Tensor *tensor = new Tensor(5, 5, Gpu);
	tensor->set_all_rand(5.0f);
	tensor->print();
	delete tensor;

	std::vector<int> layer_config = {10, 5, 1};
	NN *nn = new NN(layer_config, ReLU, ReLU, MSE, 0.001f);
	delete nn;

	return 0;
}