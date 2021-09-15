#include <iostream>

#include "NN.cuh"

int main(int argc, char **argv)
{
	srand(time(NULL));

	Tensor *x = new Tensor(1, 10, Gpu);
	x->set_all(0.5f);

	Tensor *y = new Tensor(1, 1, Gpu);
	y->set_all(1.0f);

	std::vector<int> layer_config = {10, 5, 1};
	NN *nn = new NN(layer_config, ReLU, ReLU, MSE, 0.001f);

	nn->check_gradient(x, y, true);

	delete nn;

	delete x;
	delete y;

	return 0;
}