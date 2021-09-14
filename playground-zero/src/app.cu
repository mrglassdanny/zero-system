#include <iostream>

#include "Tensor.cuh"

int main(int argc, char **argv)
{

	Tensor *tensor = new Tensor(5, 5, 1);
	tensor->set_all_rand(5.0f);
	tensor->print();
	delete tensor;

	return 0;
}