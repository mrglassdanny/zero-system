#include "util.cuh"

using namespace zero_v2::core;
using namespace zero_v2::nn;

void Initializer::initialize(InitializationFunction init_fn, Tensor *tensor,
                             int fan_in, int fan_out)
{
    switch (init_fn)
    {
    case InitializationFunction::He:
        tensor->set_all_rand(0.0f, sqrt(2.0f / fan_in));
        break;
    case InitializationFunction::Xavier:
        tensor->set_all_rand(0.0f, sqrt(1.0f / fan_in));
        break;
    case InitializationFunction::Zeros:
        tensor->reset();
        break;
    default:
        tensor->set_all_rand(0.0f, 1.0f);
        break;
    }
}