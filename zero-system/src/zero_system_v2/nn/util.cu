#include "util.cuh"

using namespace zero_v2::core;
using namespace zero_v2::nn;

void Initializer::initialize(InitializationFunction init_fn, Tensor *out)
{
    switch (init_fn)
    {
    case InitializationFunction::He:
        out->set_all_rand(0.0f, sqrt(2.0f / n_cnt));
        break;
    case InitializationFunction::Xavier:
        out->set_all_rand(0.0f, sqrt(1.0f / n_cnt));
        break;
    case InitializationFunction::Zeros:
        out->reset();
        break;
    default:
        out->set_all_rand(0.0f, 1.0f);
        break;
    }
}