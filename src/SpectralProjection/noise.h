#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <utility>
#include <wtypes.h>

enum NoiseType
{
    POISSON,
    NORMAL
};

class RadonNoiseGenerator {
    curandState* states = nullptr;
    unsigned int seed;

public:
    RadonNoiseGenerator(unsigned int _seed);

    void set_seed(const unsigned int seed);

    curandState* get();

    void add_noise(float* projection, const int dose, const enum NoiseType noiseType,
        const int width, const int height);


    void free();

    ~RadonNoiseGenerator();
};
