#pragma once
#include "noise.h"
#include <iostream>

__global__ void initialize_random_states(curandState *state, const int seed) {
    const int sequence_id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, sequence_id, 0, &state[sequence_id]);
}

__global__ void
radon_sinogram_noise(float *projection, curandState *state,
                     const unsigned int width, const unsigned int height, int dose, enum NoiseType noiseType) {


    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid = y * blockDim.x * gridDim.x + x;
    const int y_step = blockDim.y * gridDim.y;
    float N;

    if (tid < 128 * 1024) {
        // load curand state in local memory
        curandState localState = state[tid];

        // loop down the sinogram adding noise
        for (int yy = y; yy < height; yy += y_step) {
            if (x < width) {
                int pos = yy * width + x;

                switch (noiseType)
                {
                case POISSON:
                    N = fmaxf(curand_poisson(&localState, projection[pos]), 1.0f);
                    break;
                case NORMAL:
                    N = fmaxf(curand_poisson(&localState, projection[pos]), 1.0f);
                default:
                    break;
                }

                projection[pos] = log(dose/N);
            }
        }

        state[tid] = localState;
    }

}



RadonNoiseGenerator::RadonNoiseGenerator(unsigned int _seed) : seed(_seed) {
    this->states = (curandState *) malloc(sizeof(curandState * ));
    this->states = nullptr;
}


void RadonNoiseGenerator::set_seed(const unsigned int seed) {
    initialize_random_states << < 128, 1024 >> > (this->get(), seed);
}


curandState *RadonNoiseGenerator::get() {
    if (this->states == nullptr) {
#ifdef VERBOSE
        std::cout << "[TORCH RADON] Allocating Random states on device " << device << std::endl;
#endif

        // allocate random states
        cudaMalloc((void **)&states, 128 * 1024 * sizeof(curandState));
        this->set_seed(seed);
    }
    return this->states;
}


void RadonNoiseGenerator::add_noise(float *sinogram, const int dose, const enum NoiseType noiseType, const int width, const int height) {

     radon_sinogram_noise << < dim3(width / 16, 8 * 1024 / width), dim3(16, 16) >> >(sinogram, this->get(), width, height, dose, noiseType);
}


void RadonNoiseGenerator::free() {
        if (this->states != nullptr) {
#ifdef VERBOSE
            std::cout << "[TORCH RADON] Freeing Random states on device " << i << " " << this->states[i] << std::endl;
#endif
            cudaFree(this->states);
            this->states = nullptr;
#ifdef VERBOSE
            std::cout << "[TORCH RADON] DONE Freeing Random states on device " << i << std::endl;
#endif
        }
}

RadonNoiseGenerator::~RadonNoiseGenerator() {
    this->free();
}
