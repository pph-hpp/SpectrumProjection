#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void phone_count_1material(float* sgm_m1, float* sinogram, float* spec,
	float* u1, const int width, const int height, int energyNum, float N, float allSpec, bool noise);

__global__ void phone_count_2materials(float* sgm_m1, float* sgm_m2, float* sinogram, float* spec,
	float* u1, float* u2, const int width, const int height, int energyNum, float N, float allSpec, bool noise);

__global__ void phone_count_3materials(float* sgm_m1, float* sgm_m2, float* sgm_m3, float* sinogram, float* spec,
	float* u1, float* u2, float* u3, const int width, const int height, int energyNum, float N, float allSpec, bool noise);
