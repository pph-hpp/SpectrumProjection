#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include <iostream>


void phone_count_agent(std::vector<float*>sgm, float* sinogram, float* spectrum, float* material, std::string energy, const int width,
	const int height, int energyNum, float N, float allSpec, bool noise, cudaStream_t stream);


void bind_spectrum_texture(float* spectrum, int m_energy_num, int energy_num);

void bind_coefficient_texture(float* coefficient, int m_energy_num, int material_num);

__global__ void phone_count_1material(float* sgm_m1, float* sinogram, int spec_idx,
	cudaTextureObject_t specObj, cudaTextureObject_t coefObj, const int width, const int height,
	int energyNum, float N, float allSpec, bool noise);


__global__ void phone_count_2materials(float* sgm_m1, float* sgm_m2, float* sinogram, int spec_idx,
	cudaTextureObject_t specObj, cudaTextureObject_t coefObj, const int width, const int height,
	int energyNum, float N, float allSpec, bool noise);


__global__ void phone_count_3materials(float* sgm_m1, float* sgm_m2, float* sgm_m3, float* sinogram,
	int spec_idx, cudaTextureObject_t specObj, cudaTextureObject_t coefObj, const int width, const int height, int energyNum,
	float N, float allSpec, bool noise);


//Use shared memory to accelerate

__global__ void phone_count_1materials_shared_mem(float* sgm_m1, float* sinogram,
	float* spectrum, float* coef, const int width, const int height,
	int energyNum, float N, float allSpec, bool noise);

__global__ void phone_count_2materials_shared_mem(float* sgm_m1, float* sgm_m2, float* sinogram,
	float* spectrum, float* coef, const int width, const int height,
	int energyNum, float N, float allSpec, bool noise);

__global__ void phone_count_3materials_shared_mem(float* sgm_m1, float* sgm_m2, float *sgm_m3, float* sinogram,
	float* spectrum, float* coef, const int width, const int height,
	int energyNum, float N, float allSpec, bool noise);






