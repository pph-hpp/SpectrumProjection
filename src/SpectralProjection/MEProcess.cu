#include "MEProcess.cuh"
#include <stdio.h>
#include <cmath>

__global__ void phone_count_1material(float* sgm_m1, float* sinogram, float* spec,
	float* u1, const int width, const int height, int energyNum, float N, float allSpec, bool noise) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = row * width + col;

	if (row < height && col < width) {
		sinogram[idx] = 0;
		float Sp = 0;

		for (size_t i = 0; i < energyNum; i++) {
			Sp += N * 1.0 * spec[i] * (exp(-u1[i] * sgm_m1[idx] / 10)) / allSpec;
		}

		if (noise) {
			sinogram[idx] = Sp;
		}
		else {
			sinogram[idx] = log(N / Sp);
		}
	}
	/*cudaDeviceSynchronize();*/
}

__global__ void phone_count_2materials(float* sgm_m1, float* sgm_m2, float* sinogram, float* spec,
	float* u1, float* u2, const int width, const int height, int energyNum, float N, float allSpec, bool noise) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = row * width + col;

	if (row < height && col < width){
		sinogram[idx] = 0;
		float Sp = 0;

		for (size_t i = 0; i < energyNum; i++){
			Sp += N * 1.0 * spec[i] * (exp(-u1[i] * sgm_m1[idx] / 10 - u2[i] * sgm_m2[idx] / 10)) / allSpec;
		}

		if (noise){
			sinogram[idx] = Sp;
		}
		else{
			sinogram[idx] = log(N / Sp);
		}
	}
	/*cudaDeviceSynchronize();*/
}

__global__ void phone_count_3materials(float* sgm_m1, float* sgm_m2, float* sgm_m3, float* sinogram, float* spec,
	float* u1, float* u2, float* u3, const int width, const int height, int energyNum, float N, float allSpec, bool noise) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = row * width + col;

	if (row < height && col < width) {
		sinogram[idx] = 0;
		float Sp = 0;

		for (size_t i = 0; i < energyNum; i++) {
			Sp += N * 1.0 * spec[i] * (exp(-u1[i] * sgm_m1[idx] / 10 - u2[i] * sgm_m2[idx] / 10 - u3[i] * sgm_m3[idx] / 10)) / allSpec;
		}

		if (noise) {
			sinogram[idx] = Sp;
		}
		else {
			sinogram[idx] = log(N / Sp);
		}
	}
	/*cudaDeviceSynchronize();*/
}