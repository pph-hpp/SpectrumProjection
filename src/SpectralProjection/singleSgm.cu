#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "singleSgm.cuh"
#include <stdio.h>
#include <cmath>
#include <random>
#include<iostream>
#include <fstream>

__global__ void compute(float* sgm, float* sinogram, float uwater, const int width, const int height, float N) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = row * width + col;

	if (row < height && col < width)
	{
		sinogram[idx] = 0;
		int Sp = N * 1.0*(exp(-uwater * sgm[idx]));
		Sp = int(Sp);
		std::random_device rd;
		std::mt19937 gen(rd());
		std::poisson_distribution<float>d(Sp);
		Sp = d(gen);
		//sinogram[idx] = 10;
		sinogram[idx] = log(N / Sp);
	}
	/*cudaDeviceSynchronize();*/
}

void freeMemory(float*& p)
{
	cudaFree(p);
	p = nullptr;
}

singleSgm::~singleSgm() {
	freeMemory(sgm);
	freeMemory(sinogram);
	if (sgmCpu != nullptr)
	{
		delete[] sgmCpu;
	}
	if (sinogramCpu != nullptr)
	{
		delete[] sinogramCpu;
	}
}

void singleSgm::readSgm(const char* filename) {
	FILE* fp = fopen(filename, "rb");
	if (fp == NULL)
	{
		fprintf(stderr, "Cannot open file %s!\n", filename);
		exit(3);
	}

	fread(sgmCpu, sizeof(float), sgmWidth * sgmHeight, fp);
	/*printf("img: %f\n", boneCpu[2048 * 512+1024]);*/
	cudaMemcpy(sgm, sgmCpu, sizeof(float) * sgmWidth * sgmHeight, cudaMemcpyHostToDevice);
	fclose(fp);
}

void singleSgm::sgmToSinogram() {

	dim3 grid((sgmWidth + 15) / 16, (sgmHeight + 15) / 16);
	dim3 block(16, 16);
	//compute(float* sgm, float* sinogram, float uwater, const int width, const int height, float N)
	compute << <grid, block >> > (sgm, sinogram, mu, sgmWidth, sgmHeight, Dose);
	cudaDeviceSynchronize();
}
void singleSgm::saveSinogram(const char* filename, float* sinogram) {

	float* sinogramCpu = new float[sgmWidth * sgmHeight];
	cudaMemcpy(sinogramCpu, sinogram, sizeof(float) * sgmWidth * sgmHeight, cudaMemcpyDeviceToHost);

	FILE* fp = NULL;
	fp = fopen(filename, "wb");

	if (fp == NULL)
	{
		fprintf(stderr, "Cannot save to file %s!\n", filename);
		exit(4);
	}
	fwrite(sinogramCpu, sizeof(float), sgmWidth * sgmHeight, fp);
	delete[] sinogramCpu;
	fclose(fp);
}
void singleSgm::InitSgm() {
	cudaMalloc((void**)&sgm, sizeof(float) * sgmWidth * sgmHeight);
	cudaMalloc((void**)&sinogram, sizeof(float) * sgmWidth * sgmHeight);
	sgmCpu = new float[sgmWidth * sgmHeight];
	sinogramCpu = new float[sgmWidth * sgmHeight];
}