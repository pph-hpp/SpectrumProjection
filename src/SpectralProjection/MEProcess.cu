#include "MEProcess.cuh"
#include <stdio.h>
#include <cmath>

// use texture memory
__device__ cudaTextureObject_t spectrumObj;
__device__ cudaTextureObject_t coefficientObj;


void phone_count_agent(std::vector<float*>sgm, float* sinogram, float* spec, float *material, std::string energy, const int width,
	const int height, int energyNum, float N, float allSpec, bool noise, cudaStream_t stream) {
	dim3 grid((width + 31) / 32, (height + 15) / 16);
	dim3 block(32, 16);
	if (sgm.size() == 1){
		if (stream){
			phone_count_1materials_shared_mem << < grid, block, energyNum * 3 * sizeof(float), stream >> > (sgm[0], sinogram,
				spec, material, width, height, energyNum, N, allSpec, noise);
		}
		else{
			phone_count_1materials_shared_mem << < grid, block, energyNum * 3 * sizeof(float) >> > (sgm[0], sinogram,
				spec, material, width, height, energyNum, N, allSpec, noise);
		}
		
		cudaDeviceSynchronize();
	}
	else if (sgm.size() == 2){
		/*phone_count_2materials << < grid, block >> > (sgm[0], sgm[1], sinogram, spec_idx,
			spectrumObj, coefficientObj, width, height, energyNum, arraySize, N, allSpec, noise);*/
		if (stream){
			phone_count_2materials_shared_mem << < grid, block, energyNum * 3 * sizeof(float), stream >> > (sgm[0], sgm[1], sinogram,
				spec, material, width, height, energyNum, N, allSpec, noise);
		}
		else{
			phone_count_2materials_shared_mem << < grid, block, energyNum * 3 * sizeof(float) >> > (sgm[0], sgm[1], sinogram,
				spec, material, width, height, energyNum, N, allSpec, noise);
		}
		cudaDeviceSynchronize();
	}
	else if (sgm.size() == 3) {
		if (stream){
			phone_count_3materials_shared_mem << < grid, block, energyNum * 3 * sizeof(float), stream >> > (sgm[0], sgm[1], sgm[3], sinogram,
				spec, material, width, height, energyNum, N, allSpec, noise);
		}
		else{
			phone_count_3materials_shared_mem << < grid, block, energyNum * 3 * sizeof(float) >> > (sgm[0], sgm[1], sgm[3], sinogram,
				spec, material, width, height, energyNum, N, allSpec, noise);
		}
		cudaDeviceSynchronize();
	}
	else{
		printf("Error: Materials' count error");
	}
}


__global__ void phone_count_1material(float* sgm_m1, float* sinogram, int spec_idx,
	cudaTextureObject_t specObj, cudaTextureObject_t coefObj, const int width, const int height,
	int energyNum, float N, float allSpec, bool noise) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = row * width + col;

	if (row < height && col < width) {
		float Sp = 0;
		for (size_t i = 0; i < energyNum; i++) {
			Sp += N * 1.0 * tex2D<float>(specObj, i, spec_idx)
				* (exp(-tex2D<float>(coefObj, i, 0) * sgm_m1[idx] / 10)) / allSpec;
		}

		if (noise) {
			sinogram[idx] = Sp;
		}
		else {
			sinogram[idx] = log(N / Sp);
		}
	}
	__syncthreads();
}

__global__ void phone_count_2materials(float* sgm_m1, float* sgm_m2, float* sinogram, int spec_idx,
	cudaTextureObject_t specObj, cudaTextureObject_t coefObj, const int width, const int height,
	int energyNum, float N, float allSpec, bool noise) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = row * width + col;


	if (row < height && col < width){
		float sgm1 = sgm_m1[idx];
		float sgm2 = sgm_m2[idx];
		float Sp = 0;
		for (size_t i = 0; i < energyNum; i++) {
			Sp += N * 1.0 * tex2D<float>(specObj, i, spec_idx)
				* (exp(-tex2D<float>(coefObj, i, 0) * sgm1 / 10 -
					tex2D<float>(coefObj, i, 1) * sgm2 / 10)) / allSpec;
		}
		if (noise) {
			sinogram[idx] = Sp * N;
		}
		else {
			sinogram[idx] = log(1 / Sp);
		}
		//printf("Sp: %f\n", Sp);
		
	}
	__syncthreads();
}


__global__ void phone_count_3materials(float* sgm_m1, float* sgm_m2, float* sgm_m3, float* sinogram,
	int spec_idx, cudaTextureObject_t specObj, cudaTextureObject_t coefObj, const int width,
	const int height, int energyNum, float N, float allSpec, bool noise) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = row * width + col;

	if (row < height && col < width) {
		float Sp = 0;

		for (size_t i = 0; i < energyNum; i++) {
			Sp += N * 1.0 * tex2D<float>(specObj, i, spec_idx) * (exp(-tex2D<float>(coefObj, i, 0)
				* sgm_m1[idx] / 10 - tex2D<float>(coefObj, i, 1) * sgm_m2[idx] / 10
				- tex2D<float>(coefObj, i, 2) * sgm_m3[idx] / 10)) / allSpec;
		}

		if (noise) {
			sinogram[idx] = Sp;
		}
		else {
			sinogram[idx] = log(N / Sp);
		}
	}
	__syncthreads();
}

void bind_spectrum_texture(float* spectrum, int m_energy_num, int energy_num) {

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray_t cudaArray;
	cudaMallocArray(&cudaArray, &channelDesc, m_energy_num, energy_num);

	cudaMemcpy2DToArray(cudaArray, 0, 0, spectrum, m_energy_num * sizeof(float),
		m_energy_num * sizeof(float), energy_num, cudaMemcpyHostToDevice);
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));

	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cudaArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	cudaError_t err = cudaCreateTextureObject(&spectrumObj, &resDesc, &texDesc, nullptr);
	if (err != cudaSuccess) {
		std::cerr << "Error creating texture object: " << cudaGetErrorString(err) << std::endl;
	}

}

void bind_coefficient_texture(float* coefficient, int m_energy_num, int material_num) {
	// 绑定数据到纹理内存

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray_t cudaArray;
	cudaMallocArray(&cudaArray, &channelDesc, m_energy_num, material_num);

	cudaMemcpy2DToArray(cudaArray, 0, 0, coefficient, m_energy_num * sizeof(float),
		m_energy_num * sizeof(float), material_num, cudaMemcpyHostToDevice);
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));

	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cudaArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp; // X
	texDesc.addressMode[1] = cudaAddressModeClamp; // Y
	texDesc.filterMode = cudaFilterModePoint; 
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;                   // close normalize

	cudaError_t err = cudaCreateTextureObject(&coefficientObj, &resDesc, &texDesc, NULL);
	if (err != cudaSuccess) {
		std::cerr << "Error creating texture object: " << cudaGetErrorString(err) << std::endl;
	}

}



//Use shared memory to accelerate

__global__ void phone_count_1materials_shared_mem(float* sgm_m1, float* sinogram,
	float* spectrum, float* coef, const int width, const int height,
	int energyNum, float N, float allSpec, bool noise) {

	extern __shared__ float s_array[];
	float* spec = s_array;
	float* coef1 = (float*)&spec[energyNum];

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = row * width + col;

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		for (int e = 0; e < energyNum; ++e) {
			spec[e] = spectrum[e]; // 赋值
			coef1[e] = coef[e];
		}
	}
	__syncthreads();

	if (row < height && col < width) {
		float sgm1 = sgm_m1[idx];
		float Sp = 0;
		for (size_t e = 0; e < energyNum; e++) {
			Sp += 1.0 * spec[e] * (exp(-coef1[e] * sgm1 / 10)) / allSpec;
		}
		if (noise) {
			sinogram[idx] = Sp * N;
		}
		else {
			sinogram[idx] = log(1 / Sp);
		}
	}
	__syncthreads();
}

__global__ void phone_count_2materials_shared_mem(float* sgm_m1, float* sgm_m2, float* sinogram,
	float* spectrum, float* coef, const int width, const int height,
	int energyNum, float N, float allSpec, bool noise) {

	extern __shared__ float s_array[];
	float* spec = s_array;
	float* coef1 = (float*)&spec[energyNum];
	float* coef2 = (float*)&coef1[energyNum];

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = row * width + col;

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		for (int e = 0; e < energyNum; ++e) {
			spec[e] = spectrum[e]; // 赋值
			coef1[e] = coef[e];
			coef2[e] = coef[e + energyNum];
		}
	}
	__syncthreads();
	
	if (row < height && col < width) {
		float sgm1 = sgm_m1[idx];
		float sgm2 = sgm_m2[idx];
		float Sp = 0;
		for (size_t e = 0; e < energyNum; e++) {
			Sp += 1.0 * spec[e] * (exp(-coef1[e] * sgm1 / 10 -
				coef2[e] * sgm2 / 10)) / allSpec;
		}
		if (noise) {
			sinogram[idx] = Sp * N;
		}
		else {
			sinogram[idx] = log(1 / Sp);
		}
	}
	__syncthreads();
}

__global__ void phone_count_3materials_shared_mem(float* sgm_m1, float* sgm_m2, float *sgm_m3, float* sinogram,
	float* spectrum, float* coef, const int width, const int height,
	int energyNum, float N, float allSpec, bool noise) {

	extern __shared__ float s_array[];
	float* spec = s_array;
	float* coef1 = (float*)&spec[energyNum];
	float* coef2 = (float*)&coef1[energyNum];
	float* coef3 = (float*)&coef2[energyNum];

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = row * width + col;

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		for (int e = 0; e < energyNum; ++e) {
			spec[e] = spectrum[e]; // 赋值
			coef1[e] = coef[e];
			coef2[e] = coef[e + energyNum];
			coef3[e] = coef[e + energyNum * 2];
		}
	}
	__syncthreads();

	if (row < height && col < width) {
		float sgm1 = sgm_m1[idx];
		float sgm2 = sgm_m2[idx];
		float sgm3 = sgm_m3[idx];
		float Sp = 0;
		for (size_t e = 0; e < energyNum; e++) {
			Sp += 1.0 * spec[e] * (exp(-coef1[e] * sgm1 / 10 - coef2[e] * sgm2 / 10
			- coef3[e] * sgm3 / 10)) / allSpec;
		}
		if (noise) {
			sinogram[idx] = Sp * N;
		}
		else {
			sinogram[idx] = log(1 / Sp);
		}
	}
	__syncthreads();
}
