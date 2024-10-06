#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lightSource.h"
#include "../config.h"
#include "../cudaFunction.hpp"
#include <vector>

__constant__ float angles[2];	//startAngle, endAngle
__constant__ int size[2];	//N, V

// Initialize sdd or sid, the array of sdd or sid across views
// V: number of views
void InitializeDistance_Agent(float*& distance_array, const float distance, const int V);


// Initialize u, the array of each detector element coordiante
// u: array of detector elements
// N: number of detector elements
// du: detector element size [mm]
// offcenter: detector off-center [mm]
void InitializeU_Agent(float*& u, const int N, const float du);

void InitializeParams_Agent(std::vector<float> h_angles, std::vector<int> h_size);

// Initialize beta, the array of each view angle
// beta: array of view angles [radius]
// V: number of views
void InitializeBeta_Agent(float*& beta, const int V, const float startAngle, const float totalScanAngle);

// Initialize beta from an external jsonc file
// The purpose is for non uniform beta distribution
// V: number of views
// rotation: rotate the reconstructed image [degree]
// scanAngleFile: name of the jsonc file to save the angles 

// Forward projection, using bilinear interpolation
void ForwardProjectionBilinear_Agent(float* &image, float* &sinogram, const float sid, const float sdd, \
	const float* u, const float* v, const float* beta, const FPConfig& config, int z_element_idx,
	cudaStream_t stream);

// Bin the sinogram data along detector direction
void BinSinogram(float*& sinogram_large, float*& sinogram, const FPConfig& config);

// Save one slice of the sinogram data
void SaveSinogramSlice(const char* filename, float*& sinogram_slice, int z_element_idx, const FPConfig& config);

// Malloc the memory as a given size
void MallocManaged_Agent(float*& p, const int size);

// free memory
void FreeMemory_Agent(float*& p);
