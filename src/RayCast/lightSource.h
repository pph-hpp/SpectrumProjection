#pragma once
#include "../config.h"
#include "cuda_runtime.h"

class lightSource
{
public:
	static FPConfig config;

private:
	// array of detector element coordinate in Z direction
	static float* v;
	// array of detector element coordinate
	static float* u;

	// array of each view angle [radius]
	static float* beta;

private:

public:
	lightSource();
	lightSource(FPConfig config);
	~lightSource();

	// Initialize parameters
	void InitParam();

	// Forward projection, using bilinear interpolation
	void ForwardProjectionBilinear(float *image, float *sgm, cudaStream_t stream);


};
