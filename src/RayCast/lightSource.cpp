
#include "ray.cuh"
#include "../stdafx.h"
#include "lightSource.h"
#include "../config.h"


FPConfig lightSource::config;
float* lightSource::v = nullptr;
float* lightSource::u = nullptr;
float* lightSource::beta = nullptr;


lightSource::lightSource()
{
}

lightSource::lightSource(FPConfig config)
{
	//wait add
	this->config = config;
	InitParam();
}

lightSource::~lightSource()
{
	FreeMemory_Agent(u);
	FreeMemory_Agent(beta);
}



void lightSource::InitParam()
{

	InitializeU_Agent(u, config.detEltCount, config.detEltSize);
	InitializeBeta_Agent(beta, config.views, config.startAngle, config.totalScanAngle);
	cudaDeviceSynchronize();

}



void lightSource::ForwardProjectionBilinear(float* image, float* sgm)
{
	printf("\nProcessing slice# ");
	for (int z_idx = 0; z_idx < config.detZEltCount; z_idx++){
		if (z_idx != 0){
			printf("\b\b\b\b\b\b\b");
		}
		printf("%3d/%3d", z_idx + 1, config.detZEltCount);
		ForwardProjectionBilinear_Agent(image, sgm, config.sid, config.sdd, u, v, beta, config, 0);

		//BinSinogram(sgm, sgm, config);
	}
}
