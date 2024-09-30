#include "../stdafx.h"
#include "ray.cuh"

#define PI 3.1415926536f
#define STEPSIZE 0.2f

__global__ void InitDistance(float* distance_array, const float distance, const int V)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < V)
	{
		distance_array[tid] = distance;
	}
}

__global__ void InitU(float* u, const int N, const float du)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < N)
	{
		u[tid] = (tid - (N - 1) / 2.0f) * du;
	}
}

__global__ void InitBeta(float* beta, const int V, const float startAngle, const float totalScanAngle)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < V)
	{
		beta[tid] = (totalScanAngle / V * tid + startAngle) * PI / 180.0f;
	}
}

// img: image data
// sgm: sinogram data
// u: array of each detector element position
// beta: array of each view angle [radian]
// M: image dimension
// S: number of image slices
// N_z: number of detector elements in Z direction
// N: number of detector elements (sinogram width)
// V: number of views (sinogram height)
// dx: image pixel size [mm]
// dz: image slice thickness [mm]
// sid: source to isocenter distance
// sdd: source to detector distance
__global__ void ForwardProjectionBilinear_device(float* img, float* sgm, const float* u, const float* v,\
	const float* beta, int M, int S, int N, int N_z, int V, float dx, float dz, const float sid,\
	const float sdd, bool conebeam, int z_element_begin_idx, int z_element_end_idx)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;


	if (col < N && row < V && z_element_end_idx <= N_z)
	{
		// half of image side length
		float D = M * dx / 2.0f;
		// half of image thickness
		float D_z = 0.0f;
		if (conebeam)
		{
			D_z = float(S) * dz / 2.0f;
		}
		else
		{
			dz = 0;
		}
		//get the sid and sdd for a given view

		// current source position
		float xs = sid * cosf(beta[row]);
		float ys = sid * sinf(beta[row]);
		float zs = 0;

		// current detector element position
		float xd = -(sdd - sid) * cosf(beta[row]) + u[col] * cosf(beta[row] - PI / 2.0f);
		float yd = -(sdd - sid) * sinf(beta[row]) + u[col] * sinf(beta[row] - PI / 2.0f);
		float zd = 0;

		// step point region
		float L_min = sid - sqrt(2 * D * D + D_z * D_z);
		float L_max = sid + sqrt(2 * D * D + D_z * D_z);

		// source to detector element distance
		float sed = sqrtf((xs - xd) * (xs - xd) + (ys - yd) * (ys - yd));// for fan beam case

		// the point position
		float x, y, z;
		// the point index
		int kx, ky, kz;
		// weighting factor for linear interpolation
		float wx, wy, wz;
		float v1, v2;

		// the most upper left image pixel position
		float x0 = -D + dx / 2.0f;
		float y0 = D - dx / 2.0f;
		float z0 = 0;
		if (conebeam)
		{
			z0 = -D_z + dz / 2.0f;// first slice is at the bottom
		}

		// repeat for each slice
		for (int slice = z_element_begin_idx; slice < z_element_end_idx; slice++)
		{
			// initialization
			//sgm[row*N + col + N * V * slice] = 0;
			sgm[row * N + col] = 0;
			if (conebeam)
			{

				zd = v[slice];

				sed = sqrtf((xs - xd) * (xs - xd) + (ys - yd) * (ys - yd) + (zs - zd) * (zs - zd));
			}

			// calculate line integration
			for (float L = L_min; L <= L_max; L += STEPSIZE * sqrt(dx * dx + dz * dz / 2.0f))
			{
				// get the current point position 
				x = xs + (xd - xs) * L / sed;
				y = ys + (yd - ys) * L / sed;
				if (conebeam)
				{
					z = zs + (zd - zs) * L / sed;
				}

				// get the current point index
				kx = floorf((x - x0) / dx);
				ky = floorf((y0 - y) / dx);

				if (conebeam)
					kz = floorf((z - z0) / dz);

				// get the image pixel value at the current point
				if (kx >= 0 && kx + 1 < M && ky >= 0 && ky + 1 < M)
				{
					// get the weighting factor
					/*wx = ((x - x0) - kx * dx) / dx;
					wy = ((y0 - y) - ky * dx) / dx;*/
					wx = (x - kx * dx - x0) / dx;
					wy = (y0 - y - ky * dx) / dx;

					// perform bilinear interpolation
					if (conebeam == false)
					{
						//˫���Բ�ֵ
						/*v1 = (1 - wx) * img[ky * M + kx + M * M * slice]
							+ wx * img[ky * M + kx + 1 + M * M * slice];
						v2 = (1 - wx) * img[(ky + 1) * M + kx + M * M * slice]
							+ wx * img[(ky + 1) * M + kx + 1 + M * M * slice];
						sgm[row * N + col] += (1 - wy) * v1 + wy * v2;*/
						sgm[row * N + col] += (1 - wx) * (1 - wy) * img[ky * M + kx + M * M * slice] // upper left
							+ wx * (1 - wy) * img[ky * M + kx + 1 + M * M * slice] // upper right
							+ (1 - wx) * wy * img[(ky + 1) * M + kx + M * M * slice] // bottom left
							+ wx * wy * img[(ky + 1) * M + kx + 1 + M * M * slice];	// bottom right
						
					}
					else if (conebeam == true && kz >= 0 && kz + 1 < S)
					{
						wz = (z - kz * dz - z0) / dz;
						float sgm_val_lowerslice = (1 - wx) * (1 - wy) * img[ky * M + kx + M * M * kz] // upper left
							+ wx * (1 - wy) * img[ky * M + kx + 1 + M * M * kz] // upper right
							+ (1 - wx) * wy * img[(ky + 1) * M + kx + M * M * kz] // bottom left
							+ wx * wy * img[(ky + 1) * M + kx + 1 + M * M * kz];	// bottom right
						float sgm_val_upperslice = (1 - wx) * (1 - wy) * img[ky * M + kx + M * M * (kz + 1)] // upper left
							+ wx * (1 - wy) * img[ky * M + kx + 1 + M * M * (kz + 1)] // upper right
							+ (1 - wx) * wy * img[(ky + 1) * M + kx + M * M * (kz + 1)] // bottom left
							+ wx * wy * img[(ky + 1) * M + kx + 1 + M * M * (kz + 1)];	// bottom right

						sgm[row * N + col] += (1 - wz) * sgm_val_lowerslice + wz * sgm_val_upperslice;
					}

				}
			}

			sgm[row * N + col] *= STEPSIZE * sqrt(dx * dx + dz * dz);

		}
	}
}

// sgm_large: sinogram data before binning
// sgm: sinogram data after binning
// N: number of detector elements (after binning)
// V: number of views
// S: number of slices
// binSize: bin size
__global__ void BinSinogram(float* sgm_large, float* sgm, int N, int V, int S, int binSize)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if (col < N && row < V)
	{
		// repeat for each slice
		for (int slice = 0; slice < S; slice++)
		{
			// initialization
			sgm[row * N + col + N * V * slice] = 0;

			// sum over each bin
			for (int i = 0; i < binSize; i++)
			{
				sgm[row * N + col + N * V * slice] += sgm_large[row * N * binSize + col * binSize + i + slice * N * binSize * V];
			}
			// take average
			sgm[row * N + col + N * V * slice] /= binSize;
		}
	}
}


//new function with Value member to suit all non uniform parameters

void InitializeU_Agent(float*& u, const int N, const float du)
{
	if (u != nullptr)
		cudaFree(u);

	cudaMalloc((void**)&u, N * sizeof(float));
	InitU << <(N + 511) / 512, 512 >> > (u, N, du);
}

void InitializeBeta_Agent(float*& beta, const int V, const float startAngle, const float totalScanAngle)
{
	if (beta != nullptr)
		cudaFree(beta);

	cudaMalloc((void**)&beta, V * sizeof(float));
	InitBeta << < (V + 511) / 512, 512 >> > (beta, V, startAngle, totalScanAngle);
}


void ForwardProjectionBilinear_Agent(float*& image, float*& sinogram, const float sid, const float sdd, \
	const float* u, const float* v, const float* beta, const FPConfig& config, int z_element_idx)
{
	dim3 grid((config.detEltCount + 15) / 16, (config.views + 15) / 16);
	dim3 block(16, 16);

	ForwardProjectionBilinear_device << <grid, block >> > (image, sinogram, u, v, beta, config.imgDim, config.sliceCount, \
		config.detEltCount, config.detZEltCount, config.views, config.pixelSize, config.sliceThickness, sid, sdd, config.coneBeam, z_element_idx, z_element_idx + 1);
	
	cudaDeviceSynchronize();
}

void BinSinogram(float*& sinogram_large, float*& sinogram, const FPConfig& config)
{
	dim3 grid((config.detEltCount + 7) / 8, (config.views + 7) / 8);
	dim3 block(8, 8);

	BinSinogram << <grid, block >> > (sinogram_large, sinogram, config.detEltCount, config.views, 1, 1);
	// since the sinogram has only one slice, the z_element count is 1

	cudaDeviceSynchronize();
}

void SaveSinogramSlice(const char* filename, float*& sinogram_slice, int z_element_idx, const FPConfig& config)
{
	FILE* fp = nullptr;
	if (z_element_idx == 0)
		fp = fopen(filename, "wb");
	else
		fp = fopen(filename, "ab");

	if (fp == nullptr)
	{
		fprintf(stderr, "Cannot save to file %s!\n", filename);
		exit(4);
	}
	fwrite(sinogram_slice, sizeof(float), config.detEltCount * config.views, fp);
	fclose(fp);
}

void MallocManaged_Agent(float*& p, const int size)
{
	cudaMallocManaged((void**)&p, size);
}

void FreeMemory_Agent(float*& p)
{
	cudaFree(p);
	p = nullptr;
}
