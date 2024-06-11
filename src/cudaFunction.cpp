#include "cudaFunction.hpp"

void MemoryAgent::FreeMemory(float*& p)
{
	if (p != nullptr) {
		cudaFree(p);
		p = nullptr;
	}
}

void MemoryAgent::FreeMemoryCpu(float*& p) {
	if (p != nullptr) {
		delete[] p;
		p = nullptr;
	}
}