#include "cudaFunction.hpp"
#include <Windows.h>

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

cudaDeviceProp MemoryAgent::getDeviceProperties() {

    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    std::cout << "CUDA Driver Version: " << driverVersion << std::endl;
    std::cout << "CUDA Runtime Version: " << runtimeVersion << std::endl;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp prop;
    for (int i = 0; i < deviceCount; ++i) {
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "Total Global Memory: " << prop.totalGlobalMem << " bytes" << std::endl;
        std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max 1D Texture Size: " << prop.maxTexture1D << std::endl;
        std::cout << "Max 2D Texture Size: " << prop.maxTexture2D[0] << " x " << prop.maxTexture2D[1] << std::endl;
        std::cout << "Max 3D Texture Size: " << prop.maxTexture3D[0] << " x " << prop.maxTexture3D[1] << " x " << prop.maxTexture3D[2] << std::endl;
        std::cout << std::endl;
    }
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    return prop;
}

int MemoryAgent::LockMemory(float* data, int size) {
    if (VirtualLock(data, size * sizeof(float))) {
        return 0;
    }
    else {
        std::cerr << "Failed to lock memory." << std::endl;
        return -1;
    }
}

void MemoryAgent::GetWorkingSetSize() {
    HANDLE hProcess = GetCurrentProcess();

    SIZE_T minWorkingSetSize = 0;
    SIZE_T maxWorkingSetSize = 0;

    // 调用 GetProcessWorkingSetSize 获取工作集大小
    if (GetProcessWorkingSetSize(hProcess, &minWorkingSetSize, &maxWorkingSetSize)) {
        std::cout << "最小工作集大小: " << minWorkingSetSize / 1024 << " KB" << std::endl;
        std::cout << "最大工作集大小: " << maxWorkingSetSize / 1024 << " KB" << std::endl;
    }
    else {
        std::cerr << "GetProcessWorkingSetSize 调用失败, 错误码: " << GetLastError() << std::endl;
    }

}

void MemoryAgent::SetWorkingSetSize(int minNum, int maxNum) {
    HANDLE hProcess = GetCurrentProcess();

    // 定义新的工作集大小，单位为字节
    SIZE_T minWorkingSetSize = minNum * 1024 * 1024;  // 50 MB
    SIZE_T maxWorkingSetSize = maxNum * 1024 * 1024; // 200 MB

    // 调用 SetProcessWorkingSetSize 设置工作集大小
    if (SetProcessWorkingSetSize(hProcess, minWorkingSetSize, maxWorkingSetSize)) {
        std::cout << "成功设置工作集大小。" << std::endl;
    }
    else {
        std::cerr << "SetProcessWorkingSetSize 调用失败, 错误码: " << GetLastError() << std::endl;
    }
}
