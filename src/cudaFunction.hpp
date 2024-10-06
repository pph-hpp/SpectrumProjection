#pragma once
#include <iostream>
#include <chrono>

#ifndef CUDA_ENABLED
#define CUDA_ENABLED
#endif // !CUDA_ENABLED

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif
//#include "device_launch_parameters.h"

#define CHECK_CUDA_ERROR(err) {                           \
    cudaError_t error = err;                             \
    if (error != cudaSuccess) {                          \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) \
                  << " in file " << __FILE__           \
                  << " at line " << __LINE__ << std::endl; \
        exit(error);                                     \
    }                                                    \
}

class MemoryAgent {
public:
	static void FreeMemory(float*& p);

	static void FreeMemoryCpu(float*& p);

    static cudaDeviceProp getDeviceProperties();

    static int LockMemory(float* data, int size);

    static void GetWorkingSetSize();
    static void SetWorkingSetSize(int minNum, int maxNum);  //MB
};


class Timer {
public:
    void startCPU() {
        start_time_cpu = std::chrono::high_resolution_clock::now();
    }

    void stopCPU() {
        end_time_cpu = std::chrono::high_resolution_clock::now();
    }

    double getElapsedTimeCPU() {
        std::chrono::duration<double, std::milli> duration = end_time_cpu - start_time_cpu;
        return duration.count();
    }

#ifdef CUDA_ENABLED
    void startCUDA() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event);
    }

    void stopCUDA() {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
    }

    float getElapsedTimeCUDA() {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        return milliseconds;
    }

#endif

private:
    std::chrono::high_resolution_clock::time_point start_time_cpu, end_time_cpu;
#ifdef CUDA_ENABLED
    cudaEvent_t start_event, stop_event;
#endif
};

// 用法示例
/*
int main() {
    Timer timer;

    // CPU 计时示例
    timer.startCPU();
    // 执行 CPU 任务
    timer.stopCPU();
    std::cout << "CPU Time: " << timer.getElapsedTimeCPU() << " milliseconds" << std::endl;

#ifdef CUDA_ENABLED
    // CUDA 计时示例
    timer.startCUDA();
    // 执行 CUDA 任务
    timer.stopCUDA();
    std::cout << "CUDA Time: " << timer.getElapsedTimeCUDA() << " milliseconds" << std::endl;
#endif

    return 0;
}
*/