// The CUDA runtime names used by the benchmarks, mapped to HIP when CFD_BENCH_HIP is defined,
// so one source builds for both backends.
#pragma once

#ifdef CFD_BENCH_HIP
#include <hip/hip_runtime.h>

#define CFD_BENCH_BACKEND "hip"
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaSuccess hipSuccess
#else
#include <cuda_runtime.h>

#define CFD_BENCH_BACKEND "cuda"
#endif
