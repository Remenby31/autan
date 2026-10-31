#include <cuda_runtime.h>

#include "probe.hpp"

namespace autan_probe {
namespace {

__global__ void axpy(int n, float a, const float* x, float* y) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

std::string describe(cudaError_t status) {
  return std::string(cudaGetErrorName(status)) + ": " + cudaGetErrorString(status);
}

// y = 2 x + y on small integers is exact in FP32, so the check needs no tolerance.
bool run_axpy(std::string& error) {
  constexpr int n = 4096;
  std::vector<float> x(n);
  std::vector<float> y(n);
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i);
    y[i] = 1.0F;
  }
  float* dx = nullptr;
  float* dy = nullptr;
  cudaError_t status = cudaMalloc(&dx, n * sizeof(float));
  if (status == cudaSuccess) status = cudaMalloc(&dy, n * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(dy, y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    axpy<<<(n + 255) / 256, 256>>>(n, 2.0F, dx, dy);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess)
    status = cudaMemcpy(y.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dx);
  cudaFree(dy);
  if (status != cudaSuccess) {
    error = describe(status);
    return false;
  }
  for (int i = 0; i < n; ++i) {
    if (y[i] != 2.0F * static_cast<float>(i) + 1.0F) {
      error = "axpy returned a wrong value at index " + std::to_string(i);
      return false;
    }
  }
  return true;
}

}  // namespace

CudaReport probe_cuda() {
  CudaReport report;
  report.compiled = true;
  report.architectures = AUTAN_CUDA_ARCHITECTURES;
  cudaRuntimeGetVersion(&report.runtime_version);
  cudaDriverGetVersion(&report.driver_version);
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    report.error_code = static_cast<int>(status);
    report.error = describe(status);
    return report;
  }
  for (int index = 0; index < count; ++index) {
    Device device;
    device.index = index;
    cudaDeviceProp prop{};
    cudaError_t s = cudaGetDeviceProperties(&prop, index);
    if (s == cudaSuccess) s = cudaSetDevice(index);
    if (s != cudaSuccess) {
      device.error = describe(s);
      report.devices.push_back(device);
      continue;
    }
    device.name = prop.name;
    device.major = prop.major;
    device.minor = prop.minor;
    device.memory_gb = static_cast<double>(prop.totalGlobalMem) / 1e9;
    cudaFuncAttributes attributes{};
    s = cudaFuncGetAttributes(&attributes, axpy);
    if (s != cudaSuccess) {
      // cudaErrorNoKernelImageForDevice: the fatbin holds neither SASS nor PTX this GPU runs.
      device.error = describe(s);
      report.devices.push_back(device);
      continue;
    }
    device.kernel_ok = run_axpy(device.error);
    report.devices.push_back(device);
  }
  return report;
}

}  // namespace autan_probe
