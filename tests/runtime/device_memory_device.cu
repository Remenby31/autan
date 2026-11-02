#include <cuda_runtime.h>

#include "autan/core/device_memory.hpp"
#include "autan/kernels/device.cuh"
#include "device_memory_device.hpp"

namespace autan_test {

std::int64_t counted_while_holding(std::size_t n) {
  const autan::kernels::gpu::DeviceArray<double> held(n);
  return autan::device_memory::current();
}

void* raw_device_allocation(std::size_t bytes) {
  void* p = nullptr;
  return cudaMalloc(&p, bytes) == cudaSuccess ? p : nullptr;
}

void raw_device_release(void* p) { (void)cudaFree(p); }

}  // namespace autan_test
