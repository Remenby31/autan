#include "device.hpp"

#include <stdexcept>

#ifdef AUTAN_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace autan::runtime {

#ifdef AUTAN_HAVE_CUDA

namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

int visible_gpus() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return count;
}

DeviceMemory device_memory(int local_rank) {
  const int index = local_rank % visible_gpus();
  check(cudaSetDevice(index), "cudaSetDevice");
  std::size_t free = 0, total = 0;
  check(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
  int integrated = 0;
  check(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, index),
        "cudaDeviceGetAttribute");
  return {.free_bytes = static_cast<std::int64_t>(free), .integrated = integrated != 0};
}

Gpu::Gpu(int local_rank) : index_(local_rank % visible_gpus()) {
  check(cudaSetDevice(index_), "cudaSetDevice");
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, index_), "cudaGetDeviceProperties");
  name_ = properties.name;
  cudaStream_t stream = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  stream_ = stream;
}

Gpu::~Gpu() {
  if (stream_ != nullptr) (void)cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

#else

int visible_gpus() { return 0; }

DeviceMemory device_memory(int /*local_rank*/) {
  throw std::runtime_error("this build of autan-run has no CUDA");
}

Gpu::Gpu(int /*local_rank*/) { throw std::runtime_error("this build of autan-run has no CUDA"); }

Gpu::~Gpu() = default;

#endif

}  // namespace autan::runtime
