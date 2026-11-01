#include <cuda_runtime.h>

#include "autan/runtime/exact_sum.cuh"
#include "exact_sum_device.hpp"

namespace autan_test {
namespace {

struct Terms {
  const double* a;
  const double* b;
  __device__ double operator()(std::int64_t i, int k) const { return k == 0 ? a[i] : b[i]; }
};

void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

__global__ void round_on_device(const autan::runtime::gpu::ExactWords* sums, double* values) {
  values[0] = autan::runtime::gpu::exact_value(sums[0]);
  values[1] = autan::runtime::gpu::exact_value(sums[1]);
}

}  // namespace

std::array<autan::runtime::ExactSum, 2> device_sums(const std::vector<double>& a,
                                                    const std::vector<double>& b, int blocks,
                                                    std::array<double, 2>* values) {
  const auto n = a.size();
  double *da = nullptr, *db = nullptr;
  autan::runtime::gpu::ExactWords* out = nullptr;
  check(cudaMalloc(&da, n * sizeof(double)));
  check(cudaMalloc(&db, n * sizeof(double)));
  check(cudaMalloc(&out, 2 * sizeof(autan::runtime::gpu::ExactWords)));
  check(cudaMemcpy(da, a.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  check(cudaMemcpy(db, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream));
  autan::runtime::gpu::exact_sums<2>(static_cast<std::int64_t>(n), Terms{da, db}, out, stream,
                                     blocks);
  const auto sums = autan::runtime::gpu::to_host<2>(out, stream);
  if (values != nullptr) {
    double* dv = nullptr;
    check(cudaMalloc(&dv, 2 * sizeof(double)));
    round_on_device<<<1, 1, 0, stream>>>(out, dv);
    check(cudaGetLastError());
    check(cudaMemcpyAsync(values->data(), dv, 2 * sizeof(double), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    check(cudaFree(dv));
  }
  check(cudaStreamDestroy(stream));
  check(cudaFree(da));
  check(cudaFree(db));
  check(cudaFree(out));
  return sums;
}

double device_value(const autan::runtime::ExactSum& sum) {
  autan::runtime::gpu::ExactWords words{};
  autan::runtime::ExactSum copy = sum;
  const auto data = copy.data();
  for (int j = 0; j < autan::runtime::ExactSum::words; ++j)
    words.w[j] = data[static_cast<std::size_t>(j)];
  autan::runtime::gpu::ExactWords* dw = nullptr;
  double* dv = nullptr;
  check(cudaMalloc(&dw, 2 * sizeof(words)));
  check(cudaMalloc(&dv, 2 * sizeof(double)));
  check(cudaMemcpy(dw, &words, sizeof(words), cudaMemcpyHostToDevice));
  check(cudaMemcpy(dw + 1, &words, sizeof(words), cudaMemcpyHostToDevice));
  round_on_device<<<1, 1>>>(dw, dv);
  check(cudaGetLastError());
  double value = 0.0;
  check(cudaMemcpy(&value, dv, sizeof(double), cudaMemcpyDeviceToHost));
  check(cudaFree(dw));
  check(cudaFree(dv));
  return value;
}

}  // namespace autan_test
