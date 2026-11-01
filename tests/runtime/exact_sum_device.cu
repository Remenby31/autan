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

struct CountedTerms {
  const double* a;
  const double* b;
  unsigned long long* calls;
  __device__ double operator()(std::int64_t i, int k) const {
    atomicAdd(calls, 1ULL);
    return k == 0 ? a[i] : b[i];
  }
};

__global__ void decide(int* flag, int value) { *flag = value; }

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

std::array<autan::runtime::ExactSum, 2> device_sums_if(const std::vector<double>& a,
                                                       const std::vector<double>& b, int active,
                                                       unsigned long long& evaluated) {
  const auto n = a.size();
  double *da = nullptr, *db = nullptr;
  int* flag = nullptr;
  unsigned long long* calls = nullptr;
  autan::runtime::gpu::ExactWords* out = nullptr;
  check(cudaMalloc(&da, n * sizeof(double)));
  check(cudaMalloc(&db, n * sizeof(double)));
  check(cudaMalloc(&flag, sizeof(int)));
  check(cudaMalloc(&calls, sizeof(unsigned long long)));
  check(cudaMalloc(&out, 2 * sizeof(autan::runtime::gpu::ExactWords)));
  check(cudaMemcpy(da, a.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  check(cudaMemcpy(db, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  check(cudaMemcpy(flag, &active, sizeof(int), cudaMemcpyHostToDevice));
  check(cudaMemset(calls, 0, sizeof(unsigned long long)));
  check(cudaMemset(out, 0x5a, 2 * sizeof(autan::runtime::gpu::ExactWords)));
  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream));
  autan::runtime::gpu::exact_sums<2>(static_cast<std::int64_t>(n), CountedTerms{da, db, calls}, out,
                                     stream, 0, flag);
  const auto sums = autan::runtime::gpu::to_host<2>(out, stream);
  check(cudaMemcpy(&evaluated, calls, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  check(cudaStreamDestroy(stream));
  for (void* p : {static_cast<void*>(da), static_cast<void*>(db), static_cast<void*>(flag),
                  static_cast<void*>(calls), static_cast<void*>(out)})
    check(cudaFree(p));
  return sums;
}

std::vector<Replay> graph_replays(const std::vector<double>& a, const std::vector<double>& b,
                                  const std::vector<int>& decisions) {
  const auto n = a.size();
  double *da = nullptr, *db = nullptr;
  int* flag = nullptr;
  unsigned long long* calls = nullptr;
  autan::runtime::gpu::ExactWords* out = nullptr;
  check(cudaMalloc(&da, n * sizeof(double)));
  check(cudaMalloc(&db, n * sizeof(double)));
  check(cudaMalloc(&flag, sizeof(int)));
  check(cudaMalloc(&calls, sizeof(unsigned long long)));
  check(cudaMalloc(&out, 2 * sizeof(autan::runtime::gpu::ExactWords)));
  check(cudaMemcpy(da, a.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  check(cudaMemcpy(db, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  check(cudaMemset(calls, 0, sizeof(unsigned long long)));
  check(cudaMemset(out, 0x5a, 2 * sizeof(autan::runtime::gpu::ExactWords)));
  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  autan::runtime::gpu::exact_sums<2>(static_cast<std::int64_t>(n), CountedTerms{da, db, calls}, out,
                                     stream, 64, flag);
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&exec, graph, 0));
  std::vector<Replay> replays;
  for (const int d : decisions) {
    decide<<<1, 1, 0, stream>>>(flag, d);
    check(cudaGetLastError());
    check(cudaGraphLaunch(exec, stream));
    Replay r{autan::runtime::gpu::to_host<2>(out, stream), 0};
    check(cudaMemcpy(&r.evaluated, calls, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    replays.push_back(r);
  }
  check(cudaGraphExecDestroy(exec));
  check(cudaGraphDestroy(graph));
  check(cudaStreamDestroy(stream));
  for (void* p : {static_cast<void*>(da), static_cast<void*>(db), static_cast<void*>(flag),
                  static_cast<void*>(calls), static_cast<void*>(out)})
    check(cudaFree(p));
  return replays;
}

}  // namespace autan_test
