#include <stdexcept>

#include <cuda_runtime.h>

#include "autan/core/wall_distance_device.hpp"
#include "wall_distance_device_test.hpp"

namespace autan::test {

std::vector<double> device_distances(const WallSurface& surface,
                                     const std::vector<std::array<double, 3>>& points) {
  const auto n = points.size();
  std::vector<double> x(n), y(n), z(n), out(n);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = points[i][0];
    y[i] = points[i][1];
    z[i] = points[i][2];
  }
  const auto ok = [](cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
  };
  double *dx = nullptr, *dy = nullptr, *dz = nullptr, *dout = nullptr;
  for (double** p : {&dx, &dy, &dz, &dout}) ok(cudaMalloc(p, n * sizeof(double)));
  ok(cudaMemcpy(dx, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  ok(cudaMemcpy(dy, y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  ok(cudaMemcpy(dz, z.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  cudaStream_t stream = nullptr;
  ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const DeviceWallSurface device(surface);
  device.distance(dx, dy, dz, static_cast<std::int64_t>(n), dout, stream);
  ok(cudaStreamSynchronize(stream));
  ok(cudaMemcpy(out.data(), dout, n * sizeof(double), cudaMemcpyDeviceToHost));
  ok(cudaStreamDestroy(stream));
  for (double* p : {dx, dy, dz, dout}) ok(cudaFree(p));
  return out;
}

}  // namespace autan::test
