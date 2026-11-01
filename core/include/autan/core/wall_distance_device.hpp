#pragma once

#include <cstdint>

#include "autan/core/wall_distance.hpp"

namespace autan {

/// A WallSurface on the current CUDA device (CUDA builds only): its tree,
/// triangles and periodic directions, uploaded once. distance() runs the
/// host's query (wall_distance_query.hpp) per point, compiled without FMA
/// contraction, so each distance has the bits of WallSurface::distance.
class DeviceWallSurface {
 public:
  explicit DeviceWallSurface(const WallSurface& host);
  DeviceWallSurface(const DeviceWallSurface&) = delete;
  DeviceWallSurface& operator=(const DeviceWallSurface&) = delete;
  ~DeviceWallSurface();

  /// out[i] = the distance of point (x[i], y[i], z[i]), i < n, all in device
  /// memory, on `stream` (a cudaStream_t, null for the default stream).
  /// Throws std::runtime_error when CUDA fails.
  void distance(const double* x, const double* y, const double* z, std::int64_t n, double* out,
                void* stream) const;

 private:
  wall::View view_{};  // device pointers
};

}  // namespace autan
