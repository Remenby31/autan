#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include "autan/solvers/gcr.hpp"

namespace autan::solvers::gpu {

/// solvers::Gcr on a CUDA or HIP device, one rank: the same iteration, operation for operation,
/// with its reductions as the physics gives them (Euler steady step, `owned_dot`, `owned_max_abs`):
/// each dot product the exact sum of the rounded products, rounded once (runtime::ExactSum), the
/// largest |entry| with NaN counted as +inf. Given operators with the host's bits, x, r and the
/// report are the host's bit for bit. Every scalar stays on the device; the iteration's branches
/// are a device flag that the operators and kernels read, so the host reads the report once.
class Gcr {
 public:
  /// y = A x, or z ~ A^-1 r into a zeroed z, on device vectors of the solve's length, queued on
  /// `stream` (the solve's). Does nothing while *active is 0 (device memory).
  using Operator = std::function<void(const double* x, double* y, const int* active, void* stream)>;

  /// `m` entries per vector; up to `max_iterations` directions kept on the device.
  Gcr(std::int64_t m, int max_iterations, void* stream = nullptr);
  ~Gcr();
  Gcr(Gcr&&) noexcept;
  Gcr& operator=(Gcr&&) noexcept;
  Gcr(const Gcr&) = delete;
  Gcr& operator=(const Gcr&) = delete;

  /// As solvers::Gcr::solve, on device vectors r (the residual b - A x0, overwritten with the
  /// final residual) and x; `max_iterations` at most the constructor's.
  KrylovReport solve(const Operator& apply, const Operator& precondition, double* r, double* x,
                     int max_iterations, double tolerance);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace autan::solvers::gpu
