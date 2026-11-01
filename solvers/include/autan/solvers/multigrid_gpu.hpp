#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include "autan/solvers/agglomeration.hpp"

namespace autan::solvers::gpu {

/// Restriction and prolongation between the levels of an agglomeration hierarchy, on the GPU
/// (docs/design/multigrid-gpu.md#transfers). Vectors are device arrays of FP64 values in
/// structure of arrays, 5 components, component c of node i at c * n + i. Each output value is
/// written by one thread from a list in the host multigrid's order, so results are bitwise
/// those of the host. Calls are asynchronous on `stream` (a cudaStream_t; null is the default
/// stream). Throws std::out_of_range for a level without a coarser one.
class Transfers {
 public:
  /// Uploads the lists of `levels`, finest first, as `agglomerate` returns them.
  explicit Transfers(std::span<const Level> levels);
  ~Transfers();
  Transfers(Transfers&&) noexcept;
  Transfers& operator=(Transfers&&) noexcept;
  Transfers(const Transfers&) = delete;
  Transfers& operator=(const Transfers&) = delete;

  /// b_c = Pᵀ r from level l to level l + 1: each coarse value is the sum, from +0.0, of its
  /// children's values in increasing fine index.
  void restrict_residual(std::size_t l, const double* r, double* b_coarse,
                         void* stream = nullptr) const;
  /// e = P x_c from level l + 1 to level l: e_i = x_c of the coarse node holding i.
  void prolong(std::size_t l, const double* x_coarse, double* e, void* stream = nullptr) const;
  /// x += α P x_c on level l, each value rounded as α · e_i then x_i + α · e_i, without fused
  /// multiply-add; α is a device scalar.
  void correct(std::size_t l, const double* alpha, const double* x_coarse, double* x,
               void* stream = nullptr) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace autan::solvers::gpu
