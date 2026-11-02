#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include "autan/solvers/agglomeration.hpp"
#include "autan/solvers/galerkin_lists.hpp"

namespace autan::solvers::gpu {

/// Restriction and prolongation between the levels of an agglomeration hierarchy, on the GPU
/// (docs/design/multigrid-gpu.md#transfers). Vectors are device arrays of FP64 values in
/// structure of arrays, 5 components, component c of node i at c * n + i, in the device
/// numbering of their level (`order`). Each output value is written by one thread from a list in
/// the host multigrid's order, so results are bitwise those of the host, permuted. Calls are
/// asynchronous on `stream` (a cudaStream_t; null is the default stream). Throws
/// std::out_of_range for a level without a coarser one.
class Transfers {
 public:
  /// Uploads the lists of `levels`, finest first, as `agglomerate` returns them.
  explicit Transfers(std::span<const Level> levels);
  ~Transfers();
  Transfers(Transfers&&) noexcept;
  Transfers& operator=(Transfers&&) noexcept;
  Transfers(const Transfers&) = delete;
  Transfers& operator=(const Transfers&) = delete;

  /// Host index of each device node of level l: level 0 keeps the mesh's numbering, a coarse
  /// node is ranked by its least child in the finer level's device numbering.
  [[nodiscard]] std::span<const index_t> order(std::size_t l) const;
  /// b_c = Pᵀ r from level l to level l + 1: each coarse value is the sum, from +0.0, of its
  /// children's values in increasing host fine index.
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

/// The Galerkin coarse operator A_c = Pᵀ A P on the GPU (docs/design/multigrid-gpu.md#galerkin-
/// coarse-operator): each coarse block is the sum, from +0.0 and in list order, of the blocks of
/// its `GalerkinLists` list, in FP64, so the coarse operator is bitwise the host multigrid's.
/// Blocks are 5 x 5 row-major, stored in structure of arrays: entry q of the block of node i at
/// q * n_nodes + i (diagonal), of slot s at q * n_slots + s (off-diagonal). The fine diagonal is
/// FP64; the fine off-diagonal blocks are FP32 or FP64 (converted exactly). Calls are
/// asynchronous on `stream`.
class Galerkin {
 public:
  /// Uploads the lists from `n_fine_nodes` fine nodes and `n_fine_slots` fine slots to their
  /// coarse level. Throws std::invalid_argument when a list entry is out of range.
  Galerkin(const GalerkinLists& lists, std::size_t n_fine_nodes, std::size_t n_fine_slots);
  ~Galerkin();
  Galerkin(Galerkin&&) noexcept;
  Galerkin& operator=(Galerkin&&) noexcept;
  Galerkin(const Galerkin&) = delete;
  Galerkin& operator=(const Galerkin&) = delete;

  [[nodiscard]] std::size_t n_coarse_nodes() const;
  [[nodiscard]] std::size_t n_coarse_slots() const;
  void apply(const double* fine_diag, const double* fine_off, double* coarse_diag,
             double* coarse_off, void* stream = nullptr) const;
  void apply(const double* fine_diag, const float* fine_off, double* coarse_diag,
             double* coarse_off, void* stream = nullptr) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// The coarsest level's direct solve on the GPU (docs/design/multigrid-gpu.md#coarsest-level):
/// the host multigrid's dense matrix of the level (unknown c of node h at c * n + h, host
/// numbering), factored by `factor_dense` as the host does, then solved on the device in
/// `solve_dense`'s operation order, so x is the host's bit for bit. Vectors are in the level's
/// device numbering (`order`), 5 components in structure of arrays. Throws std::domain_error on a
/// singular matrix, as factor_dense does, and std::invalid_argument beyond 4096 unknowns.
class CoarsestSolve {
 public:
  /// `diag`: one block per host node; `upper`, `lower`: A(owner, neighbour) and A(neighbour,
  /// owner) per edge of `level`, row-major 5 x 5, FP64, host numbering.
  CoarsestSolve(const Level& level, std::span<const index_t> order, std::span<const double> diag,
                std::span<const double> upper, std::span<const double> lower);
  ~CoarsestSolve();
  CoarsestSolve(CoarsestSolve&&) noexcept;
  CoarsestSolve& operator=(CoarsestSolve&&) noexcept;
  CoarsestSolve(const CoarsestSolve&) = delete;
  CoarsestSolve& operator=(const CoarsestSolve&) = delete;

  /// Unknowns, 5 per node.
  [[nodiscard]] std::size_t size() const;
  void solve(const double* b, double* x, void* stream = nullptr) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace autan::solvers::gpu
