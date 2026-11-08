#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include "autan/core/mesh.hpp"
#include "autan/solvers/line_smoother.hpp"

namespace autan::solvers::gpu {

/// The physics' line-implicit block Gauss-Seidel (solvers::LineSmoother, `linear_solver =
/// "lines"`) on a CUDA or HIP device, one rank: its lines and its lone cells, colour by colour,
/// forwards then backwards, and its product A x. Every operation is FP64 in the host's order
/// (LineSmoother::set_operator, solve_unit, multiply), without fused multiply-adds or atomics, so
/// the results are the host's bit for bit when the host is given the same blocks. Blocks are
/// read on the device: `diag` 25 x n_cells in structure of arrays (entry q of cell c at
/// q n_cells + c), `off` one block per entry of the layout given at construction (entry q of
/// slot s at q n_slots + s), FP64 or FP32 and converted exactly. Vectors are 5 x n_cells in
/// structure of arrays, as the host's.
class LineSweep {
 public:
  /// `host` gives the units, colours and summation orders; `incidence_offsets` and
  /// `incidence_faces` (CSR over the cells, faces of every kind) the slot of each off-diagonal
  /// block: entry k of cell c holds A(c, m) for the cell m across face incidence_faces[k].
  /// Throws std::invalid_argument when an owned cell's interior face has no slot, or with ghost
  /// cells (one rank only).
  LineSweep(const solvers::LineSmoother& host, std::span<const offset_t> incidence_offsets,
            std::span<const index_t> incidence_faces);
  ~LineSweep();
  LineSweep(LineSweep&&) noexcept;
  LineSweep& operator=(LineSweep&&) noexcept;
  LineSweep(const LineSweep&) = delete;
  LineSweep& operator=(const LineSweep&) = delete;

  /// Factors every unit from the blocks, which are referenced until the next call; waits for
  /// the device. Throws std::domain_error on a singular block, as the host does.
  void set_operator(const double* diag, const double* off, void* stream = nullptr);
  void set_operator(const double* diag, const float* off, void* stream = nullptr);
  /// y = A x on every cell. Throws std::logic_error before set_operator.
  void multiply(const double* x, double* y, void* stream = nullptr) const;
  /// `sweeps` symmetric sweeps on A x = b from x. Throws std::logic_error before set_operator.
  void sweep(const double* b, double* x, int sweeps, void* stream = nullptr) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace autan::solvers::gpu
