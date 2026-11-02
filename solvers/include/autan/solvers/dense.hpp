#pragma once

#include <cstddef>
#include <span>

namespace autan::solvers {

/// LU factorisation of the m x m row-major matrix `a`, in place, by Gaussian elimination with
/// partial pivoting (the multigrid's coarsest level): `a` then holds L below its diagonal (unit
/// diagonal implied) and U on and above it, and row k was swapped with row pivot[k] at step k.
/// `pivot` has m entries. Throws std::domain_error on a zero or non-finite pivot.
void factor_dense(std::span<double> a, std::size_t m, std::span<std::size_t> pivot);

/// Solves A x = b from factor_dense's `lu` and `pivot`, with b in `x` on entry and the solution
/// in `x` on return (pivot.size() entries).
void solve_dense(std::span<const double> lu, std::span<const std::size_t> pivot,
                 std::span<double> x);

}  // namespace autan::solvers
