// The device GCR (gpu::Gcr) against the host's (solvers::Gcr) with the physics' reductions, each
// preconditioned by its line smoother: x, r and the report bit for bit.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "autan/runtime/exact_sum.hpp"
#include "autan/solvers/gcr.hpp"
#include "autan/solvers/gcr_gpu.hpp"
#include "autan/solvers/line_smoother.hpp"
#include "autan/solvers/line_sweep_gpu.hpp"
#include "line_sweep_fixture.hpp"

using namespace line_sweep_fixture;

TEMPLATE_TEST_CASE("device GCR has the host GCR's bits, iteration for iteration", "[gcr][gpu]",
                   double, float) {
  // The host reduces as the physics does (EulerPhysics::owned_dot, owned_max_abs): exact dot
  // products rounded once, the largest |entry| with NaN as +inf. The preconditioner is two line
  // sweeps, as the physics' "lines" solver uses four.
  using Off = TestType;
  const auto [source, kind] =
      GENERATE(std::pair{Source{"hybrid/hybrid.ascii.msh"}, LineSource::found},
               std::pair{Source{"poly/polyMesh"}, LineSource::found},
               std::pair{Source{6}, LineSource::half});
  // The physics' tolerance, and one the solve never reaches, so that every iteration runs.
  const double tolerance = GENERATE(0.1, 1e-300);
  const auto mesh = mesh_of(source);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto lines = lines_of(mesh, g, kind);
  std::int32_t n_colours = 0;
  const auto colours = unit_colours(mesh, lines, n_colours);
  autan::solvers::LineSmoother host(mesh, lines.offsets, lines.cells, colours, n_colours,
                                    scrambled_ids(ni));
  const auto layout = reversed_layout(mesh);
  const auto a = dominant<Off>(mesh, layout, 11);
  host.set_operator(a.diag, a.upper, a.lower);
  const std::size_t m = 5 * n;
  std::vector<double> r0(m);
  for (std::size_t k = 0; k < m; ++k) r0[k] = 1e3 * std::sin(0.23 * static_cast<double>(k));
  constexpr int max_iterations = 10, sweeps = 2;

  autan::solvers::Gcr::Reductions reductions{
      .dot =
          [](std::span<const double> u, std::span<const double> v) {
            autan::runtime::ExactSum sum;
            for (std::size_t k = 0; k < u.size(); ++k) sum.add(u[k] * v[k]);
            return sum.value();
          },
      .max_abs =
          [](std::span<const double> u) {
            double best = 0.0;
            for (const double x : u) {
              const double v = std::abs(x);
              best = std::isnan(v) ? std::numeric_limits<double>::infinity() : std::max(best, v);
            }
            return best;
          }};
  autan::solvers::Gcr gcr;
  std::vector<double> r_host = r0, x_host(m, 0.0);
  const auto report_host =
      gcr.solve([&](std::span<const double> x, std::span<double> y) { host.multiply(x, y); },
                [&](std::span<const double> r, std::span<double> z) {
                  host.sweep(r, z, sweeps, [](std::span<double>) {});
                },
                r_host, x_host, max_iterations, tolerance, reductions);

  autan::solvers::gpu::LineSweep device(host, layout.offsets, layout.faces);
  const DeviceArray<double> d_diag(a.diag_soa);
  const DeviceArray<Off> d_off(a.off);
  device.set_operator(d_diag.get(), d_off.get());
  const DeviceArray<double> d_r(r0), d_x(std::vector<double>(m, 0.0));
  autan::solvers::gpu::Gcr device_gcr(static_cast<std::int64_t>(m), max_iterations);
  const auto report_device =
      device_gcr.solve([&](const double* x, double* y, const int* active,
                           void* stream) { device.multiply(x, y, stream, active); },
                       [&](const double* r, double* z, const int* active, void* stream) {
                         device.sweep(r, z, sweeps, stream, active);
                       },
                       d_r.get(), d_x.get(), max_iterations, tolerance);
  const auto r_device = d_r.host();
  const auto x_device = d_x.host();

  CAPTURE(report_host.cycles, report_host.residual_drop, tolerance);
  // The solve did work: a drop below one, after more than one cycle when every iteration runs.
  REQUIRE(report_host.residual_drop < 1.0);
  if (tolerance < 1e-200) REQUIRE(report_host.cycles == max_iterations);
  CHECK(report_device.cycles == report_host.cycles);
  CHECK(bits(report_device.residual_drop) == bits(report_host.residual_drop));
  std::size_t x_differ = 0, r_differ = 0;
  for (std::size_t k = 0; k < m; ++k) {
    if (bits(x_device[k]) != bits(x_host[k])) ++x_differ;
    if (bits(r_device[k]) != bits(r_host[k])) ++r_differ;
  }
  CHECK(x_differ == 0);
  CHECK(r_differ == 0);
}

TEST_CASE("device GCR stops where the host's does: zero or non-finite residuals and images",
          "[gcr][gpu]") {
  // A residual that is zero or not a number ends the solve before any iteration; an image that is
  // zero or not a number ends it at the first, failed when not finite. The preconditioner copies,
  // the operator copies or writes zeros or NaN. x starts away from zero, so that a solve that
  // moved it, or left r scaled, would show.
  enum class Case { zero_residual, nan_residual, zero_image, nan_image };
  const auto c =
      GENERATE(Case::zero_residual, Case::nan_residual, Case::zero_image, Case::nan_image);
  CAPTURE(static_cast<int>(c));
  constexpr std::size_t m = 40;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> r0(m), x0(m);
  for (std::size_t k = 0; k < m; ++k) {
    r0[k] = 3.0 + std::sin(static_cast<double>(k));
    x0[k] = 0.5 * static_cast<double>(k);
  }
  if (c == Case::zero_residual) std::fill(r0.begin(), r0.end(), 0.0);
  if (c == Case::nan_residual) r0[7] = nan;
  const double image = c == Case::nan_image ? nan : 0.0;
  const bool copy_image = c == Case::zero_residual || c == Case::nan_residual;

  autan::solvers::Gcr gcr;
  std::vector<double> r_host = r0, x_host = x0;
  const auto report_host = gcr.solve(
      [&](std::span<const double> x, std::span<double> y) {
        if (copy_image) {
          std::copy(x.begin(), x.end(), y.begin());
        } else {
          std::fill(y.begin(), y.end(), image);
        }
      },
      [](std::span<const double> r, std::span<double> z) {
        std::copy(r.begin(), r.end(), z.begin());
      },
      r_host, x_host, 5, 0.1);

  const DeviceArray<double> d_r(r0), d_x(x0), d_image(std::vector<double>(m, image));
  autan::solvers::gpu::Gcr device_gcr(static_cast<std::int64_t>(m), 5);
  const auto copy = [](const double* from, double* to, void* stream) {
    REQUIRE(cudaMemcpyAsync(to, from, m * sizeof(double), cudaMemcpyDeviceToDevice,
                            static_cast<cudaStream_t>(stream)) == cudaSuccess);
  };
  // The operators here ignore `active`: after the solve has stopped they write only directions
  // and images, which no later step reads.
  const auto report_device = device_gcr.solve(
      [&](const double* x, double* y, const int*, void* stream) {
        copy(copy_image ? x : d_image.get(), y, stream);
      },
      [&](const double* r, double* z, const int*, void* stream) { copy(r, z, stream); }, d_r.get(),
      d_x.get(), 5, 0.1);

  CAPTURE(report_host.cycles, report_host.residual_drop);
  if (c == Case::zero_image || c == Case::nan_image) REQUIRE(report_host.cycles == 1);
  CHECK(report_device.cycles == report_host.cycles);
  CHECK(bits(report_device.residual_drop) == bits(report_host.residual_drop));
  const auto r_device = d_r.host();
  const auto x_device = d_x.host();
  std::size_t differ = 0;
  for (std::size_t k = 0; k < m; ++k) {
    if (bits(r_device[k]) != bits(r_host[k]) || bits(x_device[k]) != bits(x_host[k])) ++differ;
  }
  CHECK(differ == 0);
}
