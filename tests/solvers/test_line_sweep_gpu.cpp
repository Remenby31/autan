// The physics' line smoother on the device (gpu::LineSweep) against the host's
// (solvers::LineSmoother), bit for bit: lines, lone cells and the product, FP64 and FP32 blocks.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "autan/solvers/line_smoother.hpp"
#include "autan/solvers/line_sweep_gpu.hpp"
#include "line_sweep_fixture.hpp"

using namespace line_sweep_fixture;

TEMPLATE_TEST_CASE("device line sweeps and products have the host line smoother's bits",
                   "[lines][gpu]", double, float) {
  // The host is given the device's blocks, converted exactly (FP32 off-diagonal blocks to
  // FP64), so both run the same FP64 operations in the same order.
  using Off = TestType;
  const auto [source, kind] =
      GENERATE(std::pair{Source{"tet-gmsh/tet-gmsh.ascii.msh"}, LineSource::found},
               std::pair{Source{"hex/hex.ascii.msh"}, LineSource::found},
               std::pair{Source{"poly/polyMesh"}, LineSource::found},
               std::pair{Source{"hybrid/hybrid.ascii.msh"}, LineSource::found},
               std::pair{Source{6}, LineSource::found}, std::pair{Source{6}, LineSource::half},
               std::pair{Source{4}, LineSource::columns},
               std::pair{Source{"hex/hex.ascii.msh"}, LineSource::none});
  CAPTURE(static_cast<int>(kind));
  const auto mesh = mesh_of(source);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto lines = lines_of(mesh, g, kind);
  std::int32_t n_colours = 0;
  const auto colours = unit_colours(mesh, lines, n_colours);
  // The host sums each cell's faces by a global id that follows neither the face index nor the
  // device's slot order, so that a device reading blocks by position could not pass.
  const auto face_id = scrambled_ids(ni);
  autan::solvers::LineSmoother host(mesh, lines.offsets, lines.cells, colours, n_colours, face_id);
  const std::size_t in_lines = lines.cells.size();
  switch (kind) {
    case LineSource::found:
      if (std::holds_alternative<int>(source)) REQUIRE(lines.offsets.size() > 2);
      break;
    case LineSource::half:
      REQUIRE(lines.offsets.size() > 2);
      REQUIRE(in_lines < n);
      break;
    case LineSource::columns:
      REQUIRE(in_lines == n);
      break;
    case LineSource::none:
      REQUIRE(in_lines == 0);
      break;
  }

  const auto layout = reversed_layout(mesh);
  const auto a = dominant<Off>(mesh, layout, 7);
  std::vector<double> b(5 * n), x0(5 * n);
  for (std::size_t k = 0; k < 5 * n; ++k) {
    b[k] = std::sin(0.31 * static_cast<double>(k));
    x0[k] = std::cos(0.17 * static_cast<double>(k));
  }

  host.set_operator(a.diag, a.upper, a.lower);
  std::vector<double> x_host = x0, y_host(5 * n, 0.0);
  host.sweep(b, x_host, 2, [](std::span<double>) {});
  host.multiply(x0, y_host);

  autan::solvers::gpu::LineSweep device(host, layout.offsets, layout.faces);
  const DeviceArray<double> d_diag(a.diag_soa);
  const DeviceArray<Off> d_off(a.off);
  const DeviceArray<double> d_b(b), d_x(x0), d_x0(x0), d_y(std::vector<double>(5 * n, 0.0));
  device.set_operator(d_diag.get(), d_off.get());
  // With *active 0 nothing is written; then the same calls with *active 1.
  const DeviceArray<int> off_flag(std::vector<int>{0}), on_flag(std::vector<int>{1});
  device.sweep(d_b.get(), d_x.get(), 2, nullptr, off_flag.get());
  device.multiply(d_x0.get(), d_y.get(), nullptr, off_flag.get());
  CHECK(d_x.host() == x0);
  CHECK(d_y.host() == std::vector<double>(5 * n, 0.0));
  device.sweep(d_b.get(), d_x.get(), 2, nullptr, on_flag.get());
  device.multiply(d_x0.get(), d_y.get(), nullptr, on_flag.get());
  const auto x_device = d_x.host();
  const auto y_device = d_y.host();

  std::size_t x_differ = 0, y_differ = 0;
  for (std::size_t k = 0; k < 5 * n; ++k) {
    REQUIRE(std::isfinite(x_host[k]));
    if (bits(x_device[k]) != bits(x_host[k])) ++x_differ;
    if (bits(y_device[k]) != bits(y_host[k])) ++y_differ;
  }
  // The sweep moved x: a device that did nothing could not pass.
  REQUIRE(std::ranges::any_of(std::views::iota(std::size_t{0}, 5 * n),
                              [&](std::size_t k) { return x_host[k] != x0[k]; }));
  CHECK(x_differ == 0);
  CHECK(y_differ == 0);
}

TEST_CASE("the device line sweep refuses a singular block and a sweep before its operator",
          "[lines][gpu]") {
  const auto mesh = galerkin_fixture::wall_layers(3);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto lines = autan::smoother_lines(mesh, g);
  std::int32_t n_colours = 0;
  const auto colours = unit_colours(mesh, lines, n_colours);
  const autan::solvers::LineSmoother host(mesh, lines.offsets, lines.cells, colours, n_colours);
  const auto layout = reversed_layout(mesh);
  autan::solvers::gpu::LineSweep device(host, layout.offsets, layout.faces);
  const DeviceArray<double> d_b(std::vector<double>(5 * n, 1.0)), d_x(std::vector<double>(5 * n));
  CHECK_THROWS_AS(device.sweep(d_b.get(), d_x.get(), 1), std::logic_error);
  const DeviceArray<double> zero(std::vector<double>(b2 * n, 0.0));
  const DeviceArray<double> off(std::vector<double>(b2 * layout.faces.size(), 0.0));
  CHECK_THROWS_AS(device.set_operator(zero.get(), off.get()), std::domain_error);
  CHECK_THROWS_AS(device.sweep(d_b.get(), d_x.get(), 1), std::logic_error);
}
