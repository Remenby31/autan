// The physics' line smoother on the device (gpu::LineSweep) against the host's
// (solvers::LineSmoother), bit for bit: lines, lone cells and the product, FP64 and FP32 blocks.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_runtime.h>

#include "autan/core/lines.hpp"
#include "autan/solvers/line_smoother.hpp"
#include "autan/solvers/line_sweep_gpu.hpp"
#include "galerkin_fixture.hpp"

namespace {

using galerkin_fixture::b2;

template <class T>
class DeviceArray {
 public:
  explicit DeviceArray(const std::vector<T>& host) : n_(host.size()) {
    REQUIRE(cudaMalloc(reinterpret_cast<void**>(&p_), std::max<std::size_t>(n_, 1) * sizeof(T)) ==
            cudaSuccess);
    if (n_ > 0) {
      REQUIRE(cudaMemcpy(p_, host.data(), n_ * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    }
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray() { cudaFree(p_); }
  [[nodiscard]] T* get() const { return p_; }
  [[nodiscard]] std::vector<T> host() const {
    std::vector<T> v(n_);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    if (n_ > 0) {
      REQUIRE(cudaMemcpy(v.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
    }
    return v;
  }

 private:
  T* p_ = nullptr;
  std::size_t n_;
};

using Source = std::variant<const char*, int>;
autan::Mesh mesh_of(const Source& s) {
  if (const auto* file = std::get_if<const char*>(&s)) {
    return autan::io::read_mesh(galerkin_fixture::mesh_dir / *file);
  }
  return galerkin_fixture::wall_layers(std::get<int>(s));
}

// Where the lines come from: smoother_lines; every column of cells (all cells in lines); the
// columns over x < 1/2 (lines and lone cells); none.
enum class LineSource { found, columns, half, none };

autan::Lines lines_of(const autan::Mesh& mesh, const autan::Geometry& g, LineSource kind) {
  if (kind == LineSource::found) return autan::smoother_lines(mesh, g);
  autan::Lines out;
  if (kind == LineSource::none) return out;
  // Cells over the same point of the plane, by increasing z.
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  auto key = [&](std::size_t c) {
    return std::tuple{std::round(g.cell_centroid_x[c] * 1e6),
                      std::round(g.cell_centroid_y[c] * 1e6), g.cell_centroid_z[c]};
  };
  std::ranges::sort(order, [&](std::size_t a, std::size_t b) { return key(a) < key(b); });
  for (std::size_t k = 0; k < n; ++k) {
    const auto c = order[k];
    if (kind == LineSource::half && g.cell_centroid_x[c] > 0.5) continue;
    out.cells.push_back(static_cast<autan::index_t>(c));
    const bool last = k + 1 == n || std::get<0>(key(order[k + 1])) != std::get<0>(key(c)) ||
                      std::get<1>(key(order[k + 1])) != std::get<1>(key(c));
    if (last) out.offsets.push_back(static_cast<autan::offset_t>(out.cells.size()));
  }
  return out;
}

// A colouring of the units (lines, then each cell outside them) in which no two units of one
// colour share a face: each unit takes the least colour its coloured neighbours do not have.
std::vector<std::int32_t> unit_colours(const autan::Mesh& mesh, const autan::Lines& lines,
                                       std::int32_t& n_colours) {
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  std::vector<std::size_t> unit(n, SIZE_MAX);
  std::size_t n_units = 0;
  for (std::size_t l = 0; l + 1 < lines.offsets.size(); ++l, ++n_units) {
    for (auto k = lines.offsets[l]; k < lines.offsets[l + 1]; ++k) {
      unit[static_cast<std::size_t>(lines.cells[static_cast<std::size_t>(k)])] = n_units;
    }
  }
  for (std::size_t c = 0; c < n; ++c) {
    if (unit[c] == SIZE_MAX) unit[c] = n_units++;
  }
  std::vector<std::vector<std::size_t>> adjacent(n_units);
  for (std::size_t f = 0; f < ni; ++f) {
    const auto a = unit[static_cast<std::size_t>(mesh.owner()[f])];
    const auto b = unit[static_cast<std::size_t>(mesh.neighbour()[f])];
    if (a != b) {
      adjacent[a].push_back(b);
      adjacent[b].push_back(a);
    }
  }
  std::vector<std::int32_t> of_unit(n_units, -1);
  n_colours = 0;
  for (std::size_t u = 0; u < n_units; ++u) {
    std::vector<bool> taken(static_cast<std::size_t>(n_colours) + 1, false);
    for (const auto v : adjacent[u]) {
      if (of_unit[v] >= 0) taken[static_cast<std::size_t>(of_unit[v])] = true;
    }
    std::int32_t c = 0;
    while (taken[static_cast<std::size_t>(c)]) ++c;
    of_unit[u] = c;
    n_colours = std::max(n_colours, c + 1);
  }
  std::vector<std::int32_t> colours(n);
  for (std::size_t c = 0; c < n; ++c) colours[c] = of_unit[unit[c]];
  return colours;
}

std::int64_t bits(double v) { return std::bit_cast<std::int64_t>(v); }

}  // namespace

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
  const auto nf = static_cast<std::size_t>(mesh.n_faces());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto lines = lines_of(mesh, g, kind);
  std::int32_t n_colours = 0;
  const auto colours = unit_colours(mesh, lines, n_colours);
  // The host sums each cell's faces by a global id that follows neither the face index nor the
  // device's slot order (below), so that a device reading blocks by position could not pass.
  std::vector<std::int64_t> face_id(ni);
  for (std::size_t f = 0; f < ni; ++f) face_id[f] = static_cast<std::int64_t>((f * 7919) % ni);
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

  // The device layout: per cell, its faces of every kind in decreasing face index.
  std::vector<autan::offset_t> offsets(n + 1, 0);
  for (std::size_t f = 0; f < nf; ++f) {
    ++offsets[static_cast<std::size_t>(mesh.owner()[f]) + 1];
    if (f < ni) ++offsets[static_cast<std::size_t>(mesh.neighbour()[f]) + 1];
  }
  for (std::size_t c = 0; c < n; ++c) offsets[c + 1] += offsets[c];
  std::vector<autan::index_t> faces(static_cast<std::size_t>(offsets[n]));
  std::vector<autan::offset_t> fill(offsets.begin(), offsets.end() - 1);
  for (std::size_t f = nf; f-- > 0;) {
    faces[static_cast<std::size_t>(fill[static_cast<std::size_t>(mesh.owner()[f])]++)] =
        static_cast<autan::index_t>(f);
    if (f < ni) {
      faces[static_cast<std::size_t>(fill[static_cast<std::size_t>(mesh.neighbour()[f])]++)] =
          static_cast<autan::index_t>(f);
    }
  }
  const std::size_t ns = faces.size();

  // A diagonally dominant operator: diagonal blocks per cell, a block per slot.
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> diag_soa(b2 * n), diag(b2 * n), upper(b2 * ni), lower(b2 * ni);
  std::vector<Off> off(b2 * ns, Off(0));
  for (std::size_t c = 0; c < n; ++c) {
    for (std::size_t q = 0; q < b2; ++q) {
      const double v = u(rng) + (q % 6 == 0 ? 60.0 : 0.0);
      diag[b2 * c + q] = v;
      diag_soa[q * n + c] = v;
    }
    for (auto k = offsets[c]; k < offsets[c + 1]; ++k) {
      const auto s = static_cast<std::size_t>(k);
      const auto f = static_cast<std::size_t>(faces[s]);
      if (f >= ni) continue;
      const bool own = static_cast<std::size_t>(mesh.owner()[f]) == c;
      for (std::size_t q = 0; q < b2; ++q) {
        const auto v = static_cast<Off>(u(rng));
        off[q * ns + s] = v;
        (own ? upper : lower)[b2 * f + q] = static_cast<double>(v);
      }
    }
  }
  std::vector<double> b(5 * n), x0(5 * n);
  for (std::size_t k = 0; k < 5 * n; ++k) {
    b[k] = std::sin(0.31 * static_cast<double>(k));
    x0[k] = std::cos(0.17 * static_cast<double>(k));
  }

  host.set_operator(diag, upper, lower);
  std::vector<double> x_host = x0, y_host(5 * n, 0.0);
  host.sweep(b, x_host, 2, [](std::span<double>) {});
  host.multiply(x0, y_host);

  autan::solvers::gpu::LineSweep device(host, offsets, faces);
  const DeviceArray<double> d_diag(diag_soa);
  const DeviceArray<Off> d_off(off);
  const DeviceArray<double> d_b(b), d_x(x0), d_x0(x0), d_y(std::vector<double>(5 * n, 0.0));
  device.set_operator(d_diag.get(), d_off.get());
  device.sweep(d_b.get(), d_x.get(), 2);
  device.multiply(d_x0.get(), d_y.get());
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
  const auto nf = static_cast<std::size_t>(mesh.n_faces());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto lines = autan::smoother_lines(mesh, g);
  std::int32_t n_colours = 0;
  const auto colours = unit_colours(mesh, lines, n_colours);
  const autan::solvers::LineSmoother host(mesh, lines.offsets, lines.cells, colours, n_colours);
  std::vector<autan::offset_t> offsets{0};
  std::vector<autan::index_t> faces;
  for (std::size_t c = 0; c < n; ++c) {
    for (std::size_t f = 0; f < nf; ++f) {
      if (static_cast<std::size_t>(mesh.owner()[f]) == c ||
          (f < ni && static_cast<std::size_t>(mesh.neighbour()[f]) == c)) {
        faces.push_back(static_cast<autan::index_t>(f));
      }
    }
    offsets.push_back(static_cast<autan::offset_t>(faces.size()));
  }
  autan::solvers::gpu::LineSweep device(host, offsets, faces);
  const DeviceArray<double> d_b(std::vector<double>(5 * n, 1.0)), d_x(std::vector<double>(5 * n));
  CHECK_THROWS_AS(device.sweep(d_b.get(), d_x.get(), 1), std::logic_error);
  const DeviceArray<double> zero(std::vector<double>(b2 * n, 0.0));
  const DeviceArray<double> off(std::vector<double>(b2 * faces.size(), 0.0));
  CHECK_THROWS_AS(device.set_operator(zero.get(), off.get()), std::domain_error);
  CHECK_THROWS_AS(device.sweep(d_b.get(), d_x.get(), 1), std::logic_error);
}
