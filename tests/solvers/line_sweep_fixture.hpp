#pragma once

// Shared by the device line smoother and GCR tests: device buffers, meshes and their lines, a
// colouring of the units, and the per-cell slot layout of the off-diagonal blocks.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <ranges>
#include <tuple>
#include <variant>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include "autan/core/lines.hpp"
#include "galerkin_fixture.hpp"

namespace line_sweep_fixture {

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
inline autan::Mesh mesh_of(const Source& s) {
  if (const auto* file = std::get_if<const char*>(&s)) {
    return autan::io::read_mesh(galerkin_fixture::mesh_dir / *file);
  }
  return galerkin_fixture::wall_layers(std::get<int>(s));
}

// Where the lines come from: smoother_lines; every column of cells (all cells in lines); the
// columns over x < 1/2 (lines and lone cells); none.
enum class LineSource { found, columns, half, none };

inline autan::Lines lines_of(const autan::Mesh& mesh, const autan::Geometry& g, LineSource kind) {
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
inline std::vector<std::int32_t> unit_colours(const autan::Mesh& mesh, const autan::Lines& lines,
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

inline std::int64_t bits(double v) { return std::bit_cast<std::int64_t>(v); }

// The device layout: per cell, its faces of every kind in decreasing face index.
struct Layout {
  std::vector<autan::offset_t> offsets;
  std::vector<autan::index_t> faces;
};

inline Layout reversed_layout(const autan::Mesh& mesh) {
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto nf = static_cast<std::size_t>(mesh.n_faces());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  Layout l;
  l.offsets.assign(n + 1, 0);
  for (std::size_t f = 0; f < nf; ++f) {
    ++l.offsets[static_cast<std::size_t>(mesh.owner()[f]) + 1];
    if (f < ni) ++l.offsets[static_cast<std::size_t>(mesh.neighbour()[f]) + 1];
  }
  for (std::size_t c = 0; c < n; ++c) l.offsets[c + 1] += l.offsets[c];
  l.faces.resize(static_cast<std::size_t>(l.offsets[n]));
  std::vector<autan::offset_t> fill(l.offsets.begin(), l.offsets.end() - 1);
  for (std::size_t f = nf; f-- > 0;) {
    l.faces[static_cast<std::size_t>(fill[static_cast<std::size_t>(mesh.owner()[f])]++)] =
        static_cast<autan::index_t>(f);
    if (f < ni) {
      l.faces[static_cast<std::size_t>(fill[static_cast<std::size_t>(mesh.neighbour()[f])]++)] =
          static_cast<autan::index_t>(f);
    }
  }
  return l;
}

// A host face global id that follows neither the face index nor the reversed layout.
inline std::vector<std::int64_t> scrambled_ids(std::size_t ni) {
  std::vector<std::int64_t> id(ni);
  for (std::size_t f = 0; f < ni; ++f) id[f] = static_cast<std::int64_t>((f * 7919) % ni);
  return id;
}

// A diagonally dominant operator in both forms: the device's (diagonal blocks in structure of
// arrays, a block per slot) and the host LineSmoother's (row-major diagonal, upper, lower).
template <class Off>
struct Operator {
  std::vector<double> diag_soa, diag, upper, lower;
  std::vector<Off> off;
};

template <class Off>
Operator<Off> dominant(const autan::Mesh& mesh, const Layout& l, std::uint32_t seed) {
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const std::size_t ns = l.faces.size();
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Operator<Off> a;
  a.diag_soa.resize(b2 * n);
  a.diag.resize(b2 * n);
  a.upper.resize(b2 * ni);
  a.lower.resize(b2 * ni);
  a.off.assign(b2 * ns, Off(0));
  for (std::size_t c = 0; c < n; ++c) {
    for (std::size_t q = 0; q < b2; ++q) {
      const double v = u(rng) + (q % 6 == 0 ? 60.0 : 0.0);
      a.diag[b2 * c + q] = v;
      a.diag_soa[q * n + c] = v;
    }
    for (auto k = l.offsets[c]; k < l.offsets[c + 1]; ++k) {
      const auto s = static_cast<std::size_t>(k);
      const auto f = static_cast<std::size_t>(l.faces[s]);
      if (f >= ni) continue;
      const bool own = static_cast<std::size_t>(mesh.owner()[f]) == c;
      for (std::size_t q = 0; q < b2; ++q) {
        const auto v = static_cast<Off>(u(rng));
        a.off[q * ns + s] = v;
        (own ? a.upper : a.lower)[b2 * f + q] = static_cast<double>(v);
      }
    }
  }
  return a;
}

}  // namespace line_sweep_fixture
