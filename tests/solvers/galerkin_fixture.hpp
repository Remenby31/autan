#pragma once

// Shared by the Galerkin tests: a wall-layer mesh with lines, hierarchies, slot maps, operators in
// host form, the host multigrid's Galerkin order (the oracle) and the lists applied on the host.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "autan/core/cellwise.hpp"
#include "autan/core/geometry.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/solvers/agglomeration.hpp"
#include "autan/solvers/device_order.hpp"
#include "autan/solvers/galerkin_lists.hpp"

namespace galerkin_fixture {

namespace fs = std::filesystem;
using autan::index_t;
using autan::offset_t;
using autan::solvers::Level;
using autan::solvers::gpu::EdgeSlots;
using autan::solvers::gpu::GalerkinLists;

inline constexpr std::size_t b2 = 25;
inline const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
inline const std::array<const char*, 4> meshes = {
    "tet-gmsh/tet-gmsh.ascii.msh", "hex/hex.ascii.msh", "poly/polyMesh", "hybrid/hybrid.ascii.msh"};

// Hexahedra on [0,1]^2 x [0,1/2], nx x nx in the plane, layers growing by 1.2 from 1e-4 at the
// wall z = 0: anisotropic wall cells, so the hierarchy holds lines.
inline autan::Mesh wall_layers(int nx) {
  autan::CellwiseMesh m;
  autan::CellZone zone;
  zone.name = "fluid";
  zone.kind = autan::CellKind::fluid;
  m.cell_zones.push_back(zone);
  autan::FaceZone wall;
  wall.name = "wall";
  wall.kind = autan::BcKind::wall;
  m.face_zones.push_back(wall);
  const double dx = 1.0 / nx;
  std::vector<double> z{0.0};
  for (double d = 1e-4; z.back() < 0.5; d = std::min(dx, 1.2 * d)) z.push_back(z.back() + d);
  const int nz = static_cast<int>(z.size()) - 1, p = nx + 1;
  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= nx; ++j) {
      for (int i = 0; i <= nx; ++i) {
        m.x.push_back(i * dx);
        m.y.push_back(j * dx);
        m.z.push_back(z[static_cast<std::size_t>(k)]);
      }
    }
  }
  auto v = [&](int i, int j, int k) { return static_cast<index_t>(i + p * (j + p * k)); };
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < nx; ++j) {
      for (int i = 0; i < nx; ++i) {
        m.add_cell(autan::CellShape::hexa,
                   std::array<index_t, 8>{v(i, j, k), v(i + 1, j, k), v(i + 1, j + 1, k),
                                          v(i, j + 1, k), v(i, j, k + 1), v(i + 1, j, k + 1),
                                          v(i + 1, j + 1, k + 1), v(i, j + 1, k + 1)},
                   0);
        if (k == 0) {
          m.add_labelled_face(std::array<index_t, 4>{v(i, j, 0), v(i, j + 1, 0), v(i + 1, j + 1, 0),
                                                     v(i + 1, j, 0)},
                              0);
        }
      }
    }
  }
  return autan::assemble(autan::to_face_based(std::move(m)));
}

inline std::vector<Level> hierarchy(const autan::Mesh& mesh) {
  autan::solvers::AgglomerationOptions opt;
  opt.coarsest_nodes = 4;
  return autan::solvers::agglomerate(mesh, autan::compute_geometry(mesh), opt);
}

// Slots of a level laid out as a CSR over its device nodes, each node's edges in increasing
// edge index, followed by `padding` slots no edge uses.
inline EdgeSlots csr_slots(const Level& l, const std::vector<index_t>& order, offset_t padding,
                           offset_t& n_slots) {
  const std::size_t n = l.n_nodes, ne = l.owner.size();
  std::vector<index_t> rank(n);
  for (std::size_t j = 0; j < n; ++j)
    rank[static_cast<std::size_t>(order[j])] = static_cast<index_t>(j);
  std::vector<offset_t> start(n + 1, 0);
  for (std::size_t e = 0; e < ne; ++e) {
    ++start[static_cast<std::size_t>(rank[static_cast<std::size_t>(l.owner[e])]) + 1];
    ++start[static_cast<std::size_t>(rank[static_cast<std::size_t>(l.neighbour[e])]) + 1];
  }
  for (std::size_t j = 0; j < n; ++j) start[j + 1] += start[j];
  EdgeSlots s;
  s.owner.resize(ne);
  s.neighbour.resize(ne);
  std::vector<offset_t> fill(start.begin(), start.end() - 1);
  for (std::size_t e = 0; e < ne; ++e) {
    s.owner[e] = fill[static_cast<std::size_t>(rank[static_cast<std::size_t>(l.owner[e])])]++;
    s.neighbour[e] =
        fill[static_cast<std::size_t>(rank[static_cast<std::size_t>(l.neighbour[e])])]++;
  }
  n_slots = start.back() + padding;
  return s;
}

// A level's operator in host form: diagonal per node, upper = A(owner, neighbour) and
// lower = A(neighbour, owner) per edge, row-major 5 x 5.
struct Blocks {
  std::vector<double> diag, upper, lower;
};

// Values over 120 binades with both signs, so that any other summation order rounds otherwise.
inline Blocks spread(const Level& l, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> mantissa(1.0, 2.0);
  std::uniform_int_distribution<int> exponent(-60, 60), sign(0, 1);
  auto fill = [&](std::size_t n) {
    std::vector<double> v(n);
    for (double& x : v) x = (sign(rng) ? -1.0 : 1.0) * std::ldexp(mantissa(rng), exponent(rng));
    return v;
  };
  return {fill(b2 * l.n_nodes), fill(b2 * l.owner.size()), fill(b2 * l.owner.size())};
}

// The host multigrid's coarse operator (Multigrid::set_operator): diagonals first, then the
// edges inside agglomerates (upper, then lower), then the coarse edges, each fine edge's blocks
// in increasing edge index, a flipped edge's swapped.
inline Blocks host_galerkin(const Level& f, const Level& c, const Blocks& a) {
  Blocks g;
  g.diag.assign(b2 * c.n_nodes, 0.0);
  g.upper.assign(b2 * c.owner.size(), 0.0);
  g.lower.assign(b2 * c.owner.size(), 0.0);
  auto add = [](double* to, const double* from) {
    for (std::size_t k = 0; k < b2; ++k) to[k] += from[k];
  };
  for (std::size_t i = 0; i < f.n_nodes; ++i) {
    add(&g.diag[b2 * static_cast<std::size_t>(f.coarse_node[i])], &a.diag[b2 * i]);
  }
  for (std::size_t e = 0; e < f.owner.size(); ++e) {
    const index_t E = f.coarse_edge[e];
    if (E < 0) {
      double* d = &g.diag[b2 * static_cast<std::size_t>(
                                   f.coarse_node[static_cast<std::size_t>(f.owner[e])])];
      add(d, &a.upper[b2 * e]);
      add(d, &a.lower[b2 * e]);
      continue;
    }
    const bool flip = f.coarse_flipped[e] != 0;
    add(&g.upper[b2 * static_cast<std::size_t>(E)], flip ? &a.lower[b2 * e] : &a.upper[b2 * e]);
    add(&g.lower[b2 * static_cast<std::size_t>(E)], flip ? &a.upper[b2 * e] : &a.lower[b2 * e]);
  }
  return g;
}

// The lists applied to the fine level in its device layout: each block the sum, from +0.0, of
// its list's blocks in list order.
struct Device {
  std::vector<double> diag, off;
};

inline Device to_device(const Blocks& a, const std::vector<index_t>& order, const EdgeSlots& s,
                        offset_t n_slots) {
  Device d;
  d.diag.resize(b2 * order.size());
  for (std::size_t j = 0; j < order.size(); ++j) {
    std::copy_n(&a.diag[b2 * static_cast<std::size_t>(order[j])], b2, &d.diag[b2 * j]);
  }
  d.off.assign(b2 * static_cast<std::size_t>(n_slots), std::numeric_limits<double>::quiet_NaN());
  for (std::size_t e = 0; e < s.owner.size(); ++e) {
    std::copy_n(&a.upper[b2 * e], b2, &d.off[b2 * static_cast<std::size_t>(s.owner[e])]);
    std::copy_n(&a.lower[b2 * e], b2, &d.off[b2 * static_cast<std::size_t>(s.neighbour[e])]);
  }
  return d;
}

inline Device apply(const GalerkinLists& g, const Device& fine) {
  Device c;
  const std::size_t nc = g.diag_offsets.size() - 1, ns = g.off_offsets.size() - 1;
  c.diag.assign(b2 * nc, 0.0);
  c.off.assign(b2 * ns, 0.0);
  for (std::size_t J = 0; J < nc; ++J) {
    for (auto k = g.diag_offsets[J]; k < g.diag_offsets[J + 1]; ++k) {
      const std::int64_t t = g.diag[static_cast<std::size_t>(k)];
      const double* b = t >= 0 ? &fine.diag[b2 * static_cast<std::size_t>(t)]
                               : &fine.off[b2 * static_cast<std::size_t>(-(t + 1))];
      for (std::size_t q = 0; q < b2; ++q) c.diag[b2 * J + q] += b[q];
    }
  }
  for (std::size_t s = 0; s < ns; ++s) {
    for (auto k = g.off_offsets[s]; k < g.off_offsets[s + 1]; ++k) {
      const double* b =
          &fine.off[b2 * static_cast<std::size_t>(g.off[static_cast<std::size_t>(k)])];
      for (std::size_t q = 0; q < b2; ++q) c.off[b2 * s + q] += b[q];
    }
  }
  return c;
}

inline bool same_bits(const double* a, const double* b) {
  for (std::size_t q = 0; q < b2; ++q) {
    if (std::bit_cast<std::uint64_t>(a[q]) != std::bit_cast<std::uint64_t>(b[q])) return false;
  }
  return true;
}

// A permutation of a level's slots: the same blocks, written at other positions of an array
// with `padding` unused slots, so that a reader ignoring the map reads other blocks.
inline EdgeSlots permuted_slots(const EdgeSlots& s, offset_t n_slots, std::uint32_t seed) {
  std::vector<offset_t> to(static_cast<std::size_t>(n_slots));
  std::iota(to.begin(), to.end(), offset_t{0});
  std::shuffle(to.begin(), to.end(), std::mt19937(seed));
  EdgeSlots p = s;
  for (auto& v : p.owner) v = to[static_cast<std::size_t>(v)];
  for (auto& v : p.neighbour) v = to[static_cast<std::size_t>(v)];
  return p;
}

}  // namespace galerkin_fixture
