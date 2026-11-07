// A generated hexahedral slab for the physics tests: a wall-normal stretching as in the NASA TMR
// grids, optionally one cell thick in z as the 2D cases are.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include "autan/core/mesh.hpp"

namespace autan::test {

struct Slab {
  double first = 1.0e-5;  // spacing in y at y = 0
  int nz = 1;             // layers in z
  double angle = 0.0;     // turned about z (radians)
  double shear = 0.0;     // x shifted by shear y
  double warp = 0.0;      // the nodes of y = 0 at y = warp (2 z - 1) x: the wall's faces warped
  bool periodic = false;  // xmin and xmax a periodic pair (translation (1, 0, 0))
};

// Hexahedra over [0,1]^3, 8 along x, 12 along y with the spacing growing geometrically from
// `first` at y = 0 (1e-5: aspect ratio 10^4 at the wall, as in the wall-resolved 2D grids of the
// NASA TMR), `nz` along z; zones xmin ... ymax and `sides` for z = 0 and z = 1.
inline autan::Mesh stretched_slab(const Slab& slab = {}) {
  constexpr int nx = 8;
  constexpr int ny = 12;
  const int nz = slab.nz;
  double lo = 1.0, hi = 4.0;
  for (int it = 0; it < 200; ++it) {
    const double ratio = 0.5 * (lo + hi);
    (slab.first * (std::pow(ratio, ny) - 1.0) / (ratio - 1.0) > 1.0 ? hi : lo) = ratio;
  }
  autan::MeshInput in;
  double y = 0.0, dy = slab.first;
  for (int j = 0; j <= ny; ++j) {
    for (int k = 0; k <= nz; ++k) {
      const double zk = static_cast<double>(k) / nz;
      for (int i = 0; i <= nx; ++i) {
        const double xi = static_cast<double>(i) / nx;
        const double yj = j == ny ? 1.0 : j == 0 ? slab.warp * (2 * zk - 1) * xi : y;
        const double xs = xi + slab.shear * yj;
        in.x.push_back(std::cos(slab.angle) * xs - std::sin(slab.angle) * yj);
        in.y.push_back(std::sin(slab.angle) * xs + std::cos(slab.angle) * yj);
        in.z.push_back(zk);
      }
    }
    y += dy;
    dy *= lo;
  }
  auto p = [&](int i, int j, int k) {
    return static_cast<index_t>(i + (nx + 1) * (k + (nz + 1) * j));
  };
  auto cell = [&](int i, int j, int k) { return static_cast<index_t>(i + nx * (k + nz * j)); };
  in.face_node_offsets.push_back(0);
  auto face = [&](std::array<index_t, 4> nodes, bool flip, index_t c0, index_t c1, index_t zone) {
    if (flip) std::reverse(nodes.begin(), nodes.end());
    in.face_nodes.insert(in.face_nodes.end(), nodes.begin(), nodes.end());
    in.face_node_offsets.push_back(static_cast<offset_t>(in.face_nodes.size()));
    in.face_cell0.push_back(c0);
    in.face_cell1.push_back(c1);
    in.face_zone.push_back(zone);
  };
  constexpr index_t interior = 5;
  for (int j = 0; j < ny; ++j) {
    for (int k = 0; k < nz; ++k) {
      for (int i = 0; i <= nx; ++i) {
        const std::array<index_t, 4> q{p(i, j, k), p(i, j + 1, k), p(i, j + 1, k + 1),
                                       p(i, j, k + 1)};
        if (i == 0)
          face(q, true, cell(0, j, k), autan::no_cell, 0);
        else if (i == nx)
          face(q, false, cell(nx - 1, j, k), autan::no_cell, 1);
        else
          face(q, false, cell(i - 1, j, k), cell(i, j, k), interior);
      }
    }
  }
  for (int j = 0; j <= ny; ++j) {
    for (int k = 0; k < nz; ++k) {
      for (int i = 0; i < nx; ++i) {
        const std::array<index_t, 4> q{p(i, j, k), p(i, j, k + 1), p(i + 1, j, k + 1),
                                       p(i + 1, j, k)};
        if (j == 0)
          face(q, true, cell(i, 0, k), autan::no_cell, 2);
        else if (j == ny)
          face(q, false, cell(i, ny - 1, k), autan::no_cell, 3);
        else
          face(q, false, cell(i, j - 1, k), cell(i, j, k), interior);
      }
    }
  }
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      for (int k = 0; k <= nz; ++k) {
        const std::array<index_t, 4> q{p(i, j, k), p(i + 1, j, k), p(i + 1, j + 1, k),
                                       p(i, j + 1, k)};
        if (k == 0)
          face(q, true, cell(i, j, 0), autan::no_cell, 4);
        else if (k == nz)
          face(q, false, cell(i, j, nz - 1), autan::no_cell, 4);
        else
          face(q, false, cell(i, j, k - 1), cell(i, j, k), interior);
      }
    }
  }
  in.cell_zone.assign(static_cast<std::size_t>(nx * ny * nz), 0);
  for (const char* name : {"xmin", "xmax", "ymin", "ymax", "sides"}) {
    const bool pair = slab.periodic && name[0] == 'x';
    in.face_zones.push_back(
        autan::FaceZone{.name = name,
                        .bc_type = "",
                        .kind = pair ? autan::BcKind::periodic : autan::BcKind::unspecified,
                        .provenance = {}});
  }
  if (slab.periodic) in.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1, 0, 0}});
  in.face_zones.push_back(autan::FaceZone{.name = "interior",
                                          .bc_type = "",
                                          .kind = autan::BcKind::unspecified,
                                          .boundary = false,
                                          .provenance = {}});
  in.cell_zones = {autan::CellZone{
      .name = "fluid", .type = "", .kind = autan::CellKind::unspecified, .provenance = {}}};
  in.source = autan::MeshSource{.format = "memory", .path = ""};
  return autan::assemble(std::move(in));
}

}  // namespace autan::test
