#include "fixtures.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace cfd::test {
namespace {


// Labels every cell face whose nodes all lie on one side of the unit box
// [0,lx]x[0,1]x[0,1]. Boundary nodes are never perturbed, so the test is exact.
void label_box_sides(CellwiseMesh& mesh, double lx) {
  const std::array<const char*, 6> names{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
  for (std::size_t s = 0; s < names.size(); ++s) {
    mesh.face_zones.push_back(FaceZone{.name = names[s],
                                       .bc_type = "",
                                       .kind = BcKind::unspecified,
                                       .provenance = Provenance{.source_id = static_cast<std::int64_t>(s),
                                                                .detail = "test fixture"}});
  }
  const std::size_t n_faces = mesh.face_node_offsets.size() - 1;
  for (std::size_t f = 0; f < n_faces; ++f) {
    const auto begin = mesh.face_nodes.begin() + mesh.face_node_offsets[f];
    const auto end = mesh.face_nodes.begin() + mesh.face_node_offsets[f + 1];
    for (std::size_t s = 0; s < names.size(); ++s) {
      const std::vector<double>& coord = s < 2 ? mesh.x : (s < 4 ? mesh.y : mesh.z);
      const double side = s % 2 == 0 ? 0.0 : (s < 2 ? lx : 1.0);
      if (std::all_of(begin, end, [&](index_t p) { return coord[static_cast<std::size_t>(p)] == side; })) {
        mesh.add_labelled_face(std::vector<index_t>(begin, end), static_cast<index_t>(s));
      }
    }
  }
}

index_t add_point(CellwiseMesh& mesh, double x, double y, double z) {
  mesh.x.push_back(x);
  mesh.y.push_back(y);
  mesh.z.push_back(z);
  return static_cast<index_t>(mesh.x.size() - 1);
}

}  // namespace

std::string to_string(BoxFill fill) {
  switch (fill) {
    case BoxFill::hexa: return "hexa";
    case BoxFill::tetra: return "tetra";
    case BoxFill::prism: return "prism";
    case BoxFill::pyramid: return "pyramid";
    case BoxFill::hybrid: return "hybrid";
  }
  return "?";
}

CellwiseMesh box(int n, BoxFill fill, double perturbation, bool label_sides) {
  CellwiseMesh mesh;
  mesh.source = MeshSource{.format = "memory", .path = ""};
  mesh.cell_zones.push_back(CellZone{.name = "fluid",
                                     .type = "fluid",
                                     .kind = CellKind::fluid,
                                     .provenance = Provenance{.source_id = 0, .detail = "test fixture"}});
  const int m = n + 1;
  const double h = 1.0 / n;
  std::mt19937 rng(12345);
  auto jitter = [&]() { return perturbation * h * (2.0 * (static_cast<double>(rng()) / 4294967296.0) - 1.0); };
  for (int k = 0; k < m; ++k) {
    for (int j = 0; j < m; ++j) {
      for (int i = 0; i < m; ++i) {
        const bool interior = i > 0 && i < n && j > 0 && j < n && k > 0 && k < n;
        double x = i * h, y = j * h, z = k * h;
        if (interior && perturbation > 0.0) {
          x += jitter();
          y += jitter();
          z += jitter();
        }
        add_point(mesh, x, y, z);
      }
    }
  }
  auto p = [m](int i, int j, int k) { return static_cast<index_t>(i + m * (j + m * k)); };

  const int half = n / 2;
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        auto v = [&](int bx, int by, int bz) { return p(i + bx, j + by, k + bz); };
        const std::array<index_t, 8> hex{v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(0, 1, 0),
                                         v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1)};
        BoxFill local = fill;
        if (fill == BoxFill::hybrid) {
          if (i < half && k < half) local = (i + j + k) % 2 == 0 ? BoxFill::hexa : BoxFill::pyramid;
          else if (k < half) local = BoxFill::prism;
          else if (i >= half) local = BoxFill::tetra;
          else local = BoxFill::hybrid;  // x-extruded prisms
        }
        switch (local) {
          case BoxFill::hexa:
            mesh.add_cell(CellShape::hexa, hex, 0);
            break;
          case BoxFill::tetra: {
            // Kuhn subdivision: one tetrahedron per order of the three axis
            // steps from v000 to v111; odd permutations are mirrored.
            const std::array<std::array<int, 3>, 6> orders{
                {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {0, 2, 1}, {2, 1, 0}, {1, 0, 2}}};
            for (std::size_t o = 0; o < orders.size(); ++o) {
              std::array<int, 3> b{0, 0, 0};
              std::array<index_t, 4> tet{};
              tet[0] = v(0, 0, 0);
              for (std::size_t s = 0; s < 3; ++s) {
                b[static_cast<std::size_t>(orders[o][s])] = 1;
                tet[s + 1] = v(b[0], b[1], b[2]);
              }
              if (o >= 3) std::swap(tet[1], tet[2]);
              mesh.add_cell(CellShape::tetra, tet, 0);
            }
            break;
          }
          case BoxFill::prism:
            mesh.add_cell(CellShape::prism,
                          std::array<index_t, 6>{v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(0, 0, 1),
                                                 v(1, 0, 1), v(1, 1, 1)},
                          0);
            mesh.add_cell(CellShape::prism,
                          std::array<index_t, 6>{v(0, 0, 0), v(1, 1, 0), v(0, 1, 0), v(0, 0, 1),
                                                 v(1, 1, 1), v(0, 1, 1)},
                          0);
            break;
          case BoxFill::hybrid:
            mesh.add_cell(CellShape::prism,
                          std::array<index_t, 6>{v(0, 0, 0), v(0, 1, 0), v(0, 1, 1), v(1, 0, 0),
                                                 v(1, 1, 0), v(1, 1, 1)},
                          0);
            mesh.add_cell(CellShape::prism,
                          std::array<index_t, 6>{v(0, 0, 0), v(0, 1, 1), v(0, 0, 1), v(1, 0, 0),
                                                 v(1, 1, 1), v(1, 0, 1)},
                          0);
            break;
          case BoxFill::pyramid: {
            double cx = 0, cy = 0, cz = 0;
            for (const index_t q : hex) {
              cx += mesh.x[static_cast<std::size_t>(q)];
              cy += mesh.y[static_cast<std::size_t>(q)];
              cz += mesh.z[static_cast<std::size_t>(q)];
            }
            const index_t centre = add_point(mesh, cx / 8, cy / 8, cz / 8);
            // Bases are the outward hex faces reversed, so that their normal
            // points to the apex as in the Gmsh reference pyramid.
            const std::array<std::array<int, 4>, 6> bases{
                {{1, 2, 3, 0}, {7, 6, 5, 4}, {4, 5, 1, 0}, {5, 6, 2, 1}, {6, 7, 3, 2}, {7, 4, 0, 3}}};
            for (const auto& base : bases) {
              mesh.add_cell(CellShape::pyramid,
                            std::array<index_t, 5>{hex[static_cast<std::size_t>(base[0])],
                                                   hex[static_cast<std::size_t>(base[1])],
                                                   hex[static_cast<std::size_t>(base[2])],
                                                   hex[static_cast<std::size_t>(base[3])], centre},
                            0);
            }
            break;
          }
        }
      }
    }
  }
  if (label_sides) label_box_sides(mesh, 1.0);
  return mesh;
}

CellwiseMesh hanging_node_box() {
  CellwiseMesh mesh;
  mesh.source = MeshSource{.format = "memory", .path = ""};
  for (const char* name : {"coarse", "fine"}) {
    mesh.cell_zones.push_back(CellZone{.name = name,
                                       .type = "fluid",
                                       .kind = CellKind::fluid,
                                       .provenance = Provenance{.source_id = -1, .detail = "test fixture"}});
  }
  constexpr int n = 10;
  std::array<std::array<index_t, 2>, 2> q{};  // x = 0 corners, q[y][z]
  for (int y = 0; y < 2; ++y)
    for (int z = 0; z < 2; ++z) q[static_cast<std::size_t>(y)][static_cast<std::size_t>(z)] = add_point(mesh, 0.0, y, z);
  std::array<std::vector<index_t>, 2> grid;  // planes x = 1 and x = 2
  for (std::size_t plane = 0; plane < 2; ++plane) {
    for (int k = 0; k <= n; ++k)
      for (int j = 0; j <= n; ++j)
        grid[plane].push_back(add_point(mesh, 1.0 + static_cast<double>(plane), double(j) / n, double(k) / n));
  }
  auto g = [&](std::size_t plane, int j, int k) { return grid[plane][static_cast<std::size_t>(j + (n + 1) * k)]; };

  std::vector<offset_t> offsets{0};
  std::vector<index_t> nodes;
  auto close_face = [&]() { offsets.push_back(static_cast<offset_t>(nodes.size())); };
  nodes.insert(nodes.end(), {q[0][0], q[0][1], q[1][1], q[1][0]});  // x = 0
  close_face();
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      nodes.insert(nodes.end(), {g(0, j, k), g(0, j + 1, k), g(0, j + 1, k + 1), g(0, j, k + 1)});
      close_face();
    }
  }
  nodes.push_back(q[0][0]);  // y = 0
  for (int k = 0; k <= n; ++k) nodes.push_back(g(0, 0, k));
  nodes.push_back(q[0][1]);
  close_face();
  nodes.insert(nodes.end(), {q[1][0], q[1][1]});  // y = 1
  for (int k = n; k >= 0; --k) nodes.push_back(g(0, n, k));
  close_face();
  nodes.insert(nodes.end(), {q[0][0], q[1][0]});  // z = 0
  for (int j = n; j >= 0; --j) nodes.push_back(g(0, j, 0));
  close_face();
  nodes.push_back(q[0][1]);  // z = 1
  for (int j = 0; j <= n; ++j) nodes.push_back(g(0, j, n));
  nodes.push_back(q[1][1]);
  close_face();
  mesh.add_polyhedron(offsets, nodes, 0);

  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      mesh.add_cell(CellShape::hexa,
                    std::array<index_t, 8>{g(0, j, k), g(1, j, k), g(1, j + 1, k), g(0, j + 1, k),
                                           g(0, j, k + 1), g(1, j, k + 1), g(1, j + 1, k + 1),
                                           g(0, j + 1, k + 1)},
                    1);
    }
  }
  mesh.face_zones.push_back(FaceZone{.name = "inlet",
                                     .bc_type = "velocity-inlet",
                                     .kind = BcKind::inlet,
                                     .provenance = Provenance{.source_id = 7, .detail = "test fixture"}});
  mesh.add_labelled_face(std::array<index_t, 4>{q[0][0], q[0][1], q[1][1], q[1][0]}, 0);
  return mesh;
}

ReferenceCell affine_cell(CellShape shape, const std::array<std::array<double, 3>, 3>& a,
                          const std::array<double, 3>& b) {
  std::vector<std::array<double, 3>> ref;
  double volume = 0.0;
  std::array<double, 3> centroid{};
  switch (shape) {
    case CellShape::tetra:
      ref = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
      volume = 1.0 / 6.0;
      centroid = {0.25, 0.25, 0.25};
      break;
    case CellShape::pyramid:
      ref = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.5, 0.5, 1}};
      volume = 1.0 / 3.0;
      centroid = {0.5, 0.5, 0.25};
      break;
    case CellShape::prism:
      ref = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}};
      volume = 0.5;
      centroid = {1.0 / 3.0, 1.0 / 3.0, 0.5};
      break;
    case CellShape::hexa:
      ref = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
      volume = 1.0;
      centroid = {0.5, 0.5, 0.5};
      break;
  }
  auto map = [&](const std::array<double, 3>& r) {
    std::array<double, 3> out{};
    for (std::size_t i = 0; i < 3; ++i) out[i] = a[i][0] * r[0] + a[i][1] * r[1] + a[i][2] * r[2] + b[i];
    return out;
  };
  const double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                     a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                     a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  ReferenceCell cell{.mesh = {}, .volume = det * volume, .centroid = map(centroid)};
  cell.mesh.source = MeshSource{.format = "memory", .path = ""};
  cell.mesh.cell_zones.push_back(CellZone{.name = "fluid", .type = "", .kind = CellKind::unspecified, .provenance = {}});
  std::vector<index_t> nodes;
  for (const auto& r : ref) {
    const auto x = map(r);
    nodes.push_back(add_point(cell.mesh, x[0], x[1], x[2]));
  }
  cell.mesh.add_cell(shape, nodes, 0);
  return cell;
}

}  // namespace cfd::test
