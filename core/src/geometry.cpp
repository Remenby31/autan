#include "autan/core/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#include "autan/core/error.hpp"

namespace autan {
namespace {

struct Vec3 {
  double x, y, z;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(double s, Vec3 a) { return {s * a.x, s * a.y, s * a.z}; }
double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double norm(Vec3 a) { return std::hypot(a.x, a.y, a.z); }

std::size_t to_size(index_t value) { return static_cast<std::size_t>(value); }

Vec3 point(const Mesh& mesh, index_t p) {
  const auto i = to_size(p);
  return {mesh.x()[i], mesh.y()[i], mesh.z()[i]};
}

Vec3 node_mean(const Mesh& mesh, std::span<const index_t> nodes) {
  // Accumulate relative to the first node so the mean of far-from-origin
  // points keeps the precision of their differences.
  const Vec3 origin = point(mesh, nodes[0]);
  Vec3 sum{0.0, 0.0, 0.0};
  for (const index_t n : nodes) sum = sum + (point(mesh, n) - origin);
  return origin + (1.0 / static_cast<double>(nodes.size())) * sum;
}

}  // namespace

Geometry compute_geometry(const Mesh& mesh) {
  const auto n_faces = to_size(mesh.n_faces());
  const auto n_cells = to_size(mesh.n_cells());
  Geometry g;
  g.face_area_x.resize(n_faces);
  g.face_area_y.resize(n_faces);
  g.face_area_z.resize(n_faces);
  g.face_area.resize(n_faces);
  g.face_centroid_x.resize(n_faces);
  g.face_centroid_y.resize(n_faces);
  g.face_centroid_z.resize(n_faces);

  for (std::size_t f = 0; f < n_faces; ++f) {
    const auto nodes = mesh.face_nodes(static_cast<index_t>(f));
    const std::size_t n = nodes.size();
    const Vec3 m = node_mean(mesh, nodes);
    Vec3 area{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < n; ++i) {
      area = area + 0.5 * cross(point(mesh, nodes[i]) - m, point(mesh, nodes[(i + 1) % n]) - m);
    }
    const double magnitude = norm(area);
    if (!(magnitude > 0.0) || !std::isfinite(magnitude)) {
      throw Error(ErrorCode::geometry_degenerate_face,
                  "face " + std::to_string(f) + " has zero or non-finite area");
    }
    const Vec3 normal = (1.0 / magnitude) * area;
    // Fan triangles are weighted by their area projected on the face normal,
    // so a triangle folded back by warping counts negatively.
    Vec3 moment{0.0, 0.0, 0.0};
    double weight = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const Vec3 a = point(mesh, nodes[i]) - m;
      const Vec3 b = point(mesh, nodes[(i + 1) % n]) - m;
      const double w = dot(0.5 * cross(a, b), normal);
      moment = moment + (w / 3.0) * (a + b);
      weight += w;
    }
    const Vec3 centroid = m + (1.0 / weight) * moment;
    g.face_area_x[f] = area.x;
    g.face_area_y[f] = area.y;
    g.face_area_z[f] = area.z;
    g.face_area[f] = magnitude;
    g.face_centroid_x[f] = centroid.x;
    g.face_centroid_y[f] = centroid.y;
    g.face_centroid_z[f] = centroid.z;
  }

  // Ghost cells have only the faces they share with owned cells; their
  // geometry comes from the owning partition through the halo exchange.
  constexpr double unset = std::numeric_limits<double>::quiet_NaN();
  g.cell_volume.assign(n_cells, unset);
  g.cell_centroid_x.assign(n_cells, unset);
  g.cell_centroid_y.assign(n_cells, unset);
  g.cell_centroid_z.assign(n_cells, unset);
  for (std::size_t c = 0; c < to_size(mesh.n_owned_cells()); ++c) {
    const auto cell = static_cast<index_t>(c);
    const auto faces = mesh.cell_faces(cell);
    // Apex of the decomposition: any point gives the exact signed volume and
    // centroid; the mean of the face centroids keeps the terms small.
    Vec3 apex{0.0, 0.0, 0.0};
    for (const index_t f : faces) {
      const auto i = to_size(f);
      apex = apex + Vec3{g.face_centroid_x[i], g.face_centroid_y[i], g.face_centroid_z[i]};
    }
    apex = (1.0 / static_cast<double>(faces.size())) * apex;

    // Work in units of the cell size so that volume moments neither overflow
    // nor underflow at extreme length scales.
    double length = 0.0;
    for (const index_t f : faces) length = std::max(length, std::sqrt(g.face_area[to_size(f)]));
    const double inv = 1.0 / length;
    double scaled_volume = 0.0;
    Vec3 moment{0.0, 0.0, 0.0};
    for (const index_t f : faces) {
      const double sign = mesh.owner()[to_size(f)] == cell ? 1.0 : -1.0;
      const auto nodes = mesh.face_nodes(f);
      const std::size_t n = nodes.size();
      const Vec3 m = inv * (node_mean(mesh, nodes) - apex);
      for (std::size_t i = 0; i < n; ++i) {
        const Vec3 a = inv * (point(mesh, nodes[i]) - apex);
        const Vec3 b = inv * (point(mesh, nodes[(i + 1) % n]) - apex);
        // Tetrahedron (apex, m, a, b): V = (1/3) A_tri . (m - apex).
        const double v = sign * dot(0.5 * cross(a - m, b - m), m) / 3.0;
        scaled_volume += v;
        moment = moment + (v / 4.0) * (m + a + b);
      }
    }
    const double volume = scaled_volume * length * length * length;
    if (!(scaled_volume > 0.0) || !std::isfinite(volume) || !(volume > 0.0)) {
      throw Error(ErrorCode::geometry_non_positive_volume,
                  "cell " + std::to_string(c) + " has non-positive or non-finite volume");
    }
    const Vec3 centroid = apex + (length / scaled_volume) * moment;
    g.cell_volume[c] = volume;
    g.cell_centroid_x[c] = centroid.x;
    g.cell_centroid_y[c] = centroid.y;
    g.cell_centroid_z[c] = centroid.z;
  }
  return g;
}

GeometryView view(const Geometry& g) noexcept {
  return GeometryView{
      .face_area_x = g.face_area_x.data(),
      .face_area_y = g.face_area_y.data(),
      .face_area_z = g.face_area_z.data(),
      .face_area = g.face_area.data(),
      .face_centroid_x = g.face_centroid_x.data(),
      .face_centroid_y = g.face_centroid_y.data(),
      .face_centroid_z = g.face_centroid_z.data(),
      .cell_volume = g.cell_volume.data(),
      .cell_centroid_x = g.cell_centroid_x.data(),
      .cell_centroid_y = g.cell_centroid_y.data(),
      .cell_centroid_z = g.cell_centroid_z.data(),
  };
}

std::vector<double> cell_closure_error(const Mesh& mesh, const Geometry& g) {
  const auto n_faces = to_size(mesh.n_faces());
  if (g.face_area_x.size() != n_faces || g.face_area_y.size() != n_faces ||
      g.face_area_z.size() != n_faces || g.face_area.size() != n_faces) {
    throw Error(ErrorCode::mesh_inconsistent_sizes, "geometry does not belong to this mesh");
  }
  std::vector<double> error(to_size(mesh.n_owned_cells()));
  for (index_t c = 0; c < mesh.n_owned_cells(); ++c) {
    Vec3 sum{0.0, 0.0, 0.0};
    double magnitude = 0.0;
    for (const index_t f : mesh.cell_faces(c)) {
      const auto i = to_size(f);
      const double sign = mesh.owner()[i] == c ? 1.0 : -1.0;
      sum = sum + sign * Vec3{g.face_area_x[i], g.face_area_y[i], g.face_area_z[i]};
      magnitude += g.face_area[i];
    }
    error[to_size(c)] = norm(sum) / magnitude;
  }
  return error;
}

}  // namespace autan
