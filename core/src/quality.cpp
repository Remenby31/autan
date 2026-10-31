#include "autan/core/quality.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>

#include "autan/core/error.hpp"

namespace autan {
namespace {

struct Vec3 {
  double x, y, z;
};

Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator*(double s, Vec3 a) { return {s * a.x, s * a.y, s * a.z}; }
double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double norm(Vec3 a) { return std::hypot(a.x, a.y, a.z); }

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

double angle_deg(Vec3 a, Vec3 b) {
  const double c = std::clamp(dot(a, b) / (norm(a) * norm(b)), -1.0, 1.0);
  return std::acos(c) * 180.0 / std::numbers::pi;
}

}  // namespace

MeshQuality compute_quality(const Mesh& mesh, const Geometry& g) {
  const auto n_faces = at(mesh.n_faces());
  const auto n_interior = at(mesh.n_interior_faces());
  if (g.face_area.size() != n_faces || g.cell_volume.size() != at(mesh.n_cells())) {
    throw Error(ErrorCode::mesh_inconsistent_sizes, "geometry does not belong to this mesh");
  }
  auto cell_centre = [&](index_t c) {
    return Vec3{g.cell_centroid_x[at(c)], g.cell_centroid_y[at(c)], g.cell_centroid_z[at(c)]};
  };
  MeshQuality q;
  q.non_orthogonality.resize(n_faces);
  q.skewness.resize(n_faces);
  q.volume_ratio.resize(n_interior);
  constexpr double infinite = std::numeric_limits<double>::infinity();
  for (std::size_t f = 0; f < n_faces; ++f) {
    const Vec3 s{g.face_area_x[f], g.face_area_y[f], g.face_area_z[f]};
    const Vec3 cf{g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]};
    const Vec3 cp = cell_centre(mesh.owner()[f]);
    if (f < n_interior) {
      const index_t neighbour = mesh.neighbour()[f];
      const Vec3 d = cell_centre(neighbour) - cp;
      q.non_orthogonality[f] = angle_deg(s, d);
      const double ds = dot(d, s);
      const Vec3 crossing = cp + (dot(cf - cp, s) / ds) * d;
      q.skewness[f] = ds > 0.0 ? norm(cf - crossing) / norm(d) : infinite;
      const double vp = g.cell_volume[at(mesh.owner()[f])];
      const double vn = g.cell_volume[at(neighbour)];
      q.volume_ratio[f] = std::max(vp, vn) / std::min(vp, vn);
    } else {
      const Vec3 d = cf - cp;
      q.non_orthogonality[f] = angle_deg(s, d);
      const Vec3 n = (1.0 / norm(s)) * s;
      const Vec3 normal_step = dot(n, d) * n;
      const double length = norm(normal_step);
      q.skewness[f] = dot(n, d) > 0.0 ? norm(cf - (cp + normal_step)) / length : infinite;
    }
  }

  q.aspect_ratio.resize(at(mesh.n_owned_cells()));
  for (index_t c = 0; c < mesh.n_owned_cells(); ++c) {
    Vec3 a{0.0, 0.0, 0.0};
    for (const index_t f : mesh.cell_faces(c)) {
      a = a + Vec3{std::abs(g.face_area_x[at(f)]), std::abs(g.face_area_y[at(f)]),
                   std::abs(g.face_area_z[at(f)])};
    }
    const double a_max = std::max({a.x, a.y, a.z});
    const double a_min = std::min({a.x, a.y, a.z});
    const double closedness =
        (a.x + a.y + a.z) / (6.0 * std::cbrt(g.cell_volume[at(c)] * g.cell_volume[at(c)]));
    q.aspect_ratio[at(c)] = std::max(a_max / a_min, closedness);
  }
  return q;
}

}  // namespace autan
