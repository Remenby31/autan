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
  const auto cell_centre = [&](index_t c) {
    return Vec3{g.cell_centroid_x[at(c)], g.cell_centroid_y[at(c)], g.cell_centroid_z[at(c)]};
  };
  MeshQuality q;
  q.non_orthogonality.resize(n_faces);
  q.skewness.resize(n_faces);
  q.volume_ratio.resize(n_interior);
  constexpr double infinite = std::numeric_limits<double>::infinity();
  // OpenFOAM primitiveMeshTools::faceSkewness / boundaryFaceSkewness: the
  // skewness vector is normalised by the face extent along it, at least
  // 0.2 |d| (interior) or 0.4 |d| (boundary).
  const auto skewness = [&](std::size_t f, Vec3 cpf, Vec3 d, Vec3 s, Vec3 cf, double floor) {
    const double ds = dot(s, d);
    if (!(ds > 0.0)) return infinite;
    const Vec3 sv = cpf - (dot(s, cpf) / ds) * d;
    const double length = norm(sv);
    if (length == 0.0) return 0.0;
    const Vec3 dir = (1.0 / length) * sv;
    double extent = floor * norm(d);
    for (const index_t p : mesh.face_nodes(static_cast<index_t>(f))) {
      const Vec3 x{mesh.x()[at(p)], mesh.y()[at(p)], mesh.z()[at(p)]};
      extent = std::max(extent, std::abs(dot(dir, x - cf)));
    }
    return length / extent;
  };
  for (std::size_t f = 0; f < n_faces; ++f) {
    const Vec3 s{g.face_area_x[f], g.face_area_y[f], g.face_area_z[f]};
    const Vec3 cf{g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]};
    const Vec3 cp = cell_centre(mesh.owner()[f]);
    if (f < n_interior) {
      const index_t neighbour = mesh.neighbour()[f];
      const Vec3 d = cell_centre(neighbour) - cp;
      q.non_orthogonality[f] = angle_deg(s, d);
      q.skewness[f] = skewness(f, cf - cp, d, s, cf, 0.2);
      const double vp = g.cell_volume[at(mesh.owner()[f])];
      const double vn = g.cell_volume[at(neighbour)];
      q.volume_ratio[f] = std::min(vp, vn) / std::max(vp, vn);
    } else {
      const Vec3 cpf = cf - cp;
      q.non_orthogonality[f] = angle_deg(s, cpf);
      const Vec3 n = (1.0 / norm(s)) * s;
      q.skewness[f] = skewness(f, cpf, dot(n, cpf) * n, s, cf, 0.4);
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
    const double edge = std::cbrt(g.cell_volume[at(c)]);
    const double closedness = (a.x + a.y + a.z) / (6.0 * edge * edge);
    q.aspect_ratio[at(c)] = std::max(a_max / a_min, closedness);
  }
  return q;
}

}  // namespace autan
