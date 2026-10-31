#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "autan/core/quality.hpp"
#include "fixtures.hpp"

using autan::index_t;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

constexpr double eps = 1e-12;

}  // namespace

TEST_CASE("A uniform hexahedral box has ideal quality", "[quality]") {
  const autan::Mesh mesh =
      autan::assemble(autan::to_face_based(autan::test::box(3, autan::test::BoxFill::hexa)));
  const autan::MeshQuality q = autan::compute_quality(mesh, autan::compute_geometry(mesh));
  for (const double v : q.non_orthogonality) CHECK_THAT(v, WithinAbs(0.0, 1e-6));
  for (const double v : q.skewness) CHECK_THAT(v, WithinAbs(0.0, eps));
  for (const double v : q.volume_ratio) CHECK_THAT(v, WithinRel(1.0, eps));
  for (const double v : q.aspect_ratio) CHECK_THAT(v, WithinRel(1.0, eps));
}

TEST_CASE("Sheared and stretched boxes have their analytic quality", "[quality]") {
  // x -> x + t z shears every face family by atan(t); scaling z by 4 makes
  // every cell a 1 x 1 x 4 box (aspect ratio 4).
  const double t = GENERATE(0.0, 0.5, 2.0);
  CAPTURE(t);
  auto cw = autan::test::box(3, autan::test::BoxFill::hexa);
  for (std::size_t p = 0; p < cw.x.size(); ++p) {
    cw.z[p] *= 4.0;
    cw.x[p] += t * cw.z[p] / 4.0;
  }
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(cw));
  const autan::MeshQuality q = autan::compute_quality(mesh, autan::compute_geometry(mesh));
  // Faces z = const: normal e_z, centroid step (t h, 0, 4 h) -> atan(t / 4).
  // Faces x - t z / 4 = const: normal (1, 0, -t/4), step (h, 0, 0) -> atan(t / 4).
  const double expected = std::atan(t / 4.0) * 180.0 / std::numbers::pi;
  const double max_angle = *std::max_element(q.non_orthogonality.begin(),
                                             q.non_orthogonality.begin() + mesh.n_interior_faces());
  CHECK_THAT(max_angle, WithinAbs(expected, 1e-9));
  for (index_t f = 0; f < mesh.n_interior_faces(); ++f)
    CHECK_THAT(q.skewness[at(f)], WithinAbs(0.0, eps));
  // zmin, zmax faces (h x h squares): the cell centroid sits t h / 2 off the
  // face centroid along x and 2 h above it; skewness (t h / 2) / (0.4 * 2 h).
  const auto offsets = mesh.boundary_zone_offsets();
  for (index_t f = offsets[4]; f < offsets[6]; ++f)
    CHECK_THAT(q.skewness[at(f)], WithinAbs(0.625 * t, eps));
  if (t == 0.0) {
    for (const double v : q.aspect_ratio) CHECK_THAT(v, WithinRel(4.0, eps));
  }
}

TEST_CASE("Faces across a hanging-node interface have their analytic quality", "[quality]") {
  // Coarse centroid P = (1/2, 1/2, 1/2); fine centroid N = (3/2, y, z); face
  // centroid (1, y, z); normal e_x. With d = (1, y - 1/2, z - 1/2):
  // angle = acos(1 / |d|); the line P -> N crosses x = 1 at P + d / 2, so the
  // skewness vector has length r / 2, r = |(y, z) - (1/2, 1/2)|; the face
  // (side 0.1) extends at most 0.05 sqrt(2) < 0.2 |d| along it, so the
  // skewness is (r / 2) / (0.2 |d|); volume ratio 0.01 / 1.
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(autan::test::hanging_node_box()));
  const autan::Geometry g = autan::compute_geometry(mesh);
  const autan::MeshQuality q = autan::compute_quality(mesh, g);
  int checked = 0;
  for (index_t f = 0; f < mesh.n_interior_faces(); ++f) {
    if (mesh.owner()[at(f)] != 0) continue;
    const double y = g.face_centroid_y[at(f)] - 0.5;
    const double z = g.face_centroid_z[at(f)] - 0.5;
    const double d = std::sqrt(1.0 + y * y + z * z);
    CHECK_THAT(q.non_orthogonality[at(f)],
               WithinAbs(std::acos(1.0 / d) * 180.0 / std::numbers::pi, 1e-9));
    CHECK_THAT(q.skewness[at(f)], WithinAbs(std::sqrt(y * y + z * z) / 2.0 / (0.2 * d), eps));
    CHECK_THAT(q.volume_ratio[at(f)], WithinRel(0.01, eps));
    ++checked;
  }
  CHECK(checked == 100);
}

TEST_CASE("Quality metrics are scale invariant", "[quality]") {
  auto cw = autan::test::box(2, autan::test::BoxFill::hexa);
  for (std::size_t p = 0; p < cw.x.size(); ++p) {
    cw.x[p] *= 1e-60;
    cw.y[p] *= 1e-60;
    cw.z[p] *= 1e-60;
  }
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(cw));
  const autan::MeshQuality q = autan::compute_quality(mesh, autan::compute_geometry(mesh));
  for (const double v : q.aspect_ratio) CHECK_THAT(v, WithinRel(1.0, eps));
  for (const double v : q.volume_ratio) CHECK_THAT(v, WithinRel(1.0, eps));
}
