#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "autan/core/geometry.hpp"
#include "autan/core/lines.hpp"
#include "autan/core/renumber.hpp"
#include "fixtures.hpp"

using autan::index_t;

namespace {

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

// The unit cube cut into n^3 hexahedra, stretched by `stretch` along x and y.
autan::Mesh stretched_box(int n, double stretch) {
  auto cw = autan::test::box(n, autan::test::BoxFill::hexa);
  for (auto& v : cw.x) v *= stretch;
  for (auto& v : cw.y) v *= stretch;
  return autan::assemble(autan::to_face_based(cw));
}

// Each line as the sorted global ids of its cells.
std::set<std::vector<std::int64_t>> line_sets(const autan::Mesh& mesh, const autan::Lines& lines) {
  std::set<std::vector<std::int64_t>> out;
  for (std::size_t l = 0; l + 1 < lines.offsets.size(); ++l) {
    std::vector<std::int64_t> ids;
    for (auto k = lines.offsets[l]; k < lines.offsets[l + 1]; ++k) {
      ids.push_back(mesh.cell_global_id()[at(lines.cells[static_cast<std::size_t>(k)])]);
    }
    std::sort(ids.begin(), ids.end());
    out.insert(ids);
  }
  return out;
}

}  // namespace

TEST_CASE("lines follow the strong coupling of stretched cells, whatever the cell order",
          "[lines]") {
  // 4^3 hexahedra of 2.5 x 2.5 x 0.25 (x and y stretched tenfold): a z face couples A/d =
  // 6.25 / 0.25 = 25, an x or y face 0.625 / 2.5 = 0.25, a ratio of 100 >= 4. Every z column of
  // four cells is one line, its cells consecutive in z; the isotropic box has none.
  const auto mesh = stretched_box(4, 10.0);
  const auto g = autan::compute_geometry(mesh);
  const auto lines = autan::smoother_lines(mesh, g);
  REQUIRE(lines.offsets.size() == 17);
  for (std::size_t l = 0; l + 1 < lines.offsets.size(); ++l) {
    REQUIRE(lines.offsets[l + 1] - lines.offsets[l] == 4);
    double previous_z = -1.0;
    bool rising = true, falling = true;
    for (auto k = lines.offsets[l]; k < lines.offsets[l + 1]; ++k) {
      const auto c = at(lines.cells[static_cast<std::size_t>(k)]);
      const auto c0 = at(lines.cells[static_cast<std::size_t>(lines.offsets[l])]);
      CHECK(g.cell_centroid_x[c] == g.cell_centroid_x[c0]);
      CHECK(g.cell_centroid_y[c] == g.cell_centroid_y[c0]);
      if (previous_z >= 0.0) {
        rising = rising && g.cell_centroid_z[c] > previous_z;
        falling = falling && g.cell_centroid_z[c] < previous_z;
        CHECK(std::abs(std::abs(g.cell_centroid_z[c] - previous_z) - 0.25) <=
              64 * std::numeric_limits<double>::epsilon());
      }
      previous_z = g.cell_centroid_z[c];
    }
    CHECK((rising || falling));
  }
  const auto isotropic = stretched_box(4, 1.0);
  CHECK(autan::smoother_lines(isotropic, autan::compute_geometry(isotropic)).cells.empty());
  // The same lines, as sets of cells, after a random renumbering and after RCM.
  std::vector<index_t> order(at(mesh.n_cells()));
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(7);
  std::shuffle(order.begin(), order.end(), rng);
  for (const auto& renumbered :
       {autan::renumber_cells(mesh, order), autan::renumber_cells(mesh, autan::rcm_order(mesh))}) {
    const auto again = autan::smoother_lines(renumbered, autan::compute_geometry(renumbered));
    CHECK(line_sets(renumbered, again) == line_sets(mesh, lines));
  }
}
