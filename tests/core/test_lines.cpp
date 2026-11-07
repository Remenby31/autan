#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
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

// Hexahedra given as axis-aligned boxes {lo, hi}, sharing nodes where they touch; cell i is box i.
autan::Mesh boxes(const std::vector<std::array<std::array<double, 3>, 2>>& cells) {
  autan::CellwiseMesh m;
  std::map<std::array<double, 3>, index_t> node;
  for (const auto& [lo, hi] : cells) {
    std::array<index_t, 8> hex{};
    for (std::size_t k = 0; k < 8; ++k) {
      const bool i = k % 4 == 1 || k % 4 == 2, j = k % 4 >= 2;
      const std::array<double, 3> p{i ? hi[0] : lo[0], j ? hi[1] : lo[1], k >= 4 ? hi[2] : lo[2]};
      const auto [it, added] = node.try_emplace(p, static_cast<index_t>(m.x.size()));
      if (added) {
        m.x.push_back(p[0]);
        m.y.push_back(p[1]);
        m.z.push_back(p[2]);
      }
      hex[k] = it->second;
    }
    m.add_cell(autan::CellShape::hexa, hex, 0);
  }
  m.cell_zones.push_back(
      {.name = "fluid", .type = "", .kind = autan::CellKind::fluid, .provenance = {}});
  return autan::assemble(autan::to_face_based(std::move(m)));
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

TEST_CASE("lines follow the strong coupling of stretched cells", "[lines]") {
  // 4^3 hexahedra of 2.5 x 2.5 x 0.25 (x and y stretched tenfold): a z face couples A/d =
  // 6.25 / 0.25 = 25, an x or y face 0.625 / 2.5 = 0.25, a ratio of 100 >= 4. Every z column of
  // four cells is one line, its cells consecutive in z; the isotropic box has none.
  const auto mesh = stretched_box(4, 10.0);
  const auto g = autan::compute_geometry(mesh);
  const auto lines = autan::smoother_lines(mesh, g);
  REQUIRE(lines.offsets.size() == 17);
  CHECK(autan::cells_in_lines(lines) == 64);
  CHECK(autan::cells_in_lines(lines, autan::long_line) == 64);
  CHECK(autan::cells_in_lines(lines, 5) == 0);
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
  // 2^3 cells of 5 x 5 x 0.5: a z face couples 25 / 0.5 = 50, an x or y face 2.5 / 5 = 0.5, and
  // each z column of two cells is one line.
  const auto pairs = stretched_box(2, 10.0);
  const auto two = autan::smoother_lines(pairs, autan::compute_geometry(pairs));
  CHECK(two.offsets.size() == 5);
  CHECK(autan::cells_in_lines(two) == 8);
  const auto isotropic = stretched_box(4, 1.0);
  CHECK(autan::smoother_lines(isotropic, autan::compute_geometry(isotropic)).cells.empty());
  // No two candidates tie here (each cell has one z neighbour on each side), so the lines, as
  // sets of cells, do not depend on the cell order: the same after a random renumbering and after
  // RCM. Where candidates tie, the smaller index wins and the order can matter (#365).
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

TEST_CASE("lines are seeded at the boundary first and start there", "[lines]") {
  using Box = std::array<std::array<double, 3>, 2>;
  const auto lines_of = [](const autan::Mesh& mesh) {
    return autan::smoother_lines(mesh, autan::compute_geometry(mesh));
  };
  // Cell 0, 1 x 1 x 1/32, has cells of its shape on its x and y faces (1-4), a cell 1/8 tall above
  // (5) and one 1/2 tall below (6); it alone has no boundary face. Couplings: 0 to 1-4 1/32, to 5
  // 1 / (5/64) = 12.8, to 6 1 / (17/64) = 3.76. Strongest / weakest: 1-4 32 / (1/32) = 1024
  // (growing nothing: 1/32 < 4 / 32), 0 12.8 * 32 = 410, 5 12.8 / (1/8) = 102, 6 3.76 / (1/2) =
  // 7.5. Seed 5 comes before the more anisotropic 0 and grows through 0 to 6.
  const double h = 1.0 / 32.0;
  const auto cross = lines_of(
      boxes({Box{{{0, 0, 0}, {1, 1, h}}}, Box{{{-1, 0, 0}, {0, 1, h}}}, Box{{{1, 0, 0}, {2, 1, h}}},
             Box{{{0, -1, 0}, {1, 0, h}}}, Box{{{0, 1, 0}, {1, 2, h}}},
             Box{{{0, 0, h}, {1, 1, h + 0.125}}}, Box{{{0, 0, -0.5}, {1, 1, 0}}}}));
  CHECK(cross.offsets == std::vector<autan::offset_t>{0, 3});
  CHECK(cross.cells == std::vector<index_t>{5, 0, 6});
  // Cell 0, the unit cube, has cubes 7 long on five faces (1-5, coupling 1/4) and below it cell 6,
  // 1/4 tall, then cell 7, 1/2 tall. Cell 6 couples 1.6 to 0 and 8/3 to 7, its sides 1/4: the
  // strongest / weakest 10.7 of 6 is above 7's 8/3 / (1/2) = 5.3 and 0's 1.6 / (1/4) = 6.4 (1-5:
  // 7 / (1/7) = 49, growing nothing as 1/4 < 4 / 7). Seed 6 grows to 7, then to 0, which stops
  // there (1/4 < 4 / 4): 0, 6, 7 has a boundary face at its last cell only.
  const auto column = lines_of(
      boxes({Box{{{0, 0, 0}, {1, 1, 1}}}, Box{{{-7, 0, 0}, {0, 1, 1}}}, Box{{{1, 0, 0}, {8, 1, 1}}},
             Box{{{0, -7, 0}, {1, 0, 1}}}, Box{{{0, 1, 0}, {1, 8, 1}}}, Box{{{0, 0, 1}, {1, 1, 8}}},
             Box{{{0, 0, -0.25}, {1, 1, 0}}}, Box{{{0, 0, -0.75}, {1, 1, -0.25}}}}));
  CHECK(column.offsets == std::vector<autan::offset_t>{0, 3});
  CHECK(column.cells == std::vector<index_t>{7, 6, 0});
}
