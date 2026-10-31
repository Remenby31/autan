#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/partition/partition.hpp"
#include "fixtures.hpp"

using autan::index_t;

namespace {

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

void require_error(autan::ErrorCode expected, const std::function<void()>& action) {
  try {
    action();
    FAIL("expected " << autan::error_code_string(expected));
  } catch (const autan::Error& e) {
    CHECK(e.code() == expected);
  }
}

autan::Mesh box_mesh(int n, autan::test::BoxFill fill) {
  return autan::assemble(autan::to_face_based(autan::test::box(n, fill)));
}

/// Parts as slabs of equal width along x, by cell centroid.
std::vector<std::int32_t> slabs(const autan::Geometry& g, std::int32_t n_parts, double x_max) {
  std::vector<std::int32_t> parts(g.cell_volume.size());
  for (std::size_t c = 0; c < parts.size(); ++c) {
    const auto p = static_cast<std::int32_t>(std::floor(g.cell_centroid_x[c] / x_max * n_parts));
    parts[c] = std::clamp(p, 0, n_parts - 1);
  }
  return parts;
}

bool same_bits(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

/// Checks every rank of `parts` against the global mesh: owned cells cover the
/// mesh once, faces and owned cells carry the global geometry bit for bit,
/// ghosts are NaN, and each send list matches the receiver's ghost list.
std::vector<autan::LocalPartition> check_split(const autan::Mesh& global,
                                               const std::vector<std::int32_t>& parts,
                                               std::int32_t n_parts) {
  const autan::Geometry gg = autan::compute_geometry(global);
  std::vector<autan::LocalPartition> locals;
  std::vector<int> seen(at(global.n_cells()), 0);
  for (std::int32_t r = 0; r < n_parts; ++r) {
    locals.push_back(autan::extract_partition(global, gg, parts, n_parts, r));
    const autan::LocalPartition& lp = locals.back();
    const autan::Mesh& m = lp.mesh;
    const auto gid = m.cell_global_id();
    for (index_t c = 0; c < m.n_owned_cells(); ++c) {
      const auto global_cell = at(gid[at(c)]);
      ++seen[global_cell];
      CHECK(parts[global_cell] == r);
      CHECK(same_bits(lp.geometry.cell_volume[at(c)], gg.cell_volume[global_cell]));
      CHECK(same_bits(lp.geometry.cell_centroid_x[at(c)], gg.cell_centroid_x[global_cell]));
    }
    for (index_t c = m.n_owned_cells(); c < m.n_cells(); ++c) {
      CHECK(parts[at(gid[at(c)])] != r);
      CHECK(std::isnan(lp.geometry.cell_volume[at(c)]));
    }
    for (index_t f = 0; f < m.n_faces(); ++f) {
      const auto source = at(lp.face_global_id[at(f)]);
      const bool same_orientation = gid[at(m.owner()[at(f)])] == global.owner()[source];
      const double sign = same_orientation ? 1.0 : -1.0;
      CHECK(same_bits(lp.geometry.face_area_x[at(f)], sign * gg.face_area_x[source]));
      CHECK(same_bits(lp.geometry.face_area_z[at(f)], sign * gg.face_area_z[source]));
      CHECK(same_bits(lp.geometry.face_centroid_y[at(f)], gg.face_centroid_y[source]));
    }
    const auto& h = lp.halo;
    REQUIRE(h.recv_offsets.back() == m.n_cells() - m.n_owned_cells());
    for (const index_t f : h.interface_faces) CHECK(m.neighbour()[at(f)] >= m.n_owned_cells());
  }
  CHECK(std::all_of(seen.begin(), seen.end(), [](int s) { return s == 1; }));

  // What r sends to q are, in order, the cells q holds as ghosts from r.
  for (std::int32_t r = 0; r < n_parts; ++r) {
    const auto& hr = locals[at(r)].halo;
    for (std::size_t i = 0; i < hr.neighbours.size(); ++i) {
      const std::int32_t q = hr.neighbours[i];
      const auto& lq = locals[at(q)];
      const auto& hq = lq.halo;
      const auto j =
          at(std::find(hq.neighbours.begin(), hq.neighbours.end(), r) - hq.neighbours.begin());
      REQUIRE(j < hq.neighbours.size());
      std::vector<std::int64_t> sent, ghosts;
      for (index_t s = hr.send_offsets[i]; s < hr.send_offsets[i + 1]; ++s) {
        sent.push_back(locals[at(r)].mesh.cell_global_id()[at(hr.send_cells[at(s)])]);
      }
      for (index_t g = hq.recv_offsets[j]; g < hq.recv_offsets[j + 1]; ++g) {
        ghosts.push_back(lq.mesh.cell_global_id()[at(lq.mesh.n_owned_cells() + g)]);
      }
      CHECK(sent == ghosts);
    }
  }
  return locals;
}

}  // namespace

TEST_CASE("Slab partitions of a hex box have the analytic halo", "[partition]") {
  constexpr int n = 8;
  const std::int32_t n_parts = GENERATE(1, 2, 4, 8);
  const autan::Mesh global = box_mesh(n, autan::test::BoxFill::hexa);
  const auto parts = slabs(autan::compute_geometry(global), n_parts, 1.0);

  const auto q = autan::partition_quality(global, parts, n_parts);
  const std::int64_t layer = n * n;
  CHECK(q.cut_faces == (n_parts - 1) * layer);
  CHECK(q.imbalance == 1.0);
  CHECK(q.max_neighbours == std::min(n_parts - 1, 2));
  CHECK(q.ghost_ratio * n * n * n == static_cast<double>(2 * (n_parts - 1) * layer));
  CHECK(q.max_ghost_ratio ==
        (n_parts == 1 ? 0.0 : (n_parts == 2 ? 1.0 : 2.0) * layer * n_parts / (n * n * n)));

  const auto locals = check_split(global, parts, n_parts);
  for (std::int32_t r = 0; r < n_parts; ++r) {
    const auto& m = locals[at(r)].mesh;
    const bool end = r == 0 || r == n_parts - 1;
    const std::int64_t ghosts = n_parts == 1 ? 0 : (end ? layer : 2 * layer);
    CHECK(m.n_owned_cells() == n * n * n / n_parts);
    CHECK(m.n_cells() - m.n_owned_cells() == ghosts);
    CHECK(q.ghost_cells[at(r)] == ghosts);
    CHECK(std::ssize(locals[at(r)].halo.interface_faces) == ghosts);
  }
}

TEST_CASE("Every cell type splits into valid partitions", "[partition]") {
  using autan::test::BoxFill;
  const BoxFill fill =
      GENERATE(BoxFill::tetra, BoxFill::hexa, BoxFill::prism, BoxFill::pyramid, BoxFill::hybrid);
  const std::int32_t n_parts = GENERATE(2, 3, 8);
  const autan::Mesh global = box_mesh(6, fill);
  check_split(global, slabs(autan::compute_geometry(global), n_parts, 1.0), n_parts);
}

TEST_CASE("A 105-face polyhedron with hanging nodes splits from its neighbours", "[partition]") {
  const autan::Mesh global = autan::assemble(autan::to_face_based(autan::test::hanging_node_box()));
  const auto locals = check_split(global, slabs(autan::compute_geometry(global), 2, 2.0), 2);
  CHECK(locals[0].mesh.n_owned_cells() == 1);
  CHECK(locals[0].mesh.n_cells() == 101);
}

TEST_CASE("Cell cost weights count faces", "[partition]") {
  const autan::Mesh global = autan::assemble(autan::to_face_based(autan::test::hanging_node_box()));
  const auto w = autan::cell_cost_weights(global);
  CHECK(*std::max_element(w.begin(), w.end()) == 105);
  CHECK(std::all_of(w.begin(), w.end(), [](std::int32_t x) { return x == 6 || x == 105; }));
}

#if defined(AUTAN_HAVE_METIS) || defined(AUTAN_HAVE_SCOTCH)
TEST_CASE("Graph partitioners return balanced, reproducible parts", "[partition]") {
  std::vector<autan::PartitionMethod> methods;
#ifdef AUTAN_HAVE_METIS
  methods.push_back(autan::PartitionMethod::metis);
#endif
#ifdef AUTAN_HAVE_SCOTCH
  methods.push_back(autan::PartitionMethod::scotch);
#endif
  const auto method = GENERATE_COPY(Catch::Generators::from_range(methods));
  const std::int32_t n_parts = GENERATE(1, 2, 4, 8);
  CAPTURE(method, n_parts);
  const autan::Mesh global = box_mesh(12, autan::test::BoxFill::hybrid);
  autan::PartitionOptions options;
  options.method = method;
  options.imbalance_tolerance = 0.05;
  options.cell_weights = autan::cell_cost_weights(global);

  const auto parts = autan::partition_cells(global, n_parts, options);
  CHECK(parts == autan::partition_cells(global, n_parts, options));
  const auto q = autan::partition_quality(global, parts, n_parts, options.cell_weights);
  CHECK(q.imbalance <= 1.0 + options.imbalance_tolerance + 1e-12);
  CHECK(std::all_of(q.owned_cells.begin(), q.owned_cells.end(), [](auto c) { return c > 0; }));
  // A partitioner beats slabs, whose cut is (n_parts - 1) layers of the box.
  const auto slab_cut = autan::partition_quality(
                            global, slabs(autan::compute_geometry(global), n_parts, 1.0), n_parts)
                            .cut_faces;
  CHECK(q.cut_faces <= slab_cut);
  check_split(global, parts, n_parts);
}
#endif

TEST_CASE("Invalid partition requests are rejected with their documented code", "[partition]") {
  using autan::ErrorCode;
  const autan::Mesh global = box_mesh(2, autan::test::BoxFill::hexa);
  const autan::Geometry gg = autan::compute_geometry(global);
  const std::vector<std::int32_t> two(8, 0);
  require_error(ErrorCode::partition_invalid_request, [&] { autan::partition_cells(global, 0); });
  require_error(ErrorCode::partition_invalid_request, [&] { autan::partition_cells(global, 9); });
  autan::PartitionOptions short_weights;
  short_weights.cell_weights.assign(7, 1);
  require_error(ErrorCode::partition_invalid_request,
                [&] { autan::partition_cells(global, 1, short_weights); });
  autan::PartitionOptions zero_weights;
  zero_weights.cell_weights.assign(8, 0);
  require_error(ErrorCode::partition_invalid_request,
                [&] { autan::partition_cells(global, 1, zero_weights); });
  require_error(ErrorCode::partition_invalid_request,
                [&] { autan::extract_partition(global, gg, two, 2, 1); });
  require_error(ErrorCode::partition_invalid_request,
                [&] { autan::extract_partition(global, gg, two, 1, 1); });
  require_error(ErrorCode::partition_invalid_request,
                [&] { autan::partition_quality(global, std::vector<std::int32_t>(8, 3), 2); });

  const auto parts = slabs(gg, 2, 1.0);
  const auto half = autan::extract_partition(global, gg, parts, 2, 0);
  require_error(ErrorCode::partition_input_has_ghosts,
                [&] { autan::partition_cells(half.mesh, 1); });
  require_error(ErrorCode::partition_input_has_ghosts,
                [&] { autan::extract_partition(half.mesh, half.geometry, parts, 2, 0); });
#ifndef AUTAN_HAVE_SCOTCH
  autan::PartitionOptions scotch;
  scotch.method = autan::PartitionMethod::scotch;
  require_error(ErrorCode::partition_backend_failed,
                [&] { autan::partition_cells(global, 2, scotch); });
#endif
}
