#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/partition/partition.hpp"
#include "comm.hpp"
#include "fixtures.hpp"

#ifdef AUTAN_HAVE_MPI
#include <mpi.h>
#endif

namespace {

std::unique_ptr<autan::runtime::Comm> comm;

}  // namespace

TEST_CASE("argmin takes the least value and, among equals, the least id", "[runtime][ranks]") {
  // Rank r offers id 100 - r: the least ids sit on the highest ranks.
  const int r = comm->rank(), n = comm->size();
  const std::int64_t id = 100 - r;
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  constexpr double inf = std::numeric_limits<double>::infinity();
  // Every rank but 0 holds the least value: the highest rank's id wins.
  const auto a = comm->argmin(r == 0 ? 1.0 : 0.0, id);
  CHECK(a.first == (n == 1 ? 1.0 : 0.0));
  CHECK(a.second == 100 - (n - 1));
  // Ties everywhere: the least id.
  CHECK(comm->argmin(2.5, id) == std::pair<double, std::int64_t>{2.5, 100 - (n - 1)});
  // NaN never beats a number; all NaN gives +inf and the least id.
  const auto b = comm->argmin(r == 0 ? nan : 5.0, id);
  CHECK(b.first == (n == 1 ? inf : 5.0));
  CHECK(b.second == (n == 1 ? 100 : 100 - (n - 1)));
  CHECK(comm->argmin(nan, id) == std::pair<double, std::int64_t>{inf, 100 - (n - 1)});
  // ids at the ends of the range do not overflow.
  CHECK(comm->argmin(0.0, r == n - 1 ? 0 : std::numeric_limits<std::int64_t>::max()).second == 0);
}

TEST_CASE("max over the ranks, in double", "[runtime][ranks]") {
  CHECK(comm->max(static_cast<double>(comm->rank()) + 0.5) ==
        static_cast<double>(comm->size()) - 0.5);
}

TEST_CASE("all_gather concatenates every rank's values in rank order", "[runtime][ranks]") {
  // Rank r gives r values 10 r + k: rank 0 gives none.
  const int r = comm->rank(), n = comm->size();
  std::vector<double> mine;
  for (int k = 0; k < r; ++k) mine.push_back(10.0 * r + k);
  std::vector<double> expected;
  for (int q = 0; q < n; ++q) {
    for (int k = 0; k < q; ++k) expected.push_back(10.0 * q + k);
  }
  CHECK(comm->all_gather(mine) == expected);
  CHECK(comm->all_gather({}).empty());
}

TEST_CASE("halo_exchange fills a pattern's ghosts from their owners", "[runtime][ranks]") {
  // A ring: rank r owns items 10 r and 10 r + 1 and holds as its one ghost the second item of
  // rank r + 1 (mod n), itself on one rank.
  const int r = comm->rank(), n = comm->size();
  const int next = (r + 1) % n, previous = (r + n - 1) % n;
  autan::HaloPattern pattern;
  pattern.rank = r;
  pattern.n_ranks = n;
  pattern.neighbours = {previous, next};
  std::ranges::sort(pattern.neighbours);
  const auto [last, end] = std::ranges::unique(pattern.neighbours);
  pattern.neighbours.erase(last, end);
  for (const int q : pattern.neighbours) {
    if (q == previous) pattern.send_cells.push_back(1);
    pattern.send_offsets.push_back(static_cast<autan::index_t>(pattern.send_cells.size()));
    pattern.recv_offsets.push_back(pattern.recv_offsets.back() + (q == next ? 1 : 0));
  }
  std::array<double, 3> a{10.0 * r, 10.0 * r + 1, -1.0}, b{-10.0 * r, -10.0 * r - 1, -1.0};
  std::array<double*, 2> components{a.data(), b.data()};
  const auto exchange = comm->halo_exchange(pattern, 2);
  exchange->forward(std::span<double* const>(components), autan::MemorySpace::host, nullptr);
  CHECK(a[2] == 10.0 * next + 1);
  CHECK(b[2] == -10.0 * next - 1);
  CHECK(a[0] == 10.0 * r);
}

TEST_CASE("one rank's ghost exchange gives no fit rows to a part without ghosts",
          "[runtime][ranks]") {
  // A one-rank run on a part (reproducibility "ranks") with a physics that receives its ghosts'
  // rows: without periodic images there is no ghost, so no row and a mask of no byte. Periodic
  // images still refuse rows (E0144), and a mask of another length is the caller's error.
  if (comm->size() != 1) SKIP("the exchange of a one-rank run");
  const autan::Mesh box =
      autan::assemble(autan::to_face_based(autan::test::box(3, autan::test::BoxFill::hexa)));
  const autan::Geometry g = autan::compute_geometry(box);
  const auto part =
      autan::extract_partition(box, g, std::vector<std::int32_t>(g.cell_volume.size(), 0), 1, 0,
                               autan::GhostStencil::vertex3);
  REQUIRE(part.mesh.n_cells() == part.mesh.n_owned_cells());
  autan::CellRows owned;
  owned.stride = 19;
  owned.offsets.assign(static_cast<std::size_t>(part.mesh.n_owned_cells()) + 1, 0);
  const auto exchange = comm->ghost_exchange(part);
  const autan::CellRows got = exchange->forward_rows(owned, {});
  CHECK(got.stride == 19);
  CHECK(got.offsets == std::vector<autan::offset_t>{0});
  CHECK(got.keys.empty());
  CHECK(got.values.empty());
  const std::vector<std::uint8_t> one{1};
  CHECK_THROWS_AS(exchange->forward_rows(owned, one), std::logic_error);

  autan::CellwiseMesh cells = autan::test::box(3, autan::test::BoxFill::hexa);
  cells.face_zones[0].kind = autan::BcKind::periodic;
  cells.face_zones[1].kind = autan::BcKind::periodic;
  cells.periodic.push_back(
      {.zone_a = 0, .zone_b = 1, .translation = std::array<double, 3>{1, 0, 0}});
  const autan::Mesh periodic = autan::assemble(autan::to_face_based(std::move(cells)));
  const autan::Geometry pg = autan::compute_geometry(periodic);
  const auto images =
      autan::extract_partition(periodic, pg, std::vector<std::int32_t>(pg.cell_volume.size(), 0), 1,
                               0, autan::GhostStencil::vertex3);
  REQUIRE(images.mesh.n_cells() > images.mesh.n_owned_cells());
  autan::CellRows rows;
  rows.stride = 19;
  rows.offsets.assign(static_cast<std::size_t>(images.mesh.n_owned_cells()) + 1, 0);
  const std::vector<std::uint8_t> mask(
      static_cast<std::size_t>(images.mesh.n_cells() - images.mesh.n_owned_cells()), 0);
  try {
    (void)comm->ghost_exchange(images)->forward_rows(rows, mask);
    FAIL("expected E0144");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::halo_inconsistent);
  }
}

int main(int argc, char** argv) {
#ifdef AUTAN_HAVE_MPI
  if (autan::runtime::launched_by_mpi()) MPI_Init(&argc, &argv);
#endif
  comm = autan::runtime::make_comm();
  // Every rank must call the collectives in the same order: no shuffling.
  Catch::Session session;
  session.configData().runOrder = Catch::TestRunOrder::Declared;
  const int status = session.run(argc, argv);
  comm.reset();
#ifdef AUTAN_HAVE_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized != 0) MPI_Finalize();
#endif
  return status;
}
