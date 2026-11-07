#include <cmath>
#include <cstdint>
#include <limits>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "comm.hpp"

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
