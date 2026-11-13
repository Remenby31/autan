#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <mpi.h>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/ghost_stencil.hpp"
#include "autan/partition/halo_exchange.hpp"
#include "autan/partition/partition.hpp"
#include "fixtures.hpp"

#ifdef AUTAN_HAVE_CUDA
#include <cuda_runtime.h>
#endif

using autan::index_t;

namespace {

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

int world_size() {
  int s = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &s);
  return s;
}

enum class MeshKind { tetra, hexa, prism, pyramid, hybrid, polyhedral, periodic };

autan::Mesh make_mesh(MeshKind kind) {
  using autan::test::BoxFill;
  if (kind == MeshKind::polyhedral) {
    return autan::assemble(autan::to_face_based(autan::test::hanging_node_box()));
  }
  if (kind == MeshKind::periodic) {
    // Tetrahedra, periodic in x and y: every rank holds image ghosts.
    autan::CellwiseMesh cells = autan::test::box(6, BoxFill::tetra, 0.2);
    for (int k = 0; k < 2; ++k) {
      cells.face_zones[at(2 * k)].kind = autan::BcKind::periodic;
      cells.face_zones[at(2 * k + 1)].kind = autan::BcKind::periodic;
      std::array<double, 3> t{};
      t[at(k)] = 1.0;
      cells.periodic.push_back({.zone_a = 2 * k, .zone_b = 2 * k + 1, .translation = t});
    }
    return autan::assemble(autan::to_face_based(std::move(cells)));
  }
  const BoxFill fill = kind == MeshKind::tetra     ? BoxFill::tetra
                       : kind == MeshKind::hexa    ? BoxFill::hexa
                       : kind == MeshKind::prism   ? BoxFill::prism
                       : kind == MeshKind::pyramid ? BoxFill::pyramid
                                                   : BoxFill::hybrid;
  return autan::assemble(autan::to_face_based(autan::test::box(6, fill, 0.2)));
}

/// The global mesh, its parts (cut by rank 0, broadcast) and this rank's share.
struct Setup {
  autan::Mesh global;
  autan::Geometry global_geometry;
  std::vector<std::int32_t> parts;
  autan::LocalPartition local;
};

Setup setup(MeshKind kind, autan::GhostStencil ghosts = autan::GhostStencil::face2) {
  autan::Mesh global = make_mesh(kind);
  autan::Geometry gg = autan::compute_geometry(global);
  const std::int32_t n = world_size();
  std::vector<std::int32_t> parts(at(global.n_cells()));
  if (world_rank() == 0) {
    autan::PartitionOptions options;
    options.imbalance_tolerance = 0.05;
    options.cell_weights = autan::cell_cost_weights(global);
    parts = autan::partition_cells(global, n, options);
  }
  MPI_Bcast(parts.data(), static_cast<int>(parts.size()), MPI_INT32_T, 0, MPI_COMM_WORLD);
  autan::LocalPartition local =
      autan::extract_partition(global, gg, parts, n, world_rank(), ghosts);
  return Setup{std::move(global), std::move(gg), std::move(parts), std::move(local)};
}

std::vector<autan::MemorySpace> spaces() {
  std::vector<autan::MemorySpace> s{autan::MemorySpace::host};
#ifdef AUTAN_HAVE_CUDA
  s.push_back(autan::MemorySpace::cuda);
#endif
  return s;
}

/// Components of a cell field, held on the host and mirrored in `space` for
/// the exchange.
template <class T>
class Field {
 public:
  Field(std::size_t n_cells, int n_components, autan::MemorySpace space)
      : host_(at(n_components), std::vector<T>(n_cells)), space_(space) {
    for (auto& c : host_) pointers_.push_back(c.data());
#ifdef AUTAN_HAVE_CUDA
    if (space_ == autan::MemorySpace::cuda) {
      pointers_.clear();
      for (std::size_t k = 0; k < host_.size(); ++k) {
        void* p = nullptr;
        REQUIRE(cudaMalloc(&p, n_cells * sizeof(T)) == cudaSuccess);
        pointers_.push_back(static_cast<T*>(p));
      }
    }
#endif
  }
  ~Field() {
#ifdef AUTAN_HAVE_CUDA
    if (space_ == autan::MemorySpace::cuda) {
      for (T* p : pointers_) cudaFree(p);
    }
#endif
  }
  Field(const Field&) = delete;
  Field& operator=(const Field&) = delete;

  std::vector<T>& operator[](std::size_t k) { return host_[k]; }
  std::span<T* const> to_space() {
#ifdef AUTAN_HAVE_CUDA
    if (space_ == autan::MemorySpace::cuda) {
      for (std::size_t k = 0; k < host_.size(); ++k) {
        REQUIRE(cudaMemcpy(pointers_[k], host_[k].data(), host_[k].size() * sizeof(T),
                           cudaMemcpyHostToDevice) == cudaSuccess);
      }
    }
#endif
    return pointers_;
  }
  void from_space() {
#ifdef AUTAN_HAVE_CUDA
    if (space_ == autan::MemorySpace::cuda) {
      for (std::size_t k = 0; k < host_.size(); ++k) {
        REQUIRE(cudaMemcpy(host_[k].data(), pointers_[k], host_[k].size() * sizeof(T),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
      }
    }
#endif
  }

 private:
  std::vector<std::vector<T>> host_;
  std::vector<T*> pointers_;
  autan::MemorySpace space_;
};

bool require_device(autan::MemorySpace space) {
#ifdef AUTAN_HAVE_CUDA
  if (space == autan::MemorySpace::cuda) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      // A CUDA build without a visible GPU fails rather than skips, so CI on
      // the GPU runners cannot pass with the device path untested.
      FAIL("built with CUDA but no device is visible");
    }
  }
#else
  (void)space;
#endif
  return true;
}

template <class T>
bool same_bits(T a, T b) {
  if constexpr (sizeof(T) == 8) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
  } else {
    return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
  }
}

/// Green-Gauss gradient of `phi` in owned cell `c`, the faces visited in
/// ascending global face id and summed in FP64: the order does not depend on
/// the partition, so neither do the bits.
template <class T>
std::array<T, 3> gradient(const autan::Mesh& m, const autan::Geometry& g,
                          std::span<const std::int64_t> face_gid, const std::vector<T>& phi,
                          index_t c) {
  const auto faces = m.cell_faces(c);
  std::vector<index_t> order(faces.begin(), faces.end());
  std::sort(order.begin(), order.end(),
            [&](index_t a, index_t b) { return face_gid[at(a)] < face_gid[at(b)]; });
  std::array<double, 3> sum{0.0, 0.0, 0.0};
  for (const index_t f : order) {
    const auto i = at(f);
    const index_t o = m.owner()[i];
    const double value = f < m.n_interior_faces()
                             ? 0.5 * (static_cast<double>(phi[at(o)]) +
                                      static_cast<double>(phi[at(m.neighbour()[i])]))
                             : static_cast<double>(phi[at(o)]);
    const double sign = o == c ? 1.0 : -1.0;
    sum[0] += value * (sign * g.face_area_x[i]);
    sum[1] += value * (sign * g.face_area_y[i]);
    sum[2] += value * (sign * g.face_area_z[i]);
  }
  const double v = g.cell_volume[at(c)];
  return {static_cast<T>(sum[0] / v), static_cast<T>(sum[1] / v), static_cast<T>(sum[2] / v)};
}

double field_value(double x, double y, double z) {
  return 1.0 + 2.0 * x - 3.0 * y + 0.5 * z + x * y;
}

/// Deterministic value in [-1, 1) from a 64-bit key (splitmix64).
double noise(std::uint64_t key) {
  key += 0x9e3779b97f4a7c15ULL;
  key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
  key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
  key ^= key >> 31;
  return static_cast<double>(key >> 11) * 0x1.0p-52 - 1.0;
}

double global_sum(double local) {
  double all = 0.0;
  MPI_Allreduce(&local, &all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return all;
}

}  // namespace

TEST_CASE("Ghost geometry arrives bit for bit from the owning rank", "[halo]") {
  const MeshKind kind =
      GENERATE(MeshKind::tetra, MeshKind::hybrid, MeshKind::polyhedral, MeshKind::periodic);
  Setup s = setup(kind);
  autan::HaloExchange exchange(s.local.halo, s.local.mesh.n_owned_cells(), MPI_COMM_WORLD,
                               autan::MemorySpace::host);
  autan::exchange_ghost_geometry(exchange, s.local.mesh, s.local.geometry);
  const auto& m = s.local.mesh;
  for (index_t c = 0; c < m.n_cells(); ++c) {
    const auto global_cell = at(m.cell_global_id()[at(c)]);
    // A periodic image: the owner's centroid moved by its translation.
    const std::array<double, 3> t = c < m.n_owned_cells()
                                        ? std::array<double, 3>{}
                                        : m.ghost_translation()[at(c - m.n_owned_cells())];
    CHECK(
        same_bits(s.local.geometry.cell_volume[at(c)], s.global_geometry.cell_volume[global_cell]));
    CHECK(same_bits(s.local.geometry.cell_centroid_x[at(c)],
                    s.global_geometry.cell_centroid_x[global_cell] + t[0]));
    CHECK(same_bits(s.local.geometry.cell_centroid_y[at(c)],
                    s.global_geometry.cell_centroid_y[global_cell] + t[1]));
    CHECK(same_bits(s.local.geometry.cell_centroid_z[at(c)],
                    s.global_geometry.cell_centroid_z[global_cell] + t[2]));
  }
}

TEMPLATE_TEST_CASE("A gradient on N ranks matches the single-rank one bit for bit", "[halo]", float,
                   double) {
  using T = TestType;
  const MeshKind kind = GENERATE(MeshKind::tetra, MeshKind::hexa, MeshKind::prism,
                                 MeshKind::pyramid, MeshKind::hybrid, MeshKind::polyhedral);
  const autan::MemorySpace space = GENERATE_COPY(Catch::Generators::from_range(spaces()));
  require_device(space);
  Setup s = setup(kind);
  const autan::Mesh& m = s.local.mesh;
  autan::HaloExchange geometry_exchange(s.local.halo, m.n_owned_cells(), MPI_COMM_WORLD,
                                        autan::MemorySpace::host);
  autan::exchange_ghost_geometry(geometry_exchange, s.local.mesh, s.local.geometry);
  autan::HaloExchange exchange(s.local.halo, m.n_owned_cells(), MPI_COMM_WORLD, space);

  const auto& gg = s.global_geometry;
  std::vector<T> global_phi(at(s.global.n_cells()));
  for (std::size_t c = 0; c < global_phi.size(); ++c) {
    global_phi[c] = static_cast<T>(
        field_value(gg.cell_centroid_x[c], gg.cell_centroid_y[c], gg.cell_centroid_z[c]));
  }
  Field<T> phi(at(m.n_cells()), 1, space);
  for (index_t c = 0; c < m.n_cells(); ++c) {
    phi[0][at(c)] = c < m.n_owned_cells() ? global_phi[at(m.cell_global_id()[at(c)])]
                                          : std::numeric_limits<T>::quiet_NaN();
  }

  // Cells with an interface face wait for the exchange; the others overlap it.
  std::vector<std::uint8_t> waits(at(m.n_owned_cells()), 0);
  for (const index_t f : s.local.halo.interface_faces) waits[at(m.owner()[at(f)])] = 1;
  std::vector<std::array<T, 3>> grad(at(m.n_owned_cells()));
  exchange.begin_forward<T>(phi.to_space());
  for (index_t c = 0; c < m.n_owned_cells(); ++c) {
    if (waits[at(c)] == 0)
      grad[at(c)] = gradient(m, s.local.geometry, s.local.face_global_id, phi[0], c);
  }
  exchange.end_forward();
  phi.from_space();
  for (index_t c = 0; c < m.n_owned_cells(); ++c) {
    if (waits[at(c)] != 0)
      grad[at(c)] = gradient(m, s.local.geometry, s.local.face_global_id, phi[0], c);
  }

  std::vector<std::int64_t> identity(at(s.global.n_faces()));
  for (std::size_t f = 0; f < identity.size(); ++f) identity[f] = static_cast<std::int64_t>(f);
  std::size_t mismatches = 0;
  for (index_t c = 0; c < m.n_owned_cells(); ++c) {
    const auto global_cell = static_cast<index_t>(m.cell_global_id()[at(c)]);
    const auto expected = gradient(s.global, gg, identity, global_phi, global_cell);
    for (std::size_t d = 0; d < 3; ++d)
      mismatches += same_bits(grad[at(c)][d], expected[d]) ? 0 : 1;
  }
  CHECK(mismatches == 0);
}

TEMPLATE_TEST_CASE("The reverse exchange is the transpose of the forward exchange", "[halo]", float,
                   double) {
  using T = TestType;
  const MeshKind kind = GENERATE(MeshKind::tetra, MeshKind::hexa, MeshKind::prism,
                                 MeshKind::pyramid, MeshKind::hybrid, MeshKind::polyhedral);
  const autan::MemorySpace space = GENERATE_COPY(Catch::Generators::from_range(spaces()));
  require_device(space);
  Setup s = setup(kind);
  const autan::Mesh& m = s.local.mesh;
  autan::HaloExchange exchange(s.local.halo, m.n_owned_cells(), MPI_COMM_WORLD, space);
  constexpr int n_components = 3;
  const auto n_cells = at(m.n_cells());
  const auto rank = static_cast<std::uint64_t>(world_rank());

  // <F u, v> = <u, F^T v> over every owned and ghost entry of every rank.
  Field<T> u(n_cells, n_components, space);
  Field<T> v(n_cells, n_components, space);
  for (std::size_t k = 0; k < n_components; ++k) {
    for (std::size_t c = 0; c < n_cells; ++c) {
      const auto key = (static_cast<std::uint64_t>(m.cell_global_id()[c]) << 8) ^ (rank << 4) ^ k;
      u[k][c] = static_cast<T>(noise(key));
      v[k][c] = static_cast<T>(noise(~key));
    }
  }
  const auto u0 = std::array{u[0], u[1], u[2]};
  const auto v0 = std::array{v[0], v[1], v[2]};
  exchange.forward<T>(u.to_space());
  u.from_space();
  exchange.reverse<T>(v.to_space());
  v.from_space();
  double lhs = 0.0, rhs = 0.0, scale = 0.0;
  for (std::size_t k = 0; k < n_components; ++k) {
    for (std::size_t c = 0; c < n_cells; ++c) {
      lhs += static_cast<double>(u[k][c]) * static_cast<double>(v0[k][c]);
      rhs += static_cast<double>(u0[k][c]) * static_cast<double>(v[k][c]);
      scale += std::abs(static_cast<double>(u0[k][c]) * static_cast<double>(v0[k][c]));
    }
  }
  lhs = global_sum(lhs);
  rhs = global_sum(rhs);
  scale = global_sum(scale);
  CHECK(std::abs(lhs - rhs) <= 8.0 * std::numeric_limits<T>::epsilon() * scale);

  // Reversing the same input again gives the same bits.
  Field<T> again(n_cells, n_components, space);
  for (std::size_t k = 0; k < n_components; ++k) again[k] = v0[k];
  exchange.reverse<T>(again.to_space());
  again.from_space();
  for (std::size_t k = 0; k < n_components; ++k) {
    CHECK(std::equal(again[k].begin(), again[k].end(), v[k].begin(), same_bits<T>));
  }
}

TEST_CASE("Reverse adds one per rank that holds the cell as a ghost", "[halo]") {
  const MeshKind kind = GENERATE(MeshKind::hybrid, MeshKind::polyhedral);
  const autan::MemorySpace space = GENERATE_COPY(Catch::Generators::from_range(spaces()));
  require_device(space);
  Setup s = setup(kind);
  const autan::Mesh& m = s.local.mesh;
  autan::HaloExchange exchange(s.local.halo, m.n_owned_cells(), MPI_COMM_WORLD, space);
  Field<double> w(at(m.n_cells()), 1, space);
  for (index_t c = 0; c < m.n_cells(); ++c) w[0][at(c)] = c < m.n_owned_cells() ? 0.0 : 1.0;
  exchange.reverse<double>(w.to_space());
  w.from_space();

  // Expected from the global mesh: with two ghost layers, a cell is held by
  // every other part that owns a cell within two face steps of it.
  const auto& g = s.global;
  const auto holders = [&](index_t cell) {
    std::vector<index_t> near{cell};
    for (int step = 0; step < 2; ++step) {
      const std::vector<index_t> from = near;
      for (const index_t u : from) {
        for (const index_t f : g.cell_faces(u)) {
          if (f < g.n_interior_faces())
            near.push_back(g.owner()[at(f)] == u ? g.neighbour()[at(f)] : g.owner()[at(f)]);
        }
      }
    }
    std::vector<std::int32_t> parts;
    for (const index_t u : near) {
      if (s.parts[at(u)] != s.parts[at(cell)]) parts.push_back(s.parts[at(u)]);
    }
    std::sort(parts.begin(), parts.end());
    return static_cast<double>(std::unique(parts.begin(), parts.end()) - parts.begin());
  };
  for (index_t c = 0; c < m.n_cells(); ++c) {
    CHECK(w[0][at(c)] ==
          (c < m.n_owned_cells() ? holders(static_cast<index_t>(m.cell_global_id()[at(c)])) : 0.0));
  }
}

TEST_CASE("A halo pattern that does not fit the communicator is rejected", "[halo]") {
  Setup s = setup(MeshKind::hexa);
  autan::HaloPattern wrong = s.local.halo;
  wrong.n_ranks += 1;
  try {
    autan::HaloExchange exchange(wrong, s.local.mesh.n_owned_cells(), MPI_COMM_WORLD,
                                 autan::MemorySpace::host);
    FAIL("expected E0144");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::halo_inconsistent);
  }
#ifndef AUTAN_HAVE_CUDA
  try {
    autan::HaloExchange exchange(s.local.halo, s.local.mesh.n_owned_cells(), MPI_COMM_WORLD,
                                 autan::MemorySpace::cuda);
    FAIL("expected E0145");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::halo_device_unsupported);
  }
#endif
}

namespace {

// The row of the cell with global id `id`: (id + 1) % 4 entries (an empty row every fourth cell),
// keyed 10 id + 3 - j, so not sorted; value v of entry j is -0.0 for v = 0, 0.1 (id + 1) + j
// for v = 1 (not a float), and id + j / 4 + v / 16 otherwise.
std::int64_t row_length(std::int64_t id) { return (id + 1) % 4; }

autan::CellRows row_of(std::int64_t id, int stride) {
  autan::CellRows r;
  r.stride = stride;
  for (std::int64_t j = 0; j < row_length(id); ++j) {
    r.keys.push_back(10 * id + 3 - j);
    for (int v = 0; v < stride; ++v) {
      const double x = v == 0   ? -0.0
                       : v == 1 ? 0.1 * static_cast<double>(id + 1) + static_cast<double>(j)
                                : static_cast<double>(id) + static_cast<double>(j) / 4.0 + v / 16.0;
      r.values.push_back(x);
    }
  }
  r.offsets.push_back(static_cast<autan::offset_t>(r.keys.size()));
  return r;
}

// The rows of `mesh`'s first `n` cells, by global id, as one CellRows.
autan::CellRows rows_of(const autan::Mesh& mesh, index_t n, int stride) {
  autan::CellRows all;
  all.stride = stride;
  for (index_t c = 0; c < n; ++c) {
    const autan::CellRows r = row_of(mesh.cell_global_id()[at(c)], stride);
    all.keys.insert(all.keys.end(), r.keys.begin(), r.keys.end());
    all.values.insert(all.values.end(), r.values.begin(), r.values.end());
    all.offsets.push_back(static_cast<autan::offset_t>(all.keys.size()));
  }
  return all;
}

void require_everywhere(autan::ErrorCode expected, const std::function<void()>& action) {
  try {
    action();
    FAIL_CHECK("expected " << autan::error_code_string(expected));
  } catch (const autan::Error& e) {
    CHECK(e.code() == expected);
  }
}

// What the ranks hand to MPI_Alltoallv while armed: per call, the datatypes and the summed send
// and receive counts. The observer sits at the MPI boundary (PMPI) and passes every call on.
struct Exchange {
  MPI_Datatype send, receive;
  std::int64_t sent, received;
};
bool observing = false;
std::vector<Exchange> observed;

struct Observe {
  Observe() {
    observed.clear();
    observed.reserve(16);
    observing = true;
  }
  ~Observe() { observing = false; }
  Observe(const Observe&) = delete;
  Observe& operator=(const Observe&) = delete;
};

bool same_doubles(std::span<const double> a, std::span<const double> b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

}  // namespace

// NOLINTBEGIN(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-inconsistent-declaration-parameter-name)
extern "C" int MPI_Alltoallv(const void* sendbuf, const int sendcounts[], const int sdispls[],
                             MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
                             const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm) {
  if (observing && observed.size() < observed.capacity()) {
    int n = 0;
    PMPI_Comm_size(comm, &n);
    Exchange e{sendtype, recvtype, 0, 0};
    for (int q = 0; q < n; ++q) {
      if (sendbuf != MPI_IN_PLACE) e.sent += sendcounts[q];
      e.received += recvcounts[q];
    }
    observed.push_back(e);
  }
  return PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
                        recvtype, comm);
}
// NOLINTEND(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-inconsistent-declaration-parameter-name)

TEST_CASE("Every selected ghost receives its owner's row and nothing else travels",
          "[halo][rows]") {
  // forward_rows with a mask per ghost slot: a selected ghost gets its owner's whole row, bit for
  // bit; the others an empty row, and only the selected rows are sent. The mask depends on the
  // receiving rank and the ghost slot, so one cell can be selected by one receiver, or for one
  // periodic image, and not for another. The expected rows and counts come from the global ids.
  const MeshKind kind = GENERATE(MeshKind::tetra, MeshKind::hexa, MeshKind::hybrid,
                                 MeshKind::polyhedral, MeshKind::periodic);
  const autan::GhostStencil stencil =
      GENERATE(autan::GhostStencil::face2, autan::GhostStencil::vertex3);
  Setup s = setup(kind, stencil);
  const autan::Mesh& m = s.local.mesh;
  const index_t n_owned = m.n_owned_cells();
  const auto n_ghosts = at(m.n_cells() - n_owned);
  const auto gid = [&](std::size_t g) { return m.cell_global_id()[at(n_owned) + g]; };
  const int me = world_rank();
  // The polyhedron (the cell with the most faces) is always selected where it is a ghost.
  std::vector<int> faces(at(s.global.n_cells()), 0);
  for (std::size_t f = 0; f < s.global.owner().size(); ++f) {
    ++faces[at(s.global.owner()[f])];
    if (f < s.global.neighbour().size()) ++faces[at(s.global.neighbour()[f])];
  }
  const auto polyhedron =
      static_cast<std::int64_t>(std::distance(faces.begin(), std::ranges::max_element(faces)));
  std::vector<std::uint8_t> wanted(n_ghosts), all(n_ghosts, 1), none(n_ghosts, 0);
  for (std::size_t g = 0; g < n_ghosts; ++g) {
    const std::int64_t id = gid(g);
    wanted[g] = (kind == MeshKind::polyhedral && id == polyhedron) ||
                (7 * id + me + static_cast<std::int64_t>(g)) % 3 != 0;
  }

  // Expected traffic, independent of the slot correspondence under test: each ghost names its
  // owner (the global partition) and whether it is selected; each rank sums what it owes.
  std::vector<std::int64_t> mine;  // owner, global id, selected: one triple per ghost
  for (std::size_t g = 0; g < n_ghosts; ++g) {
    mine.insert(mine.end(), {s.parts[at(gid(g))], gid(g), wanted[g]});
  }
  const int n = world_size();
  std::vector<int> counts(at(n)), displs(at(n));
  const int count = static_cast<int>(mine.size());
  MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  for (int q = 1; q < n; ++q) displs[at(q)] = displs[at(q - 1)] + counts[at(q - 1)];
  std::vector<std::int64_t> every(at(displs.back() + counts.back()));
  MPI_Allgatherv(mine.data(), count, MPI_INT64_T, every.data(), counts.data(), displs.data(),
                 MPI_INT64_T, MPI_COMM_WORLD);
  std::int64_t owed = 0, owed_all = 0, due = 0, due_all = 0;
  for (std::size_t t = 0; t < every.size(); t += 3) {
    if (every[t] != me) continue;
    owed += every[t + 2] != 0 ? row_length(every[t + 1]) : 0;
    owed_all += row_length(every[t + 1]);
  }
  for (std::size_t g = 0; g < n_ghosts; ++g) {
    due += wanted[g] != 0 ? row_length(gid(g)) : 0;
    due_all += row_length(gid(g));
  }
  const auto n_sent = static_cast<std::int64_t>(s.local.halo.send_cells.size());
  const auto n_recv = static_cast<std::int64_t>(n_ghosts);

  for (const int stride : {0, 3}) {
    const autan::CellRows owned = rows_of(m, n_owned, stride);
    const auto run = [&](std::span<const std::uint8_t> mask, std::int64_t sent,
                         std::int64_t received) {
      autan::CellRows out;
      {
        const Observe watch;
        out = autan::forward_rows(s.local.halo, owned, mask, MPI_COMM_WORLD);
      }
      // G667-b': the mask (1 B, receiver to owner), the lengths, keys and values, in that order.
      const std::vector<Exchange> want{{MPI_UINT8_T, MPI_UINT8_T, n_recv, n_sent},
                                       {MPI_INT64_T, MPI_INT64_T, n_sent, n_recv},
                                       {MPI_INT64_T, MPI_INT64_T, sent, received},
                                       {MPI_DOUBLE, MPI_DOUBLE, sent * stride, received * stride}};
      CHECK(observed.size() == want.size());
      for (std::size_t i = 0; i < std::min(observed.size(), want.size()); ++i) {
        CHECK(observed[i].send == want[i].send);
        CHECK(observed[i].receive == want[i].receive);
        CHECK(observed[i].sent == want[i].sent);
        CHECK(observed[i].received == want[i].received);
      }
      return out;
    };
    const autan::CellRows full = run(all, owed_all, due_all);
    const autan::CellRows some = run(wanted, owed, due);
    const autan::CellRows nothing = run(none, 0, 0);
    CHECK(full.stride == stride);
    CHECK(some.stride == stride);
    CHECK(nothing.stride == stride);
    // G667-b, c: the CSR of each output against its independent sizes. A rank whose output is
    // misshapen skips the slices but stays in the collectives.
    std::vector<autan::offset_t> full_offsets{0}, some_offsets{0};
    for (std::size_t g = 0; g < n_ghosts; ++g) {
      full_offsets.push_back(full_offsets.back() + row_length(gid(g)));
      some_offsets.push_back(some_offsets.back() + (wanted[g] != 0 ? row_length(gid(g)) : 0));
    }
    const auto sized = [&](const autan::CellRows& r, const std::vector<autan::offset_t>& o) {
      return r.offsets == o && r.keys.size() == at(o.back()) &&
             r.values.size() == at(o.back()) * at(stride);
    };
    const bool shaped = sized(full, full_offsets) && sized(some, some_offsets) &&
                        sized(nothing, std::vector<autan::offset_t>(n_ghosts + 1, 0));
    CHECK(shaped);
    if (!shaped) continue;
    // G667-a: each selected row equals, bit for bit, the all-ones row and the oracle row.
    const auto keys_of = [](const autan::CellRows& r, std::size_t g) {
      return std::span(r.keys).subspan(at(r.offsets[g]), at(r.offsets[g + 1] - r.offsets[g]));
    };
    const auto values_of = [&](const autan::CellRows& r, std::size_t g) {
      return std::span(r.values).subspan(at(r.offsets[g]) * at(stride),
                                         at(r.offsets[g + 1] - r.offsets[g]) * at(stride));
    };
    for (std::size_t g = 0; g < n_ghosts; ++g) {
      const autan::CellRows oracle = row_of(gid(g), stride);
      CHECK(std::ranges::equal(keys_of(full, g), oracle.keys));
      CHECK(same_doubles(values_of(full, g), oracle.values));
      if (wanted[g] == 0) continue;
      CHECK(std::ranges::equal(keys_of(some, g), keys_of(full, g)));
      CHECK(same_doubles(values_of(some, g), values_of(full, g)));
    }
  }

  // The selection covers what it must, on cells whose row is not empty: a cell selected by one
  // receiver and not by another (4 and 8 ranks), two images of one cell selected differently
  // (the periodic mesh on 1 rank), the polyhedron selected somewhere (2 ranks or more), a
  // selected and an unselected non-empty row, and a selected multi-entry row (unsorted keys).
  std::array<int, 6> seen{};
  for (std::size_t g = 0; g < n_ghosts; ++g) {
    if (row_length(gid(g)) == 0) {
      seen[0] |= wanted[g];
      continue;
    }
    seen[1] |= wanted[g] == 0 ? 1 : 0;
    seen[2] |= wanted[g] != 0 && row_length(gid(g)) > 1 ? 1 : 0;
    seen[3] |= wanted[g] != 0 && gid(g) == polyhedron ? 1 : 0;
    for (std::size_t h = g + 1; h < n_ghosts; ++h) {
      seen[4] |= gid(h) == gid(g) && wanted[h] != wanted[g] ? 1 : 0;
    }
  }
  // The receiving rank of each gathered triple, from the gather's segments.
  std::vector<int> receiver(every.size() / 3);
  for (int q = 0; q < n; ++q) {
    for (int t = displs[at(q)]; t < displs[at(q)] + counts[at(q)]; t += 3) receiver[at(t / 3)] = q;
  }
  std::int64_t split = 0;
  for (std::size_t t = 0; t < every.size(); t += 3) {
    for (std::size_t u = t + 3; u < every.size() && split == 0; u += 3) {
      split = receiver[u / 3] != receiver[t / 3] && every[u + 1] == every[t + 1] &&
                      every[u + 2] != every[t + 2] && row_length(every[t + 1]) > 0
                  ? 1
                  : 0;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, seen.data(), static_cast<int>(seen.size()), MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  if (n > 1 || kind == MeshKind::periodic) {
    CHECK(seen[0] == 1);
    CHECK(seen[1] == 1);
    CHECK(seen[2] == 1);
  }
  if (kind == MeshKind::polyhedral && n > 1) CHECK(seen[3] == 1);
  if (kind == MeshKind::periodic && n == 1) CHECK(seen[4] == 1);
  if (n >= 4) CHECK(split == 1);
}

TEST_CASE("forward_rows refuses together and bounds the selected rows", "[halo][rows]") {
  // G667-d and the count range: every rank refuses when only rank 0's input is wrong; the
  // bound applies to the selected rows, sent and received.
  Setup s = setup(MeshKind::periodic);
  const autan::Mesh& m = s.local.mesh;
  const index_t n_owned = m.n_owned_cells();
  const auto n_ghosts = at(m.n_cells() - n_owned);
  const std::vector<std::uint8_t> all(n_ghosts, 1);
  const int me = world_rank();
  if (world_size() > 1) {
    require_everywhere(autan::ErrorCode::halo_inconsistent, [&] {
      (void)autan::forward_rows(s.local.halo, rows_of(m, n_owned, me == 0 ? 2 : 3), all,
                                MPI_COMM_WORLD);
    });
    require_everywhere(autan::ErrorCode::halo_inconsistent, [&] {
      autan::HaloPattern bent = s.local.halo;
      if (me == 0) bent.send_offsets.front() = 1;
      (void)autan::forward_rows(bent, rows_of(m, n_owned, 3), all, MPI_COMM_WORLD);
    });
  }
  require_everywhere(autan::ErrorCode::partition_invalid_request, [&] {
    autan::CellRows bad = rows_of(m, n_owned, 3);
    if (me == 0) bad.values.push_back(0.0);
    (void)autan::forward_rows(s.local.halo, bad, all, MPI_COMM_WORLD);
  });
  // The mask one byte short, one byte long, and a byte 2 (rank 0 holds periodic images).
  REQUIRE(n_ghosts > 0);
  for (const int change : {-1, 1, 0}) {
    std::vector<std::uint8_t> mask = all;
    if (me == 0 && change < 0) mask.pop_back();
    if (me == 0 && change > 0) mask.push_back(1);
    if (me == 0 && change == 0) mask.front() = 2;
    require_everywhere(autan::ErrorCode::partition_invalid_request, [&] {
      (void)autan::forward_rows(s.local.halo, rows_of(m, n_owned, 3), mask, MPI_COMM_WORLD);
    });
  }
  autan::HaloPattern self;
  self.neighbours = {0};
  self.send_cells = {0};
  self.send_offsets = {1, 2};
  self.recv_offsets = {0, 1};
  const std::vector<std::uint8_t> one_slot{1};
  require_everywhere(autan::ErrorCode::halo_inconsistent, [&] {
    (void)autan::forward_rows(self, row_of(1, 1), one_slot, MPI_COMM_SELF);
  });

  // 2048 slots of one entry at 2^20 values from rank 0 to itself: 2^31 values if all are
  // selected (refused), none or one selected (sent).
  constexpr int wide = 1 << 20;
  constexpr index_t slots = 2048;
  autan::HaloPattern large;
  large.rank = me;
  large.n_ranks = world_size();
  autan::CellRows one;
  one.stride = wide;
  if (me == 0) {
    large.neighbours = {0};
    large.send_cells.assign(slots, 0);
    large.send_offsets = {0, slots};
    large.recv_offsets = {0, slots};
    one = row_of(0, wide);
  }
  const auto n_large = me == 0 ? at(slots) : 0;
  require_everywhere(autan::ErrorCode::partition_invalid_request, [&] {
    (void)autan::forward_rows(large, one, std::vector<std::uint8_t>(n_large, 1), MPI_COMM_WORLD);
  });
  for (const int selected : {0, 1}) {
    std::vector<std::uint8_t> mask(n_large, 0);
    if (!mask.empty()) mask[at(slots / 2)] = static_cast<std::uint8_t>(selected);
    const autan::CellRows got = autan::forward_rows(large, one, mask, MPI_COMM_WORLD);
    const std::int64_t entries = me == 0 ? selected * row_length(0) : 0;
    CHECK(got.keys.size() == at(entries));
    CHECK(got.values.size() == at(entries) * at(wide));
  }

  if (world_size() < 2) return;
  // Rank 0 receives 1024 slots from itself and 1024 from rank 1: each sender within the range,
  // their sum beyond it (refused). Then rank 0 sends one row to rank 1 only: a send-only and a
  // receive-only rank in a transfer that succeeds, the other ranks idle.
  autan::HaloPattern gather;
  gather.rank = me;
  gather.n_ranks = world_size();
  autan::CellRows from;
  from.stride = wide;
  std::size_t n_gather = 0;
  if (me == 0) {
    gather.neighbours = {0, 1};
    gather.send_cells.assign(slots / 2, 0);
    gather.send_offsets = {0, slots / 2, slots / 2};
    gather.recv_offsets = {0, slots / 2, slots};
    from = row_of(0, wide);
    n_gather = at(slots);
  } else if (me == 1) {
    gather.neighbours = {0};
    gather.send_cells.assign(slots / 2, 0);
    gather.send_offsets = {0, slots / 2};
    gather.recv_offsets = {0, 0};
    from = row_of(4, wide);
  }
  require_everywhere(autan::ErrorCode::partition_invalid_request, [&] {
    (void)autan::forward_rows(gather, from, std::vector<std::uint8_t>(n_gather, 1), MPI_COMM_WORLD);
  });
  autan::HaloPattern pass;
  pass.rank = me;
  pass.n_ranks = world_size();
  autan::CellRows given;
  given.stride = 3;
  std::size_t n_pass = 0;
  if (me == 0) {
    pass.neighbours = {1};
    pass.send_cells = {0};
    pass.send_offsets = {0, 1};
    pass.recv_offsets = {0, 0};
    given = row_of(6, 3);
  } else if (me == 1) {
    pass.neighbours = {0};
    pass.send_offsets = {0, 0};
    pass.recv_offsets = {0, 1};
    n_pass = 1;
  }
  const autan::CellRows got =
      autan::forward_rows(pass, given, std::vector<std::uint8_t>(n_pass, 1), MPI_COMM_WORLD);
  const autan::CellRows oracle = row_of(6, 3);
  if (me == 1) {
    CHECK(got.keys == oracle.keys);
    CHECK(same_doubles(got.values, oracle.values));
  } else {
    CHECK(got.keys.empty());
  }
}

TEST_CASE("Graph levels exchange their nodes and edges through HaloExchange", "[halo][graph]") {
  // The dual graph of a box, METIS parts, coarse nodes of 4 consecutive cells: after a forward
  // exchange of the node ids and of the edge ids, every ghost holds its own id.
  const autan::Mesh mesh =
      autan::assemble(autan::to_face_based(autan::test::box(6, autan::test::BoxFill::tetra, 0.2)));
  const auto n = world_size(), r = world_rank();
  const auto parts = autan::partition_cells(mesh, n);
  std::vector<index_t> owner(mesh.owner().begin(), mesh.owner().begin() + mesh.n_interior_faces());
  std::vector<index_t> neighbour(mesh.neighbour().begin(), mesh.neighbour().end());
  std::vector<std::int64_t> ids(mesh.cell_global_id().begin(), mesh.cell_global_id().end());
  std::vector<index_t> coarse(at(mesh.n_cells()));
  for (std::size_t c = 0; c < coarse.size(); ++c) coarse[c] = static_cast<index_t>(c / 4);
  const auto cp = autan::coarse_parts(parts, coarse, ids, (coarse.size() + 3) / 4);
  const autan::LocalGraph g = autan::extract_graph(
      {.n_nodes = at(mesh.n_cells()), .owner = owner, .neighbour = neighbour, .global_id = ids},
      parts, n, r, coarse, cp);
  const auto exchange = [&](const autan::HaloPattern& halo, std::size_t n_owned,
                            const std::vector<std::int64_t>& truth) {
    std::vector<double> value(truth.size(), -1.0);
    for (std::size_t i = 0; i < n_owned; ++i) value[i] = static_cast<double>(truth[i]);
    autan::HaloExchange x(halo, static_cast<index_t>(n_owned), MPI_COMM_WORLD,
                          autan::MemorySpace::host);
    double* const component = value.data();
    x.forward(std::span<double* const>(&component, 1));
    for (std::size_t i = 0; i < truth.size(); ++i) CHECK(value[i] == static_cast<double>(truth[i]));
  };
  exchange(g.node_halo, g.n_owned, g.node_global_id);
  exchange(g.edge_halo, g.n_owned_edges, g.edge_global_id);
}

TEST_CASE("Parent imports bring each coarse node's owner value through HaloExchange",
          "[halo][graph]") {
  // #259 D2. The dual graph of a box, METIS parts, coarse nodes pairing cells from both ends of
  // the cell order (so that parents lie on other ranks), owned as level_parts gives them (no
  // lines): after a forward exchange of the owners' coarse ids, every imported parent holds its
  // own id. A rank whose owned coarse nodes another rank imports, and which imports none, only
  // sends.
  const autan::Mesh mesh =
      autan::assemble(autan::to_face_based(autan::test::box(6, autan::test::BoxFill::tetra, 0.2)));
  const auto n = world_size(), r = world_rank();
  const auto parts = autan::partition_cells(mesh, n);
  std::vector<std::int64_t> ids(mesh.cell_global_id().begin(), mesh.cell_global_id().end());
  std::vector<index_t> coarse(at(mesh.n_cells()));
  const std::size_t n_coarse = coarse.size() / 2;
  for (std::size_t c = 0; c < coarse.size(); ++c)
    coarse[c] = static_cast<index_t>(std::min(c, coarse.size() - 1 - c));
  std::vector<std::int64_t> coarse_ids(n_coarse);
  for (std::size_t c = 0; c < n_coarse; ++c) coarse_ids[c] = static_cast<std::int64_t>(3 * c + 1);
  const std::vector<autan::offset_t> no_lines{0};
  const auto cp = autan::level_parts(parts, coarse, ids, n_coarse, no_lines, {});
  const autan::ParentImport im = autan::parent_import(parts, coarse, cp, coarse_ids, n, r);
  std::vector<double> value(im.owned.size() + im.parents.size(), -1.0);
  for (std::size_t i = 0; i < im.owned.size(); ++i)
    value[i] = static_cast<double>(coarse_ids[at(im.owned[i])]);
  autan::HaloExchange x(im.halo, static_cast<index_t>(im.owned.size()), MPI_COMM_WORLD,
                        autan::MemorySpace::host);
  double* const component = value.data();
  x.forward(std::span<double* const>(&component, 1));
  for (std::size_t k = 0; k < im.parents.size(); ++k)
    CHECK(value[im.owned.size() + k] == static_cast<double>(coarse_ids[at(im.parents[k])]));
  long imported = static_cast<long>(im.parents.size()), total = 0;
  MPI_Allreduce(&imported, &total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (n > 1) CHECK(total > 0);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Catch::Session session;
  int result = session.applyCommandLine(argc, argv);
  // Every rank must run the same tests in the same order: their exchanges pair up.
  session.configData().runOrder = Catch::TestRunOrder::Declared;
  if (result == 0) result = session.run();
  int worst = 0;
  MPI_Allreduce(&result, &worst, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return worst;
}
