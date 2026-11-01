#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
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

Setup setup(MeshKind kind) {
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
  autan::LocalPartition local = autan::extract_partition(global, gg, parts, n, world_rank());
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
