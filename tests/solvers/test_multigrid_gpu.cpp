// Device transfers of the agglomeration multigrid, against their definition in the host
// multigrid's order (docs/design/multigrid-gpu.md#transfers).
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cuda_runtime.h>

#include "autan/core/geometry.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/solvers/agglomeration.hpp"
#include "autan/solvers/multigrid_gpu.hpp"

namespace {

namespace fs = std::filesystem;
using autan::solvers::Level;

const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
const std::array<const char*, 4> meshes = {"tet-gmsh/tet-gmsh.ascii.msh", "hex/hex.ascii.msh",
                                           "poly/polyMesh", "hybrid/hybrid.ascii.msh"};

class Device {
 public:
  explicit Device(const std::vector<double>& host) : n_(host.size()) {
    REQUIRE(cudaMalloc(reinterpret_cast<void**>(&p_), n_ * sizeof(double)) == cudaSuccess);
    REQUIRE(cudaMemcpy(p_, host.data(), n_ * sizeof(double), cudaMemcpyHostToDevice) ==
            cudaSuccess);
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device() { cudaFree(p_); }
  [[nodiscard]] double* get() const { return p_; }
  [[nodiscard]] std::vector<double> host() const {
    std::vector<double> v(n_);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaMemcpy(v.data(), p_, n_ * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess);
    return v;
  }

 private:
  double* p_ = nullptr;
  std::size_t n_;
};

// Values over 120 binades with both signs, so that a sum in any other order or with a fused
// multiply-add rounds differently; a few signed zeros.
std::vector<double> spread(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> mantissa(1.0, 2.0);
  std::uniform_int_distribution<int> exponent(-60, 60), sign(0, 1), zero(0, 49);
  std::vector<double> v(n);
  for (double& x : v) {
    x = zero(rng) == 0 ? -0.0 : std::ldexp(mantissa(rng), exponent(rng));
    if (sign(rng) == 1) x = -x;
  }
  return v;
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t k = 0; k < a.size(); ++k) {
    if (std::bit_cast<std::uint64_t>(a[k]) != std::bit_cast<std::uint64_t>(b[k])) return false;
  }
  return true;
}

// A host-numbered vector in a level's device numbering, and back: device node j is host node
// order[j].
std::vector<double> to_device(const std::vector<double>& v, std::span<const autan::index_t> order) {
  const std::size_t n = order.size();
  std::vector<double> d(v.size());
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t j = 0; j < n; ++j)
      d[c * n + j] = v[c * n + static_cast<std::size_t>(order[j])];
  }
  return d;
}

std::vector<double> to_host(const std::vector<double>& d, std::span<const autan::index_t> order) {
  const std::size_t n = order.size();
  std::vector<double> v(d.size());
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t j = 0; j < n; ++j)
      v[c * n + static_cast<std::size_t>(order[j])] = d[c * n + j];
  }
  return v;
}

std::vector<Level> hierarchy(const char* file) {
  const auto mesh = autan::io::read_mesh(mesh_dir / file);
  const auto g = autan::compute_geometry(mesh);
  autan::solvers::AgglomerationOptions opt;
  opt.coarsest_nodes = 4;
  return autan::solvers::agglomerate(mesh, g, opt);
}

}  // namespace

TEST_CASE("device transfers are bitwise the host multigrid's restriction and prolongation",
          "[multigrid][gpu]") {
  const char* file = GENERATE(from_range(meshes));
  INFO(file);
  const auto levels = hierarchy(file);
  REQUIRE(levels.size() >= 2);
  const autan::solvers::gpu::Transfers transfers(levels);
  for (std::size_t l = 0; l + 1 < levels.size(); ++l) {
    INFO("level " << l);
    const Level& f = levels[l];
    const std::size_t n = f.n_nodes, nc = levels[l + 1].n_nodes;
    const auto fine = transfers.order(l), coarse = transfers.order(l + 1);
    REQUIRE(fine.size() == n);
    REQUIRE(coarse.size() == nc);
    // Device numbering, from its definition: level 0 is the mesh's; coarse nodes are a
    // permutation ranked by their least child in the finer level's device numbering.
    if (l == 0) {
      for (std::size_t j = 0; j < n; ++j) CHECK(fine[j] == static_cast<autan::index_t>(j));
    }
    std::vector<std::size_t> fine_rank(n);
    for (std::size_t j = 0; j < n; ++j) fine_rank[static_cast<std::size_t>(fine[j])] = j;
    std::vector<std::size_t> least(nc, n);
    for (std::size_t h = 0; h < n; ++h) {
      std::size_t& m = least[static_cast<std::size_t>(f.coarse_node[h])];
      m = std::min(m, fine_rank[h]);
    }
    std::vector<bool> seen(nc, false);
    for (std::size_t j = 0; j < nc; ++j) {
      const auto H = static_cast<std::size_t>(coarse[j]);
      REQUIRE(H < nc);
      CHECK_FALSE(seen[H]);
      seen[H] = true;
      if (j > 0) CHECK(least[static_cast<std::size_t>(coarse[j - 1])] < least[H]);
    }

    // Restriction: each coarse value from +0.0, children added in increasing host fine index.
    const auto r = spread(5 * n, static_cast<std::uint32_t>(17 + l));
    std::vector<double> bc(5 * nc, 0.0);
    for (std::size_t c = 0; c < 5; ++c) {
      for (std::size_t i = 0; i < n; ++i) {
        bc[c * nc + static_cast<std::size_t>(f.coarse_node[i])] += r[c * n + i];
      }
    }
    const Device dr(to_device(r, fine));
    const Device dbc(std::vector<double>(5 * nc, 1.0));
    transfers.restrict_residual(l, dr.get(), dbc.get());
    CHECK(same_bits(to_host(dbc.host(), coarse), bc));

    // Prolongation by injection, and the correction x + alpha * e without contraction.
    const auto xc = spread(5 * nc, static_cast<std::uint32_t>(91 + l));
    std::vector<double> e(5 * n);
    for (std::size_t c = 0; c < 5; ++c) {
      for (std::size_t i = 0; i < n; ++i) {
        e[c * n + i] = xc[c * nc + static_cast<std::size_t>(f.coarse_node[i])];
      }
    }
    const Device dxc(to_device(xc, coarse));
    const Device de(std::vector<double>(5 * n, 1.0));
    transfers.prolong(l, dxc.get(), de.get());
    CHECK(same_bits(to_host(de.host(), fine), e));

    const double alpha = 0.7310585786300049;
    auto x = spread(5 * n, static_cast<std::uint32_t>(43 + l));
    const Device dx(to_device(x, fine));
    const Device dalpha(std::vector<double>{alpha});
    transfers.correct(l, dalpha.get(), dxc.get(), dx.get());
    for (std::size_t k = 0; k < x.size(); ++k) {
      const volatile double product = alpha * e[k];
      x[k] = x[k] + product;
    }
    CHECK(same_bits(to_host(dx.host(), fine), x));
  }
  CHECK_THROWS_AS(transfers.prolong(levels.size() - 1, nullptr, nullptr), std::out_of_range);
}

TEST_CASE("restricting signed zeros gives +0.0, as a sum that starts from +0.0",
          "[multigrid][gpu]") {
  // Three fine nodes in one agglomerate, a fourth alone.
  std::vector<Level> levels(2);
  levels[0].n_nodes = 4;
  levels[0].coarse_node = {0, 0, 1, 0};
  levels[1].n_nodes = 2;
  const autan::solvers::gpu::Transfers transfers(levels);
  std::vector<double> r(20, -0.0);
  const Device dr(r);
  const Device dbc(std::vector<double>(10, 1.0));
  transfers.restrict_residual(0, dr.get(), dbc.get());
  for (const double v : dbc.host()) {
    CHECK(v == 0.0);
    CHECK_FALSE(std::signbit(v));
  }
}

TEST_CASE("large agglomerates restrict in increasing host index", "[multigrid][gpu]") {
  // Coarse node 1 holds every fine node but two, interleaved with coarse node 0's: beyond eight
  // children, and beyond what a block stages in shared memory.
  const std::size_t n = GENERATE(14, 1100);
  INFO(n << " fine nodes");
  std::vector<Level> levels(2);
  levels[0].n_nodes = n;
  levels[0].coarse_node.assign(n, 1);
  levels[0].coarse_node[2] = levels[0].coarse_node[n - 3] = 0;
  levels[1].n_nodes = 2;
  const autan::solvers::gpu::Transfers transfers(levels);
  // Coarse node 1 holds fine node 0, coarse node 0 holds fine node 2 first: device order 1, 0.
  const auto order = transfers.order(1);
  REQUIRE(order.size() == 2);
  CHECK(order[0] == 1);
  CHECK(order[1] == 0);
  const auto r = spread(5 * n, 7);
  std::vector<double> bc(10, 0.0);
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t i = 0; i < n; ++i) {
      bc[c * 2 + static_cast<std::size_t>(levels[0].coarse_node[i])] += r[c * n + i];
    }
  }
  const Device dr(r);
  const Device dbc(std::vector<double>(10, 1.0));
  transfers.restrict_residual(0, dr.get(), dbc.get());
  CHECK(same_bits(to_host(dbc.host(), transfers.order(1)), bc));
}
