// The device Galerkin product and coarsest solve, against the host multigrid's order and its
// dense LU (docs/design/multigrid-gpu.md#galerkin-coarse-operator, #coarsest-level).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cuda_runtime.h>

#include "autan/solvers/dense.hpp"
#include "autan/solvers/multigrid_gpu.hpp"
#include "galerkin_fixture.hpp"

namespace {

using namespace galerkin_fixture;

template <class T>
class DeviceArray {
 public:
  explicit DeviceArray(const std::vector<T>& host) : n_(host.size()) {
    REQUIRE(cudaMalloc(reinterpret_cast<void**>(&p_), std::max<std::size_t>(n_, 1) * sizeof(T)) ==
            cudaSuccess);
    if (n_ > 0) {
      REQUIRE(cudaMemcpy(p_, host.data(), n_ * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    }
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray() { cudaFree(p_); }
  [[nodiscard]] T* get() const { return p_; }
  [[nodiscard]] std::vector<T> host() const {
    std::vector<T> v(n_);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    if (n_ > 0) {
      REQUIRE(cudaMemcpy(v.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
    }
    return v;
  }

 private:
  T* p_ = nullptr;
  std::size_t n_;
};

// Fine blocks in the device layout: structure of arrays, entry q of node j's diagonal at
// q * n + j (device numbering), of slot s at q * n_slots + s; padding slots NaN.
template <class Off>
struct Fine {
  std::vector<double> diag;
  std::vector<Off> off;
};

template <class Off>
Fine<Off> device_layout(const Blocks& a, const std::vector<index_t>& order, const EdgeSlots& s,
                        offset_t n_slots) {
  const std::size_t n = order.size(), ns = static_cast<std::size_t>(n_slots);
  Fine<Off> f;
  f.diag.resize(b2 * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t q = 0; q < b2; ++q) {
      f.diag[q * n + j] = a.diag[b2 * static_cast<std::size_t>(order[j]) + q];
    }
  }
  f.off.assign(b2 * ns, std::numeric_limits<Off>::quiet_NaN());
  for (std::size_t e = 0; e < s.owner.size(); ++e) {
    for (std::size_t q = 0; q < b2; ++q) {
      f.off[q * ns + static_cast<std::size_t>(s.owner[e])] = static_cast<Off>(a.upper[b2 * e + q]);
      f.off[q * ns + static_cast<std::size_t>(s.neighbour[e])] =
          static_cast<Off>(a.lower[b2 * e + q]);
    }
  }
  return f;
}

// Block of slot or node `i` from a structure-of-arrays array of `n` entries.
std::array<double, b2> soa_block(const std::vector<double>& v, std::size_t n, std::size_t i) {
  std::array<double, b2> b{};
  for (std::size_t q = 0; q < b2; ++q) b[q] = v[q * n + i];
  return b;
}

template <class Off>
void check_galerkin(const std::vector<Level>& levels) {
  REQUIRE(levels.size() >= 2);
  const auto orders = autan::solvers::gpu::device_orders(levels);
  Blocks a = spread(levels[0], 5);
  if constexpr (std::is_same_v<Off, float>) {
    // The host sees the FP32 blocks the device reads, converted exactly.
    for (double& v : a.upper) v = static_cast<float>(v);
    for (double& v : a.lower) v = static_cast<float>(v);
  }
  offset_t n_fine_slots = 0;
  EdgeSlots fine_slots = csr_slots(levels[0], orders[0], 2, n_fine_slots);
  fine_slots = permuted_slots(fine_slots, n_fine_slots, 17);
  for (std::size_t l = 0; l + 1 < levels.size(); ++l) {
    INFO("level " << l);
    const Level &f = levels[l], &c = levels[l + 1];
    offset_t n_coarse_slots = 0;
    const EdgeSlots csr = csr_slots(c, orders[l + 1], 3, n_coarse_slots);
    const EdgeSlots coarse_slots =
        permuted_slots(csr, n_coarse_slots, 29 + static_cast<std::uint32_t>(l));
    const GalerkinLists lists = autan::solvers::gpu::galerkin_lists(
        f, c, orders[l], orders[l + 1], fine_slots, coarse_slots, n_coarse_slots);
    const autan::solvers::gpu::Galerkin galerkin(lists, f.n_nodes,
                                                 static_cast<std::size_t>(n_fine_slots));
    REQUIRE(galerkin.n_coarse_nodes() == c.n_nodes);
    REQUIRE(galerkin.n_coarse_slots() == static_cast<std::size_t>(n_coarse_slots));

    // Fine level 0 in Off; coarser fine levels are the device's own FP64 output.
    const Fine<Off> fine0 = device_layout<Off>(a, orders[l], fine_slots, n_fine_slots);
    const Fine<double> fine64 = device_layout<double>(a, orders[l], fine_slots, n_fine_slots);
    const DeviceArray<double> d_diag(l == 0 ? fine0.diag : fine64.diag);
    const DeviceArray<double> c_diag(std::vector<double>(b2 * c.n_nodes, 7.0));
    const DeviceArray<double> c_off(
        std::vector<double>(b2 * static_cast<std::size_t>(n_coarse_slots), 7.0));
    if (l == 0) {
      const DeviceArray<Off> d_off(fine0.off);
      galerkin.apply(d_diag.get(), d_off.get(), c_diag.get(), c_off.get());
    } else {
      const DeviceArray<double> d_off(fine64.off);
      galerkin.apply(d_diag.get(), d_off.get(), c_diag.get(), c_off.get());
    }
    const auto got_diag = c_diag.host(), got_off = c_off.host();
    const Blocks expected = host_galerkin(f, c, a);
    const std::size_t ns = static_cast<std::size_t>(n_coarse_slots);
    for (std::size_t J = 0; J < c.n_nodes; ++J) {
      const auto b = soa_block(got_diag, c.n_nodes, J);
      CHECK(same_bits(b.data(), &expected.diag[b2 * static_cast<std::size_t>(orders[l + 1][J])]));
    }
    std::vector<bool> used(ns, false);
    std::size_t through_csr_wrong = 0;
    for (std::size_t E = 0; E < c.owner.size(); ++E) {
      const auto so = static_cast<std::size_t>(coarse_slots.owner[E]);
      const auto sn = static_cast<std::size_t>(coarse_slots.neighbour[E]);
      used[so] = used[sn] = true;
      CHECK(same_bits(soa_block(got_off, ns, so).data(), &expected.upper[b2 * E]));
      CHECK(same_bits(soa_block(got_off, ns, sn).data(), &expected.lower[b2 * E]));
      // A reader ignoring the permutation (the CSR position) reads another block.
      const auto wrong = soa_block(got_off, ns, static_cast<std::size_t>(csr.owner[E]));
      if (!same_bits(wrong.data(), &expected.upper[b2 * E])) ++through_csr_wrong;
    }
    CHECK(through_csr_wrong > 0);
    for (std::size_t s = 0; s < ns; ++s) {
      if (used[s]) continue;
      const auto b = soa_block(got_off, ns, s);
      CHECK(std::all_of(b.begin(), b.end(), [](double v) { return v == 0.0 && !std::signbit(v); }));
    }
    a = expected;
    fine_slots = coarse_slots;
    n_fine_slots = n_coarse_slots;
  }
}

// A coarsest operator in host form with a dominant diagonal, so that it is well conditioned.
Blocks coarsest_blocks(const Level& l, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Blocks a;
  a.diag.resize(b2 * l.n_nodes);
  a.upper.resize(b2 * l.owner.size());
  a.lower.resize(b2 * l.owner.size());
  for (std::size_t i = 0; i < l.n_nodes; ++i) {
    for (std::size_t q = 0; q < b2; ++q) a.diag[b2 * i + q] = u(rng) + (q % 6 == 0 ? 80.0 : 0.0);
  }
  for (double& v : a.upper) v = u(rng);
  for (double& v : a.lower) v = u(rng);
  return a;
}

// The host multigrid's coarsest solve (Multigrid::Smoother::factor_dense, solve_dense), host
// numbering.
std::vector<double> host_solve(const Level& l, const Blocks& a, std::vector<double> b) {
  const std::size_t n = l.n_nodes, m = 5 * n;
  std::vector<double> dense(m * m, 0.0);
  auto put = [&](std::size_t i, std::size_t j, const double* blk) {
    for (std::size_t p = 0; p < 5; ++p) {
      for (std::size_t q = 0; q < 5; ++q) dense[(p * n + i) * m + q * n + j] += blk[5 * p + q];
    }
  };
  for (std::size_t i = 0; i < n; ++i) put(i, i, &a.diag[b2 * i]);
  for (std::size_t e = 0; e < l.owner.size(); ++e) {
    const auto o = static_cast<std::size_t>(l.owner[e]),
               nb = static_cast<std::size_t>(l.neighbour[e]);
    put(o, nb, &a.upper[b2 * e]);
    put(nb, o, &a.lower[b2 * e]);
  }
  std::vector<std::size_t> pivot(m);
  autan::solvers::factor_dense(dense, m, pivot);
  autan::solvers::solve_dense(dense, pivot, b);
  return b;
}

}  // namespace

TEST_CASE("device Galerkin products are bitwise the host multigrid's coarse operators",
          "[multigrid][gpu]") {
  const bool fp32 = GENERATE(false, true);
  INFO((fp32 ? "FP32" : "FP64") << " fine off-diagonal blocks");
  SECTION("reference meshes") {
    const char* file = GENERATE(from_range(meshes));
    INFO(file);
    const auto levels = hierarchy(autan::io::read_mesh(mesh_dir / file));
    fp32 ? check_galerkin<float>(levels) : check_galerkin<double>(levels);
  }
  SECTION("wall layers, with lines") {
    const auto levels = hierarchy(wall_layers(8));
    fp32 ? check_galerkin<float>(levels) : check_galerkin<double>(levels);
  }
}

TEST_CASE("the device coarsest solve matches the host dense LU within 1e-10", "[multigrid][gpu]") {
  const char* file = GENERATE(from_range(meshes));
  INFO(file);
  const auto levels = hierarchy(autan::io::read_mesh(mesh_dir / file));
  const auto orders = autan::solvers::gpu::device_orders(levels);
  const Level& l = levels.back();
  const auto& order = orders.back();
  const std::size_t n = l.n_nodes, m = 5 * n;
  const Blocks a = coarsest_blocks(l, 23);
  const autan::solvers::gpu::CoarsestSolve solve(l, order, a.diag, a.upper, a.lower);
  REQUIRE(solve.size() == m);

  std::mt19937 rng(31);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> b(m);
  for (double& v : b) v = u(rng);
  const std::vector<double> x_host = host_solve(l, a, b);

  // Device numbering: unknown c of device node j is host unknown c * n + order[j].
  std::vector<double> b_device(m);
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t j = 0; j < n; ++j)
      b_device[c * n + j] = b[c * n + static_cast<std::size_t>(order[j])];
  }
  const DeviceArray<double> d_b(b_device);
  const DeviceArray<double> d_x(std::vector<double>(m, 0.0)), d_x2(std::vector<double>(m, 0.0));
  solve.solve(d_b.get(), d_x.get());
  solve.solve(d_b.get(), d_x2.get());
  const auto x = d_x.host(), x2 = d_x2.host();
  CHECK(std::equal(x.begin(), x.end(), x2.begin(), [](double p, double q) {
    return std::bit_cast<std::uint64_t>(p) == std::bit_cast<std::uint64_t>(q);
  }));
  double scale = 0.0, worst = 0.0;
  for (const double v : x_host) scale = std::max(scale, std::abs(v));
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t j = 0; j < n; ++j) {
      worst = std::max(worst,
                       std::abs(x[c * n + j] - x_host[c * n + static_cast<std::size_t>(order[j])]));
    }
  }
  REQUIRE(scale > 0.0);
  CHECK(worst <= 1e-10 * scale);
}

TEST_CASE("a singular coarsest operator is refused as the host refuses it", "[multigrid][gpu]") {
  const auto levels = hierarchy(autan::io::read_mesh(mesh_dir / "hex/hex.ascii.msh"));
  const Level& l = levels.back();
  const auto orders = autan::solvers::gpu::device_orders(levels);
  const std::vector<double> zero_diag(b2 * l.n_nodes, 0.0), zero_off(b2 * l.owner.size(), 0.0);
  CHECK_THROWS_AS(
      autan::solvers::gpu::CoarsestSolve(l, orders.back(), zero_diag, zero_off, zero_off),
      std::domain_error);
}
