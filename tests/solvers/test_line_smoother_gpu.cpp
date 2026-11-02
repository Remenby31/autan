// The line part of the device smoother, against the host multigrid's Smoother
// (docs/design/multigrid-gpu.md#smoother): bit for bit, on every level, FP64 and FP32 blocks.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cuda_runtime.h>

#include "autan/solvers/multigrid.hpp"
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

// Hexahedra nx x nx x nz of aspect ratio 50 (dz = dx / 50) on [0,1]^2 x [0, nz dx / 50], the wall
// at z = 0: every cell is anisotropic, so every cell lies in a line of several cells.
autan::Mesh thin_layers(int nx, int nz) {
  autan::CellwiseMesh m;
  autan::CellZone zone;
  zone.name = "fluid";
  zone.kind = autan::CellKind::fluid;
  m.cell_zones.push_back(zone);
  autan::FaceZone wall;
  wall.name = "wall";
  wall.kind = autan::BcKind::wall;
  m.face_zones.push_back(wall);
  const double dx = 1.0 / nx, dz = dx / 50.0;
  const int p = nx + 1;
  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= nx; ++j) {
      for (int i = 0; i <= nx; ++i) {
        m.x.push_back(i * dx);
        m.y.push_back(j * dx);
        m.z.push_back(k * dz);
      }
    }
  }
  auto v = [&](int i, int j, int k) { return static_cast<index_t>(i + p * (j + p * k)); };
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < nx; ++j) {
      for (int i = 0; i < nx; ++i) {
        m.add_cell(autan::CellShape::hexa,
                   std::array<index_t, 8>{v(i, j, k), v(i + 1, j, k), v(i + 1, j + 1, k),
                                          v(i, j + 1, k), v(i, j, k + 1), v(i + 1, j, k + 1),
                                          v(i + 1, j + 1, k + 1), v(i, j + 1, k + 1)},
                   0);
        if (k == 0) {
          m.add_labelled_face(std::array<index_t, 4>{v(i, j, 0), v(i, j + 1, 0), v(i + 1, j + 1, 0),
                                                     v(i + 1, j, 0)},
                              0);
        }
      }
    }
  }
  return autan::assemble(autan::to_face_based(std::move(m)));
}

// A level's operator in host form, diagonally dominant so that every line factors.
Blocks dominant(const Level& l, std::uint32_t seed, bool fp32) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Blocks a;
  a.diag.resize(b2 * l.n_nodes);
  a.upper.resize(b2 * l.owner.size());
  a.lower.resize(b2 * l.owner.size());
  for (std::size_t i = 0; i < l.n_nodes; ++i) {
    for (std::size_t q = 0; q < b2; ++q) a.diag[b2 * i + q] = u(rng) + (q % 6 == 0 ? 60.0 : 0.0);
  }
  for (double& v : a.upper) v = fp32 ? static_cast<float>(u(rng)) : u(rng);
  for (double& v : a.lower) v = fp32 ? static_cast<float>(u(rng)) : u(rng);
  return a;
}

// The host multigrid's line smoother (Multigrid::Smoother: factor, relax_line, sweep) over the
// lines of several nodes only, in its operation order, host numbering.
struct HostLines {
  const Level& l;
  const Blocks& a;
  std::vector<std::vector<std::size_t>> edges;
  std::vector<index_t> prev, next;
  std::vector<std::array<double, b2>> dinv, lower, gain;

  HostLines(const Level& level, const Blocks& blocks)
      : l(level),
        a(blocks),
        edges(level.n_nodes),
        prev(level.n_nodes, -1),
        next(level.n_nodes, -1),
        dinv(level.n_nodes),
        lower(level.n_nodes),
        gain(level.n_nodes) {
    for (std::size_t e = 0; e < l.owner.size(); ++e) {
      edges[static_cast<std::size_t>(l.owner[e])].push_back(e);
      edges[static_cast<std::size_t>(l.neighbour[e])].push_back(e);
    }
    for (std::size_t li = 0; li < l.n_lines(); ++li) {
      const auto f = static_cast<std::size_t>(l.line_offsets[li]);
      const auto t = static_cast<std::size_t>(l.line_offsets[li + 1]);
      for (std::size_t k = f; k < t; ++k) {
        const auto i = static_cast<std::size_t>(l.line_nodes[k]);
        if (k > f) prev[i] = l.line_nodes[k - 1];
        if (k + 1 < t) next[i] = l.line_nodes[k + 1];
      }
    }
  }
  const double* off(std::size_t e, std::size_t i) const {
    return static_cast<std::size_t>(l.owner[e]) == i ? &a.upper[b2 * e] : &a.lower[b2 * e];
  }
  std::size_t other(std::size_t e, std::size_t i) const {
    return static_cast<std::size_t>(static_cast<std::size_t>(l.owner[e]) == i ? l.neighbour[e]
                                                                              : l.owner[e]);
  }
  void coupling(std::size_t i, std::size_t j, double* out) const {
    std::fill(out, out + b2, 0.0);
    for (const std::size_t e : edges[i]) {
      if (other(e, i) != j) continue;
      const double* blk = off(e, i);
      for (std::size_t q = 0; q < b2; ++q) out[q] += blk[q];
    }
  }
  static void multiply5(const double* x, const double* y, double* z) {
    for (std::size_t r = 0; r < 5; ++r) {
      for (std::size_t c = 0; c < 5; ++c) {
        double s = 0.0;
        for (std::size_t k = 0; k < 5; ++k) s += x[5 * r + k] * y[5 * k + c];
        z[5 * r + c] = s;
      }
    }
  }
  static void invert5(const double* x, double* inv) {
    std::array<double, 50> m{};
    for (std::size_t i = 0; i < 5; ++i) {
      for (std::size_t j = 0; j < 5; ++j) {
        m[10 * i + j] = x[5 * i + j];
        m[10 * i + 5 + j] = i == j ? 1.0 : 0.0;
      }
    }
    for (std::size_t k = 0; k < 5; ++k) {
      std::size_t p = k;
      for (std::size_t i = k + 1; i < 5; ++i) {
        if (std::abs(m[10 * i + k]) > std::abs(m[10 * p + k])) p = i;
      }
      REQUIRE(m[10 * p + k] != 0.0);
      if (p != k) {
        for (std::size_t j = 0; j < 10; ++j) std::swap(m[10 * k + j], m[10 * p + j]);
      }
      const double d = 1.0 / m[10 * k + k];
      for (std::size_t j = 0; j < 10; ++j) m[10 * k + j] *= d;
      for (std::size_t i = 0; i < 5; ++i) {
        if (i == k) continue;
        const double f = m[10 * i + k];
        for (std::size_t j = 0; j < 10; ++j) m[10 * i + j] -= f * m[10 * k + j];
      }
    }
    for (std::size_t i = 0; i < 5; ++i) {
      for (std::size_t j = 0; j < 5; ++j) inv[5 * i + j] = m[10 * i + 5 + j];
    }
  }
  bool in_long_line(std::size_t li) const {
    return l.line_offsets[li + 1] - l.line_offsets[li] > 1;
  }
  void factor() {
    for (std::size_t li = 0; li < l.n_lines(); ++li) {
      if (!in_long_line(li)) continue;
      for (auto k = l.line_offsets[li]; k < l.line_offsets[li + 1]; ++k) {
        const auto i = static_cast<std::size_t>(l.line_nodes[static_cast<std::size_t>(k)]);
        std::array<double, b2> d{}, t{}, u{};
        std::copy_n(&a.diag[b2 * i], b2, d.data());
        if (prev[i] >= 0) {
          const auto p = static_cast<std::size_t>(prev[i]);
          coupling(i, p, lower[i].data());
          multiply5(lower[i].data(), gain[p].data(), t.data());
          for (std::size_t q = 0; q < b2; ++q) d[q] -= t[q];
        }
        invert5(d.data(), dinv[i].data());
        if (next[i] >= 0) {
          coupling(i, static_cast<std::size_t>(next[i]), u.data());
          multiply5(dinv[i].data(), u.data(), gain[i].data());
        }
      }
    }
  }
  void relax_line(std::size_t li, const std::vector<double>& b, std::vector<double>& x) const {
    const std::size_t n = l.n_nodes;
    const auto f = static_cast<std::size_t>(l.line_offsets[li]);
    const auto t = static_cast<std::size_t>(l.line_offsets[li + 1]);
    std::vector<std::array<double, 5>> w(t - f);
    for (std::size_t k = f; k < t; ++k) {
      const auto i = static_cast<std::size_t>(l.line_nodes[k]);
      double* r = w[k - f].data();
      for (std::size_t c = 0; c < 5; ++c) r[c] = b[c * n + i];
      for (const std::size_t e : edges[i]) {
        const std::size_t j = other(e, i);
        if (static_cast<index_t>(j) == prev[i] || static_cast<index_t>(j) == next[i]) continue;
        const double* blk = off(e, i);
        for (std::size_t p = 0; p < 5; ++p) {
          for (std::size_t q = 0; q < 5; ++q) r[p] -= blk[5 * p + q] * x[q * n + j];
        }
      }
    }
    for (std::size_t k = f; k < t; ++k) {
      const auto i = static_cast<std::size_t>(l.line_nodes[k]);
      double* r = w[k - f].data();
      if (k > f) {
        for (std::size_t p = 0; p < 5; ++p) {
          for (std::size_t c = 0; c < 5; ++c) r[p] -= lower[i][5 * p + c] * w[k - 1 - f][c];
        }
      }
      std::array<double, 5> s{};
      for (std::size_t p = 0; p < 5; ++p) {
        for (std::size_t c = 0; c < 5; ++c) s[p] += dinv[i][5 * p + c] * r[c];
      }
      std::copy_n(s.data(), 5, r);
    }
    for (std::size_t k = t; k-- > f;) {
      const auto i = static_cast<std::size_t>(l.line_nodes[k]);
      double* y = w[k - f].data();
      if (k + 1 < t) {
        for (std::size_t p = 0; p < 5; ++p) {
          for (std::size_t c = 0; c < 5; ++c) y[p] -= gain[i][5 * p + c] * w[k + 1 - f][c];
        }
      }
      for (std::size_t c = 0; c < 5; ++c) x[c * n + i] = y[c];
    }
  }
  void sweep(const std::vector<double>& b, std::vector<double>& x, int sweeps) const {
    const auto colours = autan::solvers::line_colours(l);
    for (int s = 0; s < sweeps; ++s) {
      for (const auto& colour : colours) {
        for (const std::size_t li : colour) {
          if (in_long_line(li)) relax_line(li, b, x);
        }
      }
      for (auto it = colours.rbegin(); it != colours.rend(); ++it) {
        for (const std::size_t li : *it) {
          if (in_long_line(li)) relax_line(li, b, x);
        }
      }
    }
  }
};

std::vector<double> to_device_vector(const std::vector<double>& v, std::span<const index_t> order) {
  const std::size_t n = order.size();
  std::vector<double> d(v.size());
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t j = 0; j < n; ++j)
      d[c * n + j] = v[c * n + static_cast<std::size_t>(order[j])];
  }
  return d;
}

std::vector<double> to_host_vector(const std::vector<double>& d, std::span<const index_t> order) {
  const std::size_t n = order.size();
  std::vector<double> v(d.size());
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t j = 0; j < n; ++j)
      v[c * n + static_cast<std::size_t>(order[j])] = d[c * n + j];
  }
  return v;
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](double p, double q) {
           return std::bit_cast<std::uint64_t>(p) == std::bit_cast<std::uint64_t>(q);
         });
}

std::vector<double> random_vector(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> v(n);
  for (double& x : v) x = u(rng);
  return v;
}

// The device line sweep of one level, from host-form blocks: x after `sweeps` sweeps, host
// numbering.
template <class Off>
std::vector<double> device_sweep(const Level& l, std::span<const index_t> order, const Blocks& a,
                                 const std::vector<double>& b, const std::vector<double>& x0,
                                 int sweeps, std::uint32_t seed) {
  const std::size_t n = l.n_nodes;
  offset_t n_slots = 0;
  const std::vector<index_t> ord(order.begin(), order.end());
  const EdgeSlots slots = permuted_slots(csr_slots(l, ord, 2, n_slots), n_slots, seed);
  std::vector<double> diag(b2 * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t q = 0; q < b2; ++q)
      diag[q * n + j] = a.diag[b2 * static_cast<std::size_t>(order[j]) + q];
  }
  const auto ns = static_cast<std::size_t>(n_slots);
  std::vector<Off> off(b2 * ns, std::numeric_limits<Off>::quiet_NaN());
  for (std::size_t e = 0; e < l.owner.size(); ++e) {
    for (std::size_t q = 0; q < b2; ++q) {
      off[q * ns + static_cast<std::size_t>(slots.owner[e])] =
          static_cast<Off>(a.upper[b2 * e + q]);
      off[q * ns + static_cast<std::size_t>(slots.neighbour[e])] =
          static_cast<Off>(a.lower[b2 * e + q]);
    }
  }
  autan::solvers::gpu::LineSmoother smoother(l, order, slots, ns);
  const DeviceArray<double> d_diag(diag), d_b(to_device_vector(b, order)),
      d_x(to_device_vector(x0, order));
  const DeviceArray<Off> d_off(off);
  smoother.factor(d_diag.get(), d_off.get());
  smoother.sweep(d_off.get(), d_b.get(), d_x.get(), sweeps);
  return to_host_vector(d_x.host(), order);
}

}  // namespace

TEST_CASE(
    "device line sweeps are the host multigrid's smoothing bit for bit when every cell "
    "lies in a line",
    "[multigrid][gpu]") {
  const bool fp32 = GENERATE(false, true);
  INFO((fp32 ? "FP32" : "FP64") << " off-diagonal blocks");
  const auto mesh = thin_layers(6, 6);
  const auto geometry = autan::compute_geometry(mesh);
  autan::solvers::Multigrid mg(mesh, geometry);
  const Level& l = mg.levels()[0];
  for (std::size_t li = 0; li < l.n_lines(); ++li) {
    REQUIRE(l.line_offsets[li + 1] - l.line_offsets[li] >= 2);
  }
  const Blocks a = dominant(l, 5, fp32);
  mg.set_operator(a.diag, a.upper, a.lower);
  const auto b = random_vector(5 * l.n_nodes, 7), x0 = random_vector(5 * l.n_nodes, 11);
  std::vector<double> x_host = x0;
  mg.smooth(b, x_host, 2);
  // The test's transcription of the host order agrees with the host itself here.
  HostLines lines(l, a);
  lines.factor();
  std::vector<double> x_spec = x0;
  lines.sweep(b, x_spec, 2);
  CHECK(same_bits(x_spec, x_host));
  std::vector<index_t> identity(l.n_nodes);
  std::iota(identity.begin(), identity.end(), 0);
  const auto x = fp32 ? device_sweep<float>(l, identity, a, b, x0, 2, 3)
                      : device_sweep<double>(l, identity, a, b, x0, 2, 3);
  CHECK(same_bits(x, x_host));
}

TEST_CASE("device line sweeps give the host order's bits on every level", "[multigrid][gpu]") {
  const bool fp32 = GENERATE(false, true);
  INFO((fp32 ? "FP32" : "FP64") << " off-diagonal blocks");
  auto run = [&](const std::vector<Level>& levels) {
    const auto orders = autan::solvers::gpu::device_orders(levels);
    std::size_t with_lines = 0;
    for (std::size_t lv = 0; lv < levels.size(); ++lv) {
      INFO("level " << lv);
      const Level& l = levels[lv];
      const Blocks a = dominant(l, 13 + static_cast<std::uint32_t>(lv), fp32);
      HostLines lines(l, a);
      lines.factor();
      const auto b = random_vector(5 * l.n_nodes, 17), x0 = random_vector(5 * l.n_nodes, 19);
      std::vector<double> x_spec = x0;
      lines.sweep(b, x_spec, 2);
      const auto x = fp32 ? device_sweep<float>(l, orders[lv], a, b, x0, 2, 23)
                          : device_sweep<double>(l, orders[lv], a, b, x0, 2, 23);
      CHECK(same_bits(x, x_spec));
      for (std::size_t li = 0; li < l.n_lines(); ++li) {
        if (l.line_offsets[li + 1] - l.line_offsets[li] > 1) ++with_lines;
      }
    }
    return with_lines;
  };
  SECTION("reference meshes") {
    const char* file = GENERATE(from_range(meshes));
    INFO(file);
    run(hierarchy(autan::io::read_mesh(mesh_dir / file)));
  }
  SECTION("wall layers, with lines on every level") { CHECK(run(hierarchy(wall_layers(8))) > 0); }
}

TEST_CASE("the device line factorisation refuses a singular block and an unfactored sweep",
          "[multigrid][gpu]") {
  const auto levels = hierarchy(thin_layers(4, 6));
  const Level& l = levels[0];
  std::vector<index_t> identity(l.n_nodes);
  std::iota(identity.begin(), identity.end(), 0);
  offset_t n_slots = 0;
  const EdgeSlots slots = csr_slots(l, identity, 0, n_slots);
  autan::solvers::gpu::LineSmoother smoother(l, identity, slots, static_cast<std::size_t>(n_slots));
  REQUIRE(smoother.n_lines() > 0);
  const DeviceArray<double> zero_diag(std::vector<double>(b2 * l.n_nodes, 0.0));
  const DeviceArray<double> zero_off(
      std::vector<double>(b2 * static_cast<std::size_t>(n_slots), 0.0));
  const DeviceArray<double> v(std::vector<double>(5 * l.n_nodes, 0.0));
  CHECK_THROWS_AS(smoother.sweep(zero_off.get(), v.get(), v.get(), 1), std::logic_error);
  CHECK_THROWS_AS(smoother.factor(zero_diag.get(), zero_off.get()), std::domain_error);
  CHECK_THROWS_AS(smoother.sweep(zero_off.get(), v.get(), v.get(), 1), std::logic_error);
}
