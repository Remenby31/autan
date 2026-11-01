// Time per cell of the device SSP-RK3 step (DeviceStepper::step: three residuals, the stages, the
// norms and the D0107 check) and of the host residual, on a mesh given on the command line.
// Usage: euler_step_bench MESH fp64|fp32+acc64 STEPS input|rcm [GATE]; prints one JSON object.
// `rcm` renumbers the cells (autan::rcm_order) first. With GATE, a FIFO, the bench prints `ready`
// after its setup and times only once a line arrives on GATE, so that a GPU lock the caller takes
// then (autan-gpu-lock) covers the timed steps alone (docs/infra/bench.md).
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/renumber.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/physics/euler.hpp"
#include "autan/physics/euler_device.hpp"
#include "autan/runtime/ranks.hpp"

namespace {

using autan::physics::BoundaryKind;
using autan::physics::Euler;
using autan::physics::Gas;
using autan::physics::Scheme;
using Clock = std::chrono::steady_clock;

constexpr Gas air{1.4, 287.0};

// One rank, no periodic pairs: no ghost cells to fill.
class NoGhosts final : public autan::runtime::GhostExchange {
 public:
  void forward(std::span<float* const> /*c*/, autan::MemorySpace /*s*/, void* /*stream*/) override {
  }
  void forward(std::span<double* const> /*c*/, autan::MemorySpace /*s*/,
               void* /*stream*/) override {}
};

template <class P>
int bench(const autan::Mesh& mesh, const char* precision, int steps, const char* gate) {
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  // Mach 0.5 freestream on farfield faces, a smooth density and pressure perturbation inside so
  // that the limiter works.
  const std::array<double, 5> w{1.2, 170.0, 0.0, 0.0, 1.0e5};
  Scheme s;
  s.scale = {w[0], 340.0, 340.0, 340.0, w[4]};
  s.mach_ref = 0.5;
  const double q2 = w[1] * w[1];
  s.reference = {w[0], w[0] * w[1], 0.0, 0.0, w[4] / (air.gamma - 1.0) + 0.5 * w[0] * q2};
  Euler<P> host(mesh, g, air, s);
  std::vector<BoundaryKind> kind(nb, BoundaryKind::farfield);
  std::vector<double> data(5 * nb);
  for (std::size_t k = 0; k < 5; ++k) std::fill_n(data.begin() + k * nb, nb, w[k]);
  host.set_boundary(kind, data);
  host.set_second_order(true);
  std::vector<double> u(5 * n);
  for (std::size_t c = 0; c < n; ++c) {
    const double bump = 0.05 * std::sin(6.0 * g.cell_centroid_x[c]) *
                        std::cos(5.0 * g.cell_centroid_y[c] + g.cell_centroid_z[c]);
    const double rho = w[0] * (1.0 + bump), p = w[4] * (1.0 + 1.4 * bump);
    u[c] = rho;
    u[n + c] = rho * w[1];
    u[2 * n + c] = u[3 * n + c] = 0.0;
    u[4 * n + c] = p / (air.gamma - 1.0) + 0.5 * rho * q2;
  }
  std::vector<typename P::state> stored(5 * n);
  std::vector<typename P::residual> res(5 * n);
  host.state(u, stored);
  host.residual(stored, res);
  const int host_reps = 3;
  const auto t1 = Clock::now();
  for (int i = 0; i < host_reps; ++i) host.residual(stored, res);
  const double host_s = std::chrono::duration<double>(Clock::now() - t1).count() / host_reps;
  autan::runtime::OneRank rank;
  NoGhosts ghosts;
  auto device =
      autan::physics::make_device_stepper<P>(mesh, g, host, air.gamma, true, rank, ghosts, nullptr);
  device->upload(u);
  if (gate != nullptr) {
    std::printf("ready\n");
    std::fflush(stdout);
    std::FILE* fifo = std::fopen(gate, "r");
    if (fifo == nullptr) {
      std::fprintf(stderr, "cannot open %s\n", gate);
      return 1;
    }
    char line[16];
    const bool released = std::fgets(line, sizeof line, fifo) != nullptr;
    std::fclose(fifo);
    if (!released) {
      std::fprintf(stderr, "gate %s closed before a line\n", gate);
      return 1;
    }
  }
  for (int i = 0; i < 5; ++i) static_cast<void>(device->step(0.5, std::nullopt, false));
  const auto t0 = Clock::now();
  for (int i = 0; i < steps; ++i) static_cast<void>(device->step(0.5, std::nullopt, false));
  const double step_s = std::chrono::duration<double>(Clock::now() - t0).count() / steps;

  std::printf(
      "{\"precision\":\"%s\",\"cells\":%zu,\"steps\":%d,\"device_step_ns_per_cell\":%.4f,"
      "\"host_residual_ns_per_cell\":%.4f}\n",
      precision, n, steps, 1e9 * step_s / static_cast<double>(n),
      1e9 * host_s / static_cast<double>(n));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: euler_step_bench MESH fp64|fp32+acc64 STEPS input|rcm [GATE]\n");
    return 2;
  }
  auto mesh = autan::io::read_mesh(argv[1]);
  if (std::string(argv[4]) == "rcm") mesh = autan::renumber_cells(mesh, autan::rcm_order(mesh));
  const int steps = std::atoi(argv[3]);
  const char* gate = argc > 5 ? argv[5] : nullptr;
  const std::string p = argv[2];
  if (p == "fp64") return bench<autan::kernels::fp64>(mesh, argv[2], steps, gate);
  if (p == "fp32+acc64") return bench<autan::kernels::fp32_acc64>(mesh, argv[2], steps, gate);
  std::fprintf(stderr, "unknown precision %s\n", argv[2]);
  return 2;
}
