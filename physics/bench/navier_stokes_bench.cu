// Times the Navier-Stokes and Spalart-Allmaras loops on a CUDA device (kernels/navier_stokes.py,
// kernels/spalart_allmaras.py), each loop and the full residuals, through the device classes
// autan-run uses, and prints the raw JSON autan-bench reads. bytes_per_rep is each loop's minimum
// bytes touched, derived by the generator (<loop>_bytes, autan_codegen.traffic) from the real
// element counts; a sequence's is the sum of its loops'. The residuals are checked against the
// host's before timing.
//
//   navier_stokes_bench MESH [--policy fp64|fp32+acc64] [--flux roe|hllc|ausm-up]
//                            [--fit linear|quadratic] [--warmup N] [--reps N] [--batch N]
//                            [--gate 1]
//   --gate 1: after the host setup, print `autan-bench: ready` on stderr and wait for a line on
//   stdin before any CUDA call (autan-bench run --ready takes the GPU lock only then).
// Conditions: an outlet on the zone xmax, an isothermal wall on ymin, slip on ymax and the flow's
// state on every other boundary face, as tests/physics/test_navier_stokes_device.cpp; a smooth
// subsonic flow over the unit cube and a positive nt, so that SA-neg takes its fw and ft2 branch.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/renumber.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/traffic.hpp"
#include "autan/physics/navier_stokes.hpp"
#include "autan/physics/spalart_allmaras.hpp"
#include "autan/runtime/physics.hpp"
#include "autan/runtime/ranks.hpp"
#include "navier_stokes_device.cuh"
#include "revision.h"

namespace {

using autan::kernels::Counts;
using autan::physics::BoundaryKind;
using autan::physics::Flux;
using autan::physics::Gas;
using autan::physics::GradientFit;
using autan::physics::NavierStokes;
using autan::physics::Scheme;
using autan::physics::SpalartAllmaras;
using autan::physics::Transport;
namespace dev = autan::physics::device;
namespace gpu = autan::kernels::gpu;
namespace ek = autan::kernels::euler;
namespace nk = autan::kernels::navier_stokes;
namespace sk = autan::kernels::spalart_allmaras;

constexpr Gas air{1.4, 287.0};

struct Options {
  std::string mesh;
  std::string policy = "fp32+acc64";
  std::string flux = "roe";
  std::string fit = "linear";
  bool gate = false;
  int warmup = 5;
  int reps = 30;
  int batch = 5;
};

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(2);
}

void wait_for_go(bool gate) {
  if (!gate) return;
  std::fprintf(stderr, "autan-bench: ready\n");
  std::fflush(stderr);
  std::string line;
  if (!std::getline(std::cin, line)) fail("--gate: stdin closed before go");
}

Options parse(int argc, char** argv) {
  Options o;
  if (argc < 2) fail("usage: navier_stokes_bench MESH [options]");
  o.mesh = argv[1];
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string_view f(argv[i]);
    const std::string v(argv[i + 1]);
    if (f == "--policy")
      o.policy = v;
    else if (f == "--flux")
      o.flux = v;
    else if (f == "--fit")
      o.fit = v;
    else if (f == "--gate")
      o.gate = std::stoi(v) != 0;
    else if (f == "--warmup")
      o.warmup = std::stoi(v);
    else if (f == "--reps")
      o.reps = std::stoi(v);
    else if (f == "--batch")
      o.batch = std::stoi(v);
    else
      fail("unknown flag " + std::string(f));
  }
  if (o.fit != "linear" && o.fit != "quadratic") fail("--fit is linear or quadratic");
  return o;
}

class NoGhosts final : public autan::runtime::GhostExchange {
 public:
  void forward(std::span<float* const>, autan::MemorySpace, void*) override {}
  void forward(std::span<double* const>, autan::MemorySpace, void*) override {}
};

// A smooth subsonic flow over the unit cube: rho, u, v, w, p at (x, y, z).
std::array<double, 5> flow(double x, double y, double z) {
  return {1.2 + 0.1 * std::sin(2.0 * x + y), 80.0 + 15.0 * std::cos(x + 2.0 * z),
          -20.0 + 10.0 * std::sin(y + z), 5.0 * std::cos(3.0 * x),
          1.0e5 + 4.0e3 * std::sin(x + y + z)};
}

std::array<double, 5> conserved(const std::array<double, 5>& w) {
  const double q2 = w[1] * w[1] + w[2] * w[2] + w[3] * w[3];
  return {w[0], w[0] * w[1], w[0] * w[2], w[0] * w[3], w[4] / (air.gamma - 1.0) + 0.5 * w[0] * q2};
}

double nt_at(double x, double y, double z) {
  return 1.0e-4 * (1.0 + 0.5 * std::sin(2.0 * x + 3.0 * y + z));
}

template <class Launch>
std::vector<double> time_it(const Options& o, Launch launch) {
  cudaEvent_t start, stop;
  gpu::check(cudaEventCreate(&start), "event");
  gpu::check(cudaEventCreate(&stop), "event");
  std::vector<double> out;
  for (int r = 0; r < o.warmup + o.reps; ++r) {
    gpu::check(cudaEventRecord(start), "record");
    for (int i = 0; i < o.batch; ++i) launch();
    gpu::check(cudaEventRecord(stop), "record");
    gpu::check(cudaEventSynchronize(stop), "sync");
    float ms = 0.0f;
    gpu::check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    if (r >= o.warmup) out.push_back(ms * 1e-3 / o.batch);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return out;
}

struct Loop {
  std::string name;
  double bytes;
  std::function<void()> launch;
};

struct Timed {
  std::string name;
  double bytes;
  std::vector<double> seconds;
};

struct Run {
  std::vector<Timed> timed;
  bool verified = true;
  double ns_difference = 0.0;
  double sa_difference = 0.0;
  double tolerance = 0.0;
};

// max_k max_i |a - b| / max_i |b| over the equations of `b` (n values each).
template <class R>
double relative(std::span<const double> a, std::span<const R> b, std::size_t n) {
  double worst = 0.0;
  for (std::size_t k = 0; k < b.size() / n; ++k) {
    double num = 0.0, den = 0.0;
    for (std::size_t c = 0; c < n; ++c) {
      const double x = a[k * n + c], y = static_cast<double>(b[k * n + c]);
      if (!std::isfinite(x)) return std::numeric_limits<double>::infinity();
      num = std::max(num, std::abs(x - y));
      den = std::max(den, std::abs(y));
    }
    worst = std::max(worst, num / den);
  }
  return worst;
}

template <class P>
Run run(const Options& o, const autan::Mesh& mesh, const autan::Geometry& g) {
  using S = typename P::state;
  using R = typename P::residual;
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());

  Scheme s;
  s.flux = o.flux == "hllc" ? Flux::hllc : o.flux == "ausm-up" ? Flux::ausm_up : Flux::roe;
  s.gradient_fit = o.fit == "quadratic" ? GradientFit::quadratic : GradientFit::linear;
  const auto ref = flow(0.5, 0.5, 0.5);
  s.mach_ref = std::hypot(ref[1], ref[2], ref[3]) / std::sqrt(air.gamma * ref[4] / ref[0]);
  s.scale = {ref[0], 100.0, 100.0, 100.0, ref[4]};
  s.reference = conserved(ref);
  const Transport t;
  std::vector<BoundaryKind> kind(nb);
  std::vector<double> data(5 * nb, 0.0), nt_b(nb), wall(n), u(5 * n), rnu(n);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto f = ni + b;
    const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
    auto w = flow(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
    nt_b[b] = nt_at(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
    if (zone == "xmax") {
      kind[b] = BoundaryKind::outlet;
    } else if (zone == "ymin") {
      kind[b] = BoundaryKind::wall;
      w = {300.0, 0.0, 0.0, 0.0, 0.0};
    } else if (zone == "ymax") {
      kind[b] = BoundaryKind::slip;
    } else {
      kind[b] = BoundaryKind::state;
    }
    for (std::size_t k = 0; k < 5; ++k) data[k * nb + b] = w[k];
  }
  for (std::size_t c = 0; c < n; ++c) {
    const double x = g.cell_centroid_x[c], y = g.cell_centroid_y[c], z = g.cell_centroid_z[c];
    const auto w = conserved(flow(x, y, z));
    for (std::size_t k = 0; k < 5; ++k) u[k * n + c] = w[k];
    rnu[c] = w[0] * nt_at(x, y, z);
    wall[c] = y + 0.05;
  }

  NavierStokes<P> host(mesh, g, air, s, t);
  host.set_boundary(kind, data);
  host.set_second_order(true);
  SpalartAllmaras<P> sa(mesh, g, host, t);
  sa.set_boundary(nt_b);
  sa.set_wall_distance(wall);
  std::vector<S> stored(5 * n), rs(n);
  host.state(u, stored);
  for (std::size_t c = 0; c < n; ++c) rs[c] = S(rnu[c]);
  std::vector<R> r_ns(5 * n), r_sa(n);
  // The model reads the primitives of the flow's last residual: laminar first, then with the
  // eddy viscosity of rho nt.
  host.residual(stored, r_ns);
  sa.eddy_viscosity(rs);
  host.residual(stored, r_ns);
  sa.residual(rs, r_sa);

  // Element counts of the topologies each module's loops run on: the Euler loops on the
  // gradient stencil, the boundary fit on its own, SA on the flow's gradient stencil.
  const Counts euler_counts = autan::kernels::counts(host.euler().data().topo);
  const Counts fit_counts = autan::kernels::counts(host.data().topo);
  const Counts sa_counts = autan::kernels::counts(sa.data().topo);
  Counts band_counts = euler_counts;
  const bool quadratic = s.gradient_fit == GradientFit::quadratic;
  const auto band = quadratic ? host.euler().quadratic_gradient()
                              : typename autan::physics::Euler<P>::QuadraticGradient{};
  if (quadratic) {
    auto topo = host.euler().data().topo;
    topo.n_stencil = static_cast<autan::kernels::offset_t>(band.entries.size());
    topo.stencil_offsets = band.offsets.data();
    topo.stencil = band.entries.data();
    band_counts = autan::kernels::counts(topo);
  }

  wait_for_go(o.gate);
  autan::runtime::OneRank rank;
  NoGhosts ghosts;
  dev::NavierStokesStepper<P> stepper(mesh, g, host, air.gamma, true, rank, ghosts, nullptr,
                                      nullptr);
  dev::SpalartAllmarasDevice<P> model(sa, stepper);

  Run out;
  // Same loops on host and device, summed in other orders (no global keys here): each equation
  // against its own scale, 1024 eps of the compute type, as euler_bench.
  out.tolerance = 1024.0 * std::numeric_limits<typename P::compute>::epsilon();
  std::vector<double> d_ns(5 * n), d_sa(n);
  stepper.residual(u, d_ns);
  model.residual(rnu, d_sa);
  out.ns_difference = relative(std::span<const double>(d_ns), std::span<const R>(r_ns), n);
  out.sa_difference = relative(std::span<const double>(d_sa), std::span<const R>(r_sa), n);
  out.verified = out.ns_difference <= out.tolerance && out.sa_difference <= out.tolerance;
  stepper.upload(u);

  const auto& sched = stepper.mesh().schedule();
  auto d = stepper.flow();
  const auto& v = stepper.viscous();
  const auto& a = model.data();
  constexpr auto gather = gpu::Strategy::store_gather;
  const bool planar = host.euler().planar();
  const double cs = static_cast<double>(n);
  // dev::store_state: the conserved state (double) read, the stored state written.
  const double store_bytes = cs * 5.0 * (sizeof(double) + sizeof(S));

  // The Euler residual as the stepper runs it (CudaStepper::residual), then the viscous terms.
  std::vector<Loop> euler = {
      {"primitives", ek::primitives_bytes<P>(euler_counts),
       [&] { ek::gpu_loops::primitives(d, sched); }},
      {"boundary_states", ek::boundary_states_bytes<P>(euler_counts),
       [&] { ek::gpu_loops::boundary_states(d, sched); }},
  };
  if (quadratic) {
    euler.push_back({"band_gradients", ek::band_gradients_bytes<P>(band_counts), [&] {
                       auto b = d;
                       b.topo = stepper.band_topology();
                       ek::gpu_loops::band_gradients(b, sched);
                     }});
    euler.push_back({"quadratic_gradients", ek::quadratic_gradients_bytes<P>(euler_counts),
                     [&] { ek::gpu_loops::quadratic_gradients(d, sched); }});
  } else {
    euler.push_back({"gradients", ek::gradients_bytes<P>(euler_counts),
                     [&] { ek::gpu_loops::gradients(d, sched); }});
  }
  euler.push_back(
      {"limiter", ek::limiter_bytes<P>(euler_counts), [&] { ek::gpu_loops::limiter(d, sched); }});
  euler.push_back(
      {"sources", ek::sources_bytes<P>(euler_counts), [&] { ek::gpu_loops::sources(d, sched); }});
  if (planar) {
    euler.push_back(s.flux == Flux::hllc
                        ? Loop{"faces_planar", ek::hllc_faces_planar_bytes<P>(euler_counts),
                               [&] { ek::gpu_loops::hllc_faces_planar(d, gather, sched); }}
                    : s.flux == Flux::ausm_up
                        ? Loop{"faces_planar", ek::ausm_faces_planar_bytes<P>(euler_counts),
                               [&] { ek::gpu_loops::ausm_faces_planar(d, gather, sched); }}
                        : Loop{"faces_planar", ek::roe_faces_planar_bytes<P>(euler_counts),
                               [&] { ek::gpu_loops::roe_faces_planar(d, gather, sched); }});
    euler.push_back(
        s.flux == Flux::hllc
            ? Loop{"boundaries_planar", ek::hllc_boundaries_planar_bytes<P>(euler_counts),
                   [&] { ek::gpu_loops::hllc_boundaries_planar(d, gather, sched); }}
        : s.flux == Flux::ausm_up
            ? Loop{"boundaries_planar", ek::ausm_boundaries_planar_bytes<P>(euler_counts),
                   [&] { ek::gpu_loops::ausm_boundaries_planar(d, gather, sched); }}
            : Loop{"boundaries_planar", ek::roe_boundaries_planar_bytes<P>(euler_counts),
                   [&] { ek::gpu_loops::roe_boundaries_planar(d, gather, sched); }});
  } else {
    euler.push_back(
        s.flux == Flux::hllc      ? Loop{"faces", ek::hllc_faces_bytes<P>(euler_counts),
                                         [&] { ek::gpu_loops::hllc_faces(d, gather, sched); }}
        : s.flux == Flux::ausm_up ? Loop{"faces", ek::ausm_faces_bytes<P>(euler_counts),
                                         [&] { ek::gpu_loops::ausm_faces(d, gather, sched); }}
                                  : Loop{"faces", ek::roe_faces_bytes<P>(euler_counts),
                                         [&] { ek::gpu_loops::roe_faces(d, gather, sched); }});
    euler.push_back(
        s.flux == Flux::hllc      ? Loop{"boundaries", ek::hllc_boundaries_bytes<P>(euler_counts),
                                         [&] { ek::gpu_loops::hllc_boundaries(d, gather, sched); }}
        : s.flux == Flux::ausm_up ? Loop{"boundaries", ek::ausm_boundaries_bytes<P>(euler_counts),
                                         [&] { ek::gpu_loops::ausm_boundaries(d, gather, sched); }}
                                  : Loop{"boundaries", ek::roe_boundaries_bytes<P>(euler_counts),
                                         [&] { ek::gpu_loops::roe_boundaries(d, gather, sched); }});
  }
  // NavierStokesStepper::add_terms. Under the quadratic fit the linear gradient goes to the
  // viscous gradient array; here it takes the flow's limiter bounds as scratch, which the
  // residual has consumed by then.
  std::vector<Loop> viscous;
  if (quadratic) {
    viscous.push_back({"linear_gradients", ek::gradients_bytes<P>(euler_counts), [&] {
                         auto l = d;
                         l.grad = v.grad;
                         ek::gpu_loops::gradients(l, sched);
                       }});
  }
  viscous.push_back({"temperatures_velocities", nk::temperatures_velocities_bytes<P>(fit_counts),
                     [&] { nk::gpu_loops::temperatures_velocities(v, sched); }});
  viscous.push_back({"boundary_temperatures_velocities",
                     nk::boundary_temperatures_velocities_bytes<P>(fit_counts),
                     [&] { nk::gpu_loops::boundary_temperatures_velocities(v, sched); }});
  viscous.push_back({"cubic_fits", nk::cubic_fits_bytes<P>(fit_counts),
                     [&] { nk::gpu_loops::cubic_fits(v, sched); }});
  viscous.push_back({"viscous_faces", nk::viscous_faces_bytes<P>(fit_counts),
                     [&] { nk::gpu_loops::viscous_faces(v, gather, sched); }});
  viscous.push_back({"viscous_boundaries", nk::viscous_boundaries_bytes<P>(fit_counts),
                     [&] { nk::gpu_loops::viscous_boundaries(v, gather, sched); }});
  const Loop radius{"viscous_spectral_radius", nk::viscous_spectral_radius_bytes<P>(fit_counts),
                    [&] { nk::gpu_loops::viscous_spectral_radius(v, sched); }};
  // SpalartAllmarasDevice::residual and ::eddy_viscosity, from the stored rho nt.
  const std::vector<Loop> sa_residual = {
      {"nt_values", sk::nt_values_bytes<P>(sa_counts), [&] { sk::gpu_loops::nt_values(a, sched); }},
      {"boundary_values", sk::boundary_values_bytes<P>(sa_counts),
       [&] { sk::gpu_loops::boundary_values(a, sched); }},
      {"nt_gradients", sk::nt_gradients_bytes<P>(sa_counts),
       [&] { sk::gpu_loops::nt_gradients(a, sched); }},
      {"nt_sources", sk::nt_sources_bytes<P>(sa_counts),
       [&] { sk::gpu_loops::nt_sources(a, sched); }},
      {"nt_faces", sk::nt_faces_bytes<P>(sa_counts),
       [&] { sk::gpu_loops::nt_faces(a, gather, sched); }},
      {"nt_boundaries", sk::nt_boundaries_bytes<P>(sa_counts),
       [&] { sk::gpu_loops::nt_boundaries(a, gather, sched); }},
  };
  const std::vector<Loop> sa_viscosity = {
      sa_residual[0],
      sa_residual[1],
      {"eddy_viscosities", sk::eddy_viscosities_bytes<P>(sa_counts),
       [&] { sk::gpu_loops::eddy_viscosities(a, sched); }},
      {"boundary_eddy_viscosities", sk::boundary_eddy_viscosities_bytes<P>(sa_counts),
       [&] { sk::gpu_loops::boundary_eddy_viscosities(a, sched); }},
  };
  auto total = [](const std::vector<Loop>& loops) {
    double b = 0.0;
    for (const auto& l : loops) b += l.bytes;
    return b;
  };
  const double ns_bytes = store_bytes + total(euler) + total(viscous);
  out.timed.push_back(
      {"residual_navier_stokes", ns_bytes, time_it(o, [&] { stepper.evaluate_residual(); })});
  out.timed.push_back({"residual_spalart_allmaras", total(sa_residual), time_it(o, [&] {
                         for (const auto& l : sa_residual) l.launch();
                       })});
  out.timed.push_back({"eddy_viscosity", total(sa_viscosity), time_it(o, [&] {
                         for (const auto& l : sa_viscosity) l.launch();
                       })});
  const std::array<const std::vector<Loop>*, 3> groups{&euler, &viscous, &sa_residual};
  for (const auto* group : groups) {
    for (const auto& l : *group) out.timed.push_back({l.name, l.bytes, time_it(o, l.launch)});
  }
  out.timed.push_back({radius.name, radius.bytes, time_it(o, radius.launch)});
  for (std::size_t i = 2; i < sa_viscosity.size(); ++i) {
    out.timed.push_back(
        {sa_viscosity[i].name, sa_viscosity[i].bytes, time_it(o, sa_viscosity[i].launch)});
  }
  gpu::check(cudaDeviceSynchronize(), "loops");
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  const auto t0 = std::chrono::steady_clock::now();
  autan::Mesh mesh = autan::io::read_mesh(o.mesh);
  const auto t1 = std::chrono::steady_clock::now();
  mesh = autan::renumber_cells(mesh, autan::rcm_order(mesh));
  const auto t2 = std::chrono::steady_clock::now();
  const auto geometry = autan::compute_geometry(mesh);
  Run r;
  if (o.policy == "fp64") {
    r = run<autan::kernels::fp64>(o, mesh, geometry);
  } else if (o.policy == "fp32+acc64") {
    r = run<autan::kernels::fp32_acc64>(o, mesh, geometry);
  } else {
    fail("--policy is fp64 or fp32+acc64");
  }
  int device = 0;
  cudaDeviceProp prop{};
  gpu::check(cudaGetDevice(&device), "device");
  gpu::check(cudaGetDeviceProperties(&prop, device), "properties");
  int runtime = 0, driver = 0;
  cudaRuntimeGetVersion(&runtime);
  cudaDriverGetVersion(&driver);
  char uuid[37];
  const auto* u = reinterpret_cast<const unsigned char*>(prop.uuid.bytes);
  std::snprintf(uuid, sizeof uuid,
                "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1],
                u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14],
                u[15]);
  std::printf(
      "{\"device\":{\"name\":\"%s\",\"uuid\":\"GPU-%s\",\"compute_capability\":\"%d.%d\","
      "\"sm_count\":%d,\"l2_bytes\":%d,\"memory_bytes\":%zu},",
      prop.name, uuid, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize,
      prop.totalGlobalMem);
  std::printf("\"revision\":{\"commit\":\"%s\",\"dirty\":%s},", AUTAN_BENCH_COMMIT,
              AUTAN_BENCH_DIRTY);
  std::printf("\"backend\":\"cuda\",\"runtime_version\":%d,\"driver_version\":%d,", runtime,
              driver);
  const std::string sha = autan::io::mesh_hash(o.mesh);
  std::printf(
      "\"config\":{\"mesh_file\":\"%s\",\"mesh_sha256\":\"%s\",\"policy\":\"%s\",\"flux\":\"%s\","
      "\"fit\":\"%s\",\"order\":\"rcm\",\"strategy\":\"store-gather\",\"n_cells\":%d,"
      "\"n_faces\":%d,\"warmup\":%d,\"reps\":%d,\"batch\":%d,\"tolerance\":%.1e},",
      std::filesystem::path(o.mesh).filename().string().c_str(), sha.c_str(), o.policy.c_str(),
      o.flux.c_str(), o.fit.c_str(), mesh.n_cells(), mesh.n_faces(), o.warmup, o.reps, o.batch,
      r.tolerance);
  std::printf("\"setup_seconds\":{\"read\":%.3f,\"renumber\":%.3f},",
              std::chrono::duration<double>(t1 - t0).count(),
              std::chrono::duration<double>(t2 - t1).count());
  std::printf("\"verified\":%s,\"kernels\":[", r.verified ? "true" : "false");
  for (std::size_t k = 0; k < r.timed.size(); ++k) {
    const auto& t = r.timed[k];
    std::printf(
        "%s{\"name\":\"%s\",\"bytes_per_rep\":%.0f,\"cells\":%d,\"params\":{\"policy\":\"%s\","
        "\"flux\":\"%s\",\"fit\":\"%s\"},\"seconds\":[",
        k ? "," : "", t.name.c_str(), t.bytes, mesh.n_cells(), o.policy.c_str(), o.flux.c_str(),
        o.fit.c_str());
    for (std::size_t i = 0; i < t.seconds.size(); ++i)
      std::printf("%s%.9e", i ? "," : "", t.seconds[i]);
    std::printf("]}");
  }
  std::printf("]}\n");
  std::fprintf(stderr,
               "max relative difference to the host: Navier-Stokes %.3e, Spalart-Allmaras %.3e "
               "(tolerance %.1e)\n",
               r.ns_difference, r.sa_difference, r.tolerance);
  return r.verified ? 0 : 1;
}
