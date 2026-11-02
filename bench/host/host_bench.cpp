// Host loop benchmark: the generated CPU loops of kernels/euler.py, timed one by one on a mesh, for
// `autan-bench run --suite host` (bench/src/autan_bench/runner.py reads its JSON on stdout).
//   autan_host_bench MESH [--policy fp32+acc64|fp64] [--order input|rcm] [--cpu N] [--gate 1]
//                         [--warmup N] [--reps N] [--batch N]
// --cpu N pins the process to logical CPU N before any work (GB10 mixes Cortex-X925 and A725
// cores, so an unpinned loop's time depends on the scheduler); the core's model, MIDR and maximum
// frequency go into the result's device. The loops are serial: one core is the whole machine
// they use. --gate 1: after the setup, print `autan-bench: ready` on stderr and wait for a line
// on stdin, so that only the verification and the timed loops run under the GPU lock.
// Each loop is timed alone, its output buffer zeroed outside the timing, and verified: the
// residual's face and boundary loops, run on zeroed buffers, give Euler::residual's result.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <sched.h>
#include <unistd.h>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/core/renumber.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/physics/euler.hpp"
#include "euler.hpp"
#include "euler_cpu.hpp"
#include "revision.h"

namespace {

using autan::index_t;
namespace ek = autan::kernels::euler;

struct Options {
  std::string mesh;
  std::string policy = "fp32+acc64";
  std::string order = "rcm";
  int cpu = -1;
  bool gate = false;
  int warmup = 3;
  int reps = 20;
  int batch = 5;  ///< runs averaged into each sample: a short loop's timing noise, divided
};

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(2);
}

Options parse(int argc, char** argv) {
  Options o;
  if (argc < 2) fail("usage: autan_host_bench MESH [options]");
  o.mesh = argv[1];
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string_view f(argv[i]);
    const std::string v(argv[i + 1]);
    if (f == "--policy") {
      o.policy = v;
    } else if (f == "--order") {
      o.order = v;
    } else if (f == "--cpu") {
      o.cpu = std::stoi(v);
    } else if (f == "--gate") {
      o.gate = v == "1";
    } else if (f == "--warmup") {
      o.warmup = std::stoi(v);
    } else if (f == "--reps") {
      o.reps = std::stoi(v);
    } else if (f == "--batch") {
      o.batch = std::stoi(v);
    } else {
      fail("unknown option " + std::string(f));
    }
  }
  if (o.policy != "fp32+acc64" && o.policy != "fp64") fail("--policy is fp32+acc64 or fp64");
  if (o.order != "input" && o.order != "rcm") fail("--order is input or rcm");
  if (o.reps < 3) fail("--reps is at least 3");
  if (o.batch < 1) fail("--batch is at least 1");
  return o;
}

std::string read_line(const std::string& path) {
  std::ifstream in(path);
  std::string s;
  std::getline(in, s);
  return s;
}

/// The core a process pinned to `cpu` runs on, as the kernel reports it (sysfs).
struct Core {
  std::vector<int> cpus;
  std::string name = "unpinned";
  std::string midr = "";
  long max_khz = 0;
};

Core pin(int cpu) {
  Core core;
  if (cpu < 0) return core;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof set, &set) != 0) fail("cannot pin to CPU " + std::to_string(cpu));
  cpu_set_t got;
  CPU_ZERO(&got);
  if (sched_getaffinity(0, sizeof got, &got) != 0 || CPU_COUNT(&got) != 1 ||
      !CPU_ISSET(cpu, &got)) {
    fail("the affinity to CPU " + std::to_string(cpu) + " did not hold");
  }
  core.cpus = {cpu};
  const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
  core.midr = read_line(base + "/regs/identification/midr_el1");
  const std::string khz = read_line(base + "/cpufreq/cpuinfo_max_freq");
  core.max_khz = khz.empty() ? 0 : std::stol(khz);
  // MIDR_EL1 bits 15:4 are the part number (Arm: 0xd85 Cortex-X925, 0xd87 Cortex-A725).
  const unsigned long midr = core.midr.empty() ? 0 : std::stoul(core.midr, nullptr, 16);
  const unsigned part = (midr >> 4) & 0xfff;
  core.name = part == 0xd85   ? "Cortex-X925"
              : part == 0xd87 ? "Cortex-A725"
                              : "CPU part 0x" + [&] {
                                  char b[8];
                                  std::snprintf(b, sizeof b, "%03x", part);
                                  return std::string(b);
                                }();
  return core;
}

void wait_for_go(bool gate) {
  if (!gate) return;
  std::fprintf(stderr, "autan-bench: ready\n");
  std::fflush(stderr);
  std::string line;
  if (!std::getline(std::cin, line)) fail("--gate: stdin closed before go");
}

struct Timed {
  std::string name;
  double bytes;
  std::string strategy;
  std::vector<double> seconds;
};

/// `reps` samples of `loop`'s time after `warmup` runs: each the mean of `batch` timed runs,
/// `reset` (untimed) before each run.
std::vector<double> time_it(const Options& o, const std::function<void()>& reset,
                            const std::function<void()>& loop) {
  for (int i = 0; i < o.warmup; ++i) {
    reset();
    loop();
  }
  std::vector<double> out;
  for (int i = 0; i < o.reps; ++i) {
    double total = 0.0;
    for (int b = 0; b < o.batch; ++b) {
      reset();
      const auto t0 = std::chrono::steady_clock::now();
      loop();
      total += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    out.push_back(total / o.batch);
  }
  return out;
}

/// The largest difference relative to b's largest magnitude; infinite when either holds a value
/// that is not finite, so that a NaN or an overflow never verifies.
double max_rel_diff(std::span<const double> a, std::span<const double> b) {
  double scale = 0.0, diff = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      return std::numeric_limits<double>::infinity();
    scale = std::max(scale, std::abs(b[i]));
    diff = std::max(diff, std::abs(a[i] - b[i]));
  }
  return scale > 0.0 ? diff / scale : diff;
}

template <class T>
std::vector<double> as_double(const std::vector<T>& v) {
  return {v.begin(), v.end()};
}

/// The DPW-free test case: freestream at Mach 0.3 with a smooth perturbation, so that every face
/// has a non-trivial flux, and every boundary face a farfield condition.
template <class P>
void run(const Options& o, const autan::Mesh& mesh, const autan::Geometry& geometry,
         std::vector<Timed>& out, bool& verified, double& tolerance) {
  using Res = typename P::residual;
  const autan::physics::Gas gas;
  const double rho = 1.2, p = 101325.0, a = std::sqrt(gas.gamma * p / rho), u = 0.3 * a;
  autan::physics::Scheme scheme;
  scheme.reference = {rho, rho * u, 0.0, 0.0, p / (gas.gamma - 1.0) + 0.5 * rho * u * u};
  scheme.scale = {rho, a, a, a, p};
  scheme.mach_ref = 0.3;
  autan::physics::Euler<P> euler(mesh, geometry, gas, scheme);
  const index_t nc = euler.n_cells(), nb = euler.n_boundary_faces();
  std::vector<autan::physics::BoundaryKind> kind(nb, autan::physics::BoundaryKind::farfield);
  std::vector<double> bc(5 * static_cast<std::size_t>(nb));
  for (index_t b = 0; b < nb; ++b) {
    const double w[5] = {rho, u, 0.0, 0.0, p};
    for (int c = 0; c < 5; ++c) bc[c * static_cast<std::size_t>(nb) + b] = w[c];
  }
  euler.set_boundary(kind, bc);
  std::vector<double> cons(5 * static_cast<std::size_t>(nc));
  for (index_t i = 0; i < nc; ++i) {
    const double s = 1.0 + 0.05 * std::sin(0.37 * i);  // deterministic, cell-varying
    const double r = rho * s, ux = u * s;
    const double w[5] = {r, r * ux, 0.01 * r * a, 0.0,
                         p / (gas.gamma - 1.0) + 0.5 * r * (ux * ux + 1e-4 * a * a)};
    for (int c = 0; c < 5; ++c) cons[c * static_cast<std::size_t>(nc) + i] = w[c];
  }
  std::vector<typename P::state> state(cons.size());
  euler.state(cons, state);
  std::vector<Res> reference(cons.size());
  euler.residual(state, reference);  // fills prim, grad and phi, which the loops read

  // The loops write through the generated Data's pointers: a copy of it, its output redirected.
  ek::Data<P> d = euler.data();
  std::vector<Res> res(cons.size());
  d.res = res.data();
  const auto zero = [&] { std::fill(res.begin(), res.end(), Res(0)); };
  // The setup above is the host's work outside the lock; what follows is timed under it.
  wait_for_go(o.gate);

  zero();
  ek::cpu::roe_faces(d);
  ek::cpu::roe_boundaries(d);
  tolerance = 1e-12;
  if (max_rel_diff(as_double(res), as_double(reference)) > tolerance) verified = false;

  // Compulsory traffic, every array touched once per rep (counts confirmed by CFD - Euler for
  // fp32_acc64: 204 B per cell and 84 B per interior face): per cell prim, grad, phi, centroid
  // read and the residual read and written; per face centroid, normal, area, moment, owner and
  // neighbour.
  using S = typename P::state;
  using G = typename P::grad;
  using Geo = typename P::geom;
  using GV = typename P::geom_value;
  const double cell = 5.0 * sizeof(S) + 15.0 * sizeof(G) + 5.0 * sizeof(S) + 3.0 * sizeof(Geo) +
                      2.0 * 5.0 * sizeof(Res);
  const double face_geometry = 3.0 * sizeof(Geo) + 3.0 * sizeof(GV) + sizeof(GV) + 9.0 * sizeof(GV);
  const double ni = d.topo.n_owned_interior_faces;
  out.push_back({"euler.roe_faces", cell * nc + (face_geometry + 2.0 * sizeof(index_t)) * ni,
                 "scatter", time_it(o, zero, [&] { ek::cpu::roe_faces(d); })});
  // Boundary faces: geometry, owner, condition kind and data; their owners counted once.
  std::set<index_t> owners;
  for (index_t b = 0; b < nb; ++b) owners.insert(d.topo.owner[d.topo.n_interior_faces + b]);
  const double bnd = (face_geometry + sizeof(index_t) + sizeof(Geo) + 5.0 * sizeof(S)) * nb +
                     cell * static_cast<double>(owners.size());
  out.push_back({"euler.roe_boundaries", bnd, "scatter",
                 time_it(o, zero, [&] { ek::cpu::roe_boundaries(d); })});
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  const Core core = pin(o.cpu);  // before any allocation: first-touch memory and caches
  const auto t0 = std::chrono::steady_clock::now();
  autan::Mesh mesh = autan::io::read_mesh(o.mesh);
  const auto t1 = std::chrono::steady_clock::now();
  if (o.order == "rcm") mesh = autan::renumber_cells(mesh, autan::rcm_order(mesh));
  const autan::Geometry geometry = autan::compute_geometry(mesh);
  const double read_seconds = std::chrono::duration<double>(t1 - t0).count();
  const double setup_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

  std::vector<Timed> timed;
  bool verified = true;
  double tolerance = 0.0;
  if (o.policy == "fp64") {
    run<autan::kernels::fp64>(o, mesh, geometry, timed, verified, tolerance);
  } else {
    run<autan::kernels::fp32_acc64>(o, mesh, geometry, timed, verified, tolerance);
  }

  std::printf("{\"device\":{\"name\":\"%s\",\"cpus\":[", core.name.c_str());
  for (std::size_t i = 0; i < core.cpus.size(); ++i)
    std::printf("%s%d", i ? "," : "", core.cpus[i]);
  std::printf("],\"midr\":\"%s\",\"max_khz\":%ld,\"online_cpus\":%ld},", core.midr.c_str(),
              core.max_khz, sysconf(_SC_NPROCESSORS_ONLN));
  std::printf("\"revision\":{\"commit\":\"%s\",\"dirty\":%s},", AUTAN_BENCH_COMMIT,
              AUTAN_BENCH_DIRTY);
  // No runtime or driver on the host: the compiler's version stands in for the runtime's.
  std::printf("\"backend\":\"cpu\",\"runtime_version\":%d,\"driver_version\":0,",
              __GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__);
  const std::string sha = autan::io::mesh_hash(o.mesh);
  std::printf(
      "\"config\":{\"mesh_file\":\"%s\",\"mesh_sha256\":\"%s\",\"policy\":\"%s\","
      "\"order\":\"%s\",\"layout\":\"soa\",\"n_cells\":%d,\"cpus\":\"%d\",\"warmup\":%d,"
      "\"reps\":%d,\"batch\":%d,\"tolerance\":%.1e},",
      std::filesystem::path(o.mesh).filename().string().c_str(), sha.c_str(), o.policy.c_str(),
      o.order.c_str(), static_cast<int>(mesh.n_cells()), o.cpu, o.warmup, o.reps, o.batch,
      tolerance);
  std::printf("\"setup_seconds\":{\"read\":%.3f,\"renumber_geometry\":%.3f},", read_seconds,
              setup_seconds);
  std::printf("\"verified\":%s,\"kernels\":[", verified ? "true" : "false");
  for (std::size_t k = 0; k < timed.size(); ++k) {
    const auto& t = timed[k];
    std::printf(
        "%s{\"name\":\"%s\",\"bytes_per_rep\":%.0f,\"cells\":%d,\"params\":{\"strategy\":"
        "\"%s\"},\"seconds\":[",
        k ? "," : "", t.name.c_str(), t.bytes, static_cast<int>(mesh.n_cells()),
        t.strategy.c_str());
    for (std::size_t i = 0; i < t.seconds.size(); ++i) {
      std::printf("%s%.9e", i ? "," : "", t.seconds[i]);
    }
    std::printf("]}");
  }
  std::printf("]}\n");
  return 0;  // verified: false is autan-bench's VERIFY_FAILED, not a crash
}
