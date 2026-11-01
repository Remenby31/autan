// Times the multigrid's device transfers between the two finest levels of a mesh's agglomeration
// hierarchy (docs/design/multigrid-gpu.md#transfers) and prints the raw JSON autan-bench reads.
// bytes_per_rep counts every array a kernel touches read or written once; each result is checked
// bitwise against its definition in the host multigrid's order before timing.
//
//   autan_mg_transfers_bench MESH [--warmup N] [--reps N] [--batch N] [--gate 1]
//   --gate 1: after the host setup, print `autan-bench: ready` on stderr and wait for a line on
//   stdin before any CUDA call (autan-bench run --ready takes the GPU lock only then).
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "autan/core/geometry.hpp"
#include "autan/core/renumber.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/solvers/agglomeration.hpp"
#include "autan/solvers/multigrid_gpu.hpp"
#include "revision.h"

namespace {

struct Options {
  std::string mesh;
  bool gate = false;
  int warmup = 5;
  int reps = 30;
  int batch = 5;
};

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(2);
}

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
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
  if (argc < 2) fail("usage: autan_mg_transfers_bench MESH [options]");
  o.mesh = argv[1];
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string_view f(argv[i]);
    const int v = std::stoi(argv[i + 1]);
    if (f == "--gate")
      o.gate = v != 0;
    else if (f == "--warmup")
      o.warmup = v;
    else if (f == "--reps")
      o.reps = v;
    else if (f == "--batch")
      o.batch = v;
    else
      fail("unknown flag " + std::string(f));
  }
  return o;
}

class Device {
 public:
  explicit Device(const std::vector<double>& v) : n_(v.size()) {
    check(cudaMalloc(reinterpret_cast<void**>(&p_), n_ * sizeof(double)), "cudaMalloc");
    check(cudaMemcpy(p_, v.data(), n_ * sizeof(double), cudaMemcpyHostToDevice), "upload");
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device() { cudaFree(p_); }
  [[nodiscard]] double* get() const { return p_; }
  [[nodiscard]] std::vector<double> download() const {
    std::vector<double> v(n_);
    check(cudaMemcpy(v.data(), p_, n_ * sizeof(double), cudaMemcpyDeviceToHost), "download");
    return v;
  }

 private:
  double* p_ = nullptr;
  std::size_t n_;
};

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  for (std::size_t k = 0; k < a.size(); ++k) {
    if (std::bit_cast<std::uint64_t>(a[k]) != std::bit_cast<std::uint64_t>(b[k])) return false;
  }
  return a.size() == b.size();
}

std::vector<double> values(std::size_t n, double phase) {
  std::vector<double> v(n);
  for (std::size_t k = 0; k < n; ++k) v[k] = std::sin(0.37 * static_cast<double>(k) + phase);
  return v;
}

std::vector<double> time_it(const Options& o, const std::function<void()>& launch) {
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "event");
  check(cudaEventCreate(&stop), "event");
  std::vector<double> out;
  for (int r = 0; r < o.warmup + o.reps; ++r) {
    check(cudaEventRecord(start), "record");
    for (int i = 0; i < o.batch; ++i) launch();
    check(cudaEventRecord(stop), "record");
    check(cudaEventSynchronize(stop), "sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    if (r >= o.warmup) out.push_back(ms * 1e-3 / o.batch);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return out;
}

struct Timed {
  std::string name;
  double bytes;
  std::vector<double> seconds;
};

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  const auto t0 = std::chrono::steady_clock::now();
  autan::Mesh mesh = autan::io::read_mesh(o.mesh);
  mesh = autan::renumber_cells(mesh, autan::rcm_order(mesh));
  const auto geometry = autan::compute_geometry(mesh);
  const auto t1 = std::chrono::steady_clock::now();
  const auto levels = autan::solvers::agglomerate(mesh, geometry);
  const auto t2 = std::chrono::steady_clock::now();
  if (levels.size() < 2) fail("the mesh has a single level");
  const autan::solvers::Level& f = levels[0];
  const std::size_t n = f.n_nodes, nc = levels[1].n_nodes;

  // Host references, in the host multigrid's order.
  const auto r = values(5 * n, 0.1), xc = values(5 * nc, 0.7), x0 = values(5 * n, 1.3);
  const double alpha = 0.75;
  std::vector<double> bc(5 * nc, 0.0), e(5 * n), x = x0;
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t i = 0; i < n; ++i) {
      const auto I = static_cast<std::size_t>(f.coarse_node[i]);
      bc[c * nc + I] += r[c * n + i];
      e[c * n + i] = xc[c * nc + I];
    }
  }
  for (std::size_t k = 0; k < x.size(); ++k) {
    const volatile double product = alpha * e[k];
    x[k] = x[k] + product;
  }

  wait_for_go(o.gate);
  const autan::solvers::gpu::Transfers transfers(levels);
  const Device dr(r), dxc(xc), dx(x0), dalpha(std::vector<double>{alpha});
  const Device dbc(std::vector<double>(5 * nc)), de(std::vector<double>(5 * n));
  transfers.restrict_residual(0, dr.get(), dbc.get());
  transfers.prolong(0, dxc.get(), de.get());
  transfers.correct(0, dalpha.get(), dxc.get(), dx.get());
  check(cudaDeviceSynchronize(), "verify");
  const bool verified =
      same_bits(dbc.download(), bc) && same_bits(de.download(), e) && same_bits(dx.download(), x);

  const auto nf = static_cast<double>(n), ncd = static_cast<double>(nc);
  std::vector<Timed> timed;
  // r and b_c once, the children list and its offsets once.
  timed.push_back({"restrict", 40.0 * nf + 40.0 * ncd + 4.0 * nf + 8.0 * (ncd + 1.0),
                   time_it(o, [&] { transfers.restrict_residual(0, dr.get(), dbc.get()); })});
  timed.push_back({"prolong", 40.0 * ncd + 4.0 * nf + 40.0 * nf,
                   time_it(o, [&] { transfers.prolong(0, dxc.get(), de.get()); })});
  timed.push_back({"correct", 40.0 * ncd + 4.0 * nf + 80.0 * nf + 8.0,
                   time_it(o, [&] { transfers.correct(0, dalpha.get(), dxc.get(), dx.get()); })});

  int device = 0;
  cudaDeviceProp prop{};
  check(cudaGetDevice(&device), "device");
  check(cudaGetDeviceProperties(&prop, device), "properties");
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
  std::printf(
      "\"config\":{\"mesh_file\":\"%s\",\"mesh_sha256\":\"%s\",\"order\":\"rcm\",\"n_cells\":%zu,"
      "\"n_coarse\":%zu,\"levels\":%zu,\"warmup\":%d,\"reps\":%d,\"batch\":%d},",
      std::filesystem::path(o.mesh).filename().string().c_str(),
      autan::io::mesh_hash(o.mesh).c_str(), n, nc, levels.size(), o.warmup, o.reps, o.batch);
  std::printf("\"setup_seconds\":{\"mesh\":%.3f,\"agglomerate\":%.3f},",
              std::chrono::duration<double>(t1 - t0).count(),
              std::chrono::duration<double>(t2 - t1).count());
  std::printf("\"verified\":%s,\"kernels\":[", verified ? "true" : "false");
  for (std::size_t k = 0; k < timed.size(); ++k) {
    const Timed& t = timed[k];
    std::printf("%s{\"name\":\"%s\",\"bytes_per_rep\":%.0f,\"cells\":%zu,\"seconds\":[",
                k ? "," : "", t.name.c_str(), t.bytes, n);
    for (std::size_t i = 0; i < t.seconds.size(); ++i)
      std::printf("%s%.9e", i ? "," : "", t.seconds[i]);
    std::printf("]}");
  }
  std::printf("]}\n");
  return verified ? 0 : 1;
}
