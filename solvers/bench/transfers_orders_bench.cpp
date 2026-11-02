// #342 study: the multigrid transfers on several cell orders of one mesh, interleaved in one
// GPU hold. Each MESH is read in its own cell order (no renumbering) and agglomerated before
// `go`; each round times every kernel of every order, the order that starts the round rotating
// from round to round. Kernel names carry the order's tag, the file name's last "-" field
// ("tet-n096-hilbert.msh" -> "restrict_hilbert"). Results are checked bitwise against the host
// definition before timing, as autan_mg_transfers_bench does.
//
//   autan_mg_transfers_orders_bench MESH... [--warmup N] [--reps N] [--batch N] [--gate 1]
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "autan/core/geometry.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/solvers/agglomeration.hpp"
#include "autan/solvers/multigrid_gpu.hpp"
#include "revision.h"

namespace {

struct Options {
  std::vector<std::string> meshes;
  bool gate = false;
  int warmup = 5;
  int reps = 30;
  int batch = 20;
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
  int i = 1;
  for (; i < argc && std::string_view(argv[i]).substr(0, 2) != "--"; ++i)
    o.meshes.emplace_back(argv[i]);
  if (o.meshes.empty()) fail("usage: autan_mg_transfers_orders_bench MESH... [options]");
  for (; i + 1 < argc; i += 2) {
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

std::string tag_of(const std::string& path) {
  std::string stem = std::filesystem::path(path).stem().string();
  const auto dash = stem.rfind('-');
  std::string tag = dash == std::string::npos ? stem : stem.substr(dash + 1);
  for (const char ch : tag) {
    if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_')) {
      fail("order tag '" + tag + "' must be [a-z0-9_]");
    }
  }
  return tag;
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

// One cell order of the mesh: its hierarchy, host references, and (after go) device data.
struct Case {
  std::string path, tag;
  std::vector<autan::solvers::Level> levels;
  std::size_t n = 0, nc = 0;
  std::vector<double> r, xc, x0, bc, e, x;
  double alpha = 0.75;
  std::unique_ptr<autan::solvers::gpu::Transfers> transfers;
  std::unique_ptr<Device> dr, dxc, dx, dalpha, dbc, de;
  std::vector<double> seconds[3];
};

void prepare(Case& k) {
  const autan::Mesh mesh = autan::io::read_mesh(k.path);
  k.levels = autan::solvers::agglomerate(mesh, autan::compute_geometry(mesh));
  if (k.levels.size() < 2) fail(k.path + ": a single level");
  const autan::solvers::Level& f = k.levels[0];
  k.n = f.n_nodes;
  k.nc = k.levels[1].n_nodes;
  k.r = values(5 * k.n, 0.1);
  k.xc = values(5 * k.nc, 0.7);
  k.x0 = values(5 * k.n, 1.3);
  k.bc.assign(5 * k.nc, 0.0);
  k.e.resize(5 * k.n);
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t i = 0; i < k.n; ++i) {
      const auto I = static_cast<std::size_t>(f.coarse_node[i]);
      k.bc[c * k.nc + I] += k.r[c * k.n + i];
      k.e[c * k.n + i] = k.xc[c * k.nc + I];
    }
  }
  k.x = k.x0;
  for (std::size_t q = 0; q < k.x.size(); ++q) {
    const volatile double product = k.alpha * k.e[q];
    k.x[q] = k.x[q] + product;
  }
}

std::vector<double> permuted(const Case& k, const std::vector<double>& v, bool to_device) {
  const auto coarse = k.transfers->order(1);
  std::vector<double> d(v.size());
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t j = 0; j < k.nc; ++j) {
      const std::size_t h = c * k.nc + static_cast<std::size_t>(coarse[j]);
      if (to_device)
        d[c * k.nc + j] = v[h];
      else
        d[h] = v[c * k.nc + j];
    }
  }
  return d;
}

bool upload_and_verify(Case& k) {
  k.transfers = std::make_unique<autan::solvers::gpu::Transfers>(k.levels);
  k.dr = std::make_unique<Device>(k.r);
  k.dxc = std::make_unique<Device>(permuted(k, k.xc, true));
  k.dx = std::make_unique<Device>(k.x0);
  k.dalpha = std::make_unique<Device>(std::vector<double>{k.alpha});
  k.dbc = std::make_unique<Device>(std::vector<double>(5 * k.nc));
  k.de = std::make_unique<Device>(std::vector<double>(5 * k.n));
  k.transfers->restrict_residual(0, k.dr->get(), k.dbc->get());
  k.transfers->prolong(0, k.dxc->get(), k.de->get());
  k.transfers->correct(0, k.dalpha->get(), k.dxc->get(), k.dx->get());
  check(cudaDeviceSynchronize(), "verify");
  return same_bits(permuted(k, k.dbc->download(), false), k.bc) &&
         same_bits(k.de->download(), k.e) && same_bits(k.dx->download(), k.x);
}

void launch(Case& k, int kernel) {
  if (kernel == 0)
    k.transfers->restrict_residual(0, k.dr->get(), k.dbc->get());
  else if (kernel == 1)
    k.transfers->prolong(0, k.dxc->get(), k.de->get());
  else
    k.transfers->correct(0, k.dalpha->get(), k.dxc->get(), k.dx->get());
}

const char* const kernel_names[3] = {"restrict", "prolong", "correct"};

double bytes(const Case& k, int kernel) {
  const auto nf = static_cast<double>(k.n), nc = static_cast<double>(k.nc);
  if (kernel == 0) return 40.0 * nf + 40.0 * nc + 4.0 * nf + 8.0 * (nc + 1.0);
  if (kernel == 1) return 40.0 * nc + 4.0 * nf + 40.0 * nf;
  return 40.0 * nc + 4.0 * nf + 80.0 * nf + 8.0;
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  std::vector<Case> cases(o.meshes.size());
  for (std::size_t m = 0; m < cases.size(); ++m) {
    cases[m].path = o.meshes[m];
    cases[m].tag = tag_of(o.meshes[m]);
    prepare(cases[m]);
  }

  wait_for_go(o.gate);
  bool verified = true;
  for (Case& k : cases) verified = upload_and_verify(k) && verified;

  // Rounds: every order and kernel once per round; the order that starts rotates each round.
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "event");
  check(cudaEventCreate(&stop), "event");
  const std::size_t m = cases.size();
  for (int round = 0; round < o.warmup + o.reps; ++round) {
    for (std::size_t s = 0; s < m; ++s) {
      Case& k = cases[(s + static_cast<std::size_t>(round)) % m];
      for (int kernel = 0; kernel < 3; ++kernel) {
        check(cudaEventRecord(start), "record");
        for (int b = 0; b < o.batch; ++b) launch(k, kernel);
        check(cudaEventRecord(stop), "record");
        check(cudaEventSynchronize(stop), "sync");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        if (round >= o.warmup) k.seconds[kernel].push_back(ms * 1e-3 / o.batch);
      }
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

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
      "\"config\":{\"interleaved\":true,\"warmup\":%d,\"reps\":%d,\"batch\":%d,\"meshes\":[",
      o.warmup, o.reps, o.batch);
  for (std::size_t c = 0; c < m; ++c) {
    std::printf(
        "%s{\"tag\":\"%s\",\"file\":\"%s\",\"sha256\":\"%s\",\"n_cells\":%zu,\"n_coarse\":%zu}",
        c ? "," : "", cases[c].tag.c_str(),
        std::filesystem::path(cases[c].path).filename().string().c_str(),
        autan::io::mesh_hash(cases[c].path).c_str(), cases[c].n, cases[c].nc);
  }
  std::printf("]},\"verified\":%s,\"kernels\":[", verified ? "true" : "false");
  bool first = true;
  for (const Case& k : cases) {
    for (int kernel = 0; kernel < 3; ++kernel) {
      std::printf("%s{\"name\":\"%s_%s\",\"bytes_per_rep\":%.0f,\"cells\":%zu,\"seconds\":[",
                  first ? "" : ",", kernel_names[kernel], k.tag.c_str(), bytes(k, kernel), k.n);
      first = false;
      for (std::size_t i = 0; i < k.seconds[kernel].size(); ++i)
        std::printf("%s%.9e", i ? "," : "", k.seconds[kernel][i]);
      std::printf("]}");
    }
  }
  std::printf("]}\n");
  return verified ? 0 : 1;
}
