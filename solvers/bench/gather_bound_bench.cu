// #406: what the multigrid restriction's access pattern costs on its own. On one mesh (read in
// its own cell order) and its agglomeration, four kernels move the same bytes, 5 FP64 components
// per list entry (read 40 B through an index, write 40 B contiguously, read a 4 B index):
//   gather_restrict  the children list in the device numbering of the transfers (restriction's
//                    reads, entry by entry);
//   gather_identity  the same kernel over the identity list;
//   gather_random    the same kernel over a fixed random permutation (worst case, for context);
//   copy             contiguous reads and writes of the same arrays, the index read but unused.
// Each round times every kernel once, the starting kernel rotating from round to round. Every
// kernel's output is checked bitwise against a host gather before timing.
//
//   autan_gather_bound_bench MESH [--warmup N] [--reps N] [--batch N] [--gate 1]
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "autan/core/geometry.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/solvers/agglomeration.hpp"
#include "revision.h"

namespace {

struct Options {
  std::string mesh;
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
  if (argc < 2) fail("usage: autan_gather_bound_bench MESH [options]");
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

constexpr int components = 5;
constexpr int block = 256;

__global__ void gather(std::int64_t n, std::int64_t m, const std::int32_t* index, const double* in,
                       double* out) {
  const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= m) return;
  const std::int64_t i = index[k];
  for (int c = 0; c < components; ++c) out[c * m + k] = in[c * n + i];
}

// Same bytes as gather: the index is read (and kept live) but the reads are contiguous.
__global__ void copy(std::int64_t m, const std::int32_t* index, const double* in, double* out) {
  const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= m) return;
  const double bias = index[k] < 0 ? 1.0 : 0.0;
  for (int c = 0; c < components; ++c) out[c * m + k] = in[c * m + k] + bias;
}

template <class T>
T* upload(const std::vector<T>& v) {
  T* p = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&p), v.size() * sizeof(T)), "cudaMalloc");
  check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
  return p;
}

std::vector<double> download(const double* p, std::size_t n) {
  std::vector<double> v(n);
  check(cudaMemcpy(v.data(), p, n * sizeof(double), cudaMemcpyDeviceToHost), "download");
  return v;
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t k = 0; k < a.size(); ++k) {
    if (std::bit_cast<std::uint64_t>(a[k]) != std::bit_cast<std::uint64_t>(b[k])) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  const autan::Mesh mesh = autan::io::read_mesh(o.mesh);
  const auto levels = autan::solvers::agglomerate(mesh, autan::compute_geometry(mesh));
  if (levels.size() < 2) fail("the mesh has a single level");
  const autan::solvers::Level& f = levels[0];
  const std::size_t n = f.n_nodes, nc = levels[1].n_nodes;

  // Restriction's list: coarse nodes in the device numbering, each one's children in increasing
  // fine index (solvers/src/multigrid_gpu.cu, Transfers).
  // Device numbering of level 1 (least child first), as Transfers computes it.
  std::vector<std::size_t> least(nc, n);
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t& l = least[static_cast<std::size_t>(f.coarse_node[i])];
    l = std::min(l, i);
  }
  std::vector<std::int32_t> by_least(nc);
  std::iota(by_least.begin(), by_least.end(), 0);
  std::sort(by_least.begin(), by_least.end(), [&](std::int32_t a, std::int32_t b) {
    return least[static_cast<std::size_t>(a)] < least[static_cast<std::size_t>(b)];
  });
  std::vector<std::int32_t> rank(nc);
  for (std::size_t J = 0; J < nc; ++J)
    rank[static_cast<std::size_t>(by_least[J])] = static_cast<std::int32_t>(J);
  std::vector<std::int64_t> start(nc + 1, 0);
  for (std::size_t i = 0; i < n; ++i)
    ++start[static_cast<std::size_t>(rank[static_cast<std::size_t>(f.coarse_node[i])]) + 1];
  for (std::size_t J = 0; J < nc; ++J) start[J + 1] += start[J];
  std::vector<std::int32_t> restrict_list(n), identity(n), random_list(n);
  std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
  for (std::size_t i = 0; i < n; ++i) {
    restrict_list[static_cast<std::size_t>(
        fill[static_cast<std::size_t>(rank[static_cast<std::size_t>(f.coarse_node[i])])]++)] =
        static_cast<std::int32_t>(i);
  }
  std::iota(identity.begin(), identity.end(), 0);
  random_list = identity;
  std::shuffle(random_list.begin(), random_list.end(), std::mt19937(406));

  std::vector<double> in(components * n);
  for (std::size_t k = 0; k < in.size(); ++k) in[k] = std::sin(0.37 * static_cast<double>(k));
  auto host_gather = [&](const std::vector<std::int32_t>& index) {
    std::vector<double> out(components * n);
    for (std::size_t c = 0; c < components; ++c) {
      for (std::size_t k = 0; k < n; ++k)
        out[c * n + k] = in[c * n + static_cast<std::size_t>(index[k])];
    }
    return out;
  };
  const auto expected_restrict = host_gather(restrict_list);
  const auto expected_identity = host_gather(identity);
  const auto expected_random = host_gather(random_list);

  wait_for_go(o.gate);
  const auto m = static_cast<std::int64_t>(n), nn = static_cast<std::int64_t>(n);
  std::int32_t* d_restrict = upload(restrict_list);
  std::int32_t* d_identity = upload(identity);
  std::int32_t* d_random = upload(random_list);
  double* d_in = upload(in);
  double* d_out = upload(std::vector<double>(components * n, 0.0));
  const auto grid = static_cast<unsigned int>((m + block - 1) / block);

  struct Kernel {
    const char* name;
    std::int32_t* index;
    const std::vector<double>* expected;
    bool is_copy;
    std::vector<double> seconds;
  };
  std::vector<Kernel> kernels = {{"gather_restrict", d_restrict, &expected_restrict, false, {}},
                                 {"gather_identity", d_identity, &expected_identity, false, {}},
                                 {"gather_random", d_random, &expected_random, false, {}},
                                 {"copy", d_identity, &in, true, {}}};
  auto launch = [&](const Kernel& k) {
    if (k.is_copy)
      copy<<<grid, block>>>(m, k.index, d_in, d_out);
    else
      gather<<<grid, block>>>(nn, m, k.index, d_in, d_out);
  };

  bool verified = true;
  for (const Kernel& k : kernels) {
    launch(k);
    check(cudaDeviceSynchronize(), "verify");
    verified = same_bits(download(d_out, components * n), *k.expected) && verified;
  }

  cudaEvent_t t0, t1;
  check(cudaEventCreate(&t0), "event");
  check(cudaEventCreate(&t1), "event");
  for (int round = 0; round < o.warmup + o.reps; ++round) {
    for (std::size_t s = 0; s < kernels.size(); ++s) {
      Kernel& k = kernels[(s + static_cast<std::size_t>(round)) % kernels.size()];
      check(cudaEventRecord(t0), "record");
      for (int b = 0; b < o.batch; ++b) launch(k);
      check(cudaEventRecord(t1), "record");
      check(cudaEventSynchronize(t1), "sync");
      float ms = 0.0f;
      check(cudaEventElapsedTime(&ms, t0, t1), "elapsed");
      if (round >= o.warmup) k.seconds.push_back(ms * 1e-3 / o.batch);
    }
  }

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
      "\"config\":{\"mesh_file\":\"%s\",\"mesh_sha256\":\"%s\",\"order\":\"input\",\"n_cells\":%zu,"
      "\"n_coarse\":%zu,\"interleaved\":true,\"warmup\":%d,\"reps\":%d,\"batch\":%d},",
      std::filesystem::path(o.mesh).filename().string().c_str(),
      autan::io::mesh_hash(o.mesh).c_str(), n, nc, o.warmup, o.reps, o.batch);
  std::printf("\"verified\":%s,\"kernels\":[", verified ? "true" : "false");
  const double bytes = static_cast<double>(n) * (2.0 * 8.0 * components + 4.0);
  for (std::size_t q = 0; q < kernels.size(); ++q) {
    std::printf("%s{\"name\":\"%s\",\"bytes_per_rep\":%.0f,\"cells\":%zu,\"seconds\":[",
                q ? "," : "", kernels[q].name, bytes, n);
    for (std::size_t i = 0; i < kernels[q].seconds.size(); ++i)
      std::printf("%s%.9e", i ? "," : "", kernels[q].seconds[i]);
    std::printf("]}");
  }
  std::printf("]}\n");
  return verified ? 0 : 1;
}
