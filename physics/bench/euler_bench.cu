// Times the full Euler residual (kernels/euler.py: primitives, boundary states, face2 LSQ
// gradients with the limiter's bounds, limiter, sources, face and boundary fluxes) on one mesh and
// prints the raw JSON autan-bench reads. bytes_per_rep is the sum over the loops of each loop's
// analytic minimum (every array it touches read or written once); the result is checked against the
// host residual (autan::physics::Euler) before timing.
//
//   euler_bench MESH [--policy fp64|fp32+acc64] [--flux roe|hllc|ausm-up] [--block N]
//                    [--warmup N] [--reps N] [--batch N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/renumber.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/colouring.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/kernels/gpu.hpp"
#include "autan/physics/euler.hpp"
#include "euler_gpu.cuh"
#include "revision.h"

namespace {

using autan::kernels::index_t;
using autan::kernels::offset_t;
namespace gpu = autan::kernels::gpu;
namespace ek = autan::kernels::euler;

struct Options {
  std::string mesh;
  std::string policy = "fp32+acc64";
  std::string flux = "roe";
  int block = 256;
  int warmup = 5;
  int reps = 30;
  int batch = 5;
};

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(2);
}

Options parse(int argc, char** argv) {
  Options o;
  if (argc < 2) fail("usage: euler_bench MESH [options]");
  o.mesh = argv[1];
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string_view f(argv[i]);
    const std::string v(argv[i + 1]);
    if (f == "--policy")
      o.policy = v;
    else if (f == "--flux")
      o.flux = v;
    else if (f == "--block")
      o.block = std::stoi(v);
    else if (f == "--warmup")
      o.warmup = std::stoi(v);
    else if (f == "--reps")
      o.reps = std::stoi(v);
    else if (f == "--batch")
      o.batch = std::stoi(v);
    else
      fail("unknown flag " + std::string(f));
  }
  return o;
}

template <class T>
class DeviceArray {
 public:
  explicit DeviceArray(std::size_t n) : n_(n) {
    gpu::check(cudaMalloc(&p_, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    gpu::check(cudaMemset(p_, 0, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMemset");
  }
  explicit DeviceArray(const std::vector<T>& v) : DeviceArray(v.size()) {
    gpu::check(cudaMemcpy(p_, v.data(), n_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray() { cudaFree(p_); }
  [[nodiscard]] std::vector<T> download() const {
    std::vector<T> v(n_);
    gpu::check(cudaMemcpy(v.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
  }
  [[nodiscard]] T* get() const { return p_; }

 private:
  T* p_ = nullptr;
  std::size_t n_;
};

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

struct Timed {
  std::string name, strategy;
  double bytes;
  std::vector<double> seconds;
};

struct Run {
  std::vector<Timed> timed;
  bool verified = true;
  double difference = 0.0;
  double tolerance = 0.0;
};

template <class P>
Run run(const Options& o, const autan::Mesh& mesh, const autan::Geometry& geometry) {
  using S = typename P::state;
  using Gr = typename P::grad;
  using R = typename P::residual;
  using G = typename P::geom;
  const auto packed = autan::kernels::pack_geometry(mesh, geometry);
  auto topo = autan::kernels::topology(mesh.view());
  const auto stencil =
      autan::kernels::cell_stencil(mesh, packed, autan::kernels::StencilKind::face2);
  const auto nc = static_cast<std::size_t>(topo.n_cells);
  const auto nf = static_cast<std::size_t>(topo.n_faces);
  const auto ni = static_cast<std::size_t>(topo.n_interior_faces);
  const auto nb = nf - ni;

  // A uniform flow at Mach 0.5 with a smooth perturbation, farfield on every boundary face.
  const double gamma = 1.4, rho = 1.2, p = 1e5;
  const double a = std::sqrt(gamma * p / rho);
  const std::array<double, 3> v = {0.5 * a * 0.8, 0.5 * a * 0.36, 0.5 * a * 0.48};
  std::vector<double> u(5 * nc);
  for (std::size_t c = 0; c < nc; ++c) {
    const double x = geometry.cell_centroid_x[c], y = geometry.cell_centroid_y[c],
                 z = geometry.cell_centroid_z[c];
    const double s = 1.0 + 0.1 * std::sin(3.0 * x + 2.0 * y) * std::cos(4.0 * z);
    const double r = rho * s, pr = p * s * s;
    u[c] = r;
    for (std::size_t k = 0; k < 3; ++k) u[(1 + k) * nc + c] = r * v[k];
    u[4 * nc + c] = pr / (gamma - 1.0) + 0.5 * r * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }
  std::vector<autan::physics::BoundaryKind> kind(nb, autan::physics::BoundaryKind::farfield);
  std::vector<double> data(5 * nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::array<double, 5> w = {rho, v[0], v[1], v[2], p};
    for (std::size_t k = 0; k < 5; ++k) data[k * nb + b] = w[k];
  }
  autan::physics::Scheme scheme;
  scheme.flux = o.flux == "hllc"      ? autan::physics::Flux::hllc
                : o.flux == "ausm-up" ? autan::physics::Flux::ausm_up
                                      : autan::physics::Flux::roe;
  scheme.mach_ref = 0.5;
  scheme.scale = {rho, a, a, a, p};
  autan::physics::Euler<P> host(mesh, geometry, {gamma, 287.0}, scheme);
  host.set_boundary(kind, data);
  std::vector<S> us(u.begin(), u.end());
  std::vector<R> reference(5 * nc);
  host.residual(us, reference);

  // Device copies of every dataset of the module.
  DeviceArray<index_t> owner(std::vector<index_t>(topo.owner, topo.owner + nf));
  DeviceArray<index_t> neighbour(std::vector<index_t>(topo.neighbour, topo.neighbour + ni));
  DeviceArray<offset_t> offsets(
      std::vector<offset_t>(topo.cell_face_offsets, topo.cell_face_offsets + nc + 1));
  DeviceArray<index_t> cell_faces(
      std::vector<index_t>(topo.cell_faces, topo.cell_faces + topo.n_incidences));
  DeviceArray<offset_t> st_offsets(stencil.offsets);
  DeviceArray<index_t> st_entries(stencil.entries);
  DeviceArray<offset_t> st_twin(stencil.twin);
  const auto interior = autan::kernels::colour_faces(topo, 0, topo.n_interior_faces);
  const auto boundary = autan::kernels::colour_faces(topo, topo.n_interior_faces, topo.n_faces);
  DeviceArray<index_t> int_faces(interior.faces), bnd_faces(boundary.faces);
  constexpr int scratch_width = 5;
  DeviceArray<double> scratch(scratch_width * nf);
  auto conv = [](const std::vector<double>& h) {
    return DeviceArray<G>(std::vector<G>(h.begin(), h.end()));
  };
  DeviceArray<G> face_normal = conv(packed.face_normal),
                 face_area = conv(packed.face_area_magnitude),
                 face_centroid = conv(packed.face_centroid), face_moment = conv(packed.face_moment),
                 cell_centroid = conv(packed.cell_centroid), cell_volume = conv(packed.cell_volume),
                 st_weight = conv(stencil.lsq_weight);
  std::vector<double> bc_kind(nf, 0.0);
  std::vector<S> bc(5 * nf, S(0));
  for (std::size_t b = 0; b < nb; ++b) {
    bc_kind[ni + b] = static_cast<double>(kind[b]);
    for (std::size_t k = 0; k < 5; ++k) bc[k * nf + ni + b] = S(data[k * nb + b]);
  }
  DeviceArray<double> d_kind(bc_kind);
  DeviceArray<S> d_bc(bc), d_u(us), d_prim(5 * nc), d_prim_b(5 * nf), d_wmin(5 * nc),
      d_wmax(5 * nc), d_phi(5 * nc), d_source(5 * nc);
  DeviceArray<Gr> d_grad(15 * nc);
  DeviceArray<R> d_res(5 * nc);

  topo.owner = owner.get();
  topo.neighbour = neighbour.get();
  topo.cell_face_offsets = offsets.get();
  topo.cell_faces = cell_faces.get();
  topo.n_stencil = static_cast<offset_t>(stencil.entries.size());
  topo.stencil_offsets = st_offsets.get();
  topo.stencil = st_entries.get();
  topo.stencil_twin = st_twin.get();
  ek::Data<P> d{};
  d.topo = topo;
  d.u = d_u.get();
  d.prim = d_prim.get();
  d.prim_b = d_prim_b.get();
  d.grad = d_grad.get();
  d.wmin = d_wmin.get();
  d.wmax = d_wmax.get();
  d.phi = d_phi.get();
  d.source = d_source.get();
  d.res = d_res.get();
  d.bc_kind = d_kind.get();
  d.bc = d_bc.get();
  d.face_normal = face_normal.get();
  d.face_area_magnitude = face_area.get();
  d.face_centroid = face_centroid.get();
  d.face_moment = face_moment.get();
  d.cell_centroid = cell_centroid.get();
  d.cell_volume = cell_volume.get();
  d.stencil_lsq_weight = st_weight.get();
  d.gamma[0] = typename P::param(gamma);
  d.gas_constant[0] = typename P::param(287.0);
  d.mach_ref[0] = typename P::param(0.5);
  const double k3 = std::pow(scheme.venkat_k / scheme.length, 3);
  for (std::size_t c = 0; c < 5; ++c)
    d.limiter_scale[c] = typename P::param(k3 * scheme.scale[c] * scheme.scale[c]);
  gpu::Schedule sched;
  sched.block = o.block;
  sched.interior = {int_faces.get(), interior.offsets};
  sched.boundary = {bnd_faces.get(), boundary.offsets};
  sched.face_scratch = scratch.get();
  sched.face_scratch_width = scratch_width;

  // Analytic minimum of each loop: every array it touches read or written once.
  const double cs = static_cast<double>(nc), fs = static_cast<double>(nf),
               is = static_cast<double>(ni), bs = static_cast<double>(nb),
               st = static_cast<double>(stencil.entries.size());
  const double sS = sizeof(S), sG = sizeof(G), sGr = sizeof(Gr), sR = sizeof(R),
               sI = sizeof(index_t), sO = sizeof(offset_t), sK = sizeof(double);
  const double gather = cs * 5 * sS + bs * 5 * sS + st * sI + cs * sO;  // values over the stencil
  std::vector<index_t> owners(mesh.owner().begin() + static_cast<std::ptrdiff_t>(ni),
                              mesh.owner().end());
  std::sort(owners.begin(), owners.end());
  const double bo = static_cast<double>(std::unique(owners.begin(), owners.end()) - owners.begin());
  struct Loop {
    const char* name;
    double bytes;
    std::function<void(gpu::Strategy)> launch;
  };
  const std::vector<Loop> loops = {
      {"primitives", cs * 10 * sS, [&](gpu::Strategy) { ek::gpu_loops::primitives(d, sched); }},
      {"boundary_states", bs * (sI + sK + 5 * sS + 3 * sG + 5 * sS) + bo * 5 * sS,
       [&](gpu::Strategy) { ek::gpu_loops::boundary_states(d, sched); }},
      {"gradients", gather + st * 3 * sG + cs * (15 * sGr + 10 * sS),
       [&](gpu::Strategy) { ek::gpu_loops::gradients(d, sched); }},
      {"limiter",
       cs * (5 * sS + 15 * sGr + 10 * sS + sG + 3 * sG + 5 * sS + sO) +
           static_cast<double>(topo.n_incidences) * sI + fs * 3 * sG,
       [&](gpu::Strategy) { ek::gpu_loops::limiter(d, sched); }},
      {"sources", cs * (5 * sS + sG + 5 * sR),
       [&](gpu::Strategy) { ek::gpu_loops::sources(d, sched); }},
      {"faces", cs * (25 * sGr + 3 * sG + 10 * sR) + is * (2 * sI + 16 * sG),
       [&](gpu::Strategy s) {
         if (scheme.flux == autan::physics::Flux::hllc) {
           ek::gpu_loops::hllc_faces(d, s, sched);
         } else if (scheme.flux == autan::physics::Flux::ausm_up) {
           ek::gpu_loops::ausm_faces(d, s, sched);
         } else {
           ek::gpu_loops::roe_faces(d, s, sched);
         }
       }},
      {"boundaries", bs * (sI + sK + 5 * sS + 16 * sG) + bo * (25 * sGr + 3 * sG + 10 * sR),
       [&](gpu::Strategy s) {
         if (scheme.flux == autan::physics::Flux::hllc) {
           ek::gpu_loops::hllc_boundaries(d, s, sched);
         } else if (scheme.flux == autan::physics::Flux::ausm_up) {
           ek::gpu_loops::ausm_boundaries(d, s, sched);
         } else {
           ek::gpu_loops::roe_boundaries(d, s, sched);
         }
       }},
  };
  double bytes = 0.0;
  for (const auto& l : loops) bytes += l.bytes;
  auto residual = [&](gpu::Strategy s) {
    for (const auto& l : loops) l.launch(s);
  };

  Run out;
  for (auto s : {gpu::Strategy::atomic, gpu::Strategy::warp_aggregated, gpu::Strategy::store_gather,
                 gpu::Strategy::gather_recompute}) {
    residual(s);
    gpu::check(cudaDeviceSynchronize(), "residual");
    const auto r = d_res.download();
    // Same loops on host and device: only the order of floating-point operations differs. Each
    // equation against its own scale: allow 1024 eps of the compute type.
    out.tolerance = 1024.0 * std::numeric_limits<typename P::compute>::epsilon();
    double diff = 0.0;
    for (std::size_t k = 0; k < 5; ++k) {
      double num = 0.0, den = 0.0;
      for (std::size_t c = 0; c < nc; ++c) {
        const double a1 = static_cast<double>(r[k * nc + c]);
        const double b1 = static_cast<double>(reference[k * nc + c]);
        if (!std::isfinite(a1)) num = std::numeric_limits<double>::infinity();
        num = std::max(num, std::abs(a1 - b1));
        den = std::max(den, std::abs(b1));
      }
      diff = std::max(diff, num / den);
    }
    out.difference = std::max(out.difference, diff);
    out.verified = out.verified && diff <= out.tolerance;
    std::string tag = gpu::name(s);
    std::replace(tag.begin(), tag.end(), '-', '_');
    out.timed.push_back(
        {"residual_" + o.flux + "_" + tag, gpu::name(s), bytes, time_it(o, [&] { residual(s); })});
    for (const auto& l : loops) {
      const bool by_strategy =
          std::string_view(l.name) == "faces" || std::string_view(l.name) == "boundaries";
      if (!by_strategy && s != gpu::Strategy::atomic) continue;
      out.timed.push_back({std::string(l.name) + (by_strategy ? "_" + o.flux + "_" + tag : ""),
                           gpu::name(s), l.bytes, time_it(o, [&] { l.launch(s); })});
    }
  }
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
      "\"order\":\"rcm\",\"gradient_stencil\":\"face2\",\"limiter\":true,\"n_cells\":%d,"
      "\"n_faces\":%d,\"block\":%d,\"warmup\":%d,\"reps\":%d,\"batch\":%d,\"tolerance\":%.1e},",
      std::filesystem::path(o.mesh).filename().string().c_str(), sha.c_str(), o.policy.c_str(),
      o.flux.c_str(), mesh.n_cells(), mesh.n_faces(), o.block, o.warmup, o.reps, o.batch,
      r.tolerance);
  std::printf("\"setup_seconds\":{\"read\":%.3f,\"renumber\":%.3f},",
              std::chrono::duration<double>(t1 - t0).count(),
              std::chrono::duration<double>(t2 - t1).count());
  std::printf("\"verified\":%s,\"kernels\":[", r.verified ? "true" : "false");
  for (std::size_t k = 0; k < r.timed.size(); ++k) {
    const auto& t = r.timed[k];
    std::printf(
        "%s{\"name\":\"%s\",\"bytes_per_rep\":%.0f,\"cells\":%d,\"params\":{\"strategy\":\"%s\","
        "\"policy\":\"%s\",\"flux\":\"%s\"},\"seconds\":[",
        k ? "," : "", t.name.c_str(), t.bytes, mesh.n_cells(), t.strategy.c_str(), o.policy.c_str(),
        o.flux.c_str());
    for (std::size_t i = 0; i < t.seconds.size(); ++i)
      std::printf("%s%.9e", i ? "," : "", t.seconds[i]);
    std::printf("]}");
  }
  std::printf("]}\n");
  std::fprintf(stderr, "max relative difference to the host residual %.3e (tolerance %.1e)\n",
               r.difference, r.tolerance);
  return r.verified ? 0 : 1;
}
