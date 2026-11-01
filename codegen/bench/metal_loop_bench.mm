// Times the generated Metal loops of the spike on one mesh and prints the raw JSON that
// autan-bench reads, as bench/loop_bench.cu does for CUDA (one entry per loop, strategy and
// policy; bytes_per_rep is the loop's analytic minimum). Strategies the policy lacks on Metal
// (atomics on a double-float accumulator) are not timed.
//
//   autan_metal_loop_bench MESH [--policy fp32|fp32+acc64] [--order input|rcm] [--block N]
//                          [--warmup N] [--reps N] [--batch N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "advection_cpu.hpp"
#include "advection_metal.hpp"
#include "autan/core/renumber.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/metal.hpp"
#include "gradients_cpu.hpp"
#include "gradients_metal.hpp"
#include "metal_bench.hpp"
#include "metal_device.hpp"
#include "revision.h"
#include "roe_metal.hpp"
#include "spike.hpp"

namespace {

using autan::kernels::index_t;
using autan::kernels::offset_t;
using autan::kernels::Strategy;
using autan::spike::GeomCopy;
using autan::spike::SpikeMesh;
using autan::spike::metal::DeviceGeom;
using autan::spike::metal::DeviceMesh;
namespace mtl = autan::kernels::metal;
template <class T>
using DeviceArray = autan::spike::metal::Array<T>;

struct Options {
  std::string mesh;
  std::string policy = "fp32+acc64";
  std::string order = "rcm";
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
  if (argc < 2) fail("usage: loop_bench MESH [options]");
  o.mesh = argv[1];
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string_view f(argv[i]);
    const std::string v(argv[i + 1]);
    if (f == "--policy")
      o.policy = v;
    else if (f == "--order")
      o.order = v;
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

struct Timed {
  std::string name;
  double bytes;
  std::string strategy;
  std::vector<double> seconds;
  double write_bytes = 0.0;  ///< the part of `bytes` written: weights the reference peak
};

/// GPU time of one command buffer holding `batch` launches, per launch.
template <class Launch>
std::vector<double> time_it(mtl::Context& ctx, const Options& o, Launch launch) {
  std::vector<double> out;
  for (int r = 0; r < o.warmup + o.reps; ++r) {
    for (int i = 0; i < o.batch; ++i) launch();
    const double t = ctx.synchronise();
    if (r >= o.warmup) out.push_back(t / o.batch);
  }
  return out;
}

double max_rel_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
      return std::numeric_limits<double>::infinity();
    }
    num = std::max(num, std::abs(a[i] - b[i]));
    den = std::max(den, std::abs(b[i]));
  }
  return den > 0.0 ? num / den : std::numeric_limits<double>::infinity();
}

template <class T>
std::vector<double> as_double(const std::vector<T>& v) {
  return std::vector<double>(v.begin(), v.end());
}

/// The strategies of `all` that policy P has on Metal.
template <class P>
std::vector<Strategy> strategies(std::initializer_list<Strategy> all) {
  std::vector<Strategy> out;
  for (auto s : all) {
    if (P::float_atomics || autan::kernels::deterministic(s)) out.push_back(s);
  }
  return out;
}

template <class P>
void run(mtl::Context& ctx, const Options& o, const SpikeMesh& m, std::vector<Timed>& out,
         bool& verified, double& tolerance) {
  using S = typename P::state;
  using Gr = typename P::grad;
  using R = typename P::residual;
  using G = typename P::geom;
  const auto& t = m.topo;
  const auto nc = static_cast<std::size_t>(t.n_cells);
  const auto nf = static_cast<std::size_t>(t.n_faces);
  const double ni = t.n_interior_faces, nb = t.n_faces - t.n_interior_faces;
  const double cells = t.n_cells, inc = static_cast<double>(t.n_incidences);
  std::mt19937_64 rng(42);
  std::normal_distribution<double> normal;
  std::vector<S> q(nc), qb(nf);
  std::vector<Gr> grad(3 * nc);
  for (auto& v : q) v = S(normal(rng));
  for (auto& v : qb) v = S(normal(rng));
  for (auto& v : grad) v = Gr(normal(rng));

  const auto colour_start = std::chrono::steady_clock::now();
  DeviceMesh<P> dm(ctx, m, o.block);
  const double colouring_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - colour_start).count();
  GeomCopy<P> hg(m.packed);
  DeviceGeom<P> g(ctx, hg);
  DeviceArray<S> dq(ctx, q), dqb(ctx, qb), dmin(ctx, nc), dmax(ctx, nc);
  DeviceArray<Gr> dgrad(ctx, grad);
  DeviceArray<R> dres(ctx, nc), dacc(ctx, 3 * nc);
  tolerance = std::is_same_v<typename P::compute, double> ? 1e-12 : 1e-4;

  namespace av = autan::kernels::advection;
  av::Data<P> a{};
  a.topo = dm.topo;
  a.q = dq.get();
  a.grad = dgrad.get();
  a.res = dres.get();
  a.face_r_left = g.face_r_left.get();
  a.face_r_right = g.face_r_right.get();
  a.face_normal = g.face_normal.get();
  a.face_area_magnitude = g.face_area_magnitude.get();
  a.velocity[0] = typename P::param(1.0);
  a.velocity[1] = typename P::param(0.7);
  a.velocity[2] = typename P::param(0.4);

  // Host reference of one residual evaluation, in the CPU policy Metal matches to rounding
  // (state and gradient are FP32 in both, so they are shared).
  using H = typename P::host;
  GeomCopy<H> hh(m.packed);
  av::Data<H> ah{};
  ah.topo = m.topo;
  std::vector<typename H::residual> hres(nc, typename H::residual(0));
  ah.q = q.data();
  ah.grad = grad.data();
  ah.res = hres.data();
  ah.face_r_left = hh.face_r_left.data();
  ah.face_r_right = hh.face_r_right.data();
  ah.face_normal = hh.face_normal.data();
  ah.face_area_magnitude = hh.face_area_magnitude.data();
  ah.velocity[0] = typename H::param(1.0);
  ah.velocity[1] = typename H::param(0.7);
  ah.velocity[2] = typename H::param(0.4);
  av::cpu::residual(ah);

  // Analytic minimum traffic: face geometry (r_left, r_right, n, |S|) and owner/neighbour once per
  // interior face; q and grad read once, res read and written once per cell.
  const double adv_bytes = ni * (10.0 * sizeof(G) + 2 * sizeof(index_t)) +
                           cells * (sizeof(S) + 3.0 * sizeof(Gr) + 2.0 * sizeof(R));
  for (auto s : strategies<P>({Strategy::atomic, Strategy::warp_aggregated, Strategy::coloured,
                               Strategy::store_gather, Strategy::gather_recompute})) {
    dres.zero();
    av::metal_loops::residual(ctx, a, s, dm.schedule);
    ctx.synchronise();
    if (max_rel_diff(as_double(dres.download()), as_double(hres)) > tolerance) verified = false;
    std::string kernel = std::string("advection_") + autan::kernels::name(s);
    std::replace(kernel.begin(), kernel.end(), '-', '_');
    out.push_back({kernel, adv_bytes, autan::kernels::name(s),
                   time_it(ctx, o, [&] { av::metal_loops::residual(ctx, a, s, dm.schedule); }),
                   cells * sizeof(R)});
  }

  namespace gr = autan::kernels::gradients;
  gr::Data<P> d{};
  d.topo = dm.topo;
  d.q = dq.get();
  d.q_b = dqb.get();
  d.grad = dgrad.get();
  d.qmin = dmin.get();
  d.qmax = dmax.get();
  d.gg_sum = dacc.get();
  d.lsq_weight = g.lsq_weight.get();
  d.cell_volume = g.cell_volume.get();
  d.cell_centroid = g.cell_centroid.get();
  d.face_centroid = g.face_centroid.get();
  d.face_area = g.face_area.get();
  d.face_moment = g.face_moment.get();
  gr::Data<H> dh{};
  dh.topo = m.topo;
  std::vector<Gr> hgrad(3 * nc);
  std::vector<S> hmin(nc), hmax(nc);
  dh.q = q.data();
  dh.q_b = qb.data();
  dh.grad = hgrad.data();
  dh.qmin = hmin.data();
  dh.qmax = hmax.data();
  dh.lsq_weight = hh.lsq_weight.data();
  gr::cpu::lsq_gradient_bounds(dh);
  gr::metal_loops::lsq_gradient_bounds(ctx, d, dm.schedule);
  ctx.synchronise();
  if (max_rel_diff(as_double(dgrad.download()), as_double(hgrad)) > tolerance) verified = false;
  // Analytic minimum: per incidence the face index and LSQ weight; per cell its CSR offset, q,
  // and the gradient and two bounds written; owner/neighbour per interior face; q_b per boundary
  // face.
  const double lsq_bytes = inc * (sizeof(index_t) + 3.0 * sizeof(G)) +
                           cells * (sizeof(offset_t) + 3.0 * sizeof(S) + 3.0 * sizeof(Gr)) +
                           ni * 2 * sizeof(index_t) + nb * sizeof(S);
  out.push_back(
      {"lsq_gradient_bounds", lsq_bytes, "cell",
       time_it(ctx, o, [&] { gr::metal_loops::lsq_gradient_bounds(ctx, d, dm.schedule); }),
       cells * (3.0 * sizeof(Gr) + 2.0 * sizeof(S))});

  // Roe flux, its tangent and its adjoint: the adjoint cost ratio of research section 8.
  namespace rf = autan::kernels::roe;
  std::vector<S> u(5 * nc);
  for (std::size_t c = 0; c < nc; ++c) {
    const double rho = 1.0 + 0.1 * std::abs(normal(rng));
    const double vx = 0.3 * normal(rng), vy = 0.3 * normal(rng), vz = 0.3 * normal(rng);
    u[c] = S(rho);
    u[nc + c] = S(rho * vx);
    u[2 * nc + c] = S(rho * vy);
    u[3 * nc + c] = S(rho * vz);
    u[4 * nc + c] = S(2.5 + 0.5 * rho * (vx * vx + vy * vy + vz * vz));
  }
  DeviceArray<S> du(ctx, u), du_tan(ctx, u);
  // A nonzero adjoint seed, so that the adjoint does its real work.
  std::vector<R> seed(5 * nc);
  for (auto& v : seed) v = R(normal(rng));
  DeviceArray<R> dr(ctx, 5 * nc), dr_tan(ctx, 5 * nc), dr_adj(ctx, seed), du_adj(ctx, 5 * nc);
  rf::Data<P> rd{};
  rd.topo = dm.topo;
  rd.u_cons = du.get();
  rd.res = dr.get();
  rd.face_normal = g.face_normal.get();
  rd.face_area_magnitude = g.face_area_magnitude.get();
  rd.u_cons_tan = du_tan.get();
  rd.res_tan = dr_tan.get();
  rd.res_adj = dr_adj.get();
  rd.u_cons_adj = du_adj.get();
  const double roe_bytes =
      ni * (4.0 * sizeof(G) + 2 * sizeof(index_t)) + cells * (5.0 * sizeof(S) + 10.0 * sizeof(R));
  for (auto s : strategies<P>({Strategy::atomic, Strategy::warp_aggregated, Strategy::store_gather,
                               Strategy::gather_recompute})) {
    std::string tag = autan::kernels::name(s);
    std::replace(tag.begin(), tag.end(), '-', '_');
    const char* sn = autan::kernels::name(s);
    // Each mode reads and writes its 5 increments per cell; the tangent also reads the state
    // tangent, the adjoint its 5 seeds per cell.
    const double written = cells * 5.0 * sizeof(R);
    out.push_back({"roe_" + tag, roe_bytes, sn,
                   time_it(ctx, o, [&] { rf::metal_loops::flux(ctx, rd, s, dm.schedule); }),
                   written});
    out.push_back({"roe_tangent_" + tag, roe_bytes + cells * 5.0 * sizeof(S), sn,
                   time_it(ctx, o, [&] { rf::metal_loops::flux_tangent(ctx, rd, s, dm.schedule); }),
                   written});
    out.push_back({"roe_adjoint_" + tag, roe_bytes + cells * 5.0 * sizeof(R), sn,
                   time_it(ctx, o, [&] { rf::metal_loops::flux_adjoint(ctx, rd, s, dm.schedule); }),
                   written});
  }
  std::fprintf(stderr, "colouring %.3f s (%zu interior colours, %zu boundary colours)\n",
               colouring_seconds, dm.int_colours.offsets.size() - 1,
               dm.bnd_colours.offsets.size() - 1);
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  const auto t0 = std::chrono::steady_clock::now();
  autan::Mesh mesh = autan::io::read_mesh(o.mesh);
  const auto t1 = std::chrono::steady_clock::now();
  double renumber_seconds = 0.0;
  if (o.order == "rcm") {
    mesh = autan::renumber_cells(mesh, autan::rcm_order(mesh));
    renumber_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  } else if (o.order != "input") {
    fail("--order is input or rcm");
  }
  SpikeMesh m{std::move(mesh), {}, {}, {}, {}, {}};
  m.geometry = autan::compute_geometry(m.mesh);
  m.packed = autan::kernels::pack_geometry(m.mesh, m.geometry);
  m.topo = autan::kernels::topology(m.mesh.view());
  m.twins = autan::kernels::incidence_twins(m.mesh);
  m.topo.incidence_twin = m.twins.data();
  const double read_seconds = std::chrono::duration<double>(t1 - t0).count();

  std::vector<Timed> timed;
  bool verified = true;
  double tolerance = 0.0;
  mtl::Context ctx;
  autan::spike::metal::with_policy(
      o.policy, [&](auto p) { run<decltype(p)>(ctx, o, m, timed, verified, tolerance); });

  std::printf("{");
  autan::spike::metal::print_device(ctx);
  std::printf("\"revision\":{\"commit\":\"%s\",\"dirty\":%s},", AUTAN_BENCH_COMMIT,
              AUTAN_BENCH_DIRTY);
  // `config` identifies a run by content (mesh hash, not path) and holds no measurement, so that
  // autan-bench compare can match runs; one-off host costs go to `setup_seconds`.
  const std::string sha = autan::io::mesh_hash(o.mesh);
  std::printf(
      "\"config\":{\"mesh_file\":\"%s\",\"mesh_sha256\":\"%s\",\"policy\":\"%s\","
      "\"order\":\"%s\",\"layout\":\"soa\",\"n_cells\":%d,\"n_faces\":%d,"
      "\"n_interior_faces\":%d,\"n_incidences\":%lld,\"block\":%d,\"warmup\":%d,\"reps\":%d,"
      "\"batch\":%d,\"tolerance\":%.1e},",
      std::filesystem::path(o.mesh).filename().string().c_str(), sha.c_str(), o.policy.c_str(),
      o.order.c_str(), m.topo.n_cells, m.topo.n_faces, m.topo.n_interior_faces,
      static_cast<long long>(m.topo.n_incidences), o.block, o.warmup, o.reps, o.batch, tolerance);
  std::printf("\"setup_seconds\":{\"read\":%.3f,\"renumber\":%.3f,\"metal_compile\":%.3f},",
              read_seconds, renumber_seconds, ctx.compile_seconds());
  std::printf("\"verified\":%s,\"kernels\":[", verified ? "true" : "false");
  for (std::size_t k = 0; k < timed.size(); ++k) {
    const auto& t = timed[k];
    std::printf("%s{\"name\":\"%s\",\"bytes_per_rep\":%.0f,\"cells\":%d,\"params\":{\"strategy\":"
                "\"%s\",\"policy\":\"%s\",\"order\":\"%s\",\"write_bytes\":%.0f},"
                "\"seconds\":[",
                k ? "," : "", t.name.c_str(), t.bytes, m.topo.n_cells, t.strategy.c_str(),
                o.policy.c_str(), o.order.c_str(), t.write_bytes);
    for (std::size_t r = 0; r < t.seconds.size(); ++r)
      std::printf("%s%.9e", r ? "," : "", t.seconds[r]);
    std::printf("]}");
  }
  std::printf("]}\n");
  return verified ? 0 : 1;
}
