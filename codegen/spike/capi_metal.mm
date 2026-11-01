// C ABI of the Metal spike driver: the generated Metal loops on shared-memory copies of a spike
// mesh, under the same symbols as the CUDA driver (capi_gpu.cu), so that the same tests run.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "advection_diffusion_metal.hpp"
#include "advection_metal.hpp"
#include "autan/kernels/metal.hpp"
#include "autan/kernels/strategy.hpp"
#include "gradients_metal.hpp"
#include "metal_device.hpp"
#include "roe_metal.hpp"
#include "spike.hpp"

namespace {

using autan::kernels::index_t;
using autan::kernels::offset_t;
using autan::kernels::Strategy;
using autan::kernels::strategy_of;
using autan::spike::GeomCopy;
using autan::spike::guarded;
using autan::spike::SpikeMesh;
using autan::spike::metal::DeviceGeom;
using autan::spike::metal::DeviceMesh;
namespace mtl = autan::kernels::metal;

template <class T>
using DeviceArray = autan::spike::metal::Array<T>;

template <class T>
std::vector<T> converted_in(const double* p, std::size_t n) {
  return std::vector<T>(p, p + n);
}

template <class T>
void copy_out(const std::vector<T>& v, double* p) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    p[i] = static_cast<double>(v[i]);
  }
}

/// One context per process, created on first use: libraries and pipelines compile once.
std::unique_ptr<mtl::Context>& context_slot() {
  static std::unique_ptr<mtl::Context> ctx;
  return ctx;
}

mtl::Context& context() {
  auto& ctx = context_slot();
  if (!ctx) {
    ctx = std::make_unique<mtl::Context>();
  }
  return *ctx;
}

void synchronise() { context().synchronise(); }

/// guarded() for Metal calls: a request refused while recording drops what it recorded.
template <class F>
int metal_guarded(char* err, std::size_t len, F&& body) {
  return guarded(err, len, [&] {
    try {
      body();
    } catch (...) {
      if (auto& ctx = context_slot()) {
        ctx->discard();
      }
      throw;
    }
  });
}

template <class P>
void advdiff_residual(const SpikeMesh& m, Strategy strategy, int block, const double* q,
                      const double* q_b, const double* source, const double* velocity, double nu,
                      double* res, double* grad, double* norm) {
  namespace ad = autan::kernels::advection_diffusion;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  auto& ctx = context();
  DeviceMesh<P> dm(ctx, m, block);
  GeomCopy<P> hg(m.packed);
  DeviceGeom<P> g(ctx, hg);
  DeviceArray<typename P::state> qs(ctx, converted_in<typename P::state>(q, nc));
  DeviceArray<typename P::state> qbs(ctx, converted_in<typename P::state>(q_b, nf));
  DeviceArray<typename P::state> ss(ctx, converted_in<typename P::state>(source, nc));
  DeviceArray<typename P::grad> gs(ctx, 3 * nc);
  DeviceArray<typename P::residual> rs(ctx, nc);
  ad::Data<P> d{};
  d.topo = dm.topo;
  d.q = qs.get();
  d.q_b = qbs.get();
  d.lsq_weight = g.lsq_weight.get();
  d.grad = gs.get();
  d.source = ss.get();
  d.cell_volume = g.cell_volume.get();
  d.res = rs.get();
  d.cell_centroid = g.cell_centroid.get();
  d.face_centroid = g.face_centroid.get();
  d.face_area = g.face_area.get();
  d.face_moment = g.face_moment.get();
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = typename P::param(velocity[i]);
  }
  d.nu[0] = typename P::param(nu);
  ad::metal_loops::lsq_gradient(ctx, d, dm.schedule);
  ad::metal_loops::init_residual(ctx, d, dm.schedule);
  ad::metal_loops::flux(ctx, d, strategy, dm.schedule);
  ad::metal_loops::boundary_flux(ctx, d, strategy, dm.schedule);
  const auto n_owned = static_cast<std::int64_t>(m.topo.n_owned_cells);
  DeviceArray<mtl::df64> partials(ctx, static_cast<std::size_t>(mtl::norm_partials(n_owned)));
  DeviceArray<mtl::df64> sum(ctx, 2);
  mtl::sum_of_squares(ctx, ad::metal_loops::library, rs.get(), n_owned, partials.get(), sum.get());
  synchronise();
  copy_out(rs.download(), res);
  copy_out(gs.download(), grad);
  *norm = mtl::norm_of(sum.download().data());
}

template <class P>
void gradients(const SpikeMesh& m, std::string_view method, Strategy strategy, int block,
               const double* q, const double* q_b, int iterations, double* grad, double* qmin,
               double* qmax) {
  namespace gr = autan::kernels::gradients;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  auto& ctx = context();
  DeviceMesh<P> dm(ctx, m, block);
  GeomCopy<P> hg(m.packed);
  DeviceGeom<P> g(ctx, hg);
  DeviceArray<typename P::state> qs(ctx, converted_in<typename P::state>(q, nc));
  DeviceArray<typename P::state> qbs(ctx, converted_in<typename P::state>(q_b, nf));
  DeviceArray<typename P::grad> gs(ctx, converted_in<typename P::grad>(grad, 3 * nc));
  DeviceArray<typename P::state> mins(ctx, nc), maxs(ctx, nc);
  DeviceArray<typename P::residual> acc(ctx, 3 * nc);
  gr::Data<P> d{};
  d.topo = dm.topo;
  d.q = qs.get();
  d.q_b = qbs.get();
  d.grad = gs.get();
  d.qmin = mins.get();
  d.qmax = maxs.get();
  d.gg_sum = acc.get();
  d.lsq_weight = g.lsq_weight.get();
  d.cell_volume = g.cell_volume.get();
  d.cell_centroid = g.cell_centroid.get();
  d.face_centroid = g.face_centroid.get();
  d.face_area = g.face_area.get();
  d.face_moment = g.face_moment.get();
  if (method == "lsq") {
    gr::metal_loops::lsq_gradient_bounds(ctx, d, dm.schedule);
  } else {
    for (int it = 0; it < iterations; ++it) {
      synchronise();
      acc.zero();
      gr::metal_loops::green_gauss_interior(ctx, d, strategy, dm.schedule);
      gr::metal_loops::green_gauss_boundary(ctx, d, strategy, dm.schedule);
      gr::metal_loops::green_gauss_finalise(ctx, d, dm.schedule);
    }
  }
  synchronise();
  copy_out(gs.download(), grad);
  copy_out(mins.download(), qmin);
  copy_out(maxs.download(), qmax);
}

template <class P>
void advection(const SpikeMesh& m, Strategy strategy, int block, const double* q,
               const double* grad, const double* velocity, double* res) {
  namespace av = autan::kernels::advection;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  auto& ctx = context();
  DeviceMesh<P> dm(ctx, m, block);
  GeomCopy<P> hg(m.packed);
  DeviceGeom<P> g(ctx, hg);
  DeviceArray<typename P::state> qs(ctx, converted_in<typename P::state>(q, nc));
  DeviceArray<typename P::grad> gs(ctx, converted_in<typename P::grad>(grad, 3 * nc));
  DeviceArray<typename P::residual> rs(ctx, nc);
  av::Data<P> d{};
  d.topo = dm.topo;
  d.q = qs.get();
  d.grad = gs.get();
  d.res = rs.get();
  d.face_r_left = g.face_r_left.get();
  d.face_r_right = g.face_r_right.get();
  d.face_normal = g.face_normal.get();
  d.face_area_magnitude = g.face_area_magnitude.get();
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = typename P::param(velocity[i]);
  }
  av::metal_loops::residual(ctx, d, strategy, dm.schedule);
  synchronise();
  copy_out(rs.download(), res);
}

template <class P>
void roe(const SpikeMesh& m, Strategy strategy, int block, int mode, const double* u,
         const double* seed, double* out) {
  namespace rf = autan::kernels::roe;
  using S = typename P::state;
  using R = typename P::residual;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  auto& ctx = context();
  DeviceMesh<P> dm(ctx, m, block);
  GeomCopy<P> hg(m.packed);
  DeviceGeom<P> g(ctx, hg);
  DeviceArray<S> us(ctx, converted_in<S>(u, 5 * nc));
  DeviceArray<S> u_tan(ctx, converted_in<S>(seed, 5 * nc));
  DeviceArray<R> r_adj(ctx, converted_in<R>(seed, 5 * nc));
  DeviceArray<R> rs(ctx, 5 * nc), r_tan(ctx, 5 * nc), u_adj(ctx, 5 * nc);
  rf::Data<P> d{};
  d.topo = dm.topo;
  d.u_cons = us.get();
  d.res = rs.get();
  d.face_normal = g.face_normal.get();
  d.face_area_magnitude = g.face_area_magnitude.get();
  d.u_cons_tan = u_tan.get();
  d.res_tan = r_tan.get();
  d.res_adj = r_adj.get();
  d.u_cons_adj = u_adj.get();
  if (mode == 0) {
    rf::metal_loops::flux(ctx, d, strategy, dm.schedule);
  } else if (mode == 1) {
    rf::metal_loops::flux_tangent(ctx, d, strategy, dm.schedule);
  } else {
    rf::metal_loops::flux_adjoint(ctx, d, strategy, dm.schedule);
  }
  synchronise();
  copy_out((mode == 0 ? rs : mode == 1 ? r_tan : u_adj).download(), out);
}

/// Adjoint of the composed advection-diffusion residual on the device (seed R-bar -> q-bar).
template <class P>
void advdiff_adjoint(const SpikeMesh& m, Strategy strategy, int block, const double* q,
                     const double* q_b, const double* source, const double* velocity, double nu,
                     const double* seed, double* out) {
  namespace ad = autan::kernels::advection_diffusion;
  using S = typename P::state;
  using R = typename P::residual;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  auto& ctx = context();
  DeviceMesh<P> dm(ctx, m, block);
  GeomCopy<P> hg(m.packed);
  DeviceGeom<P> g(ctx, hg);
  DeviceArray<S> qs(ctx, converted_in<S>(q, nc)), qbs(ctx, converted_in<S>(q_b, nf)),
      ss(ctx, converted_in<S>(source, nc));
  DeviceArray<typename P::grad> gs(ctx, 3 * nc);
  DeviceArray<R> rs(ctx, nc), r_adj(ctx, converted_in<R>(seed, nc)), q_adj(ctx, nc),
      g_adj(ctx, 3 * nc), s_adj(ctx, nc);
  ad::Data<P> d{};
  d.topo = dm.topo;
  d.q = qs.get();
  d.q_b = qbs.get();
  d.lsq_weight = g.lsq_weight.get();
  d.grad = gs.get();
  d.source = ss.get();
  d.cell_volume = g.cell_volume.get();
  d.res = rs.get();
  d.cell_centroid = g.cell_centroid.get();
  d.face_centroid = g.face_centroid.get();
  d.face_area = g.face_area.get();
  d.face_moment = g.face_moment.get();
  d.res_adj = r_adj.get();
  d.q_adj = q_adj.get();
  d.grad_adj = g_adj.get();
  d.source_adj = s_adj.get();
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = typename P::param(velocity[i]);
  }
  d.nu[0] = typename P::param(nu);
  ad::metal_loops::lsq_gradient(ctx, d, dm.schedule);
  ad::metal_loops::boundary_flux_adjoint(ctx, d, strategy, dm.schedule);
  ad::metal_loops::flux_adjoint(ctx, d, strategy, dm.schedule);
  ad::metal_loops::init_residual_adjoint(ctx, d, dm.schedule);
  ad::metal_loops::lsq_gradient_adjoint(ctx, d, strategy, dm.schedule);
  synchronise();
  copy_out(q_adj.download(), out);
}

/// Adjoint of the LSQ gradient + bounds reduction on the device (seed 5 x n_cells -> q-bar):
/// Sum, Min and Max accumulators and the finalise body.
template <class P>
void lsq_bounds_adjoint(const SpikeMesh& m, Strategy strategy, int block, const double* q,
                        const double* q_b, const double* seed, double* out) {
  namespace gr = autan::kernels::gradients;
  using S = typename P::state;
  using R = typename P::residual;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  auto& ctx = context();
  DeviceMesh<P> dm(ctx, m, block);
  GeomCopy<P> hg(m.packed);
  DeviceGeom<P> g(ctx, hg);
  DeviceArray<S> qs(ctx, converted_in<S>(q, nc)), qbs(ctx, converted_in<S>(q_b, nf));
  DeviceArray<typename P::grad> gs(ctx, 3 * nc);
  DeviceArray<S> lo(ctx, nc), hi(ctx, nc);
  DeviceArray<R> g_adj(ctx, converted_in<R>(seed, 3 * nc)),
      lo_adj(ctx, converted_in<R>(seed + 3 * nc, nc)),
      hi_adj(ctx, converted_in<R>(seed + 4 * nc, nc)), q_adj(ctx, nc);
  gr::Data<P> d{};
  d.topo = dm.topo;
  d.q = qs.get();
  d.q_b = qbs.get();
  d.grad = gs.get();
  d.qmin = lo.get();
  d.qmax = hi.get();
  d.lsq_weight = g.lsq_weight.get();
  d.grad_adj = g_adj.get();
  d.qmin_adj = lo_adj.get();
  d.qmax_adj = hi_adj.get();
  d.q_adj = q_adj.get();
  gr::metal_loops::lsq_gradient_bounds_adjoint(ctx, d, strategy, dm.schedule);
  synchronise();
  copy_out(q_adj.download(), out);
}

}  // namespace

extern "C" {

int autan_spike_gpu_lsq_bounds_adjoint(const void* mesh, const char* policy, const char* strategy,
                                       int block, const double* q, const double* q_b,
                                       const double* seed, double* out, char* err,
                                       std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return metal_guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::metal::with_policy(
        policy, [&](auto p) { lsq_bounds_adjoint<decltype(p)>(*m, s, block, q, q_b, seed, out); });
  });
}

/// Compiles a generated module's Metal source (NUL-terminated MSL text) without running it.
int autan_spike_metal_compile(const char* name, const char* source, char* err,
                              std::size_t err_len) {
  return guarded(err, err_len, [&] {
    const char* const pieces[] = {source, nullptr};
    context().compile(mtl::Library{name, pieces});
  });
}

/// Euclidean norm of FP32 data computed on the device by autan::kernels::metal::sum_of_squares.
int autan_spike_metal_norm_f32(const float* x, std::int64_t n, double* out, char* err,
                               std::size_t err_len) {
  return metal_guarded(err, err_len, [&] {
    namespace av = autan::kernels::advection;
    auto& ctx = context();
    DeviceArray<float> dx(ctx, x, static_cast<std::size_t>(n));
    DeviceArray<mtl::df64> partials(ctx, static_cast<std::size_t>(mtl::norm_partials(n)));
    DeviceArray<mtl::df64> sum(ctx, 2);
    mtl::sum_of_squares(ctx, av::metal_loops::library, dx.get(), n, partials.get(), sum.get());
    synchronise();
    *out = mtl::norm_of(sum.download().data());
  });
}

/// FP64 sum of squares on the device: Metal has no FP64, so it is refused (the residual norm of
/// autan_spike_gpu_advdiff_residual accumulates in double-float).
int autan_spike_gpu_sum_of_squares(const double*, std::int64_t, double*, char* err,
                                   std::size_t err_len) {
  return guarded(err, err_len, [&] {
    throw std::invalid_argument(
        "E1013: Metal has no FP64: an FP64 sum of squares is not available");
  });
}

int autan_spike_gpu_roe(const void* mesh, const char* policy, const char* strategy, int block,
                        int mode, const double* u, const double* seed, double* out, char* err,
                        std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return metal_guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::metal::with_policy(
        policy, [&](auto p) { roe<decltype(p)>(*m, s, block, mode, u, seed, out); });
  });
}

int autan_spike_gpu_advdiff_adjoint(const void* mesh, const char* policy, const char* strategy,
                                    int block, const double* q, const double* q_b,
                                    const double* source, const double* velocity, double nu,
                                    const double* seed, double* out, char* err,
                                    std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return metal_guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::metal::with_policy(policy, [&](auto p) {
      advdiff_adjoint<decltype(p)>(*m, s, block, q, q_b, source, velocity, nu, seed, out);
    });
  });
}

int autan_spike_gpu_advdiff_residual(const void* mesh, const char* policy, const char* strategy,
                                     int block, const double* q, const double* q_b,
                                     const double* source, const double* velocity, double nu,
                                     double* res, double* grad, double* norm, char* err,
                                     std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return metal_guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::metal::with_policy(policy, [&](auto p) {
      advdiff_residual<decltype(p)>(*m, s, block, q, q_b, source, velocity, nu, res, grad, norm);
    });
  });
}

int autan_spike_gpu_gradients(const void* mesh, const char* policy, const char* method,
                              const char* strategy, int block, const double* q, const double* q_b,
                              int iterations, double* grad, double* qmin, double* qmax, char* err,
                              std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return metal_guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::metal::with_policy(policy, [&](auto p) {
      gradients<decltype(p)>(*m, method, s, block, q, q_b, iterations, grad, qmin, qmax);
    });
  });
}

int autan_spike_gpu_advection(const void* mesh, const char* policy, const char* strategy, int block,
                              const double* q, const double* grad, const double* velocity,
                              double* res, char* err, std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return metal_guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::metal::with_policy(
        policy, [&](auto p) { advection<decltype(p)>(*m, s, block, q, grad, velocity, res); });
  });
}

}  // extern "C"
