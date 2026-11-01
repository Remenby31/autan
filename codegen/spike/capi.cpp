// C ABI of the generator spike: builds a mesh through the core and evaluates the generated loops
// on the host, so that the Python test drivers reach the C++ solver API through ctypes.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "advection_cpu.hpp"
#include "advection_diffusion_cpu.hpp"
#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/partition/partition.hpp"
#include "face_weighted_cpu.hpp"
#include "gradients_cpu.hpp"
#include "roe_cpu.hpp"
#include "spike.hpp"

namespace {

using autan::kernels::index_t;
using autan::kernels::offset_t;
using autan::spike::GeomCopy;
using autan::spike::guarded;
using autan::spike::SpikeMesh;

template <class T>
std::vector<T> copy_in(const double* p, std::size_t n) {
  return std::vector<T>(p, p + n);
}

template <class T>
void copy_out(const std::vector<T>& v, double* p) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    p[i] = static_cast<double>(v[i]);
  }
}

}  // namespace

namespace autan::spike {

/// LSQ gradient of the advection-diffusion residual: over the cell stencil when the mesh has one
/// (autan_spike_set_stencil), else over face neighbours.
template <class D>
void lsq(const D& d, const SpikeMesh& m) {
  namespace ad = autan::kernels::advection_diffusion;
  if (m.topo.stencil != nullptr) {
    ad::cpu::lsq_gradient_stencil(d);
  } else {
    ad::cpu::lsq_gradient(d);
  }
}

/// Residual of the advection-diffusion scheme (kernels/advection_diffusion.py) in policy P:
/// LSQ gradient, then volume source, interior and boundary fluxes. Inputs and outputs in FP64.
template <class P>
void advdiff_residual(const SpikeMesh& m, const double* q, const double* q_b, const double* source,
                      const double* velocity, double nu, double* res, double* grad) {
  namespace ad = autan::kernels::advection_diffusion;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  GeomCopy<P> g(m.packed);
  auto qs = copy_in<typename P::state>(q, nc);
  auto qbs = copy_in<typename P::state>(q_b, nf);
  auto ss = copy_in<typename P::state>(source, nc);
  std::vector<typename P::grad> gs(3 * nc, typename P::grad(0));
  std::vector<typename P::residual> rs(nc, typename P::residual(0));
  ad::Data<P> d{};
  d.topo = m.topo;
  d.q = qs.data();
  d.q_b = qbs.data();
  autan::spike::bind(d.lsq_weight, g.lsq_weight);
  autan::spike::bind(d.stencil_lsq_weight, g.stencil_lsq_weight);
  d.grad = gs.data();
  d.source = ss.data();
  autan::spike::bind(d.cell_volume, g.cell_volume);
  d.res = rs.data();
  autan::spike::bind(d.cell_centroid, g.cell_centroid);
  autan::spike::bind(d.face_centroid, g.face_centroid);
  autan::spike::bind(d.face_area, g.face_area);
  autan::spike::bind(d.face_moment, g.face_moment);
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = typename P::param(velocity[i]);
  }
  d.nu[0] = typename P::param(nu);
  lsq(d, m);
  ad::cpu::init_residual(d);
  ad::cpu::flux(d);
  ad::cpu::boundary_flux(d);
  copy_out(rs, res);
  if (grad != nullptr) {
    copy_out(gs, grad);
  }
}

/// Tangent (mode 0: seed = dq, out = dR) or adjoint (mode 1: seed = R-bar, out = q-bar) of the
/// composed residual with respect to q, boundary values and sources held fixed. The adjoint runs
/// the loop adjoints in reverse order (research section 4.1).
template <class P>
void advdiff_derivative(const SpikeMesh& m, int mode, const double* q, const double* q_b,
                        const double* source, const double* velocity, double nu, const double* seed,
                        double* out) {
  namespace ad = autan::kernels::advection_diffusion;
  using S = typename P::state;
  using Gr = typename P::grad;
  using R = typename P::residual;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  GeomCopy<P> g(m.packed);
  auto qs = copy_in<S>(q, nc);
  auto qbs = copy_in<S>(q_b, nf);
  auto ss = copy_in<S>(source, nc);
  std::vector<Gr> gs(3 * nc, Gr(0)), g_tan(3 * nc, Gr(0));
  std::vector<R> rs(nc, R(0)), r_tan(nc, R(0)), r_adj(nc, R(0)), q_adj(nc, R(0)),
      g_adj(3 * nc, R(0)), s_adj(nc, R(0));
  std::vector<S> q_tan(nc, S(0)), s_tan(nc, S(0));
  ad::Data<P> d{};
  d.topo = m.topo;
  d.q = qs.data();
  d.q_b = qbs.data();
  autan::spike::bind(d.lsq_weight, g.lsq_weight);
  autan::spike::bind(d.stencil_lsq_weight, g.stencil_lsq_weight);
  d.grad = gs.data();
  d.source = ss.data();
  autan::spike::bind(d.cell_volume, g.cell_volume);
  d.res = rs.data();
  autan::spike::bind(d.cell_centroid, g.cell_centroid);
  autan::spike::bind(d.face_centroid, g.face_centroid);
  autan::spike::bind(d.face_area, g.face_area);
  autan::spike::bind(d.face_moment, g.face_moment);
  d.q_tan = q_tan.data();
  d.grad_tan = g_tan.data();
  d.source_tan = s_tan.data();
  d.res_tan = r_tan.data();
  d.res_adj = r_adj.data();
  d.q_adj = q_adj.data();
  d.grad_adj = g_adj.data();
  d.source_adj = s_adj.data();
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = typename P::param(velocity[i]);
  }
  d.nu[0] = typename P::param(nu);
  lsq(d, m);
  if (mode == 0) {
    for (std::size_t c = 0; c < nc; ++c) {
      q_tan[c] = S(seed[c]);
    }
    if (m.topo.stencil != nullptr) {
      ad::cpu::lsq_gradient_stencil_tangent(d);
    } else {
      ad::cpu::lsq_gradient_tangent(d);
    }
    ad::cpu::init_residual_tangent(d);
    ad::cpu::flux_tangent(d);
    ad::cpu::boundary_flux_tangent(d);
    copy_out(r_tan, out);
  } else {
    for (std::size_t c = 0; c < nc; ++c) {
      r_adj[c] = R(seed[c]);
    }
    ad::cpu::boundary_flux_adjoint(d);
    ad::cpu::flux_adjoint(d);
    ad::cpu::init_residual_adjoint(d);
    if (m.topo.stencil != nullptr) {
      ad::cpu::lsq_gradient_stencil_adjoint(d);
    } else {
      ad::cpu::lsq_gradient_adjoint(d);
    }
    copy_out(q_adj, out);
  }
}

/// Im(R(q + i h dq)) / h on the host in complex arithmetic: the complex-step derivative
/// (Martins, Sturdza and Alonso 2003, eq. 6).
void advdiff_complex_step(const SpikeMesh& m, const double* q, const double* q_b,
                          const double* source, const double* velocity, double nu, const double* dq,
                          double h, double* out) {
  using P = autan::kernels::complex_step;
  namespace ad = autan::kernels::advection_diffusion;
  using Z = std::complex<double>;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  GeomCopy<P> g(m.packed);
  std::vector<Z> qs(nc), qbs(q_b, q_b + nf), ss(source, source + nc), gs(3 * nc), rs(nc);
  for (std::size_t c = 0; c < nc; ++c) {
    qs[c] = Z(q[c], h * dq[c]);
  }
  ad::Data<P> d{};
  d.topo = m.topo;
  d.q = qs.data();
  d.q_b = qbs.data();
  autan::spike::bind(d.lsq_weight, g.lsq_weight);
  autan::spike::bind(d.stencil_lsq_weight, g.stencil_lsq_weight);
  d.grad = gs.data();
  d.source = ss.data();
  autan::spike::bind(d.cell_volume, g.cell_volume);
  d.res = rs.data();
  autan::spike::bind(d.cell_centroid, g.cell_centroid);
  autan::spike::bind(d.face_centroid, g.face_centroid);
  autan::spike::bind(d.face_area, g.face_area);
  autan::spike::bind(d.face_moment, g.face_moment);
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = velocity[i];
  }
  d.nu[0] = nu;
  lsq(d, m);
  ad::cpu::init_residual(d);
  ad::cpu::flux(d);
  ad::cpu::boundary_flux(d);
  for (std::size_t c = 0; c < nc; ++c) {
    out[c] = rs[c].imag() / h;
  }
}

/// Roe flux loop: mode 0 residual of U, 1 tangent (seed dU -> dR), 2 adjoint (seed R-bar -> U-bar).
/// Arrays are 5 x n_cells, structure of arrays.
template <class P>
void roe(const SpikeMesh& m, int mode, const double* u, const double* seed, double* out,
         double* jac = nullptr) {
  namespace rf = autan::kernels::roe;
  using S = typename P::state;
  using R = typename P::residual;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  GeomCopy<P> g(m.packed);
  auto us = copy_in<S>(u, 5 * nc);
  std::vector<S> u_tan(5 * nc, S(0));
  std::vector<R> rs(5 * nc, R(0)), r_tan(5 * nc, R(0)), r_adj(5 * nc, R(0)), u_adj(5 * nc, R(0));
  std::vector<typename P::jac> js(mode == 3 ? 50 * static_cast<std::size_t>(m.topo.n_faces) : 0);
  rf::Data<P> d{};
  d.topo = m.topo;
  d.face_jac = js.data();
  d.u_cons = us.data();
  d.res = rs.data();
  autan::spike::bind(d.face_normal, g.face_normal);
  autan::spike::bind(d.face_area_magnitude, g.face_area_magnitude);
  d.u_cons_tan = u_tan.data();
  d.res_tan = r_tan.data();
  d.res_adj = r_adj.data();
  d.u_cons_adj = u_adj.data();
  if (mode == 0) {
    rf::cpu::flux(d);
    copy_out(rs, out);
  } else if (mode == 3) {
    rf::cpu::flux_and_jacobian(d);
    copy_out(rs, out);
    copy_out(js, jac);
  } else if (mode == 1) {
    for (std::size_t i = 0; i < 5 * nc; ++i) u_tan[i] = S(seed[i]);
    rf::cpu::flux_tangent(d);
    copy_out(r_tan, out);
  } else {
    for (std::size_t i = 0; i < 5 * nc; ++i) r_adj[i] = R(seed[i]);
    rf::cpu::flux_adjoint(d);
    copy_out(u_adj, out);
  }
}

void roe_complex_step(const SpikeMesh& m, const double* u, const double* du, double h,
                      double* out) {
  using P = autan::kernels::complex_step;
  namespace rf = autan::kernels::roe;
  using Z = std::complex<double>;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  GeomCopy<P> g(m.packed);
  std::vector<Z> us(5 * nc), rs(5 * nc);
  for (std::size_t i = 0; i < 5 * nc; ++i) us[i] = Z(u[i], h * du[i]);
  rf::Data<P> d{};
  d.topo = m.topo;
  d.u_cons = us.data();
  d.res = rs.data();
  autan::spike::bind(d.face_normal, g.face_normal);
  autan::spike::bind(d.face_area_magnitude, g.face_area_magnitude);
  rf::cpu::flux(d);
  for (std::size_t i = 0; i < 5 * nc; ++i) out[i] = rs[i].imag() / h;
}

/// LSQ gradient and bounds: mode 1 tangent (seed dq -> (dgrad, dqmin, dqmax) as 5 x n_cells),
/// mode 2 adjoint (seed 5 x n_cells -> q-bar).
template <class P>
void lsq_bounds_derivative(const SpikeMesh& m, int mode, const double* q, const double* q_b,
                           const double* seed, double* out) {
  namespace gr = autan::kernels::gradients;
  using S = typename P::state;
  using Gr = typename P::grad;
  using R = typename P::residual;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  GeomCopy<P> g(m.packed);
  auto qs = copy_in<S>(q, nc);
  auto qbs = copy_in<S>(q_b, nf);
  std::vector<Gr> gs(3 * nc), g_tan(3 * nc);
  std::vector<S> lo(nc), hi(nc), lo_tan(nc), hi_tan(nc), q_tan(nc, S(0));
  std::vector<R> g_adj(3 * nc, R(0)), lo_adj(nc, R(0)), hi_adj(nc, R(0)), q_adj(nc, R(0));
  gr::Data<P> d{};
  d.topo = m.topo;
  d.q = qs.data();
  d.q_b = qbs.data();
  d.grad = gs.data();
  d.qmin = lo.data();
  d.qmax = hi.data();
  autan::spike::bind(d.lsq_weight, g.lsq_weight);
  autan::spike::bind(d.stencil_lsq_weight, g.stencil_lsq_weight);
  d.q_tan = q_tan.data();
  d.grad_tan = g_tan.data();
  d.qmin_tan = lo_tan.data();
  d.qmax_tan = hi_tan.data();
  d.grad_adj = g_adj.data();
  d.qmin_adj = lo_adj.data();
  d.qmax_adj = hi_adj.data();
  d.q_adj = q_adj.data();
  if (mode == 1) {
    for (std::size_t c = 0; c < nc; ++c) q_tan[c] = S(seed[c]);
    gr::cpu::lsq_gradient_bounds_tangent(d);
    for (std::size_t i = 0; i < 3 * nc; ++i) out[i] = static_cast<double>(g_tan[i]);
    for (std::size_t c = 0; c < nc; ++c) {
      out[3 * nc + c] = static_cast<double>(lo_tan[c]);
      out[4 * nc + c] = static_cast<double>(hi_tan[c]);
    }
  } else {
    for (std::size_t i = 0; i < 3 * nc; ++i) g_adj[i] = R(seed[i]);
    for (std::size_t c = 0; c < nc; ++c) {
      lo_adj[c] = R(seed[3 * nc + c]);
      hi_adj[c] = R(seed[4 * nc + c]);
    }
    gr::cpu::lsq_gradient_bounds_adjoint(d);
    copy_out(q_adj, out);
  }
}

/// Test kernel codegen/tests/kernels/face_weighted.py: mode 0 value, 1 tangent (seed dq),
/// 2 adjoint (seed w-bar -> q-bar).
template <class P>
void face_weighted(const SpikeMesh& m, int mode, const double* q, const double* q_b,
                   const double* seed, double* out) {
  namespace fw = autan::kernels::face_weighted;
  using S = typename P::state;
  using R = typename P::residual;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  GeomCopy<P> g(m.packed);
  auto qs = copy_in<S>(q, nc);
  auto qbs = copy_in<S>(q_b, nf);
  auto q_tan = copy_in<S>(seed, nc);
  std::vector<S> w(nc), w_tan(nc);
  std::vector<R> w_adj(seed, seed + nc), q_adj(nc, R(0));
  fw::Data<P> d{};
  d.topo = m.topo;
  d.q = qs.data();
  d.q_b = qbs.data();
  d.w = w.data();
  autan::spike::bind(d.face_area_magnitude, g.face_area_magnitude);
  autan::spike::bind(d.face_centroid, g.face_centroid);
  autan::spike::bind(d.cell_centroid, g.cell_centroid);
  d.q_tan = q_tan.data();
  d.w_tan = w_tan.data();
  d.w_adj = w_adj.data();
  d.q_adj = q_adj.data();
  if (mode == 0) {
    fw::cpu::face_weighted(d);
    copy_out(w, out);
  } else if (mode == 1) {
    fw::cpu::face_weighted_tangent(d);
    copy_out(w_tan, out);
  } else {
    fw::cpu::face_weighted_adjoint(d);
    copy_out(q_adj, out);
  }
}

template <class P>
void gradients(const SpikeMesh& m, const char* method, const double* q, const double* q_b,
               int iterations, double* grad, double* qmin, double* qmax) {
  namespace gr = autan::kernels::gradients;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  GeomCopy<P> g(m.packed);
  auto qs = copy_in<typename P::state>(q, nc);
  auto qbs = copy_in<typename P::state>(q_b, nf);
  auto gs = copy_in<typename P::grad>(grad, 3 * nc);
  std::vector<typename P::state> mins(nc), maxs(nc);
  std::vector<typename P::residual> acc(3 * nc);
  gr::Data<P> d{};
  d.topo = m.topo;
  d.q = qs.data();
  d.q_b = qbs.data();
  d.grad = gs.data();
  d.qmin = mins.data();
  d.qmax = maxs.data();
  d.gg_sum = acc.data();
  autan::spike::bind(d.lsq_weight, g.lsq_weight);
  autan::spike::bind(d.stencil_lsq_weight, g.stencil_lsq_weight);
  autan::spike::bind(d.cell_volume, g.cell_volume);
  autan::spike::bind(d.cell_centroid, g.cell_centroid);
  autan::spike::bind(d.face_centroid, g.face_centroid);
  autan::spike::bind(d.face_area, g.face_area);
  autan::spike::bind(d.face_moment, g.face_moment);
  if (std::string_view(method) == "lsq") {
    gr::cpu::lsq_gradient_bounds(d);
  } else {
    for (int it = 0; it < iterations; ++it) {
      std::fill(acc.begin(), acc.end(), typename P::residual(0));
      gr::cpu::green_gauss_interior(d);
      gr::cpu::green_gauss_boundary(d);
      gr::cpu::green_gauss_finalise(d);
    }
  }
  copy_out(gs, grad);
  if (qmin != nullptr) {
    copy_out(mins, qmin);
    copy_out(maxs, qmax);
  }
}

template <class P>
void advection(const SpikeMesh& m, std::string_view strategy, const double* q, const double* grad,
               const double* velocity, double* res) {
  namespace av = autan::kernels::advection;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  GeomCopy<P> g(m.packed);
  auto qs = copy_in<typename P::state>(q, nc);
  auto gs = copy_in<typename P::grad>(grad, 3 * nc);
  std::vector<typename P::residual> rs(nc, typename P::residual(0));
  av::Data<P> d{};
  d.topo = m.topo;
  d.q = qs.data();
  d.grad = gs.data();
  d.res = rs.data();
  autan::spike::bind(d.face_r_left, g.face_r_left);
  autan::spike::bind(d.face_r_right, g.face_r_right);
  autan::spike::bind(d.face_normal, g.face_normal);
  autan::spike::bind(d.face_area_magnitude, g.face_area_magnitude);
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = typename P::param(velocity[i]);
  }
  if (strategy == "scatter") {
    av::cpu::residual(d);
  } else {
    av::cpu::residual(d, autan::kernels::strategy_of(strategy));
  }
  copy_out(rs, res);
}

}  // namespace autan::spike

namespace {

template <class F>
int dispatch(const char* policy, char* err, std::size_t len, F&& f) {
  return guarded(err, len, [&] { autan::spike::with_policy(policy, f); });
}

std::unique_ptr<SpikeMesh> finish(autan::Mesh mesh, autan::Geometry geometry) {
  auto m = std::make_unique<SpikeMesh>(
      SpikeMesh{.mesh = std::move(mesh), .geometry = std::move(geometry)});
  m->packed = autan::kernels::pack_geometry(m->mesh, m->geometry);
  m->topo = autan::kernels::topology(m->mesh.view());
  m->twins = autan::kernels::incidence_twins(m->mesh);
  m->topo.incidence_twin = m->twins.data();
  return m;
}

std::unique_ptr<SpikeMesh> finish(autan::Mesh mesh) {
  auto g = autan::compute_geometry(mesh);
  return finish(std::move(mesh), std::move(g));
}

}  // namespace

extern "C" {

/// Builds a mesh from face-based arrays (face_cell1 = -1 on the boundary), the last
/// n_ghost_cells cells being ghosts; all boundary faces in one zone. Returns 0 and a handle in
/// *out, or 1 with a coded message in err.
int autan_spike_mesh_from_faces(std::int64_t n_points, const double* xyz, std::int64_t n_faces,
                                const std::int64_t* face_node_offsets,
                                const std::int32_t* face_nodes, const std::int32_t* face_cell0,
                                const std::int32_t* face_cell1, std::int32_t n_cells,
                                std::int32_t n_ghost_cells, void** out, char* err,
                                std::size_t err_len) {
  return guarded(err, err_len, [&] {
    autan::MeshInput in;
    const auto np = static_cast<std::size_t>(n_points);
    const auto nf = static_cast<std::size_t>(n_faces);
    for (std::size_t i = 0; i < np; ++i) {
      in.x.push_back(xyz[3 * i]);
      in.y.push_back(xyz[3 * i + 1]);
      in.z.push_back(xyz[3 * i + 2]);
    }
    in.face_node_offsets.assign(face_node_offsets, face_node_offsets + nf + 1);
    in.face_nodes.assign(face_nodes, face_nodes + face_node_offsets[nf]);
    in.face_cell0.assign(face_cell0, face_cell0 + nf);
    in.face_cell1.assign(face_cell1, face_cell1 + nf);
    in.face_zone.resize(nf);
    for (std::size_t f = 0; f < nf; ++f) {
      in.face_zone[f] = face_cell1[f] < 0 ? 1 : 0;
    }
    in.cell_zone.assign(static_cast<std::size_t>(n_cells), 0);
    in.n_ghost_cells = n_ghost_cells;
    in.face_zones = {{"interior", "", autan::BcKind::unspecified, false, {}},
                     {"boundary", "", autan::BcKind::unspecified, true, {}}};
    in.cell_zones = {{"fluid", "", autan::CellKind::unspecified, {}}};
    in.source = {"memory", ""};
    *out = finish(autan::assemble(std::move(in))).release();
  });
}

/// Part `rank` of `parts` (n_cells entries: one per cell of the unpartitioned `global`, checked
/// by extract_partition) with `layers` ghost
/// layers (autan::extract_partition). Ghost cells get the volume and translated centroid of their
/// global cell, as the forward halo exchange of geometry would give them.
int autan_spike_mesh_partition(const void* global, const std::int32_t* parts, std::int64_t n_cells,
                               std::int32_t n_parts, std::int32_t rank, std::int32_t layers,
                               void** out, char* err, std::size_t err_len) {
  const auto* g = static_cast<const SpikeMesh*>(global);
  return guarded(err, err_len, [&] {
    auto lp = autan::extract_partition(
        g->mesh, g->geometry,
        std::span<const std::int32_t>(parts, static_cast<std::size_t>(n_cells)), n_parts, rank,
        layers);
    // Global ids are persistent identities, not positions: find each one's cell in `global`.
    const auto global_ids = g->mesh.cell_global_id();
    std::unordered_map<std::int64_t, std::size_t> position;
    for (std::size_t c = 0; c < global_ids.size(); ++c) {
      position.emplace(global_ids[c], c);
    }
    const auto gid = lp.mesh.cell_global_id();
    const auto shift = lp.mesh.ghost_translation();
    const auto n_owned = static_cast<std::size_t>(lp.mesh.n_owned_cells());
    auto& lg = lp.geometry;
    const auto& gg = g->geometry;
    for (std::size_t c = n_owned; c < gid.size(); ++c) {
      const auto o = global_ids.empty() ? static_cast<std::size_t>(gid[c]) : position.at(gid[c]);
      const auto t = shift.empty() ? std::array<double, 3>{} : shift[c - n_owned];
      lg.cell_volume[c] = gg.cell_volume[o];
      lg.cell_centroid_x[c] = gg.cell_centroid_x[o] + t[0];
      lg.cell_centroid_y[c] = gg.cell_centroid_y[o] + t[1];
      lg.cell_centroid_z[c] = gg.cell_centroid_z[o] + t[2];
    }
    auto m = finish(std::move(lp.mesh), std::move(lg));
    m->face_global_id = std::move(lp.face_global_id);
    m->face_reversed = std::move(lp.face_reversed);
    *out = m.release();
  });
}

/// Orders the mesh by its global numbering (autan::kernels::GlobalKeys): incidences by global
/// face id, incidence LSQ weights and face moments recomputed accordingly, reversed faces evaluated
/// in their global orientation, and the cell stencil of `stencil_kind` (0 vertex, 1 face2, -1
/// none) by global keys. Results then do not depend on the partition.
int autan_spike_set_global_order(void* mesh, int stencil_kind, char* err, std::size_t err_len) {
  auto* m = static_cast<SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    const auto nc = static_cast<std::size_t>(m->mesh.n_cells());
    const auto nf = static_cast<std::size_t>(m->mesh.n_faces());
    const auto gid = m->mesh.cell_global_id();
    m->cell_key.resize(nc);
    m->face_key.resize(nf);
    for (std::size_t c = 0; c < nc; ++c) {
      m->cell_key[c] = gid.empty() ? static_cast<std::int64_t>(c) : gid[c];
    }
    for (std::size_t f = 0; f < nf; ++f) {
      m->face_key[f] =
          m->face_global_id.empty() ? static_cast<std::int64_t>(f) : m->face_global_id[f];
    }
    const auto keys = m->keys();
    m->packed = autan::kernels::pack_geometry(m->mesh, m->geometry, &keys);
    m->order = autan::kernels::global_order(m->mesh, keys);
    m->topo.cell_faces = m->order.cell_faces.data();
    m->topo.incidence_twin = m->order.twin.data();
    m->topo.face_reversed = m->face_reversed.empty() ? nullptr : m->face_reversed.data();
    m->global = true;
    if (stencil_kind >= 0) {
      m->stencil =
          autan::kernels::cell_stencil(m->mesh, m->packed,
                                       stencil_kind == 0 ? autan::kernels::StencilKind::vertex
                                                         : autan::kernels::StencilKind::face2,
                                       &keys);
      m->packed.stencil_lsq_weight = m->stencil.lsq_weight;
      m->topo.n_stencil = static_cast<offset_t>(m->stencil.entries.size());
      m->topo.stencil_offsets = m->stencil.offsets.data();
      m->topo.stencil = m->stencil.entries.data();
      m->topo.stencil_twin = m->stencil.twin.data();
    }
  });
}

/// Global cell and face of every local cell and face (identity on an unpartitioned mesh).
void autan_spike_global_ids(const void* mesh, std::int64_t* cells, std::int64_t* faces) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  const auto gid = m->mesh.cell_global_id();
  for (index_t c = 0; c < m->mesh.n_cells(); ++c) {
    cells[c] = gid.empty() ? c : gid[static_cast<std::size_t>(c)];
  }
  for (index_t f = 0; f < m->mesh.n_faces(); ++f) {
    faces[f] = m->face_global_id.empty() ? f : m->face_global_id[static_cast<std::size_t>(f)];
  }
}

/// Reads a mesh file through the core readers (autan::io::read_mesh).
int autan_spike_mesh_read(const char* path, void** out, char* err, std::size_t err_len) {
  return guarded(err, err_len, [&] { *out = finish(autan::io::read_mesh(path)).release(); });
}

/// Builds the cell stencil of `kind` (0 vertex, 1 face2) and its LSQ weights, so that the
/// advection-diffusion residual takes its gradient over it; kind -1 removes it.
int autan_spike_set_stencil(void* mesh, int kind, std::int64_t* size, char* err,
                            std::size_t err_len) {
  auto* m = static_cast<SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    if (kind < 0) {
      m->stencil = {};
      m->packed.stencil_lsq_weight.clear();
      m->topo.n_stencil = 0;
      m->topo.stencil_offsets = nullptr;
      m->topo.stencil = nullptr;
      m->topo.stencil_twin = nullptr;
    } else {
      m->stencil = autan::kernels::cell_stencil(
          m->mesh, m->packed,
          kind == 0 ? autan::kernels::StencilKind::vertex : autan::kernels::StencilKind::face2);
      m->packed.stencil_lsq_weight = m->stencil.lsq_weight;
      m->topo.n_stencil = static_cast<offset_t>(m->stencil.entries.size());
      m->topo.stencil_offsets = m->stencil.offsets.data();
      m->topo.stencil = m->stencil.entries.data();
      m->topo.stencil_twin = m->stencil.twin.data();
    }
    *size = m->topo.n_stencil;
  });
}

/// Fills the volume and centroid of ghost cells (NaN from compute_geometry until a partitioner
/// copies them from the owning partition) and repacks the geometry; call it before
/// autan_spike_set_stencil. `centroid` is 3 x n_cells (structure of arrays), `volume` n_cells;
/// owned entries are ignored.
int autan_spike_fill_ghost_geometry(void* mesh, const double* centroid, const double* volume,
                                    char* err, std::size_t err_len) {
  auto* m = static_cast<SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    const auto nc = static_cast<std::size_t>(m->mesh.n_cells());
    for (auto c = static_cast<std::size_t>(m->mesh.n_owned_cells()); c < nc; ++c) {
      m->geometry.cell_volume[c] = volume[c];
      m->geometry.cell_centroid_x[c] = centroid[c];
      m->geometry.cell_centroid_y[c] = centroid[nc + c];
      m->geometry.cell_centroid_z[c] = centroid[2 * nc + c];
    }
    m->packed = autan::kernels::pack_geometry(m->mesh, m->geometry);
  });
}

void autan_spike_stencil_arrays(const void* mesh, std::int64_t* offsets, std::int32_t* entries) {
  const auto& st = static_cast<const SpikeMesh*>(mesh)->stencil;
  std::copy(st.offsets.begin(), st.offsets.end(), offsets);
  std::copy(st.entries.begin(), st.entries.end(), entries);
}

void autan_spike_mesh_free(void* mesh) { delete static_cast<SpikeMesh*>(mesh); }

/// counts = {n_cells, n_owned_cells, n_faces, n_interior_faces, n_incidences}
void autan_spike_mesh_counts(const void* mesh, std::int64_t* counts) {
  const auto& t = static_cast<const SpikeMesh*>(mesh)->topo;
  counts[0] = t.n_cells;
  counts[1] = t.n_owned_cells;
  counts[2] = t.n_faces;
  counts[3] = t.n_interior_faces;
  counts[4] = t.n_incidences;
}

/// Copies the topology and packed geometry (layouts of autan::kernels::PackedGeometry).
void autan_spike_mesh_arrays(const void* mesh, std::int32_t* owner, std::int32_t* neighbour,
                             std::int64_t* cell_face_offsets, std::int32_t* cell_faces,
                             double* face_area, double* face_centroid, double* face_moment,
                             double* cell_centroid, double* cell_volume, double* lsq_weight) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  const auto& t = m->topo;
  std::memcpy(owner, t.owner, sizeof(index_t) * static_cast<std::size_t>(t.n_faces));
  std::memcpy(neighbour, t.neighbour,
              sizeof(index_t) * static_cast<std::size_t>(t.n_interior_faces));
  std::memcpy(cell_face_offsets, t.cell_face_offsets,
              sizeof(offset_t) * static_cast<std::size_t>(t.n_cells + 1));
  std::memcpy(cell_faces, t.cell_faces, sizeof(index_t) * static_cast<std::size_t>(t.n_incidences));
  const auto& p = m->packed;
  std::memcpy(face_area, p.face_area.data(), sizeof(double) * p.face_area.size());
  std::memcpy(face_centroid, p.face_centroid.data(), sizeof(double) * p.face_centroid.size());
  std::memcpy(face_moment, p.face_moment.data(), sizeof(double) * p.face_moment.size());
  std::memcpy(cell_centroid, p.cell_centroid.data(), sizeof(double) * p.cell_centroid.size());
  std::memcpy(cell_volume, p.cell_volume.data(), sizeof(double) * p.cell_volume.size());
  std::memcpy(lsq_weight, p.lsq_weight.data(), sizeof(double) * p.lsq_weight.size());
}

int autan_spike_advdiff_residual(const void* mesh, const char* policy, const double* q,
                                 const double* q_b, const double* source, const double* velocity,
                                 double nu, double* res, double* grad, char* err,
                                 std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return dispatch(policy, err, err_len, [&](auto p) {
    autan::spike::advdiff_residual<decltype(p)>(*m, q, q_b, source, velocity, nu, res, grad);
  });
}

int autan_spike_gradients(const void* mesh, const char* policy, const char* method, const double* q,
                          const double* q_b, int iterations, double* grad, double* qmin,
                          double* qmax, char* err, std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return dispatch(policy, err, err_len, [&](auto p) {
    autan::spike::gradients<decltype(p)>(*m, method, q, q_b, iterations, grad, qmin, qmax);
  });
}

int autan_spike_advdiff_derivative(const void* mesh, const char* policy, int mode, const double* q,
                                   const double* q_b, const double* source, const double* velocity,
                                   double nu, const double* seed, double* out, char* err,
                                   std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return dispatch(policy, err, err_len, [&](auto p) {
    autan::spike::advdiff_derivative<decltype(p)>(*m, mode, q, q_b, source, velocity, nu, seed,
                                                  out);
  });
}

int autan_spike_advdiff_complex_step(const void* mesh, const double* q, const double* q_b,
                                     const double* source, const double* velocity, double nu,
                                     const double* dq, double h, double* out, char* err,
                                     std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    autan::spike::advdiff_complex_step(*m, q, q_b, source, velocity, nu, dq, h, out);
  });
}

int autan_spike_roe(const void* mesh, const char* policy, int mode, const double* u,
                    const double* seed, double* out, char* err, std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return dispatch(policy, err, err_len,
                  [&](auto p) { autan::spike::roe<decltype(p)>(*m, mode, u, seed, out); });
}

/// Roe residual and its face Jacobians (50 x n_faces, structure of arrays; component
/// (side * 5 + i) * 5 + j is dF_i / dU_side,j) from the fused loop flux_and_jacobian.
int autan_spike_roe_jacobian(const void* mesh, const char* policy, const double* u, double* res,
                             double* jac, char* err, std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return dispatch(policy, err, err_len,
                  [&](auto p) { autan::spike::roe<decltype(p)>(*m, 3, u, u, res, jac); });
}

int autan_spike_roe_complex_step(const void* mesh, const double* u, const double* du, double h,
                                 double* out, char* err, std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] { autan::spike::roe_complex_step(*m, u, du, h, out); });
}

int autan_spike_face_weighted(const void* mesh, const char* policy, int mode, const double* q,
                              const double* q_b, const double* seed, double* out, char* err,
                              std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return dispatch(policy, err, err_len, [&](auto p) {
    autan::spike::face_weighted<decltype(p)>(*m, mode, q, q_b, seed, out);
  });
}

int autan_spike_lsq_bounds_derivative(const void* mesh, const char* policy, int mode,
                                      const double* q, const double* q_b, const double* seed,
                                      double* out, char* err, std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return dispatch(policy, err, err_len, [&](auto p) {
    autan::spike::lsq_bounds_derivative<decltype(p)>(*m, mode, q, q_b, seed, out);
  });
}

/// `strategy`: "scatter" (the face loop) or a strategy name (store-gather and gather-recompute
/// gather per cell in Topology::cell_faces order).
int autan_spike_advection(const void* mesh, const char* policy, const char* strategy,
                          const double* q, const double* grad, const double* velocity, double* res,
                          char* err, std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return dispatch(policy, err, err_len, [&](auto p) {
    autan::spike::advection<decltype(p)>(*m, strategy, q, grad, velocity, res);
  });
}

}  // extern "C"
