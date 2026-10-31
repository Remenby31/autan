// C ABI of the generator spike: builds a mesh through the core and evaluates the generated loops
// on the host, so that the Python test drivers reach the C++ solver API through ctypes.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "advection_cpu.hpp"
#include "advection_diffusion_cpu.hpp"
#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"
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
  d.lsq_weight = g.lsq_weight.data();
  d.grad = gs.data();
  d.source = ss.data();
  d.cell_volume = g.cell_volume.data();
  d.res = rs.data();
  d.cell_centroid = g.cell_centroid.data();
  d.face_centroid = g.face_centroid.data();
  d.face_area = g.face_area.data();
  d.face_moment = g.face_moment.data();
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = typename P::param(velocity[i]);
  }
  d.nu[0] = typename P::param(nu);
  ad::cpu::lsq_gradient(d);
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
  d.lsq_weight = g.lsq_weight.data();
  d.grad = gs.data();
  d.source = ss.data();
  d.cell_volume = g.cell_volume.data();
  d.res = rs.data();
  d.cell_centroid = g.cell_centroid.data();
  d.face_centroid = g.face_centroid.data();
  d.face_area = g.face_area.data();
  d.face_moment = g.face_moment.data();
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
  ad::cpu::lsq_gradient(d);
  if (mode == 0) {
    for (std::size_t c = 0; c < nc; ++c) {
      q_tan[c] = S(seed[c]);
    }
    ad::cpu::lsq_gradient_tangent(d);
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
    ad::cpu::lsq_gradient_adjoint(d);
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
  d.lsq_weight = g.lsq_weight.data();
  d.grad = gs.data();
  d.source = ss.data();
  d.cell_volume = g.cell_volume.data();
  d.res = rs.data();
  d.cell_centroid = g.cell_centroid.data();
  d.face_centroid = g.face_centroid.data();
  d.face_area = g.face_area.data();
  d.face_moment = g.face_moment.data();
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = velocity[i];
  }
  d.nu[0] = nu;
  ad::cpu::lsq_gradient(d);
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
void roe(const SpikeMesh& m, int mode, const double* u, const double* seed, double* out) {
  namespace rf = autan::kernels::roe;
  using S = typename P::state;
  using R = typename P::residual;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  GeomCopy<P> g(m.packed);
  auto us = copy_in<S>(u, 5 * nc);
  std::vector<S> u_tan(5 * nc, S(0));
  std::vector<R> rs(5 * nc, R(0)), r_tan(5 * nc, R(0)), r_adj(5 * nc, R(0)), u_adj(5 * nc, R(0));
  rf::Data<P> d{};
  d.topo = m.topo;
  d.u_cons = us.data();
  d.res = rs.data();
  d.face_normal = g.face_normal.data();
  d.face_area_magnitude = g.face_area_magnitude.data();
  d.u_cons_tan = u_tan.data();
  d.res_tan = r_tan.data();
  d.res_adj = r_adj.data();
  d.u_cons_adj = u_adj.data();
  if (mode == 0) {
    rf::cpu::flux(d);
    copy_out(rs, out);
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
  d.face_normal = g.face_normal.data();
  d.face_area_magnitude = g.face_area_magnitude.data();
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
  d.lsq_weight = g.lsq_weight.data();
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
  d.lsq_weight = g.lsq_weight.data();
  d.cell_volume = g.cell_volume.data();
  d.cell_centroid = g.cell_centroid.data();
  d.face_centroid = g.face_centroid.data();
  d.face_area = g.face_area.data();
  d.face_moment = g.face_moment.data();
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
void advection(const SpikeMesh& m, const double* q, const double* grad, const double* velocity,
               double* res) {
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
  d.face_r_left = g.face_r_left.data();
  d.face_r_right = g.face_r_right.data();
  d.face_normal = g.face_normal.data();
  d.face_area_magnitude = g.face_area_magnitude.data();
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = typename P::param(velocity[i]);
  }
  av::cpu::residual(d);
  copy_out(rs, res);
}

}  // namespace autan::spike

namespace {

template <class F>
int dispatch(const char* policy, char* err, std::size_t len, F&& f) {
  return guarded(err, len, [&] { autan::spike::with_policy(policy, f); });
}

std::unique_ptr<SpikeMesh> finish(autan::Mesh mesh) {
  auto m = std::make_unique<SpikeMesh>(SpikeMesh{std::move(mesh), {}, {}, {}});
  m->geometry = autan::compute_geometry(m->mesh);
  m->packed = autan::kernels::pack_geometry(m->mesh, m->geometry);
  m->topo = autan::kernels::topology(m->mesh.view());
  return m;
}

}  // namespace

extern "C" {

/// Builds a mesh from face-based arrays (face_cell1 = -1 on the boundary); all boundary faces in
/// one zone. Returns 0 and a handle in *out, or 1 with a coded message in err.
int autan_spike_mesh_from_faces(std::int64_t n_points, const double* xyz, std::int64_t n_faces,
                                const std::int64_t* face_node_offsets,
                                const std::int32_t* face_nodes, const std::int32_t* face_cell0,
                                const std::int32_t* face_cell1, std::int32_t n_cells, void** out,
                                char* err, std::size_t err_len) {
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
    in.face_zones = {{"interior", "", autan::BcKind::unspecified, false, {}},
                     {"boundary", "", autan::BcKind::unspecified, true, {}}};
    in.cell_zones = {{"fluid", "", autan::CellKind::unspecified, {}}};
    in.source = {"memory", ""};
    *out = finish(autan::assemble(std::move(in))).release();
  });
}

/// Reads a mesh file through the core readers (autan::io::read_mesh).
int autan_spike_mesh_read(const char* path, void** out, char* err, std::size_t err_len) {
  return guarded(err, err_len, [&] { *out = finish(autan::io::read_mesh(path)).release(); });
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

int autan_spike_roe_complex_step(const void* mesh, const double* u, const double* du, double h,
                                 double* out, char* err, std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] { autan::spike::roe_complex_step(*m, u, du, h, out); });
}

int autan_spike_lsq_bounds_derivative(const void* mesh, const char* policy, int mode,
                                      const double* q, const double* q_b, const double* seed,
                                      double* out, char* err, std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return dispatch(policy, err, err_len, [&](auto p) {
    autan::spike::lsq_bounds_derivative<decltype(p)>(*m, mode, q, q_b, seed, out);
  });
}

int autan_spike_advection(const void* mesh, const char* policy, const double* q, const double* grad,
                          const double* velocity, double* res, char* err, std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return dispatch(policy, err, err_len, [&](auto p) {
    autan::spike::advection<decltype(p)>(*m, q, grad, velocity, res);
  });
}

}  // extern "C"
