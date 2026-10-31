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

int autan_spike_advection(const void* mesh, const char* policy, const double* q, const double* grad,
                          const double* velocity, double* res, char* err, std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return dispatch(policy, err, err_len, [&](auto p) {
    autan::spike::advection<decltype(p)>(*m, q, grad, velocity, res);
  });
}

}  // extern "C"
