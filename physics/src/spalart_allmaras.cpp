#include "autan/physics/spalart_allmaras.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"
#include "spalart_allmaras_cpu.hpp"

namespace autan::physics {

namespace sk = autan::kernels::spalart_allmaras;

template <class P>
struct SpalartAllmaras<P>::Impl {
  using J = typename P::jac;

  NavierStokes<P>& flow;
  std::size_t nc, nf, ni;
  template <class Ptr>
  using stored = std::vector<std::remove_cvref_t<std::remove_pointer_t<Ptr>>>;
  using D = sk::Data<P>;
  stored<decltype(D::face_normal)> face_normal;
  stored<decltype(D::face_area_magnitude)> face_area_magnitude;
  stored<decltype(D::face_centroid)> face_centroid;
  stored<decltype(D::cell_centroid)> cell_centroid;
  stored<decltype(D::cell_volume)> cell_volume;
  stored<decltype(D::wall_distance)> wall_distance;
  std::vector<State> nt, nt_b, nt_bc, source;
  std::vector<typename P::grad> nt_grad;
  std::vector<J> cell_jac, face_jac, bnd_jac;
  std::vector<index_t> owner, neighbour;
  GhostFill* ghost_fill = nullptr;
  sk::Data<P> d{};

  Impl(const Mesh& mesh, const Geometry& geometry, NavierStokes<P>& f, const Transport& t, bool ft2,
       const kernels::GlobalKeys* keys)
      : flow(f),
        nc(static_cast<std::size_t>(mesh.n_cells())),
        nf(static_cast<std::size_t>(mesh.n_faces())),
        ni(static_cast<std::size_t>(mesh.n_interior_faces())) {
    const auto packed = kernels::pack_geometry(mesh, geometry, keys);
    face_normal =
        kernels::converted<typename decltype(face_normal)::value_type>(packed.face_normal);
    face_area_magnitude = kernels::converted<typename decltype(face_area_magnitude)::value_type>(
        packed.face_area_magnitude);
    face_centroid =
        kernels::converted<typename decltype(face_centroid)::value_type>(packed.face_centroid);
    cell_centroid =
        kernels::converted<typename decltype(cell_centroid)::value_type>(packed.cell_centroid);
    cell_volume =
        kernels::converted<typename decltype(cell_volume)::value_type>(packed.cell_volume);
    wall_distance.assign(nc, typename decltype(wall_distance)::value_type(1));
    nt.assign(nc, State(0));
    nt_b.assign(nf, State(0));
    nt_bc.assign(nf, State(0));
    source.assign(nc, State(0));
    nt_grad.assign(3 * nc, typename P::grad(0));
    cell_jac.assign(nc, J(0));
    face_jac.assign(2 * nf, J(0));
    bnd_jac.assign(nf, J(0));
    owner.assign(mesh.owner().begin(), mesh.owner().end());
    neighbour.assign(mesh.neighbour().begin(), mesh.neighbour().end());

    d.nt = nt.data();
    d.nt_b = nt_b.data();
    d.nt_bc = nt_bc.data();
    d.nt_source = source.data();
    d.nt_grad = nt_grad.data();
    d.wall_distance = wall_distance.data();
    d.nt_cell_jac = cell_jac.data();
    d.nt_face_jac = face_jac.data();
    d.nt_bnd_jac = bnd_jac.data();
    d.face_normal = face_normal.data();
    d.face_area_magnitude = face_area_magnitude.data();
    d.face_centroid = face_centroid.data();
    d.cell_centroid = cell_centroid.data();
    d.cell_volume = cell_volume.data();
    using Pa = typename P::param;
    d.mu_ref[0] = Pa(t.mu_ref);
    d.t_ref[0] = Pa(t.sutherland ? t.t_ref : 1.0);
    d.sutherland_s[0] = Pa(t.sutherland ? t.s : 0.0);
    d.sutherland[0] = Pa(t.sutherland ? 1.0 : 0.0);
    d.ft2_on[0] = Pa(ft2 ? 1.0 : 0.0);
    share();
  }

  // The flow's arrays of its last evaluation (reallocated by Euler<P> on first use).
  void share() {
    const auto a = flow.arrays();
    d.topo = a.flow.topo;
    d.prim = a.flow.prim;
    d.grad = a.flow.grad;
    d.prim_b = a.flow.prim_b;
    d.bc_kind = a.flow.bc_kind;
    d.stencil_lsq_weight = a.flow.stencil_lsq_weight;
    d.mu_t = a.mu_t;
    d.mu_t_b = a.mu_t_b;
    d.p_ref[0] = typename P::param(a.p_ref);
    d.gas_constant[0] = typename P::param(a.gas_constant);
  }

  // A face or boundary loop: its gather on the flow's rank-independent topology (global keys),
  // else its scatter.
  template <class Loop>
  void run(Loop loop) const {
    if (d.topo.rank_independent) {
      loop(kernels::Strategy::store_gather);
    } else {
      loop();
    }
  }

  // `data` over every local cell and face (the generated cell and boundary loops cover owned ones
  // only): ghosts then hold nt, its boundary values and the eddy viscosity of their owners.
  [[nodiscard]] static sk::Data<P> halo(sk::Data<P> data) {
    data.topo.n_owned_cells = data.topo.n_cells;
    data.topo.n_owned_interior_faces = data.topo.n_interior_faces;
    data.topo.n_owned_faces = data.topo.n_faces;
    return data;
  }

  void prepare(std::span<const State> s, std::span<Res> res) {
    share();
    d.rnu = const_cast<State*>(s.data());
    d.nt_res = res.data();
    const auto h = halo(d);
    sk::cpu::nt_values(h);
    sk::cpu::boundary_values(h);
    // The owned cells' gradients, filled into the ghosts: a ghost's stencil weights are not set
    // (Euler<P>::set_boundary weighs the owned cells' stencils).
    sk::cpu::nt_gradients(d);
    if (ghost_fill != nullptr) {
      std::array<typename P::grad*, 3> g{};
      for (std::size_t k = 0; k < 3; ++k) g[k] = nt_grad.data() + k * nc;
      ghost_fill->forward(std::span<typename P::grad* const>(g.data(), 3));
    }
  }
};

template <class P>
SpalartAllmaras<P>::SpalartAllmaras(const Mesh& mesh, const Geometry& geometry,
                                    NavierStokes<P>& flow, const Transport& transport, bool ft2,
                                    const kernels::GlobalKeys* keys)
    : impl_(std::make_unique<Impl>(mesh, geometry, flow, transport, ft2, keys)) {}

template <class P>
SpalartAllmaras<P>::~SpalartAllmaras() = default;

template <class P>
void SpalartAllmaras<P>::set_ghosts(GhostFill* ghosts) noexcept {
  impl_->ghost_fill = ghosts;
}

template <class P>
void SpalartAllmaras<P>::set_boundary(std::span<const double> nt) {
  auto& m = *impl_;
  if (nt.size() != m.nf - m.ni) throw std::invalid_argument("nt array does not match the faces");
  for (std::size_t b = 0; b < nt.size(); ++b) m.nt_bc[m.ni + b] = State(nt[b]);
}

template <class P>
void SpalartAllmaras<P>::set_wall_distance(std::span<const double> d) {
  auto& m = *impl_;
  if (d.size() != m.nc) throw std::invalid_argument("wall distance does not match the cells");
  for (std::size_t c = 0; c < m.nc; ++c) {
    if (!(d[c] > 0.0)) throw std::invalid_argument("wall distance must be positive");
    m.wall_distance[c] = static_cast<typename decltype(m.wall_distance)::value_type>(d[c]);
  }
}

template <class P>
void SpalartAllmaras<P>::set_source(std::span<const double> source) {
  auto& m = *impl_;
  if (source.size() != m.nc) throw std::invalid_argument("source does not match the cells");
  for (std::size_t c = 0; c < m.nc; ++c) m.source[c] = State(source[c]);
}

template <class P>
void SpalartAllmaras<P>::residual(std::span<const State> s, std::span<Res> res) {
  auto& m = *impl_;
  m.prepare(s, res);
  sk::cpu::nt_sources(m.d);
  m.run([&](auto... strategy) { sk::cpu::nt_faces(m.d, strategy...); });
  m.run([&](auto... strategy) { sk::cpu::nt_boundaries(m.d, strategy...); });
}

template <class P>
void SpalartAllmaras<P>::residual_and_jacobian(std::span<const State> s, std::span<Res> res) {
  auto& m = *impl_;
  m.prepare(s, res);
  sk::cpu::nt_sources_jacobian(m.d);
  m.run([&](auto... strategy) { sk::cpu::nt_faces_jacobian(m.d, strategy...); });
  m.run([&](auto... strategy) { sk::cpu::nt_boundaries_jacobian(m.d, strategy...); });
}

template <class P>
void SpalartAllmaras<P>::assemble(std::span<const double> shift, std::span<double> diag,
                                  std::span<double> upper, std::span<double> lower) const {
  const auto& m = *impl_;
  for (std::size_t c = 0; c < m.nc; ++c) {
    diag[c] = shift[c] + std::max(0.0, static_cast<double>(m.cell_jac[c]));
  }
  if (m.d.topo.rank_independent) {
    // Under global keys the face loops evaluate each face in its global orientation (face_jac:
    // dF/d(rho nt) of the global left, block 0, and right, block 1), each diagonal entry gathers
    // its cell's faces in Topology::cell_faces order (the global order), and the off-diagonal
    // entries are placed in the local orientation, as Euler<P>::assemble.
    const auto& t = m.d.topo;
    auto left = [&](std::size_t f) {
      const bool rev = t.face_reversed != nullptr && t.face_reversed[f] != 0;
      return static_cast<std::size_t>(rev ? t.neighbour[f] : t.owner[f]);
    };
    for (std::size_t c = 0; c < m.nc; ++c) {
      for (auto k = t.cell_face_offsets[c]; k < t.cell_face_offsets[c + 1]; ++k) {
        const auto f = static_cast<std::size_t>(t.cell_faces[k]);
        if (f >= m.ni) {
          diag[c] += static_cast<double>(m.bnd_jac[f]);
        } else if (left(f) == c) {
          diag[c] += static_cast<double>(m.face_jac[f]);
        } else {
          diag[c] -= static_cast<double>(m.face_jac[m.nf + f]);
        }
      }
    }
    for (std::size_t f = 0; f < m.ni; ++f) {
      const bool rev = left(f) != static_cast<std::size_t>(t.owner[f]);
      const double jl = static_cast<double>(m.face_jac[f]);
      const double jr = static_cast<double>(m.face_jac[m.nf + f]);
      // A(left, right) = jr, A(right, left) = -jl.
      upper[f] = rev ? -jl : jr;
      lower[f] = rev ? jr : -jl;
    }
    return;
  }
  // R_o += F, R_n -= F; F depends on rho nt of the owner (block 0) and the neighbour (block 1).
  for (std::size_t f = 0; f < m.ni; ++f) {
    const auto o = static_cast<std::size_t>(m.owner[f]);
    const auto n = static_cast<std::size_t>(m.neighbour[f]);
    const double jl = static_cast<double>(m.face_jac[f]);
    const double jr = static_cast<double>(m.face_jac[m.nf + f]);
    diag[o] += jl;
    upper[f] = jr;
    lower[f] = -jl;
    diag[n] -= jr;
  }
  for (std::size_t f = m.ni; f < m.nf; ++f) {
    diag[static_cast<std::size_t>(m.owner[f])] += static_cast<double>(m.bnd_jac[f]);
  }
}

template <class P>
void SpalartAllmaras<P>::solve(std::span<const double> diag, std::span<const double> upper,
                               std::span<const double> lower, std::span<const double> b,
                               std::span<double> dx, int sweeps,
                               const std::vector<std::vector<std::size_t>>& colours,
                               const std::function<void(std::span<double>)>& exchange) const {
  const auto& m = *impl_;
  // The global distance-1 colouring: no two cells of a colour are neighbours, so each colour
  // relaxes in any order, and the sweeps are those of one rank in any cell order. Each row sums
  // its faces in Topology::cell_faces order (the global order under keys).
  const auto& t = m.d.topo;
  auto relax_row = [&](std::size_t c) {
    double r = b[c];
    for (auto k = t.cell_face_offsets[c]; k < t.cell_face_offsets[c + 1]; ++k) {
      const auto f = static_cast<std::size_t>(t.cell_faces[k]);
      if (f >= m.ni) continue;
      const auto o = static_cast<std::size_t>(t.owner[f]);
      const auto n = static_cast<std::size_t>(t.neighbour[f]);
      r -= o == c ? upper[f] * dx[n] : lower[f] * dx[o];
    }
    dx[c] = r / diag[c];
  };
  auto colour = [&](const std::vector<std::size_t>& cells) {
    for (const auto c : cells) relax_row(c);
    if (exchange) exchange(dx);
  };
  for (int s = 0; s < sweeps; ++s) {
    for (const auto& cells : colours) colour(cells);
    for (auto it = colours.rbegin(); it != colours.rend(); ++it) colour(*it);
  }
}

template <class P>
void SpalartAllmaras<P>::eddy_viscosity(std::span<const State> s) {
  auto& m = *impl_;
  m.share();
  m.d.rnu = const_cast<State*>(s.data());
  const auto h = Impl::halo(m.d);
  sk::cpu::nt_values(h);
  sk::cpu::boundary_values(h);
  sk::cpu::eddy_viscosities(h);
  sk::cpu::boundary_eddy_viscosities(h);
}

template <class P>
const kernels::spalart_allmaras::Data<P>& SpalartAllmaras<P>::data() const {
  return impl_->d;
}

template class SpalartAllmaras<kernels::fp64>;
template class SpalartAllmaras<kernels::fp32_acc64>;

std::vector<std::vector<std::size_t>> colour_groups(const Mesh& mesh,
                                                    std::span<const std::int32_t> colours,
                                                    std::int32_t n_colours, std::size_t owned) {
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  if (colours.size() != nc || n_colours < 0 || owned > nc) {
    throw std::invalid_argument("E0720: the colouring does not give one colour per cell");
  }
  for (std::size_t c = 0; c < nc; ++c) {
    if (colours[c] < 0 || colours[c] >= n_colours) {
      throw std::invalid_argument("E0720: cell " + std::to_string(c) + " has no valid colour");
    }
  }
  for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_interior_faces()); ++f) {
    if (colours[static_cast<std::size_t>(mesh.owner()[f])] ==
        colours[static_cast<std::size_t>(mesh.neighbour()[f])]) {
      throw std::invalid_argument("E0720: interior face " + std::to_string(f) +
                                  " joins two cells of one colour");
    }
  }
  std::vector<std::vector<std::size_t>> groups(static_cast<std::size_t>(n_colours));
  for (std::size_t c = 0; c < owned; ++c) groups[static_cast<std::size_t>(colours[c])].push_back(c);
  return groups;
}

void update_rho_nt(std::span<double> q, std::span<const double> dx, std::span<const double> rho,
                   double nt_inf, double alpha) {
  if (dx.size() < q.size() || rho.size() < q.size()) {
    throw std::invalid_argument("update_rho_nt: dx and rho must cover the updated cells");
  }
  for (std::size_t c = 0; c < q.size(); ++c) {
    const double scale = std::max(std::abs(q[c]), rho[c] * nt_inf);
    q[c] += alpha * std::clamp(dx[c], -sa_max_decrease * scale, sa_max_increase * scale);
  }
}

}  // namespace autan::physics
