// The Euler residual of kernels/euler.py evaluated by the generated Metal loops equals the CPU
// physics, autan::physics::Euler<fp32_acc64>, to rounding: three fluxes, first and second order,
// both Metal policies, every face-loop strategy each policy has. The loop sequence is the one of
// Euler<P>::residual (physics/src/euler.cpp).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "autan/core/geometry.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/colouring.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/kernels/metal.hpp"
#include "autan/physics/euler.hpp"
#include "euler_metal.hpp"

namespace {

namespace fs = std::filesystem;
namespace ek = autan::kernels::euler;
namespace mtl = autan::kernels::metal;
using autan::index_t;
using autan::kernels::offset_t;
using autan::kernels::Strategy;
using autan::physics::BoundaryKind;
using autan::physics::Flux;
using autan::physics::Gas;
using autan::physics::Scheme;

const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
constexpr Gas air{1.4, 287.0};

struct Case {
  autan::Mesh mesh;
  autan::Geometry geometry;
  explicit Case(const char* file)
      : mesh(autan::io::read_mesh(mesh_dir / file)), geometry(autan::compute_geometry(mesh)) {}
  [[nodiscard]] std::size_t n_cells() const { return static_cast<std::size_t>(mesh.n_cells()); }
  [[nodiscard]] std::size_t n_faces() const { return static_cast<std::size_t>(mesh.n_faces()); }
  [[nodiscard]] std::size_t n_interior() const {
    return static_cast<std::size_t>(mesh.n_interior_faces());
  }
};

/// Boundary conditions as Euler::set_boundary takes them (kind and 5 x n_boundary data).
struct Boundaries {
  std::vector<BoundaryKind> kind;
  std::vector<double> data;
};

// Every boundary kind in turn (first order), or given states only (second order: Euler<P> then
// keeps the stencil's least-squares weights, which this test reproduces).
Boundaries boundaries(const Case& c, bool all_kinds, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> jitter(-1.0, 1.0);
  const std::size_t nb = c.n_faces() - c.n_interior();
  Boundaries b{std::vector<BoundaryKind>(nb), std::vector<double>(5 * nb, 0.0)};
  for (std::size_t i = 0; i < nb; ++i) {
    const auto kind = all_kinds ? static_cast<BoundaryKind>(i % 5) : BoundaryKind::state;
    b.kind[i] = kind;
    std::array<double, 5> v{};
    switch (kind) {
      case BoundaryKind::state:
      case BoundaryKind::farfield:
        v = {1.2 + 0.1 * jitter(rng), 80.0 + 20.0 * jitter(rng), 10.0 * jitter(rng),
             10.0 * jitter(rng), 1e5 * (1.0 + 0.05 * jitter(rng))};
        break;
      case BoundaryKind::inlet:
        v = {1.1e5, 300.0, 0.0, 0.0, 0.0};
        break;
      case BoundaryKind::outlet:
        v = {0.0, 0.0, 0.0, 0.0, 1e5};
        break;
      case BoundaryKind::slip:
        break;
    }
    for (std::size_t k = 0; k < 5; ++k) b.data[k * nb + i] = v[k];
  }
  return b;
}

/// Conservative states 5 x n_cells around a subsonic stream, varying from cell to cell.
std::vector<double> states(const Case& c, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> jitter(-1.0, 1.0);
  const std::size_t n = c.n_cells();
  std::vector<double> u(5 * n);
  for (std::size_t i = 0; i < n; ++i) {
    const double rho = 1.2 + 0.1 * jitter(rng);
    const std::array<double, 3> v = {80.0 + 20.0 * jitter(rng), 10.0 * jitter(rng),
                                     10.0 * jitter(rng)};
    const double p = 1e5 * (1.0 + 0.05 * jitter(rng));
    u[i] = rho;
    for (std::size_t k = 0; k < 3; ++k) u[(k + 1) * n + i] = rho * v[k];
    u[4 * n + i] = p / (air.gamma - 1.0) + 0.5 * rho * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }
  return u;
}

Scheme scheme(Flux flux, bool second_order) {
  Scheme s;
  s.flux = flux;
  s.second_order = second_order;
  s.mach_ref = 0.3;
  s.scale = {1.0, 100.0, 100.0, 100.0, 1e5};
  s.reference = {1.0, 0.0, 0.0, 0.0, 1e5 / (air.gamma - 1.0)};  // at rest, p_ref = 1e5
  return s;
}

std::vector<double> cpu_residual(const Case& c, const Scheme& s, const Boundaries& b,
                                 const std::vector<double>& u) {
  using P = autan::kernels::fp32_acc64;
  autan::physics::Euler<P> e(c.mesh, c.geometry, air, s);
  e.set_boundary(b.kind, b.data);
  std::vector<typename P::state> us(u.size());
  e.state(u, us);
  std::vector<typename P::residual> r(us.size());
  e.residual(us, r);
  return {r.begin(), r.end()};
}

/// Shared buffers kept alive while the context runs.
class Buffers {
 public:
  explicit Buffers(mtl::Context& ctx) : ctx_(ctx) {}
  template <class T, class V>
  T* put(const std::vector<V>& v) {
    std::vector<T> t(v.size());
    std::transform(v.begin(), v.end(), t.begin(), [](const V& x) { return T(x); });
    keep_.push_back(ctx_.upload(t));
    return keep_.back().template device<T>();
  }
  template <class T>
  T* zeros(std::size_t n) {
    keep_.push_back(ctx_.allocate(n * sizeof(T)));
    return keep_.back().template device<T>();
  }
  template <class T>
  [[nodiscard]] const T* host(const T* device) const {
    for (const auto& b : keep_) {
      if (b.template device<T>() == device) return b.template data<T>();
    }
    return nullptr;
  }

 private:
  mtl::Context& ctx_;
  std::vector<mtl::Buffer> keep_;
};

/// The sequence of Euler<P>::residual on the Metal loops, in Metal policy M.
template <class M>
std::vector<double> metal_residual(mtl::Context& ctx, const Case& c, const Scheme& s,
                                   const Boundaries& b, const std::vector<double>& u,
                                   Strategy strategy) {
  using S = typename M::state;
  using G = typename M::geom;
  using R = typename M::residual;
  const auto packed = autan::kernels::pack_geometry(c.mesh, c.geometry);
  const auto stencil =
      autan::kernels::cell_stencil(c.mesh, packed, autan::kernels::StencilKind::face2);
  const auto host_topo = autan::kernels::topology(c.mesh.view());
  const auto nc = c.n_cells(), nf = c.n_faces(), ni = c.n_interior(), nb = nf - ni;
  const auto copy = [](const auto* p, std::size_t n) { return std::vector(p, p + n); };
  Buffers buf(ctx);
  auto topo = host_topo;
  topo.owner = buf.put<index_t>(copy(host_topo.owner, nf));
  topo.neighbour = buf.put<index_t>(copy(host_topo.neighbour, ni));
  topo.cell_face_offsets = buf.put<offset_t>(copy(host_topo.cell_face_offsets, nc + 1));
  topo.cell_faces = buf.put<index_t>(
      copy(host_topo.cell_faces, static_cast<std::size_t>(host_topo.n_incidences)));
  topo.n_stencil = static_cast<offset_t>(stencil.entries.size());
  topo.stencil_offsets = buf.put<offset_t>(stencil.offsets);
  topo.stencil = buf.put<index_t>(stencil.entries);
  topo.stencil_twin = buf.put<offset_t>(stencil.twin);
  const auto interior =
      autan::kernels::colour_faces(host_topo, 0, host_topo.n_owned_interior_faces);
  const auto boundary =
      autan::kernels::colour_faces(host_topo, host_topo.n_interior_faces, host_topo.n_owned_faces);
  mtl::Schedule sched;
  sched.block = 128;
  sched.interior = {buf.put<index_t>(interior.faces), interior.offsets};
  sched.boundary = {buf.put<index_t>(boundary.faces), boundary.offsets};
  sched.face_scratch = buf.zeros<typename M::accum>(5 * nf);
  sched.face_scratch_width = 5;

  std::vector<double> kind(nf, 0.0), bc(5 * nf, 0.0);
  for (std::size_t i = 0; i < nb; ++i) {
    kind[ni + i] = static_cast<double>(b.kind[i]);
    for (std::size_t k = 0; k < 5; ++k) bc[k * nf + ni + i] = b.data[k * nb + i];
  }
  ek::Data<M> d{};
  d.topo = topo;
  // Stored state U - U_ref and gauge pressure p - p_ref (Scheme::reference).
  std::vector<double> stored(u.size());
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t i = 0; i < nc; ++i) stored[k * nc + i] = u[k * nc + i] - s.reference[k];
  }
  d.u = buf.put<S>(stored);
  d.prim = buf.zeros<S>(5 * nc);
  d.bc_kind = buf.put<G>(kind);
  d.bc = buf.put<S>(bc);
  d.prim_b = buf.zeros<S>(5 * nf);
  d.grad = buf.zeros<typename M::grad>(15 * nc);
  d.wmin = buf.zeros<S>(5 * nc);
  d.wmax = buf.zeros<S>(5 * nc);
  d.phi = buf.put<S>(std::vector<double>(5 * nc, s.second_order && !s.limiter ? 1.0 : 0.0));
  d.source = buf.zeros<S>(5 * nc);
  d.res = buf.zeros<R>(5 * nc);
  d.face_normal = buf.put<G>(packed.face_normal);
  d.face_area_magnitude = buf.put<G>(packed.face_area_magnitude);
  d.face_centroid = buf.put<G>(packed.face_centroid);
  d.face_moment = buf.put<G>(packed.face_moment);
  d.cell_centroid = buf.put<G>(packed.cell_centroid);
  d.cell_volume = buf.put<G>(packed.cell_volume);
  d.stencil_lsq_weight = buf.put<G>(stencil.lsq_weight);
  d.gamma[0] = typename M::param(air.gamma);
  d.gas_constant[0] = typename M::param(air.gas_constant);
  d.mach_ref[0] = typename M::param(s.mach_ref);
  const auto& r0 = s.reference;
  for (std::size_t k = 0; k < 5; ++k) d.u_ref[k] = typename M::param(r0[k]);
  d.p_ref[0] = typename M::param(
      (air.gamma - 1.0) * (r0[4] - 0.5 * (r0[1] * r0[1] + r0[2] * r0[2] + r0[3] * r0[3]) / r0[0]));
  const double k3 = std::pow(s.venkat_k / s.length, 3);
  for (std::size_t k = 0; k < 5; ++k) {
    d.limiter_scale[k] = typename M::param(k3 * s.scale[k] * s.scale[k]);
  }

  namespace ml = ek::metal_loops;
  ml::primitives(ctx, d, sched);
  ml::boundary_states(ctx, d, sched);
  if (s.second_order) {
    ml::gradients(ctx, d, sched);
    if (s.limiter) ml::limiter(ctx, d, sched);
  }
  ml::sources(ctx, d, sched);
  switch (s.flux) {
    case Flux::roe:
      ml::roe_faces(ctx, d, strategy, sched);
      ml::roe_boundaries(ctx, d, strategy, sched);
      break;
    case Flux::hllc:
      ml::hllc_faces(ctx, d, strategy, sched);
      ml::hllc_boundaries(ctx, d, strategy, sched);
      break;
    case Flux::ausm_up:
      ml::ausm_faces(ctx, d, strategy, sched);
      ml::ausm_boundaries(ctx, d, strategy, sched);
      break;
  }
  ctx.synchronise();
  const R* r = buf.host(d.res);
  std::vector<double> out(5 * nc);
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<double>(r[i]);
  return out;
}

mtl::Context& context() {
  static mtl::Context ctx;
  return ctx;
}

}  // namespace

TEMPLATE_TEST_CASE("the Euler residual on Metal equals the CPU physics to rounding",
                   "[euler][metal]", mtl::fp32, mtl::fp32_acc64) {
  using M = TestType;
  const Flux flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  const bool second_order = GENERATE(false, true);
  // Atomic strategies need atomics on the accumulation type: none on double-float (E1108).
  std::vector<Strategy> strategies{Strategy::coloured, Strategy::store_gather,
                                   Strategy::gather_recompute};
  if (M::float_atomics) {
    strategies.insert(strategies.end(), {Strategy::atomic, Strategy::warp_aggregated});
  }
  const Strategy strategy = GENERATE_COPY(from_range(strategies));
  const Case c("hybrid/hybrid.ascii.msh");
  std::mt19937_64 rng(7);
  const auto b = boundaries(c, !second_order, rng);
  const auto u = states(c, rng);
  const auto s = scheme(flux, second_order);
  const auto host = cpu_residual(c, s, b, u);
  const auto dev = metal_residual<M>(context(), c, s, b, u, strategy);
  double scale = 0.0, diff = 0.0;
  for (std::size_t i = 0; i < host.size(); ++i) {
    REQUIRE(std::isfinite(dev[i]));
    scale = std::max(scale, std::abs(host[i]));
    diff = std::max(diff, std::abs(dev[i] - host[i]));
  }
  REQUIRE(scale > 0.0);
  // FP32 bodies on both sides; sums in FP32 (fp32) or double-float vs FP64 (fp32+acc64).
  CHECK(diff <= 500.0 * std::numeric_limits<float>::epsilon() * scale);
}
