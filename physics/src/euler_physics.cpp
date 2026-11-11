#include "autan/physics/euler_physics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

#include "autan/core/lines.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/physics/euler.hpp"
#include "autan/physics/euler_device.hpp"
#include "autan/physics/first_order_start.hpp"
#include "autan/physics/manufactured.hpp"
#include "autan/physics/navier_stokes.hpp"
#include "autan/physics/navier_stokes_device.hpp"
#include "autan/physics/point_vortex.hpp"
#include "autan/physics/spalart_allmaras.hpp"
#include "autan/runtime/exact_sum.hpp"
#include "autan/runtime/round_off.hpp"
#include "autan/solvers/agglomeration.hpp"
#include "autan/solvers/line_smoother.hpp"
#include "autan/solvers/multigrid.hpp"

namespace autan::physics {

namespace {

using runtime::Loads;
using runtime::PhysicsContext;
using runtime::StepReport;

constexpr std::array<const char*, 6> field_names = {"Density",   "VelocityX", "VelocityY",
                                                    "VelocityZ", "Pressure",  "Temperature"};
// SIDS names of the conserved variables.
constexpr std::array<const char*, 5> conserved_names = {"Density", "MomentumX", "MomentumY",
                                                        "MomentumZ", "EnergyStagnationDensity"};

// Steady runs: pseudo-transient continuation, (V / dt + dR/dU) du = -R with the approximate
// Jacobian, local dt = CFL V / lambda, the CFL number following the residual (switched evolution
// relaxation, Mulder and van Leer 1985, J. Comput. Phys. 59:232-246). The multigrid solves the
// step's system at any CFL number, so the start is large: the pseudo-time step then no longer
// scales with the viscous time step h^2 (docs/design/multigrid.md, measurements).
constexpr double cfl_start = 1000.0;
constexpr double cfl_min = 1.0;
constexpr double cfl_max = 1e6;
// The penalty also halves after a step whose largest relative residual rose by more than this
// factor over the step before's: the defect correction's approximate Jacobian stops converging
// above a case-dependent CFL number, which no constant bound tracks (TMR flat plate, #185: a
// bound of 2000 converges, 5000 diverges, none stalls at 1.5e4). A margin of 10 % lets the small
// rises of a converging run recover; halving on every rise ratchets the penalty down and stalls.
constexpr double cfl_rise = 1.1;
// Each step's linear system is solved to this relative residual (inexact Newton), with at most
// this many multigrid-preconditioned iterations.
constexpr double linear_tolerance = 0.1;
constexpr int linear_max_cycles = 10;
// linear_solver = "gauss-seidel": symmetric sweeps over the global colours per implicit step;
// "lines": as many line sweeps per iteration of the GCR outer iteration.
constexpr int gauss_seidel_sweeps = 4;
// A solve that leaves more than this fraction of the residual has failed (the multigrid cycle can
// diverge on a system the CFL number made too weakly diagonal): the step is rejected.
constexpr double linear_failure = 0.9;
// No update may lower density or pressure by more than this fraction in one step; a step
// shortened below min_step is rejected.
constexpr double max_decrease = 0.5;
constexpr double min_step = 1e-6;
// Largest relative decrease and increase of rho nt in one step, and the symmetric Gauss-Seidel
// sweeps of its scalar system (SA).
constexpr double nt_max_decrease = 0.5;
constexpr double nt_max_increase = 2.0;
constexpr int nt_sweeps = 4;
constexpr double venkatakrishnan_k = 10.0;
// Steady cases on a device: SSP-RK3 in pseudo-time with local time steps CFL V / lambda.
constexpr double device_steady_cfl = 0.8;

Gas gas_of(const runtime::SolverInput& in) {
  std::optional<Gas> gas;
  for (const auto& z : in.cell_zones) {
    if (z.medium.kind != "ideal-gas") {
      throw std::invalid_argument("cell zone '" + z.zone +
                                  "': the Euler physics takes ideal gases only");
    }
    const Gas g{*z.medium.gamma, *z.medium.gas_constant};
    if (gas && (gas->gamma != g.gamma || gas->gas_constant != g.gas_constant)) {
      throw std::invalid_argument("the Euler physics takes a single gas for every cell zone");
    }
    gas = g;
  }
  if (!gas) throw std::invalid_argument("no fluid cell zone");
  return *gas;
}

Transport transport_of(const runtime::SolverInput& in) {
  const auto& m = in.cell_zones.front().medium;
  const auto& v = *m.viscosity;
  Transport t;
  t.sutherland = v.law == "sutherland";
  t.mu_ref = t.sutherland ? *v.mu_ref : *v.mu;
  if (t.sutherland) {
    t.t_ref = *v.t_ref;
    t.s = *v.s;
  }
  t.prandtl = *m.prandtl;
  for (const auto& z : in.cell_zones) {
    const auto& w = *z.medium.viscosity;
    if (w.law != v.law || w.mu_ref != v.mu_ref || w.t_ref != v.t_ref || w.s != v.s ||
        w.mu != v.mu || z.medium.prandtl != m.prandtl) {
      throw std::invalid_argument(
          "the Navier-Stokes physics takes a single gas for every cell zone");
    }
  }
  return t;
}

// Spalart-Allmaras (SA-neg) beside a NavierStokes<P> flow: its state rho nt in FP64 and the
// arrays of its loosely coupled implicit step.
template <class P>
struct SaState {
  SaState(const Mesh& mesh, const Geometry& geometry, NavierStokes<P>& flow, const Transport& t,
          bool ft2, const kernels::GlobalKeys* keys)
      : model(mesh, geometry, flow, t, ft2, keys) {}
  SpalartAllmaras<P> model;
  std::vector<double> rnu, rhs, dx, diag, upper, lower;
  // Under global keys or on a partition: the owned cells by global colour, which solve() sweeps
  // with an exchange after each, the same sweeps on any number of ranks.
  std::vector<std::vector<std::size_t>> colours;
  double nt_inf = 0.0;  ///< m^2/s, the freestream's
  std::vector<typename P::state> s;
  std::vector<typename P::residual> res;

  std::span<const typename P::state> stored(const std::vector<double>& v) {
    for (std::size_t c = 0; c < v.size(); ++c) s[c] = typename P::state(v[c]);
    return s;
  }
};

template <class P, class Flow>
class EulerPhysics final : public runtime::Physics {
 public:
  using State = typename P::state;
  using Res = typename P::residual;
  static constexpr bool is_viscous = !std::is_same_v<Flow, Euler<P>>;

  // The ghost stencil the flow's scheme reads (PhysicsRegistry::Entry::ghosts).
  static std::optional<GhostStencil> ghosts(const runtime::SolverInput& in) {
    if constexpr (is_viscous) {
      return navier_stokes_ghost_stencil(in.reproducibility == "ranks");
    } else {
      return ghost_stencil(flow_scheme(in));
    }
  }

  explicit EulerPhysics(const PhysicsContext& ctx)
      : exempt_(runtime::residual_names(ctx.input), ctx.input.start && ctx.input.start->woken
                                                        ? *ctx.input.start->woken
                                                        : std::vector<std::string>{}),
        ctx_(ctx),
        gas_(gas_of(ctx.input)),
        n_(static_cast<std::size_t>(ctx.mesh.n_cells())),
        owned_(static_cast<std::size_t>(ctx.mesh.n_owned_cells())),
        ghost_fill_(ctx.ghosts),
        keys_(global_keys(ctx)),
        euler_(make_flow(ctx, gas_, keys_ ? &*keys_ : nullptr)),
        reference_(scheme(ctx.input).reference) {
    // On every rank: the exchange is collective, even for a part without ghosts.
    euler_.set_ghosts(&ghost_fill_);
    // The loops without the warped-face terms where every face of every rank is planar: one
    // choice for the run, so that ranks evaluate their faces alike (#412). Inviscid only: the
    // planar boundary loops have no no-slip wall flux (#540).
    if constexpr (!is_viscous) {
      if (ctx.ranks.min(euler_.all_planar() ? 1.0 : 0.0) > 0.0) {
        euler_.set_planar(true);
        ctx.log.write("info", "solver",
                      "every face is planar: residual loops without the "
                      "warped-face terms",
                      {}, "{}");
      }
    }
    const auto& mesh = ctx.mesh;
    const auto& in = ctx.input;
    std::optional<Manufactured> mms;
    if (in.verification) mms.emplace(in.verification->text);
    set_boundaries(mms ? &*mms : nullptr);
    // A steady case on a device steps explicitly there (device_step): no host Jacobian, no linear
    // solver.
    if (in.time.kind == "steady" && ctx.device >= 0) {
      euler_.set_second_order(false);
    } else if (in.time.kind == "steady") {
      matrix_.emplace(mesh);
      // Under reproducibility "ranks", each row sums its neighbours in global face order.
      if (keys_) matrix_->order_rows(*keys_);
      if (in.time.linear_solver == "multigrid") {
        multigrid_.emplace(mesh, ctx.geometry);
        nlohmann::json nodes = nlohmann::json::array(), edges = nlohmann::json::array();
        for (const auto& l : multigrid_->levels()) {
          nodes.push_back(l.n_nodes);
          edges.push_back(l.owner.size());
        }
        ctx.log.write("info", "linear", "agglomeration multigrid", {},
                      nlohmann::json{{"nodes", nodes},
                                     {"edges", edges},
                                     {"coarsest", multigrid_->direct_coarsest() ? "lu" : "sweeps"}}
                          .dump());
      } else if (in.time.linear_solver == "lines") {
        // The multigrid's fine lines, kept whole by the partition, swept over the runtime's
        // colouring of the lines by global id: the same sweeps on any number of ranks.
        // A mesh without lines has no groups: every cell is its own, coloured as cell_colours.
        const bool grouped = !ctx.cell_group_colours.empty();
        // Each cell sums its couplings by global face id: the same sums on any partition.
        lines_.emplace(mesh, ctx.group_offsets, ctx.group_cells,
                       grouped ? ctx.cell_group_colours : ctx.cell_colours,
                       grouped ? ctx.n_group_colours : ctx.n_colours, ctx.face_global_id);
        ctx.log.write("info", "linear", "line-implicit block Gauss-Seidel", {},
                      nlohmann::json{{"colours", lines_->n_colours()},
                                     {"sweeps", gauss_seidel_sweeps},
                                     {"iterations", linear_max_cycles}}
                          .dump());
      } else {
        // The runtime's colouring by global id: the same sweeps on any number of ranks.
        colours_.assign(static_cast<std::size_t>(ctx.n_colours), {});
        for (std::size_t c = 0; c < owned_; ++c) {
          colours_[static_cast<std::size_t>(ctx.cell_colours[c])].push_back(c);
        }
        ctx.log.write(
            "info", "linear", "multicolour block Gauss-Seidel", {},
            nlohmann::json{{"colours", ctx.n_colours}, {"sweeps", gauss_seidel_sweeps}}.dump());
      }
      euler_.set_second_order(false);
    } else {
      second_order_ = true;
    }
    // case-schema.md#time: start_order "second" skips the first-order start
    // (euler.md#time-stepping).
    if (in.time.kind == "steady" && in.time.start_order == "second") {
      second_order_ = true;
      euler_.set_second_order(true);
      ctx.log.write("info", "solver", "the steady run starts at second order (time.start_order)",
                    {}, nlohmann::json{{"start_order", "second"}}.dump());
    }
    if (mms) set_source(*mms);
    std::map<std::string, const runtime::CellZoneInput*> zones;
    for (const auto& z : in.cell_zones) zones[z.zone] = &z;
    u_.assign(5 * n_, 0.0);
    for (std::size_t c = 0; c < owned_; ++c) {
      const auto zi = static_cast<std::size_t>(mesh.cell_zone()[c]);
      const auto& init = zones.at(mesh.cell_zones()[zi].name)->initial;
      set_cell(c, *init.density, *init.velocity, *init.pressure);
    }
    if (ctx.initial != nullptr) start_from_initial(*ctx.initial);
    state_.assign(5 * n_, State(0));
    res_.assign(5 * n_, Res(0));
    lambda_.assign(n_, State(0));
    if (ctx.device < 0) {
      stage_.assign(5 * n_, 0.0);
      rhs_.assign(5 * n_, 0.0);
      du_.assign(5 * n_, 0.0);
    }
    mean_.assign(6 * n_, 0.0);
    if constexpr (is_viscous) {
      if (in.turbulence.model == "sa") init_sa(mms ? &*mms : nullptr);
    }
    if (ctx.start != nullptr) start_from(*ctx.start);
    if (in.start && in.start->residual_reference) exempt_.reference(*in.start->residual_reference);
    apply_vortices();
    if (ctx.device >= 0) {
#ifdef AUTAN_PHYSICS_CUDA
      {
        // Euler<P>'s or NavierStokes<P>'s (laminar only: SA has no device entry, E0706).
        device_ = make_device_stepper<P>(mesh, ctx.geometry, euler_, gas_.gamma, second_order_,
                                         ctx.ranks, ctx.ghosts, ctx.stream);
        device_->upload(u_);
        if constexpr (is_viscous) {
          const bool exp_log = std::ranges::any_of(in.boundaries, [](const auto& b) {
            return b.kind == "farfield" || b.kind == "inlet";
          });
          if (in.precision == "fp32+acc64" && exp_log) {
            const double gain = device_stencil_gain(euler_.euler(), mesh, ctx.geometry);
            if (device_gain_warns(true, exp_log, gain)) {
              ctx.log.warn("W0717",
                           "the device comparison on this combination is not covered by a tight "
                           "bound (#586 2(b): G beyond 4.79, the largest G at which tightness was "
                           "measured)",
                           nlohmann::json{{"G", gain},
                                          { "G_covered",
                                            device_covered_gain }}
                               .dump());
            }
          }
        }
        if (in.time.kind == "steady") {
          ctx.log.write("warning", "solver",
                        "steady case on the device: explicit SSP-RK3 with local time steps, not "
                        "the implicit solver",
                        {}, nlohmann::json{{
                              "cfl",
                              device_steady_cfl
                            }}.dump());
        }
        if (!in.monitors.empty()) {
          ctx.log.write("warning", "solver",
                        "force monitors on the device: each recorded iteration downloads the state "
                        "and evaluates the residual on the host",
                        {}, nlohmann::json{{
                              "monitors",
                              in.monitors.size()
                            }}.dump());
        }
      }
#else
      throw std::logic_error("the Euler physics was built without CUDA");
#endif
    }
  }

  StepReport step(std::int64_t /*iteration*/) override {
    pressure_valid_ = false;
    // A steady step that solves the first-order start is provisional (case-schema.md#time).
    const bool provisional = ctx_.input.time.kind == "steady" && !second_order_;
    StepReport report = device_ ? device_step() : host_step();
    report.provisional = provisional;
    return report;
  }

  StepReport host_step() {
    StepReport report;
    // A state the kernels cannot evaluate (as stored, see first_invalid_cell) diverges here,
    // before a solve turns it into an exception; on every rank, also those that own no such cell.
    const auto bad = first_invalid_cell();
    if (ctx_.ranks.max(bad ? 1.0 : 0.0) > 0.0) {
      residual_of(u_);
      record_norms(report);
      report.diverged_cell = bad;
      return report;
    }
    if (ctx_.input.time.kind == "steady") {
      steady_step(report);
    } else {
      unsteady_step(report);
    }
    for (auto& v : vortices_) v.circulation = v.next;
    apply_vortices();
    if (!report.diverged_cell) report.diverged_cell = first_invalid_cell();
    return report;
  }

  void accumulate_average(double weight) override {
    sync();
    const auto f = primitive_fields();
    for (std::size_t k = 0; k < 6; ++k) {
      for (std::size_t c = 0; c < owned_; ++c) mean_[k * n_ + c] += weight * f[k][c];
    }
    mean_weight_ += weight;
  }

  // The conserved variables bit for bit, then the averaging sums and their weight.
  [[nodiscard]] std::vector<io::Field> restart_fields() const override {
    sync();
    std::vector<io::Field> out;
    for (std::size_t k = 0; k < 5; ++k) {
      out.push_back({.name = conserved_names[k], .values = owned(u_, k)});
    }
    if (mean_weight_ > 0.0) {
      for (std::size_t k = 0; k < 6; ++k) {
        out.push_back(
            {.name = std::string("AverageSum") + field_names[k], .values = owned(mean_, k)});
      }
      out.push_back({.name = "AverageWeight", .values = std::vector<double>(owned_, mean_weight_)});
    }
    if (sa_) out.push_back({.name = "DensityNuTilde", .values = owned(sa_->rnu, 0)});
    // The steady solver's own state, one uniform field per value (cell-centred, as RestartState).
    if (ctx_.input.time.kind == "steady") {
      for (const auto& [name, value] : solver_state()) {
        out.push_back({.name = name, .values = std::vector<double>(owned_, value)});
      }
    }
    for (std::size_t i = 0; i < vortices_.size(); ++i) {
      out.push_back({.name = "VortexCirculation" + std::to_string(i),
                     .values = std::vector<double>(owned_, vortices_[i].circulation)});
    }
    return out;
  }

  // Phi_k = f_k sqrt(sum_i A_i^2 / N), A_i the surface of cell i and f_k the face flux magnitude
  // of the freestream with q = |v| + c: rho q, rho q^2 (each momentum) and rho H q
  // (docs/design/euler.md, Precision); SA's rho nt q with the freestream nt.
  [[nodiscard]] std::vector<double> residual_scales() const override {
    const auto& mesh = ctx_.mesh;
    const auto& g = ctx_.geometry;
    std::vector<double> surface(n_, 0.0);
    if (keys_) {
      // Each cell's faces in global order: the same bits on any number of ranks.
      const auto faces = kernels::global_order(mesh, *keys_).cell_faces;
      const auto off = mesh.cell_face_offsets();
      for (std::size_t c = 0; c < owned_; ++c) {
        for (auto k = off[c]; k < off[c + 1]; ++k) {
          surface[c] += g.face_area[static_cast<std::size_t>(faces[static_cast<std::size_t>(k)])];
        }
      }
    } else {
      for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_faces()); ++f) {
        surface[static_cast<std::size_t>(mesh.owner()[f])] += g.face_area[f];
        if (f < static_cast<std::size_t>(mesh.n_interior_faces())) {
          surface[static_cast<std::size_t>(mesh.neighbour()[f])] += g.face_area[f];
        }
      }
    }
    // Over the owned cells of every rank, as the history's norms.
    std::array<runtime::ExactSum, 2> sums;
    for (std::size_t c = 0; c < owned_; ++c) {
      sums[0].add(surface[c] * surface[c]);
      sums[1].add(1.0);
    }
    ctx_.ranks.sum(sums);
    const double area = std::sqrt(sums[0].value() / sums[1].value());
    const auto& fs = ctx_.input.freestream;
    const double c = std::sqrt(gas_.gamma * fs.pressure / fs.density);
    const double q = fs.speed + c;
    const double h = c * c / (gas_.gamma - 1.0) + 0.5 * fs.speed * fs.speed;
    const double mass = fs.density * q * area;
    const double momentum = mass * q;
    if (sa_) return {mass, momentum, momentum, momentum, mass * h, mass * sa_->nt_inf};
    return {mass, momentum, momentum, momentum, mass * h};
  }

  [[nodiscard]] std::vector<io::Field> fields(bool mean) const override {
    sync();
    std::vector<io::Field> out;
    const auto f = primitive_fields();
    for (std::size_t k = 0; k < 6; ++k) {
      io::Field field{.name = field_names[k], .values = f[k]};
      if (mean) {
        for (std::size_t c = 0; c < owned_; ++c) field.values[c] = mean_[k * n_ + c] / mean_weight_;
      }
      out.push_back(std::move(field));
    }
    if (sa_) {
      if (mean) throw std::invalid_argument("time averages of SA runs are not supported");
      io::Field nt{.name = "TurbulentSANuTilde", .values = std::vector<double>(owned_)};
      for (std::size_t c = 0; c < owned_; ++c) nt.values[c] = sa_->rnu[c] / u_[c];
      out.push_back(std::move(nt));
    }
    return out;
  }

  [[nodiscard]] std::vector<io::Field> wall_fields(std::span<const index_t> faces) const override {
    std::vector<double> p(static_cast<std::size_t>(ctx_.mesh.n_boundary_faces()));
    face_pressure(p, {});
    const auto t = face_traction();
    const auto& g = ctx_.geometry;
    const auto ni = static_cast<std::size_t>(ctx_.mesh.n_interior_faces());
    const auto nb = static_cast<std::size_t>(ctx_.mesh.n_boundary_faces());
    std::vector<io::Field> out{{.name = "Pressure", .values = {}},
                               {.name = "WallShearStressX", .values = {}},
                               {.name = "WallShearStressY", .values = {}},
                               {.name = "WallShearStressZ", .values = {}}};
    for (const auto f : faces) {
      const auto i = static_cast<std::size_t>(f);
      const auto b = i - ni;
      out[0].values.push_back(p[b]);
      // The tangential part of the viscous traction (results.md, WallSolution).
      const double a =
          std::sqrt(g.face_area_x[i] * g.face_area_x[i] + g.face_area_y[i] * g.face_area_y[i] +
                    g.face_area_z[i] * g.face_area_z[i]);
      const std::array<double, 3> n{g.face_area_x[i] / a, g.face_area_y[i] / a,
                                    g.face_area_z[i] / a};
      const std::array<double, 3> tv{t[b], t[nb + b], t[2 * nb + b]};
      const double tn = tv[0] * n[0] + tv[1] * n[1] + tv[2] * n[2];
      for (std::size_t k = 0; k < 3; ++k) out[1 + k].values.push_back(tv[k] - tn * n[k]);
    }
    return out;
  }

  [[nodiscard]] Loads loads(std::span<const index_t> zones,
                            const std::array<double, 3>& center) const override {
    // The pressure force exactly as the residual integrates it, minus p_inf S (results.md), plus
    // the viscous force.
    const auto& mesh = ctx_.mesh;
    const auto& g = ctx_.geometry;
    const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
    std::vector<double> p(nb), face_force(3 * nb);
    face_pressure(p, face_force);
    const auto t = face_traction();
    const auto offsets = mesh.boundary_zone_offsets();
    // Exact sums over this rank's faces, then over the ranks: the same bits on any number of
    // ranks for the same face values.
    std::array<runtime::ExactSum, 6> sums;
    for (const auto z : zones) {
      for (auto f = offsets[static_cast<std::size_t>(z)];
           f < offsets[static_cast<std::size_t>(z) + 1]; ++f) {
        const auto i = static_cast<std::size_t>(f);
        const auto force = wall_force(face_force, t, i);
        const std::array<double, 3> arm{g.face_centroid_x[i] - center[0],
                                        g.face_centroid_y[i] - center[1],
                                        g.face_centroid_z[i] - center[2]};
        for (std::size_t k = 0; k < 3; ++k) sums[k].add(force[k]);
        sums[3].add(arm[1] * force[2] - arm[2] * force[1]);
        sums[4].add(arm[2] * force[0] - arm[0] * force[2]);
        sums[5].add(arm[0] * force[1] - arm[1] * force[0]);
      }
    }
    ctx_.ranks.sum(sums);
    Loads loads;
    for (std::size_t k = 0; k < 3; ++k) {
      loads.force[k] = sums[k].value();
      loads.moment[k] = sums[3 + k].value();
    }
    return loads;
  }

 private:
  // Guaranteed copy elision: Flow is neither copyable nor movable.
  static Flow make_flow(const PhysicsContext& ctx, const Gas& gas,
                        const kernels::GlobalKeys* keys) {
    if constexpr (is_viscous) {
      return Flow(ctx.mesh, ctx.geometry, gas, flow_scheme(ctx.input), transport_of(ctx.input),
                  keys, ctx.ghost_stencil);
    } else {
      return Flow(ctx.mesh, ctx.geometry, gas, flow_scheme(ctx.input), keys, ctx.ghost_stencil);
    }
  }

  static Scheme flow_scheme(const runtime::SolverInput& in) {
    Scheme s = scheme(in);
    // The quadratic gradient in every cell: with the linear one near the boundary only, or with
    // the fit to cell averages, ns-mms-order loses its order on tetrahedra or hexahedra
    // (docs/design/euler.md, Quadratic gradient).
    if constexpr (is_viscous) s.gradient_fit = GradientFit::quadratic;
    return s;
  }

  struct Vortex {
    const runtime::BoundaryInput* condition = nullptr;
    std::array<double, 3> center{};
    std::vector<index_t> zones;      // the surfaces whose lift gives the circulation
    double span = 0.0;               // their extent along y
    std::vector<std::size_t> faces;  // its farfield faces (boundary indices)
    double circulation = 0.0, next = 0.0;
  };

  static Scheme scheme(const runtime::SolverInput& in) {
    const auto& f = in.freestream;
    const double c = std::sqrt(*in.cell_zones.front().medium.gamma * f.pressure / f.density);
    const double speed = std::max(f.speed, c);
    Scheme s;
    s.flux = in.flux == "hllc"      ? Flux::hllc
             : in.flux == "ausm+up" ? Flux::ausm_up
             : in.flux == "roe"     ? Flux::roe
                                    : throw std::invalid_argument("unknown flux '" + in.flux + "'");
    // A manufactured solution is smooth by construction: verification measures the design
    // order of the unlimited scheme (docs/design/euler.md, Limiter and verification).
    s.limiter = !in.verification.has_value();
    s.venkat_k = venkatakrishnan_k;
    s.length = f.ref_length.value_or(1.0);
    s.scale = {f.density, speed, speed, speed, f.pressure};
    s.mach_ref = f.speed / c;
    const double gamma = *in.cell_zones.front().medium.gamma;
    const double q2 = f.velocity[0] * f.velocity[0] + f.velocity[1] * f.velocity[1] +
                      f.velocity[2] * f.velocity[2];
    s.reference = {f.density, f.density * f.velocity[0], f.density * f.velocity[1],
                   f.density * f.velocity[2], f.pressure / (gamma - 1.0) + 0.5 * f.density * q2};
    return s;
  }

  // From a previous run: its conserved variables and averaging sums when it left them
  // (`continue`), else its primitive fields (`warm`).
  void start_from(const runtime::StartState& start) {
    auto find = [](const std::vector<io::Field>& fields, const std::string& name) {
      for (const auto& f : fields) {
        if (f.name == name) return &f.values;
      }
      throw std::invalid_argument("start state has no field " + name);
    };
    if (!start.restart_fields.empty()) {
      for (std::size_t k = 0; k < 5; ++k) {
        const auto& v = *find(start.restart_fields, conserved_names[k]);
        std::copy(v.begin(), v.end(), u_.begin() + static_cast<std::ptrdiff_t>(k * n_));
      }
      for (const auto& f : start.restart_fields) {
        if (f.name == "AverageWeight") mean_weight_ = f.values.front();
      }
      for (std::size_t i = 0; i < vortices_.size(); ++i) {
        vortices_[i].circulation =
            find(start.restart_fields, "VortexCirculation" + std::to_string(i))->front();
        vortices_[i].next = vortices_[i].circulation;
      }
      if (ctx_.input.time.kind == "steady") {
        auto state = solver_state();
        for (auto& [name, value] : state) value = find(start.restart_fields, name)->front();
        set_solver_state(state);
      }
      if (sa_) {
        const auto& v = *find(start.restart_fields, "DensityNuTilde");
        std::copy(v.begin(), v.end(), sa_->rnu.begin());
        fill_ghost_values(sa_->rnu);
      }
      if (mean_weight_ > 0.0) {
        for (std::size_t k = 0; k < 6; ++k) {
          const auto& v = *find(start.restart_fields, std::string("AverageSum") + field_names[k]);
          std::copy(v.begin(), v.end(), mean_.begin() + static_cast<std::ptrdiff_t>(k * n_));
        }
      }
      return;
    }
    const auto& rho = *find(start.fields, "Density");
    const auto& vx = *find(start.fields, "VelocityX");
    const auto& vy = *find(start.fields, "VelocityY");
    const auto& vz = *find(start.fields, "VelocityZ");
    const auto& p = *find(start.fields, "Pressure");
    for (std::size_t c = 0; c < owned_; ++c) set_cell(c, rho[c], {vx[c], vy[c], vz[c]}, p[c]);
    if (sa_) {
      const auto& nt = *find(start.fields, "TurbulentSANuTilde");
      for (std::size_t c = 0; c < owned_; ++c) sa_->rnu[c] = rho[c] * nt[c];
      fill_ghost_values(sa_->rnu);
    }
  }

  [[nodiscard]] std::vector<std::pair<std::string, double>> solver_state() const {
    std::vector<std::pair<std::string, double>> v = {
        {"SolverPenalty", penalty_},
        {"SolverShortened", shortened_ ? 1.0 : 0.0},
        {"SolverSecondOrder", second_order_ ? 1.0 : 0.0},
        {"SolverFirstOrderSteps", static_cast<double>(first_order_.state().steps)}};
    for (std::size_t k = 0; k < 5; ++k) {
      v.emplace_back("SolverReferenceResidual" + std::to_string(k),
                     first_norms_.empty() ? -1.0 : first_norms_[k]);
    }
    v.emplace_back("SolverCflCap", cfl_cap_);
    // The first-order start's plateau (-1: no reference yet), which decides when it ends.
    const auto plateau = first_order_.state().plateau;
    v.emplace_back("SolverFirstOrderPlateauReference",
                   std::isinf(plateau.reference) ? -1.0 : plateau.reference);
    v.emplace_back("SolverFirstOrderPlateauCount", static_cast<double>(plateau.count));
    v.emplace_back("SolverFirstOrderPlateauSince", static_cast<double>(plateau.since));
    v.emplace_back("SolverPreviousDrop", previous_drop_);
    return v;
  }

  void set_solver_state(const std::vector<std::pair<std::string, double>>& v) {
    penalty_ = v[0].second;
    shortened_ = v[1].second != 0.0;
    second_order_ = v[2].second != 0.0;
    euler_.set_second_order(second_order_);

    first_norms_.clear();
    if (v[4].second >= 0.0) {
      for (std::size_t k = 0; k < 5; ++k) first_norms_.push_back(v[4 + k].second);
    }
    cfl_cap_ = v[9].second;
    first_order_.restore(
        {.steps = static_cast<int>(v[3].second),
         .plateau = {.reference = v[10].second < 0.0 ? std::numeric_limits<double>::infinity()
                                                     : v[10].second,
                     .count = static_cast<std::size_t>(v[11].second),
                     .since = static_cast<std::size_t>(v[12].second)}});
    previous_drop_ = v[13].second;
  }

  // SA-neg: rho nt from the zones' initial nt, the freestream nt at farfield faces and the ratio
  // nt / nu at inlets, the wall distance; a manufactured solution gives its own nt, wall distance
  // and source.
  void init_sa(const Manufactured* mms) {
    const auto& mesh = ctx_.mesh;
    const auto& g = ctx_.geometry;
    const auto& in = ctx_.input;
    if (in.time.kind != "steady") {
      throw std::invalid_argument("the SA physics runs steady cases only");
    }
    sa_ = std::make_unique<SaState<P>>(mesh, g, euler_, transport_of(in),
                                       mms == nullptr || mms->turbulence() != "sa-noft2",
                                       keys_ ? &*keys_ : nullptr);
    auto& t = *sa_;
    t.model.set_ghosts(&ghost_fill_);
    const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
    const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
    std::vector<std::array<double, 3>> cells(n_), bfaces(nb);
    for (std::size_t c = 0; c < n_; ++c) {
      cells[c] = {g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]};
    }
    for (std::size_t b = 0; b < nb; ++b) {
      bfaces[b] = {g.face_centroid_x[ni + b], g.face_centroid_y[ni + b], g.face_centroid_z[ni + b]};
    }
    t.nt_inf = in.freestream.nu_tilde.value_or(0.0);
    std::vector<double> nt_b(nb, t.nt_inf);
    std::vector<double> d;
    if (mms != nullptr) {
      static const std::array<std::string, 1> nt_name = {"nu_hat"};
      static const std::array<std::string, 1> d_name = {"wall_distance"};
      static const std::array<std::string, 1> s_name = {"rhonu"};
      nt_b = mms->evaluate("fields", nt_name, bfaces)[0];
      d = mms->evaluate("auxiliary", d_name, cells)[0];
      t.model.set_source(mms->evaluate("sources", s_name, cells)[0]);
    } else {
      std::map<std::string, std::string> kinds;
      for (const auto& b : in.boundaries) kinds[b.zone] = b.kind;
      // The runtime's, from the walls of the whole mesh: a cell's nearest wall may lie in
      // another part.
      if (ctx_.wall_distance.empty()) throw std::invalid_argument("the SA physics needs a wall");
      d.assign(ctx_.wall_distance.begin(), ctx_.wall_distance.end());
      for (std::size_t b = 0; b < nb; ++b) {
        const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[ni + b])];
        const auto it = kinds.find(zone.name);
        if (it != kinds.end() && it->second == "inlet")
          nt_b[b] = in.turbulence.nu_tilde_ratio.value();
      }
    }
    t.model.set_boundary(nt_b);
    t.model.set_wall_distance(d);
    std::map<std::string, const runtime::CellZoneInput*> zones;
    for (const auto& z : in.cell_zones) zones[z.zone] = &z;
    t.rnu.assign(n_, 0.0);
    for (std::size_t c = 0; c < n_; ++c) {
      const auto zi = static_cast<std::size_t>(mesh.cell_zone()[c]);
      t.rnu[c] = u_[c] * zones.at(mesh.cell_zones()[zi].name)->initial.nu_tilde.value_or(0.0);
    }
    // The ghosts' density may not be filled yet: their owners' rho nt.
    fill_ghost_values(t.rnu);
    if (keys_ || owned_ < n_) {
      t.colours.assign(static_cast<std::size_t>(ctx_.n_colours), {});
      for (std::size_t c = 0; c < owned_; ++c) {
        t.colours[static_cast<std::size_t>(ctx_.cell_colours[c])].push_back(c);
      }
    }
    t.rhs.assign(n_, 0.0);
    t.dx.assign(n_, 0.0);
    t.diag.assign(n_, 0.0);
    t.upper.assign(static_cast<std::size_t>(mesh.n_interior_faces()), 0.0);
    t.lower.assign(t.upper.size(), 0.0);
    t.s.assign(n_, typename P::state(0));
    t.res.assign(n_, typename P::residual(0));
  }

  // initial-state.md: the owned cells' state from the case's initial-state file; a NaN row (a
  // solid cell) keeps its zone's state.
  void start_from_initial(const std::vector<io::Field>& fields) {
    auto find = [&](const char* name) -> const std::vector<double>& {
      for (const auto& f : fields) {
        if (f.name == name) return f.values;
      }
      throw std::invalid_argument(std::string("initial state has no field ") + name);
    };
    const auto& rho = find("Density");
    const std::array<const std::vector<double>*, 3> v{&find("VelocityX"), &find("VelocityY"),
                                                      &find("VelocityZ")};
    const auto& p = find("Pressure");
    for (std::size_t c = 0; c < owned_; ++c) {
      if (std::isnan(rho[c])) continue;
      set_cell(c, rho[c], {(*v[0])[c], (*v[1])[c], (*v[2])[c]}, p[c]);
    }
  }

  void set_cell(std::size_t c, double rho, const std::array<double, 3>& v, double p) {
    u_[c] = rho;
    for (std::size_t k = 0; k < 3; ++k) u_[(1 + k) * n_ + c] = rho * v[k];
    u_[4 * n_ + c] = p / (gas_.gamma - 1.0) + 0.5 * rho * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }

  void set_boundaries(const Manufactured* mms) {
    make_vortices();
    const auto& mesh = ctx_.mesh;
    const auto& in = ctx_.input;
    const auto& g = ctx_.geometry;
    const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
    const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
    std::map<std::string, const runtime::BoundaryInput*> conditions;
    for (const auto& b : in.boundaries) conditions[b.zone] = &b;
    std::vector<BoundaryKind> kind(nb, BoundaryKind::state);
    std::vector<double> data(5 * nb, 0.0);
    std::vector<std::array<double, 3>> manufactured_faces;
    std::vector<std::size_t> manufactured_index;
    const auto& f = in.freestream;
    const std::array<double, 5> freestream{f.density, f.velocity[0], f.velocity[1], f.velocity[2],
                                           f.pressure};
    for (std::size_t b = 0; b < nb; ++b) {
      const auto face = ni + b;
      const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[face])].name;
      const auto it = conditions.find(zone);
      if (it == conditions.end()) {
        throw std::invalid_argument("boundary zone '" + zone + "' has no condition");
      }
      const auto& bc = *it->second;
      auto put = [&](BoundaryKind k, std::array<double, 5> v) {
        kind[b] = k;
        for (std::size_t j = 0; j < 5; ++j) data[j * nb + b] = v[j];
      };
      if (bc.kind == "slip" || bc.kind == "symmetry") {
        put(BoundaryKind::slip, {});
      } else if (bc.kind == "wall" && is_viscous) {
        put(BoundaryKind::wall, {bc.temperature.value_or(0.0), 0.0, 0.0, 0.0, 0.0});
      } else if (bc.kind == "farfield") {
        put(BoundaryKind::farfield, freestream);
        if (bc.point_vortex) vortex_of(bc).faces.push_back(b);
      } else if (bc.kind == "inlet") {
        put(BoundaryKind::inlet, {*bc.total_pressure, *bc.total_temperature, 0.0, 0.0, 0.0});
      } else if (bc.kind == "outlet") {
        put(BoundaryKind::outlet, {0.0, 0.0, 0.0, 0.0, *bc.static_pressure});
      } else if (bc.kind == "oracle-state") {
        // The runtime paired the zone's faces with the file's rows (face-states.md).
        const index_t z = mesh.face_zone()[face];
        const auto& states = ctx_.boundary_states.at(z);
        const auto off = mesh.boundary_zone_offsets();
        const auto nz = static_cast<std::size_t>(off[static_cast<std::size_t>(z) + 1] -
                                                 off[static_cast<std::size_t>(z)]);
        const auto j = face - static_cast<std::size_t>(off[static_cast<std::size_t>(z)]);
        put(BoundaryKind::state, {states[j], states[nz + j], states[2 * nz + j], states[3 * nz + j],
                                  states[4 * nz + j]});
      } else if (bc.kind == "manufactured" && mms != nullptr) {
        kind[b] = BoundaryKind::state;
        manufactured_faces.push_back(
            {g.face_centroid_x[face], g.face_centroid_y[face], g.face_centroid_z[face]});
        manufactured_index.push_back(b);
      } else {
        throw std::invalid_argument("boundary zone '" + zone + "': condition '" + bc.kind +
                                    "' is not supported by this physics");
      }
    }
    if (!manufactured_faces.empty()) {
      static const std::array<std::string, 5> names = {"rho", "u", "v", "w", "p"};
      const auto w = mms->evaluate("fields", names, manufactured_faces);
      for (std::size_t k = 0; k < manufactured_index.size(); ++k) {
        for (std::size_t j = 0; j < 5; ++j) data[j * nb + manufactured_index[k]] = w[j][k];
      }
    }
    euler_.set_boundary(kind, data);
    const auto linear = euler_.linear_cells();
    if (linear.count > 0) {
      ctx_.log.write("warning", "solver",
                     "cells keep the linear gradient: too few neighbours for the quadratic fit", {},
                     nlohmann::json{{"cells", linear.count}, {"first", linear.first}}.dump());
    }
    boundary_data_ = std::move(data);
  }

  // One vortex per farfield condition that has one, in input order on every rank: the span is
  // a collective reduction.
  void make_vortices() {
    const auto& mesh = ctx_.mesh;
    const auto off = mesh.boundary_zone_offsets();
    for (const auto& bc : ctx_.input.boundaries) {
      if (!bc.point_vortex) continue;
      Vortex v{.condition = &bc, .center = bc.point_vortex->center, .zones = {}, .faces = {}};
      double lo = std::numeric_limits<double>::infinity();
      double hi = -lo;
      for (std::size_t z = 0; z < mesh.face_zones().size(); ++z) {
        if (std::ranges::find(bc.point_vortex->zones, mesh.face_zones()[z].name) ==
            bc.point_vortex->zones.end()) {
          continue;
        }
        v.zones.push_back(static_cast<index_t>(z));
        for (auto f = off[z]; f < off[z + 1]; ++f) {
          for (const auto n : mesh.face_nodes(f)) {
            lo = std::min(lo, mesh.y()[static_cast<std::size_t>(n)]);
            hi = std::max(hi, mesh.y()[static_cast<std::size_t>(n)]);
          }
        }
      }
      v.span = -ctx_.ranks.min(-hi) - ctx_.ranks.min(lo);
      if (!(v.span > 0.0)) {
        throw std::invalid_argument("boundary zone '" + bc.zone +
                                    "': the point vortex surfaces have no extent along y");
      }
      vortices_.push_back(std::move(v));
    }
  }

  Vortex& vortex_of(const runtime::BoundaryInput& bc) {
    return *std::ranges::find(vortices_, &bc, &Vortex::condition);
  }

  // The pressure force on boundary face i (global index) minus p_inf S (results.md), plus the
  // viscous traction t (per unit area, 3 x n_boundary_faces) times the face area when given.
  [[nodiscard]] std::array<double, 3> wall_force(std::span<const double> face_force,
                                                 std::span<const double> t, std::size_t i) const {
    const auto& g = ctx_.geometry;
    const auto nb = static_cast<std::size_t>(ctx_.mesh.n_boundary_faces());
    const auto b = i - static_cast<std::size_t>(ctx_.mesh.n_interior_faces());
    const double p_inf = ctx_.input.freestream.pressure;
    std::array<double, 3> force{face_force[b] - p_inf * g.face_area_x[i],
                                face_force[nb + b] - p_inf * g.face_area_y[i],
                                face_force[2 * nb + b] - p_inf * g.face_area_z[i]};
    if (!t.empty()) {
      const double a =
          std::sqrt(g.face_area_x[i] * g.face_area_x[i] + g.face_area_y[i] * g.face_area_y[i] +
                    g.face_area_z[i] * g.face_area_z[i]);
      for (std::size_t k = 0; k < 3; ++k) force[k] += a * t[k * nb + b];
    }
    return force;
  }

  // Each vortex's circulation from the lift of the residual just evaluated, pressure and viscous
  // traction as in the force monitors, Kutta-Joukowski L' = rho V Gamma; it takes effect at the
  // end of the step.
  void measure_circulation() {
    if (vortices_.empty()) return;
    const auto& mesh = ctx_.mesh;
    const auto& fs = ctx_.input.freestream;
    const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
    std::vector<double> p(nb), face_force(3 * nb), traction;
    euler_.boundary_pressure(p, face_force);
    if constexpr (is_viscous) {
      traction.assign(3 * nb, 0.0);
      euler_.viscous_traction(traction);
    }
    const auto off = mesh.boundary_zone_offsets();
    for (auto& v : vortices_) {
      std::array<runtime::ExactSum, 2> force;
      for (const auto z : v.zones) {
        for (auto f = off[static_cast<std::size_t>(z)]; f < off[static_cast<std::size_t>(z) + 1];
             ++f) {
          const auto load = wall_force(face_force, traction, static_cast<std::size_t>(f));
          force[0].add(load[0]);
          force[1].add(load[2]);
        }
      }
      ctx_.ranks.sum(force);
      const double lift =
          -std::sin(fs.aoa) * force[0].value() + std::cos(fs.aoa) * force[1].value();
      v.next = lift / (v.span * fs.density * fs.speed);
    }
  }

  // The farfield data: the freestream plus each vortex at the current circulation.
  void apply_vortices() {
    if (vortices_.empty()) return;
    const auto& mesh = ctx_.mesh;
    const auto& g = ctx_.geometry;
    const auto& fs = ctx_.input.freestream;
    const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
    const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
    const std::array<double, 5> freestream{fs.density, fs.velocity[0], fs.velocity[1],
                                           fs.velocity[2], fs.pressure};
    std::vector<double> data = boundary_data_;
    for (const auto& v : vortices_) {
      for (const auto b : v.faces) {
        const double dx = g.face_centroid_x[ni + b] - v.center[0];
        const double dz = g.face_centroid_z[ni + b] - v.center[2];
        if (dx == 0.0 && dz == 0.0) {
          throw std::invalid_argument("boundary zone '" + v.condition->zone +
                                      "': a farfield face lies on the point vortex");
        }
        const auto w = point_vortex_state(freestream, gas_.gamma, fs.aoa, v.circulation, dx, dz);
        for (std::size_t j = 0; j < 5; ++j) data[j * nb + b] = w[j];
      }
    }
    euler_.set_boundary_data(data);
  }

  void set_source(const Manufactured& mms) {
    const auto& g = ctx_.geometry;
    std::vector<std::array<double, 3>> points(n_);
    for (std::size_t c = 0; c < n_; ++c) {
      points[c] = {g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]};
    }
    static const std::array<std::string, 5> names = {"rho", "rhou", "rhov", "rhow", "rhoE"};
    const auto s = mms.evaluate("sources", names, points);
    std::vector<double> source(5 * n_);
    for (std::size_t j = 0; j < 5; ++j) {
      std::copy(s[j].begin(), s[j].end(), source.begin() + static_cast<std::ptrdiff_t>(j * n_));
    }
    euler_.set_source(source);
  }

  // R(u) of the FP64 solution, through the policy's state type.
  void residual_of(const std::vector<double>& u) {
    euler_.state(u, state_);
    eddy_viscosity_of_state();
    euler_.residual(state_, res_);
  }

  // With SA, the eddy viscosity the flow reads is that of state_ and the current rho nt, from the
  // primitives of a flow evaluation at state_: the residual is a function of the state alone,
  // which a continued run rebuilds.
  void eddy_viscosity_of_state() {
    if (!sa_) return;
    euler_.residual(state_, res_);
    sa_->model.eddy_viscosity(sa_->stored(sa_->rnu));
  }

  // results.md#history: sqrt(sum R_i^2 / N) over the owned cells of every rank, summed exactly.
  void record_norms(StepReport& report) const {
    std::array<runtime::ExactSum, 6> sums;
    for (std::size_t k = 0; k < 5; ++k) {
      for (std::size_t c = 0; c < owned_; ++c) {
        const double r = static_cast<double>(res_[k * n_ + c]);
        sums[k].add(r * r);
      }
    }
    sums[5].add(static_cast<double>(owned_));
    ctx_.ranks.sum(sums);
    report.residuals.assign(5, 0.0);
    for (std::size_t k = 0; k < 5; ++k) {
      report.residuals[k] = std::sqrt(sums[k].value() / sums[5].value());
    }
  }

  void steady_step(StepReport& report) {
    euler_.state(u_, state_);
    eddy_viscosity_of_state();
    if (!second_order_) {
      // The history reports the second-order residual from the first step on, so that the
      // runtime never judges convergence on the first-order one.
      euler_.set_second_order(true);
      euler_.residual(state_, res_);
      record_norms(report);
      euler_.set_second_order(false);
    }
    euler_.residual_and_jacobian(state_, res_);
    jacobian_second_order_ = second_order_;
    measure_circulation();
    std::vector<double> norms = report.residuals;
    record_norms(report);
    if (!norms.empty()) std::swap(norms, report.residuals);
    if (sa_) {
      // Loosely coupled: the SA residual at the same state, after the flow's evaluation.
      auto& t = *sa_;
      t.model.residual_and_jacobian(t.stored(t.rnu), t.res);
      // As record_norms: an exact sum over the owned cells, the same bits in any order.
      std::array<runtime::ExactSum, 2> sum;
      for (std::size_t c = 0; c < owned_; ++c) {
        const double r = static_cast<double>(t.res[c]);
        sum[0].add(r * r);
      }
      sum[1].add(static_cast<double>(owned_));
      ctx_.ranks.sum(sum);
      report.residuals.push_back(std::sqrt(sum[0].value() / sum[1].value()));
    }
    euler_.spectral_radius(lambda_);
    // Largest relative residual over the equations, of the order being solved, leaving out the
    // equations at the round-off floor (the runtime's rule, decided on the reported residuals):
    // their ratio is round-off and would set the CFL number.
    const double drop = relative_drop(norms.empty() ? report.residuals : norms, report.residuals);
    switch_order(drop);
    // Switched evolution relaxation, CFL = CFL_0 / (relative residual), times a penalty halved
    // after each step the admissibility check shortened or whose relative residual rose, and
    // recovering by half otherwise.
    const bool rose = previous_drop_ > 0.0 && drop > cfl_rise * previous_drop_;
    penalty_ = (shortened_ || rose) ? 0.5 * penalty_ : std::min(1.0, 1.5 * penalty_);
    previous_drop_ = drop;
    if (drop > 0.0) {
      cfl_ = std::clamp(std::min(penalty_ * cfl_start / drop, cfl_cap_), cfl_min, cfl_max);
    }
    cfl_cap_ = cfl_max;
    std::vector<double> shift(n_);
    for (std::size_t c = 0; c < n_; ++c) shift[c] = static_cast<double>(lambda_[c]) / cfl_;
    euler_.assemble(shift, *matrix_);
    for (std::size_t i = 0; i < 5 * n_; ++i) rhs_[i] = -static_cast<double>(res_[i]);
    std::fill(du_.begin(), du_.end(), 0.0);
    solvers::Multigrid::Report linear;
    try {
      if (multigrid_) {
        multigrid_->set_operator(matrix_->diagonals(), matrix_->uppers(), matrix_->lowers());
        // As Multigrid::solve, GCR preconditioned by one cycle, on the operator with the
        // boundary faces' gradient term.
        krylov_residual_ = rhs_;
        linear = gcr_.solve(
            [&](std::span<const double> v, std::span<double> y) {
              matrix_->multiply(v, y);
              add_boundary_gradient_term(v, y);
            },
            [&](std::span<const double> r, std::span<double> z) { multigrid_->cycle(r, z); },
            krylov_residual_, du_, linear_max_cycles, linear_tolerance);
      } else if (lines_) {
        // GCR preconditioned by the line sweeps, as the multigrid's outer iteration; its inner
        // products are exact sums over the owned cells of every rank.
        // Every rank enters the GCR's reductions, or none does: a singular block anywhere fails
        // the step on every rank.
        bool singular = false;
        try {
          lines_->set_operator(matrix_->diagonals(), matrix_->uppers(), matrix_->lowers());
        } catch (const std::domain_error&) {
          singular = true;
        }
        if (ctx_.ranks.max(singular ? 1.0 : 0.0) > 0.0) throw std::domain_error("singular block");
        auto exchange = [this](std::span<double> x) { fill_ghosts(x); };
        krylov_residual_ = rhs_;
        linear = gcr_.solve(
            [&](std::span<const double> v, std::span<double> y) {
              lines_->multiply(v, y);
              add_boundary_gradient_term(v, y);
            },
            [&](std::span<const double> r, std::span<double> z) {
              lines_->sweep(r, z, gauss_seidel_sweeps, exchange);
            },
            krylov_residual_, du_, linear_max_cycles, linear_tolerance,
            {.dot = [this](std::span<const double> u,
                           std::span<const double> v) { return owned_dot(u, v); },
             .max_abs = [this](std::span<const double> u) { return owned_max_abs(u); }});
      } else {
        // Every rank sweeps, exchanging after each colour, or none does.
        bool singular = false;
        try {
          matrix_->factor(owned_);
        } catch (const std::domain_error&) {
          singular = true;
        }
        if (ctx_.ranks.max(singular ? 1.0 : 0.0) > 0.0) {
          throw std::domain_error("singular diagonal block");
        }
        // GCR preconditioned by the symmetric sweeps, as the line smoother's outer iteration, on
        // the operator with the boundary faces' gradient term; its inner products are exact sums
        // over the owned cells of every rank.
        auto exchange = [this](std::span<double> x) { fill_ghosts(x); };
        if (krylov_v_.size() != rhs_.size()) krylov_v_.assign(rhs_.size(), 0.0);
        krylov_residual_ = rhs_;
        linear = gcr_.solve(
            [&](std::span<const double> v, std::span<double> y) {
              std::copy(v.begin(), v.end(), krylov_v_.begin());
              fill_ghosts(krylov_v_);
              matrix_->multiply(krylov_v_, y);
              add_boundary_gradient_term(krylov_v_, y);
            },
            [&](std::span<const double> r, std::span<double> z) {
              std::fill(z.begin(), z.end(), 0.0);
              matrix_->gauss_seidel(r, z, gauss_seidel_sweeps, colours_, exchange);
            },
            krylov_residual_, du_, linear_max_cycles, linear_tolerance,
            {.dot = [this](std::span<const double> u,
                           std::span<const double> v) { return owned_dot(u, v); },
             .max_abs = [this](std::span<const double> u) { return owned_max_abs(u); }});
      }
    } catch (const std::domain_error&) {
      linear = {};  // a singular block: the solve failed
    }
    const bool failed = !(linear.residual_drop <= linear_failure);
    if (!(linear.residual_drop <= linear_tolerance)) {
      nlohmann::json data = {{"cycles", linear.cycles}, {"residual_drop", nullptr}};
      if (std::isfinite(linear.residual_drop)) data["residual_drop"] = linear.residual_drop;
      ctx_.log.write("warning", "linear",
                     failed ? "the implicit step's linear solve failed: step rejected"
                            : "the implicit step's linear solve stopped above its tolerance",
                     {}, data.dump());
    }
    if (failed) {
      // As a step with no admissible length: the state is kept and the next CFL number is half
      // this one, also when this one was clamped; at the smallest CFL number the run diverged.
      if (drop > 0.0) penalty_ = std::min(penalty_, cfl_ * drop / cfl_start);
      cfl_cap_ = 0.5 * cfl_;
      shortened_ = true;
      if (cfl_ <= cfl_min) report.diverged_cell = worst_cell();
      return;
    }
    const double alpha = admissible_step();
    shortened_ = alpha < 1.0;
    // The next step halves the penalty; starting from the CFL number actually used, so that the
    // CFL number halves also when it was clamped.
    if (shortened_) {
      if (drop > 0.0) penalty_ = std::min(penalty_, cfl_ * drop / cfl_start);
      cfl_cap_ = 0.5 * cfl_;
      ctx_.log.write("warning", "solver",
                     alpha < min_step ? "no admissible step: step rejected" : "step shortened", {},
                     nlohmann::json{{"alpha", alpha}, {"cfl", cfl_}}.dump());
    }
    if (alpha < min_step) {
      // No admissible step: the state is kept and the CFL number halves; at the smallest CFL
      // number the run has diverged.
      if (cfl_ <= cfl_min) report.diverged_cell = worst_cell();
      return;
    }
    for (std::size_t i = 0; i < 5 * n_; ++i) u_[i] += alpha * du_[i];
    if (sa_) {
      // The first-order start has no velocity gradients, hence no production: SA waits for the
      // second order.
      auto& t = *sa_;
      if (second_order_) {
        for (std::size_t c = 0; c < n_; ++c) t.rhs[c] = -static_cast<double>(t.res[c]);
        std::fill(t.dx.begin(), t.dx.end(), 0.0);
        t.model.assemble(shift, t.diag, t.upper, t.lower);
        if (t.colours.empty()) {
          t.model.solve(t.diag, t.upper, t.lower, t.rhs, t.dx, nt_sweeps);
        } else {
          t.model.solve(t.diag, t.upper, t.lower, t.rhs, t.dx, nt_sweeps, &t.colours,
                        [this](std::span<double> x) { fill_ghost_values(x); });
        }
        // Local under-relaxation of the SA update: rho nt changes by at most a fraction of
        // max(|rho nt|, rho nt_inf) per step, which the converged state never reaches. Owned
        // cells; the ghosts take their owners' values.
        for (std::size_t c = 0; c < owned_; ++c) {
          const double scale = std::max(std::abs(t.rnu[c]), u_[c] * t.nt_inf);
          t.rnu[c] +=
              alpha * std::clamp(t.dx[c], -nt_max_decrease * scale, nt_max_increase * scale);
        }
        fill_ghost_values(t.rnu);
      }
    }
  }

  // The cell whose density or pressure the full step lowers the most, relatively.
  // Over every rank, the cell whose density or pressure the full step lowers the most, relatively
  // (ties: the least global id); reported by the rank that owns it.
  [[nodiscard]] std::optional<index_t> worst_cell() const {
    const auto ids = ctx_.mesh.cell_global_id();
    auto id = [&](std::size_t c) { return ids.empty() ? static_cast<std::int64_t>(c) : ids[c]; };
    std::size_t worst = 0;
    double lowest = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < owned_; ++c) {
      const double r = std::min((u_[c] + du_[c]) / u_[c], pressure(c, 1.0) / pressure(c, 0.0));
      if (r < lowest || (r == lowest && id(c) < id(worst))) {
        lowest = r;
        worst = c;
      }
    }
    const auto global = ctx_.ranks.argmin(lowest, owned_ > 0 ? id(worst) : 0);
    if (owned_ == 0 || global.second != id(worst)) return std::nullopt;
    return static_cast<index_t>(worst);
  }

  // Largest fraction of du (at most 1) that keeps every cell's density and pressure above
  // (1 - max_decrease) of their current values.
  // Each owned cell's own admissible fraction (its density bound, then halved until its
  // pressure bound holds), and the least over every rank: independent of the cell order.
  [[nodiscard]] double admissible_step() const {
    double alpha = 1.0;
    for (std::size_t c = 0; c < owned_; ++c) {
      double a = 1.0;
      const double rho = u_[c], drho = du_[c];
      if (drho < 0.0) a = std::min(a, max_decrease * rho / -drho);
      const double p = pressure(c, 0.0);
      while (a >= min_step && !(pressure(c, a) >= (1.0 - max_decrease) * p)) a *= 0.5;
      alpha = std::min(alpha, a);
    }
    return ctx_.ranks.min(alpha);
  }

  // Ghost cells take their owners' du (5 components).
  // Sum of u v over the owned cells of every rank, exact: the same bits on any partition.
  // y += the boundary faces' derivative through the reconstruction's gradients, which the
  // assembled blocks drop: in large far-field cells the step without it is unstable at the CFL
  // numbers the SER reaches (ONERA M6, #244).
  void add_boundary_gradient_term(std::span<const double> v, std::span<double> y) {
    // The step that switches to second order solves the first-order system it assembled.
    if (!jacobian_second_order_) return;
    if (dv_.size() != 5 * n_) {
      dv_.assign(5 * n_, State(0));
      dres_.assign(5 * n_, Res(0));
    }
    for (std::size_t i = 0; i < 5 * n_; ++i) dv_[i] = State(v[i]);
    std::fill(dres_.begin(), dres_.end(), Res(0));
    euler_.gradient_tangent(state_, dv_, dres_, Flow::Faces::boundary);
    for (std::size_t i = 0; i < 5 * n_; ++i) y[i] += static_cast<double>(dres_[i]);
  }

  double owned_dot(std::span<const double> u, std::span<const double> v) {
    std::array<runtime::ExactSum, 1> sum;
    for (std::size_t k = 0; k < 5; ++k) {
      for (std::size_t c = 0; c < owned_; ++c) sum[0].add(u[k * n_ + c] * v[k * n_ + c]);
    }
    ctx_.ranks.sum(sum);
    return sum[0].value();
  }

  // Largest |u| over the owned cells of every rank; infinite when an entry is not a number.
  double owned_max_abs(std::span<const double> u) {
    double m = 0.0;
    for (std::size_t k = 0; k < 5; ++k) {
      for (std::size_t c = 0; c < owned_; ++c) {
        const double a = std::abs(u[k * n_ + c]);
        m = std::isnan(a) ? std::numeric_limits<double>::infinity() : std::max(m, a);
      }
    }
    return ctx_.ranks.max(m);
  }

  // Fills the ghost values of one cell array (n_ values) from their owners. On every rank: the
  // exchange is collective, even for a part without ghosts.
  void fill_ghost_values(std::span<double> x) {
    std::array<double*, 1> c{x.data()};
    ctx_.ghosts.forward(std::span<double* const>(c), MemorySpace::host, nullptr);
  }

  void fill_ghosts(std::span<double> x) {
    std::array<double*, 5> c{};
    for (std::size_t k = 0; k < 5; ++k) c[k] = x.data() + k * n_;
    ctx_.ghosts.forward(std::span<double* const>(c), MemorySpace::host, nullptr);
  }

  [[nodiscard]] double pressure(std::size_t c, double a) const {
    double v[5];
    for (std::size_t k = 0; k < 5; ++k) v[k] = u_[k * n_ + c] + a * du_[k * n_ + c];
    return (gas_.gamma - 1.0) * (v[4] - 0.5 * (v[1] * v[1] + v[2] * v[2] + v[3] * v[3]) / v[0]);
  }

  void unsteady_step(StepReport& report) {
    // Third-order strong-stability-preserving Runge-Kutta, Shu and Osher (1988), J. Comput.
    // Phys. 77:439-471, eq. 2.18, with one global time step.
    const auto& t = ctx_.input.time;
    const auto& g = ctx_.geometry;
    residual_of(u_);
    record_norms(report);
    double dt = t.dt.value_or(0.0);
    if (t.cfl) {
      euler_.spectral_radius(lambda_);
      dt = std::numeric_limits<double>::infinity();
      for (std::size_t c = 0; c < owned_; ++c) {
        dt = std::min(dt, *t.cfl * g.cell_volume[c] / static_cast<double>(lambda_[c]));
      }
      dt = ctx_.ranks.min(dt);
    }
    const std::array<std::array<double, 2>, 3> weights{
        {{0.0, 1.0}, {0.75, 0.25}, {1.0 / 3.0, 2.0 / 3.0}}};
    stage_ = u_;
    for (std::size_t s = 0; s < 3; ++s) {
      if (s > 0) residual_of(stage_);
      for (std::size_t k = 0; k < 5; ++k) {
        for (std::size_t c = 0; c < owned_; ++c) {
          const auto i = k * n_ + c;
          const double next = stage_[i] - dt / g.cell_volume[c] * static_cast<double>(res_[i]);
          stage_[i] = weights[s][0] * u_[i] + weights[s][1] * next;
        }
      }
    }
    u_ = stage_;
    report.time_step = dt;
  }

  // The first cell whose state, as the kernels read it, has a non-finite or non-positive
  // density or pressure (to_primitive of kernels/euler.py: P::state differences from the
  // reference, the gauge pressure from the differences, all in P::compute, FP32 in
  // fp32+acc64); else the first cell whose residual is not finite.
  [[nodiscard]] std::optional<index_t> first_invalid_cell() const {
    using C = typename P::compute;
    const auto& r = reference_;
    const C g = static_cast<C>(gas_.gamma);
    const C r0 = static_cast<C>(r[0]);
    const std::array<C, 3> mr{static_cast<C>(r[1]), static_cast<C>(r[2]), static_cast<C>(r[3])};
    const C p_ref = static_cast<C>((gas_.gamma - 1.0) *
                                   (r[4] - 0.5 * (r[1] * r[1] + r[2] * r[2] + r[3] * r[3]) / r[0]));
    LeastGlobal first{.ids = ctx_.mesh.cell_global_id(), .cell = {}, .best = 0};
    for (std::size_t c = 0; c < owned_; ++c) {
      std::array<C, 5> dq{};
      for (std::size_t k = 0; k < 5; ++k) {
        dq[k] = static_cast<C>(State(u_[k * n_ + c] - r[k]));
      }
      const C rho = r0 + dq[0];
      C dm_m = 0, mr_mr = 0;
      for (std::size_t k = 0; k < 3; ++k) {
        dm_m += dq[k + 1] * (C(2) * mr[k] + dq[k + 1]);
        mr_mr += mr[k] * mr[k];
      }
      const C dk = (dm_m * r0 - mr_mr * dq[0]) / (rho * r0);
      const C p = (g - C(1)) * (dq[4] - C(0.5) * dk) + p_ref;
      if (!std::isfinite(static_cast<double>(rho)) || !std::isfinite(static_cast<double>(p)) ||
          !(rho > C(0)) || !(p > C(0))) {
        first.take(c);
      }
    }
    if (first.cell) return first.cell;
    for (std::size_t c = 0; c < owned_; ++c) {
      for (std::size_t k = 0; k < 5; ++k) {
        if (!std::isfinite(static_cast<double>(res_[k * n_ + c]))) first.take(c);
      }
    }
    return first.cell;
  }

  // Of the cells offered, the one with the least cell_global_id: the same cell whatever the
  // numbering and the partition.
  struct LeastGlobal {
    std::span<const std::int64_t> ids;
    std::optional<index_t> cell;
    std::int64_t best = 0;
    void take(std::size_t c) {
      const std::int64_t id = ids.empty() ? static_cast<std::int64_t>(c) : ids[c];
      if (!cell || id < best) {
        cell = static_cast<index_t>(c);
        best = id;
      }
    }
  };

  [[nodiscard]] std::array<std::vector<double>, 6> primitive_fields() const {
    std::array<std::vector<double>, 6> f;
    for (auto& v : f) v.resize(owned_);
    for (std::size_t c = 0; c < owned_; ++c) {
      const double rho = u_[c];
      double q2 = 0.0;
      for (std::size_t k = 0; k < 3; ++k) {
        const double v = u_[(1 + k) * n_ + c] / rho;
        f[1 + k][c] = v;
        q2 += v * v;
      }
      const double p = (gas_.gamma - 1.0) * (u_[4 * n_ + c] - 0.5 * rho * q2);
      f[0][c] = rho;
      f[4][c] = p;
      f[5][c] = p / (rho * gas_.gas_constant);
    }
    return f;
  }

  // Viscous traction -tau n on every boundary face (3 x n_boundary_faces) of the current state,
  // zero for inviscid flow; from the evaluation face_pressure() caches.
  [[nodiscard]] std::vector<double> face_traction() const {
    std::vector<double> p(static_cast<std::size_t>(ctx_.mesh.n_boundary_faces()));
    face_pressure(p, {});
    return traction_;
  }

  // The boundary pressure and traction of the current state, evaluated once per state: every force
  // monitor of an iteration reads the same values (one host residual, not one per monitor).
  void face_pressure(std::span<double> p, std::span<double> force) const {
    auto& self = const_cast<EulerPhysics&>(*this);
    if (!pressure_valid_) {
      sync();
      self.residual_of(u_);
      const auto nb = static_cast<std::size_t>(ctx_.mesh.n_boundary_faces());
      self.pressure_.assign(nb, 0.0);
      self.pressure_force_.assign(3 * nb, 0.0);
      euler_.boundary_pressure(self.pressure_, self.pressure_force_);
      self.traction_.assign(3 * nb, 0.0);
      if constexpr (is_viscous) euler_.viscous_traction(self.traction_);
      self.pressure_valid_ = true;
    }
    std::copy(pressure_.begin(), pressure_.end(), p.begin());
    if (!force.empty()) std::copy(pressure_force_.begin(), pressure_force_.end(), force.begin());
  }

  // Largest relative residual over the equations, of the order being solved.
  double relative_drop(const std::vector<double>& current, const std::vector<double>& reported) {
    note_exemptions(reported);
    double drop = 0.0;
    if (first_norms_.empty()) first_norms_ = current;
    for (std::size_t k = 0; k < 5; ++k) {
      // A woken equation drops from the reference of its kind, as in the convergence rule.
      if (exempt_.exempt(k)) continue;
      const double from = exempt_.woken(k) ? exempt_.woken_reference(k) : first_norms_[k];
      if (from > 0.0) drop = std::max(drop, current[k] / from);
    }
    return drop;
  }

  // First-order start: its approximate Jacobian is exact, so the steps are robust far from the
  // solution; second order from the step after the first-order residual has dropped. The device
  // starts the same way, so that both solve the same sequence of problems.
  void switch_order(double drop) {
    if (second_order_) return;
    // Also once the drop is on a round-off plateau: in fp32+acc64 three orders can lie below it.
    if (first_order_.ends_after(drop)) {
      second_order_ = true;
      euler_.set_second_order(true);
      if (device_) device_->set_second_order(true);
      first_norms_.clear();
    }
  }

  // One step on the device; the host state is out of date until sync().
  StepReport device_step() {
    StepReport report;
    if (!device_checked_) {
      // As the host's step: a stored state the kernels cannot evaluate diverges at once.
      device_checked_ = true;
      const auto e = device_->evaluate();
      if (e.invalid) {
        report.residuals = e.residuals;
        report.diverged_cell = e.invalid;
        return report;
      }
    }
    const auto& t = ctx_.input.time;
    const bool steady = t.kind == "steady";
    if (steady && !second_order_) {
      // As the host's steady step: the history reports the second-order residual.
      device_->set_second_order(true);
      report.residuals = device_->evaluate().residuals;
      device_->set_second_order(false);
    }
    const auto r = steady ? device_->step(device_steady_cfl, std::nullopt, true)
                          : device_->step(t.cfl.value_or(0.0), t.cfl ? std::nullopt : t.dt, false);
    host_stale_ = true;
    // The exemption is decided on the residuals the runtime receives, as on the host.
    if (steady)
      switch_order(
          relative_drop(r.residuals, report.residuals.empty() ? r.residuals : report.residuals));
    if (report.residuals.empty()) report.residuals = r.residuals;
    if (!steady) report.time_step = r.time_step;
    report.diverged_cell = r.invalid;
    return report;
  }

  // The device state into the host's, before anything reads it.
  void sync() const {
    if (!device_ || !host_stale_) return;
    auto& self = const_cast<EulerPhysics&>(*this);
    device_->download(self.u_);
    self.host_stale_ = false;
  }

  std::unique_ptr<DeviceStepper<P>> device_;  // devices.kind = cuda
  bool host_stale_ = false;
  // face_pressure's values for the current state; a step invalidates them.
  std::vector<double> pressure_, pressure_force_, traction_;
  bool pressure_valid_ = false;
  bool device_checked_ = false;

  // The runtime's ghost exchange, on the host, for Euler<P>.
  class HostGhosts final : public GhostFill {
   public:
    explicit HostGhosts(runtime::GhostExchange& ghosts) : ghosts_(ghosts) {}
    void forward(std::span<float* const> components) override {
      ghosts_.forward(components, MemorySpace::host, nullptr);
    }
    void forward(std::span<double* const> components) override {
      ghosts_.forward(components, MemorySpace::host, nullptr);
    }
    CellRows forward_rows(const CellRows& owned, std::span<const std::uint8_t> wanted) override {
      return ghosts_.forward_rows(owned, wanted);
    }

   private:
    runtime::GhostExchange& ghosts_;
  };

  // Under reproducibility "ranks", the keys that order every sum: the same bits on any number
  // of ranks and in any cell order (solver-input.md).
  static std::optional<kernels::GlobalKeys> global_keys(const PhysicsContext& ctx) {
    if (ctx.input.reproducibility != "ranks") return std::nullopt;
    return kernels::GlobalKeys{.cell = ctx.mesh.cell_global_id(),
                               .face = ctx.face_global_id,
                               .reversed = ctx.face_reversed,
                               .shift = ctx.mesh.ghost_translation()};
  }

  // Component k of `v` (laid out k * n_ + c) over the owned cells.
  [[nodiscard]] std::vector<double> owned(const std::vector<double>& v, std::size_t k) const {
    const auto begin = v.begin() + static_cast<std::ptrdiff_t>(k * n_);
    return {begin, begin + static_cast<std::ptrdiff_t>(owned_)};
  }

  // The runtime's round-off exemption, kept in step with its decisions: fed the same reported
  // residuals from the same reference, on every rank.
  void note_exemptions(const std::vector<double>& reported) {
    const bool first = !exempt_logged_;
    exempt_.update(reported);
    if (first) {
      exempt_logged_ = true;
      nlohmann::json names = nlohmann::json::array();
      for (std::size_t k = 0; k < reported.size(); ++k) {
        if (exempt_.exempt(k)) names.push_back(exempt_.name(k));
      }
      if (!names.empty()) {
        ctx_.log.write("info", "solver",
                       "equations at the round-off floor do not set the CFL number", {},
                       nlohmann::json{{"equations", names}, {"floor", exempt_.floor()}}.dump());
      }
    }
  }

  runtime::RoundOffExemption exempt_;
  bool exempt_logged_ = false;

  PhysicsContext ctx_;
  Gas gas_;
  std::size_t n_;      // cells of the mesh: owned, then ghosts
  std::size_t owned_;  // owned cells: what this rank updates and reports
  HostGhosts ghost_fill_;
  std::optional<kernels::GlobalKeys> keys_;  // reproducibility "ranks" only
  Flow euler_;
  std::unique_ptr<SaState<P>> sa_;   // SA-neg runs only
  std::array<double, 5> reference_;  // the state the cells store differences from
  std::vector<double> u_, stage_, mean_, rhs_, du_, first_norms_;
  FirstOrderStart first_order_;
  std::optional<BlockMatrix> matrix_;               // steady runs only
  std::optional<solvers::Multigrid> multigrid_;     // linear_solver = "multigrid"
  std::optional<solvers::LineSmoother> lines_;      // linear_solver = "lines"
  solvers::Gcr gcr_;                                // and its outer iteration
  std::vector<double> krylov_residual_, krylov_v_;  // the GCR's residual; a vector with ghosts
  std::vector<std::vector<std::size_t>> colours_;   // linear_solver = "gauss-seidel": owned cells
  double cfl_cap_ = cfl_max;  // after a shortened or rejected step: half its CFL number
  double cfl_ = cfl_start;
  double penalty_ = 1.0;
  double previous_drop_ = 0.0;  // the last step's largest relative residual; 0: none yet
  bool second_order_ = false;
  bool shortened_ = false;
  double mean_weight_ = 0.0;
  std::vector<State> state_, lambda_, dv_;
  bool jacobian_second_order_ = false;  // the order of the last assembled Jacobian
  std::vector<Res> res_, dres_;
  std::vector<double> boundary_data_;  // set_boundary's data, before any vortex
  std::vector<Vortex> vortices_;
};

template <class P, class Flow>
runtime::PhysicsFactory factory() {
  return [](const PhysicsContext& ctx) -> std::unique_ptr<runtime::Physics> {
    return std::make_unique<EulerPhysics<P, Flow>>(ctx);
  };
}

bool same_gas(const runtime::Medium& a, const runtime::Medium& b) {
  const auto& va = *a.viscosity;
  const auto& vb = *b.viscosity;
  return a.gas_constant == b.gas_constant && a.gamma == b.gamma && a.prandtl == b.prandtl &&
         va.law == vb.law && va.mu_ref == vb.mu_ref && va.t_ref == vb.t_ref && va.s == vb.s &&
         va.mu == vb.mu;
}

// What the Euler physics runs beyond its key: every cell zone of one ideal gas, boundary
// conditions that apply to inviscid flow, and the multigrid on one rank without periodic pairs.
std::optional<runtime::PhysicsRefusal> euler_scope(const runtime::SolverInput& in, int ranks) {
  // case-schema.md#time: the opt-in steps, the coupled turbulence step SA's and the matrix-free
  // step laminar flow's, in the host's steady multigrid on one rank.
  if (in.time.kind == "steady") {
    for (const auto& [key, on, model, code] :
         {std::tuple{"coupled_turbulence", in.time.coupled_turbulence, "sa", "E0426"},
          std::tuple{"matrix_free", in.time.matrix_free, "laminar", "E0429"}}) {
      if (!on) continue;
      const std::string path = std::string("time.") + key;
      if (in.turbulence.model != model) {
        return runtime::PhysicsRefusal{
            .message = path + " needs the " + model + " model, not '" + in.turbulence.model + "'",
            .path = path,
            .data_json = nlohmann::json{{"model", in.turbulence.model}}.dump(),
            .code = code};
      }
      if (in.time.linear_solver != "multigrid" || ranks > 1) {
        return runtime::PhysicsRefusal{
            .message = path + " runs in the host multigrid on one rank",
            .path = path,
            .data_json =
                nlohmann::json{{"linear_solver", in.time.linear_solver}, {"ranks", ranks}}.dump(),
            .code = "E0427"};
      }
    }
  }
  // Under reproducibility "ranks" Gauss-Seidel and "lines" sum each row in global face order;
  // the multigrid still agglomerates, smooths and sums in local cell order.
  if (in.reproducibility == "ranks" && in.time.kind == "steady" &&
      in.time.linear_solver == "multigrid") {
    return runtime::PhysicsRefusal{
        .message =
            "the steady multigrid does not give the same bits on any number of ranks yet; "
            "linear_solver = \"gauss-seidel\" or \"lines\" does",
        .path = "reproducibility",
        .data_json = "{}",
        .code = "E1109"};
  }
  // The multigrid agglomerates one mesh's cells: on a partition, or with periodic image ghosts,
  // it would be another algorithm. Gauss-Seidel sweeps colours by global id, the same anywhere.
  if (in.time.kind == "steady" && in.time.linear_solver == "multigrid" &&
      (ranks > 1 || !in.periodic.empty())) {
    return runtime::PhysicsRefusal{
        .message =
            "the steady multigrid runs on one rank without periodic pairs; set "
            "linear_solver = \"lines\" or \"gauss-seidel\" until the distributed multigrid lands",
        .path = "time.linear_solver",
        .data_json = nlohmann::json{{"ranks", ranks}, {"pairs", in.periodic.size()}}.dump()};
  }
  // Face states are paired with the global mesh's zone faces; a partition holds only its own,
  // and the physics has no global face numbering to find their rows.
  for (std::size_t i = 0; ranks > 1 && i < in.boundaries.size(); ++i) {
    if (in.boundaries[i].kind == "oracle-state") {
      return runtime::PhysicsRefusal{
          .message = "the Euler physics imposes oracle-state faces on one rank only, not " +
                     std::to_string(ranks),
          .path = "boundaries[" + std::to_string(i) + "].kind",
          .data_json = nlohmann::json{{"zone", in.boundaries[i].zone}, {"ranks", ranks}}.dump()};
    }
  }
  const runtime::Medium* first = nullptr;
  for (std::size_t i = 0; i < in.cell_zones.size(); ++i) {
    const auto& z = in.cell_zones[i];
    const std::string path = "cell_zones[" + std::to_string(i) + "].medium";
    if (z.medium.kind != "ideal-gas") {
      return runtime::PhysicsRefusal{
          .message = "the Euler physics takes ideal gases only; cell zone '" + z.zone + "' is " +
                     z.medium.kind + " '" + z.medium.name + "'",
          .path = path,
          .data_json = nlohmann::json{{"zone", z.zone}, {"medium", z.medium.kind}}.dump()};
    }
    if (first == nullptr) {
      first = &z.medium;
    } else if (!same_gas(z.medium, *first)) {
      return runtime::PhysicsRefusal{
          .message = "the Euler physics takes one gas in every cell zone; cell zone '" + z.zone +
                     "' is '" + z.medium.name + "', not '" + first->name + "'",
          .path = path,
          .data_json = nlohmann::json{{"zone", z.zone}, {"medium", z.medium.name}}.dump()};
    }
  }
  for (std::size_t i = 0; i < in.boundaries.size(); ++i) {
    const auto& b = in.boundaries[i];
    if (b.kind != "slip" && b.kind != "symmetry" && b.kind != "farfield" && b.kind != "inlet" &&
        b.kind != "outlet" && b.kind != "manufactured" && b.kind != "oracle-state") {
      return runtime::PhysicsRefusal{
          .message = "the Euler physics has no '" + b.kind + "' condition; boundary zone '" +
                     b.zone + "' needs slip, symmetry, farfield, inlet, outlet or oracle-state",
          .path = "boundaries[" + std::to_string(i) + "].kind",
          .data_json = nlohmann::json{{"zone", b.zone}, {"condition", b.kind}}.dump()};
    }
    if (!b.point_vortex || first == nullptr) continue;
    // A steady far-field model (lagged by one step in the circulation), planar in x-z (the lift
    // plane at zero sideslip) and subsonic.
    const auto& fs = in.freestream;
    if (in.time.kind != "steady") {
      return runtime::PhysicsRefusal{
          .message = "a point vortex needs a steady case; boundary zone '" + b.zone + "'",
          .path = "time.kind",
          .data_json = nlohmann::json{{"zone", b.zone}, {"time", in.time.kind}}.dump()};
    }
    if (fs.sideslip != 0.0) {
      return runtime::PhysicsRefusal{
          .message = "a point vortex needs zero sideslip; boundary zone '" + b.zone + "'",
          .path = "freestream.sideslip",
          .data_json = nlohmann::json{{"zone", b.zone}, {"sideslip", fs.sideslip}}.dump()};
    }
    const double mach =
        fs.mach.value_or(fs.speed / std::sqrt(*first->gamma * fs.pressure / fs.density));
    if (!(mach < 1.0)) {
      return runtime::PhysicsRefusal{
          .message = "a point vortex needs a subsonic freestream; boundary zone '" + b.zone + "'",
          .path = "freestream.mach",
          .data_json = nlohmann::json{{"zone", b.zone}, {"mach", mach}}.dump()};
    }
  }
  return std::nullopt;
}

// The Navier-Stokes physics: one ideal gas as Euler, the Euler conditions and no-slip walls; SA
// runs steady (its loosely coupled step is the steady solver's) and needs a wall to measure
// distances from, unless a manufactured solution gives them.
std::optional<runtime::PhysicsRefusal> navier_stokes_scope(const runtime::SolverInput& in,
                                                           int ranks) {
  // case-schema.md#time's opt-in steady steps: neither is written for this physics yet.
  if (in.time.kind == "steady" && in.time.coupled_turbulence) {
    return runtime::PhysicsRefusal{
        .message = "the Navier-Stokes physics has no coupled turbulence step yet",
        .path = "time.coupled_turbulence",
        .data_json = "{}",
        .code = "E0426"};
  }
  if (in.time.kind == "steady" && in.time.matrix_free) {
    return runtime::PhysicsRefusal{
        .message = "the Navier-Stokes physics has no matrix-free step yet",
        .path = "time.matrix_free",
        .data_json = "{}",
        .code = "E0429"};
  }
  // Under reproducibility "ranks" the viscous and turbulence loops, the boundary fit and SA's own
  // sweeps follow the global order; the steady multigrid still sums in local cell order.
  if (in.reproducibility == "ranks" && in.time.kind == "steady" &&
      in.time.linear_solver == "multigrid") {
    return runtime::PhysicsRefusal{
        .message =
            "the steady multigrid does not give the same bits in every cell order yet; "
            "linear_solver = \"gauss-seidel\" or \"lines\" does",
        .path = "reproducibility",
        .data_json = "{}",
        .code = "E1109"};
  }
  // The viscous terms' ghost fills and boundary fits are written for partitions; across a periodic
  // pair the fits would need image ghosts' translated stencils.
  if (!in.periodic.empty()) {
    return runtime::PhysicsRefusal{
        .message = "the Navier-Stokes physics does not run with periodic pairs yet",
        .path = "periodic",
        .data_json = nlohmann::json{{"pairs", in.periodic.size()}}.dump()};
  }
  bool wall = false;
  for (std::size_t i = 0; i < in.boundaries.size(); ++i) {
    const auto& b = in.boundaries[i];
    if (b.kind == "wall") {
      wall = true;
      continue;
    }
    // case-schema.md: oracle-state faces for laminar flow only.
    const bool oracle = b.kind == "oracle-state" && in.turbulence.model == "laminar";
    if (!oracle && b.kind != "slip" && b.kind != "symmetry" && b.kind != "farfield" &&
        b.kind != "inlet" && b.kind != "outlet" && b.kind != "manufactured") {
      return runtime::PhysicsRefusal{
          .message = "the Navier-Stokes physics has no '" + b.kind +
                     "' condition here; boundary zone '" + b.zone +
                     "' needs wall, slip, symmetry, farfield, inlet, outlet, or oracle-state for "
                     "laminar flow",
          .path = "boundaries[" + std::to_string(i) + "].kind",
          .data_json = nlohmann::json{{"zone", b.zone}, {"condition", b.kind}}.dump()};
    }
  }
  if (in.turbulence.model == "sa") {
    if (in.time.kind != "steady") {
      return runtime::PhysicsRefusal{.message = "the SA physics runs steady cases only",
                                     .path = "time.kind",
                                     .data_json = nlohmann::json{{"time", in.time.kind}}.dump()};
    }
    if (!wall && !in.verification) {
      return runtime::PhysicsRefusal{
          .message = "the SA physics needs a wall boundary to measure the wall distance from",
          .path = "boundaries",
          .data_json = "{}"};
    }
  }
  // The Euler physics' checks, the gas's and the point vortex's among them, a wall standing for
  // the slip wall it is to the inviscid flux.
  runtime::SolverInput as_euler = in;
  for (auto& b : as_euler.boundaries) {
    if (b.kind == "wall") b.kind = "slip";
  }
  return euler_scope(as_euler, ranks);
}

#ifdef AUTAN_PHYSICS_CUDA
// On a device, besides euler_scope: one rank, and no point vortex (the device steps do not
// measure the circulation; the host's steady step does).
std::optional<runtime::PhysicsRefusal> euler_device_scope(const runtime::SolverInput& in,
                                                          int ranks) {
  if (in.time.kind == "steady" && (in.time.coupled_turbulence || in.time.matrix_free)) {
    const std::string path =
        in.time.coupled_turbulence ? "time.coupled_turbulence" : "time.matrix_free";
    return runtime::PhysicsRefusal{.message = path + " runs in the host multigrid, not on a device",
                                   .path = path,
                                   .data_json = "{}",
                                   .code = "E0427"};
  }
  // The device loops sum in local order: the host's global-order gathers have no device
  // counterpart in the Euler physics yet.
  if (in.reproducibility == "ranks") {
    return runtime::PhysicsRefusal{
        .message =
            "the Euler physics on a device does not give the same bits on any number of "
            "ranks yet",
        .path = "reproducibility",
        .data_json = "{}",
        .code = "E1109"};
  }
  if (ranks > 1) {
    return runtime::PhysicsRefusal{
        .message = "the Euler physics runs on one device only, not " + std::to_string(ranks),
        .path = "devices.kind",
        .data_json = nlohmann::json{{"ranks", ranks}}.dump()};
  }
  for (std::size_t i = 0; i < in.boundaries.size(); ++i) {
    if (in.boundaries[i].point_vortex) {
      return runtime::PhysicsRefusal{
          .message = "the Euler physics on a device has no point vortex yet; boundary zone '" +
                     in.boundaries[i].zone + "'",
          .path = "boundaries[" + std::to_string(i) + "].point_vortex",
          .data_json = nlohmann::json{{"zone", in.boundaries[i].zone}}.dump()};
    }
  }
  return euler_scope(in, ranks);
}

// The Navier-Stokes physics on a device: one rank, no point vortex, the host's limits otherwise;
// steady cases take the device's explicit steps. Registered for laminar flow only (ADR 0005): SA
// on a device has no entry, E0706.
std::optional<runtime::PhysicsRefusal> navier_stokes_device_scope(const runtime::SolverInput& in,
                                                                  int ranks) {
  if (ranks > 1) {
    return runtime::PhysicsRefusal{
        .message =
            "the Navier-Stokes physics runs on one device only, not " + std::to_string(ranks),
        .path = "devices.kind",
        .data_json = nlohmann::json{{"ranks", ranks}}.dump()};
  }
  for (std::size_t i = 0; i < in.boundaries.size(); ++i) {
    if (in.boundaries[i].point_vortex) {
      return runtime::PhysicsRefusal{
          .message = "the Navier-Stokes physics on a device has no point vortex; boundary zone '" +
                     in.boundaries[i].zone + "'",
          .path = "boundaries[" + std::to_string(i) + "].point_vortex",
          .data_json = nlohmann::json{{"zone", in.boundaries[i].zone}}.dump()};
    }
  }
  return navier_stokes_scope(in, ranks);
}
#endif

// EulerPhysics::restart_fields' names: the conserved variables, the averaging sums and weight,
// SA's transported variable, the steady solver's state, one circulation per point vortex.
std::vector<std::string> euler_restart_names(const runtime::SolverInput& in, bool averaging) {
  std::vector<std::string> out(conserved_names.begin(), conserved_names.end());
  if (averaging) {
    for (const char* name : field_names) out.push_back(std::string("AverageSum") + name);
    out.emplace_back("AverageWeight");
  }
  if (in.turbulence.model == "sa") out.emplace_back("DensityNuTilde");
  if (in.time.kind == "steady") {
    out.insert(out.end(),
               {"SolverPenalty", "SolverShortened", "SolverSecondOrder", "SolverFirstOrderSteps"});
    for (std::size_t k = 0; k < 5; ++k) {
      out.push_back("SolverReferenceResidual" + std::to_string(k));
    }
    out.emplace_back("SolverCflCap");
    out.insert(out.end(), {"SolverFirstOrderPlateauReference", "SolverFirstOrderPlateauCount",
                           "SolverFirstOrderPlateauSince", "SolverPreviousDrop"});
  }
  std::size_t vortices = 0;
  for (const auto& bc : in.boundaries) {
    if (bc.point_vortex) out.push_back("VortexCirculation" + std::to_string(vortices++));
  }
  return out;
}

}  // namespace

template <class P>
double device_stencil_gain(const Euler<P>& flow, const Mesh& mesh, const Geometry& g) {
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  std::vector<double> reach(nc, 0.0);
  for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_faces()); ++f) {
    auto at = [&](std::size_t c) {
      const double d = std::hypot(g.face_centroid_x[f] - g.cell_centroid_x[c],
                                  g.face_centroid_y[f] - g.cell_centroid_y[c],
                                  g.face_centroid_z[f] - g.cell_centroid_z[c]);
      reach[c] = std::max(reach[c], d);
    };
    at(static_cast<std::size_t>(mesh.owner()[f]));
    if (f < static_cast<std::size_t>(mesh.n_interior_faces())) {
      at(static_cast<std::size_t>(mesh.neighbour()[f]));
    }
  }
  auto gain = [&](auto offsets, std::size_t entries, const std::vector<double>& w) {
    double worst = 0.0;
    for (std::size_t c = 0; c < nc; ++c) {
      double sum = 0.0;
      for (auto e = static_cast<std::size_t>(offsets[c]);
           e < static_cast<std::size_t>(offsets[c + 1]); ++e) {
        sum += std::hypot(w[e], w[entries + e], w[2 * entries + e]);
      }
      worst = std::max(worst, sum * reach[c]);
    }
    return worst;
  };
  const auto& st = flow.stencil();
  double worst = gain(st.offsets, st.entries.size(), flow.gradient_weights());
  if (flow.scheme().gradient_fit == GradientFit::quadratic) {
    const auto q = flow.quadratic_gradient();
    worst = std::max(worst, gain(q.offsets, q.entries.size(), q.weights));
  }
  return worst;
}

template double device_stencil_gain(const Euler<kernels::fp64>&, const Mesh&, const Geometry&);
template double device_stencil_gain(const Euler<kernels::fp32_acc64>&, const Mesh&,
                                    const Geometry&);

// linear_solver = "lines": the multigrid's fine lines, computed on the global mesh, which the
// partition keeps whole.
runtime::CellGroups euler_groups(const runtime::SolverInput& in, const Mesh& mesh,
                                 const Geometry& geometry) {
  if (in.time.kind != "steady" || in.time.linear_solver != "lines") return {};
  auto lines = smoother_lines(mesh, geometry);
  if (lines.cells.empty()) return {};
  return {std::move(lines.offsets), std::move(lines.cells)};
}

void register_euler(runtime::PhysicsRegistry& registry) {
  registry.add({"compressible", "inviscid", "fp64", "cpu"},
               factory<kernels::fp64, Euler<kernels::fp64>>(), euler_scope,
               /*several_ranks=*/true, euler_groups, euler_restart_names,
               EulerPhysics<kernels::fp64, Euler<kernels::fp64>>::ghosts);
  registry.add({"compressible", "inviscid", "fp32+acc64", "cpu"},
               factory<kernels::fp32_acc64, Euler<kernels::fp32_acc64>>(), euler_scope,
               /*several_ranks=*/true, euler_groups, euler_restart_names,
               EulerPhysics<kernels::fp32_acc64, Euler<kernels::fp32_acc64>>::ghosts);
#ifdef AUTAN_PHYSICS_CUDA
  registry.add({"compressible", "inviscid", "fp64", "cuda"},
               factory<kernels::fp64, Euler<kernels::fp64>>(), euler_device_scope,
               /*several_ranks=*/false, {}, euler_restart_names,
               EulerPhysics<kernels::fp64, Euler<kernels::fp64>>::ghosts);
  registry.add({"compressible", "inviscid", "fp32+acc64", "cuda"},
               factory<kernels::fp32_acc64, Euler<kernels::fp32_acc64>>(), euler_device_scope,
               /*several_ranks=*/false, {}, euler_restart_names,
               EulerPhysics<kernels::fp32_acc64, Euler<kernels::fp32_acc64>>::ghosts);
#endif
}

void register_navier_stokes(runtime::PhysicsRegistry& registry) {
  for (const char* model : {"laminar", "sa"}) {
    registry.add({"compressible", model, "fp64", "cpu"},
                 factory<kernels::fp64, NavierStokes<kernels::fp64>>(), navier_stokes_scope,
                 /*several_ranks=*/true, euler_groups, euler_restart_names,
                 EulerPhysics<kernels::fp64, NavierStokes<kernels::fp64>>::ghosts, {},
                 /*wall_distance=*/std::string(model) == "sa");
    registry.add({"compressible", model, "fp32+acc64", "cpu"},
                 factory<kernels::fp32_acc64, NavierStokes<kernels::fp32_acc64>>(),
                 navier_stokes_scope, /*several_ranks=*/true, euler_groups, euler_restart_names,
                 EulerPhysics<kernels::fp32_acc64, NavierStokes<kernels::fp32_acc64>>::ghosts, {},
                 /*wall_distance=*/std::string(model) == "sa");
  }
#ifdef AUTAN_PHYSICS_CUDA
  registry.add({"compressible", "laminar", "fp64", "cuda"},
               factory<kernels::fp64, NavierStokes<kernels::fp64>>(), navier_stokes_device_scope,
               /*several_ranks=*/false, {}, euler_restart_names);
  registry.add({"compressible", "laminar", "fp32+acc64", "cuda"},
               factory<kernels::fp32_acc64, NavierStokes<kernels::fp32_acc64>>(),
               navier_stokes_device_scope, /*several_ranks=*/false, {}, euler_restart_names);
#endif
}

}  // namespace autan::physics
