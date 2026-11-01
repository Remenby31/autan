#include "autan/physics/euler_physics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

#include "autan/core/wall_distance.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/physics/euler.hpp"
#include "autan/physics/manufactured.hpp"
#include "autan/solvers/gcr.hpp"
#include "autan/physics/navier_stokes.hpp"
#include "autan/physics/spalart_allmaras.hpp"
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
// Jacobian, local dt = CFL V / lambda. The CFL number, the step length and the line search are
// ADflow's approximate Newton-Krylov (ANK) phase (Yildirim, Kenway, Mader and Martins 2019,
// J. Comput. Phys. 397:108741, secs. 3.5-3.7), with its published defaults.
constexpr double cfl_start = 5.0;          // CFL^(0)
constexpr double cfl_max = 1e5;            // CFL_max
constexpr double cfl_ramp = 10.0;          // alpha of the mRDM ramp, eq. 18
constexpr double cfl_floor_exponent = 0.5;  // beta of the SER floor, eq. 19
constexpr double omega_ramp = 0.4;          // a step longer than this ramps the CFL number up
constexpr double omega_min = 0.01;          // a step shorter than this is not taken
constexpr double cfl_cut = 0.5;             // theta_red
constexpr double backtrack = 0.7;           // theta_bt
constexpr double physical_fraction = 0.2;   // theta_phys of density and energy, eq. 15
constexpr double first_order_drop = 1e-3;
constexpr int first_order_max_steps = 200;
// Each step's linear system is solved to this relative residual (eta_lin), with at most this
// many multigrid-preconditioned iterations; a solve that misses it still gives the step.
constexpr double linear_tolerance = 0.05;
constexpr int linear_max_cycles = 10;
// Newton-Krylov phases (fp64): in second order, each step solves (V / dt + dR/dU) du = -R with
// dR/dU v by a finite difference of the residual, the limiter frozen (ANK's R1 level), the
// multigrid on the approximate Jacobian as preconditioner (ANK eq. 11); n_lin = 40.
constexpr int newton_max_iterations = 40;
// Below this relative residual, ANK hands over to Newton-Krylov with the full residual and
// HANIM's controls (Wang, Diskin, Nielsen and Liu 2021, AIAA SciTech, sec. II.D).
constexpr double newton_drop = 1e-5;
constexpr double ser_cfl_start = 1000.0;
constexpr double ser_cfl_min = 1.0;
constexpr double ser_cfl_max = 1e6;
constexpr double hanim_cfl_increase = 2.0;
constexpr double hanim_cfl_decrease = 0.1;
constexpr double hanim_cfl_min = 1e-12;
constexpr double hanim_cfl_max = 1e10;
constexpr double hanim_tolerance = 0.1;
constexpr double linear_failure = 0.9;
constexpr double max_decrease = 0.5;
template <class State>
constexpr double round_off_correction =
    std::max(1e-12, 8.0 * static_cast<double>(std::numeric_limits<State>::epsilon()));
// Largest relative decrease and increase of rho nt in one step, and the symmetric Gauss-Seidel
// sweeps of its scalar system (SA).
constexpr double nt_max_decrease = 0.5;
constexpr double nt_max_increase = 2.0;
constexpr int nt_sweeps = 4;
constexpr double venkatakrishnan_k = 10.0;

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
          bool ft2)
      : model(mesh, geometry, flow, t, ft2) {}
  SpalartAllmaras<P> model;
  std::vector<double> rnu, rhs, dx, diag, upper, lower;
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

  explicit EulerPhysics(const PhysicsContext& ctx)
      : ctx_(ctx),
        gas_(gas_of(ctx.input)),
        n_(static_cast<std::size_t>(ctx.mesh.n_cells())),
        euler_(make_flow(ctx, gas_)),
        reference_(scheme(ctx.input).reference) {
    const auto& mesh = ctx.mesh;
    const auto& in = ctx.input;
    std::optional<Manufactured> mms;
    if (in.verification) mms.emplace(in.verification->text);
    set_boundaries(mms ? &*mms : nullptr);
    if (in.time.kind == "steady") {
      matrix_.emplace(mesh);
      multigrid_.emplace(mesh, ctx.geometry);
      nlohmann::json nodes = nlohmann::json::array();
      for (const auto& l : multigrid_->levels()) nodes.push_back(l.n_nodes);
      ctx.log.write("info", "linear", "agglomeration multigrid", {},
                    nlohmann::json{{"nodes", nodes},
                                   {"coarsest", multigrid_->direct_coarsest() ? "lu" : "sweeps"}}
                        .dump());
      euler_.set_second_order(false);
    } else {
      second_order_ = true;
    }
    if (mms) set_source(*mms);
    std::map<std::string, const runtime::CellZoneInput*> zones;
    for (const auto& z : in.cell_zones) zones[z.zone] = &z;
    u_.assign(5 * n_, 0.0);
    for (std::size_t c = 0; c < n_; ++c) {
      const auto zi = static_cast<std::size_t>(mesh.cell_zone()[c]);
      const auto& init = zones.at(mesh.cell_zones()[zi].name)->initial;
      set_cell(c, *init.density, *init.velocity, *init.pressure);
    }
    state_.assign(5 * n_, State(0));
    res_.assign(5 * n_, Res(0));
    lambda_.assign(n_, State(0));
    stage_.assign(5 * n_, 0.0);
    rhs_.assign(5 * n_, 0.0);
    du_.assign(5 * n_, 0.0);
    mean_.assign(6 * n_, 0.0);
    if constexpr (is_viscous) {
      if (in.turbulence.model == "sa") init_sa(mms ? &*mms : nullptr);
    }
    if (ctx.start != nullptr) start_from(*ctx.start);
  }

  StepReport step(std::int64_t iteration) override {
    StepReport report;
    // A state the kernels cannot evaluate (as stored, see first_invalid_cell) diverges here,
    // before a solve turns it into an exception.
    if (const auto bad = first_invalid_cell()) {
      residual_of(u_);
      record_norms(report);
      report.diverged_cell = bad;
      return report;
    }
    if (ctx_.input.time.kind == "steady") {
      steady_step(report, iteration);
    } else {
      unsteady_step(report);
    }
    if (!report.diverged_cell) report.diverged_cell = first_invalid_cell();
    return report;
  }

  void accumulate_average(double weight) override {
    const auto f = primitive_fields();
    for (std::size_t k = 0; k < 6; ++k) {
      for (std::size_t c = 0; c < n_; ++c) mean_[k * n_ + c] += weight * f[k][c];
    }
    mean_weight_ += weight;
  }

  // The conserved variables bit for bit, then the averaging sums and their weight.
  [[nodiscard]] std::vector<io::Field> restart_fields() const override {
    std::vector<io::Field> out;
    for (std::size_t k = 0; k < 5; ++k) {
      out.push_back(
          {.name = conserved_names[k],
           .values = std::vector<double>(u_.begin() + static_cast<std::ptrdiff_t>(k * n_),
                                         u_.begin() + static_cast<std::ptrdiff_t>((k + 1) * n_))});
    }
    if (mean_weight_ > 0.0) {
      for (std::size_t k = 0; k < 6; ++k) {
        out.push_back({.name = std::string("AverageSum") + field_names[k],
                       .values = std::vector<double>(
                           mean_.begin() + static_cast<std::ptrdiff_t>(k * n_),
                           mean_.begin() + static_cast<std::ptrdiff_t>((k + 1) * n_))});
      }
      out.push_back({.name = "AverageWeight", .values = std::vector<double>(n_, mean_weight_)});
    }
    if (sa_) out.push_back({.name = "DensityNuTilde", .values = sa_->rnu});
    // The steady solver's own state, one uniform field per value (cell-centred, as RestartState).
    if (ctx_.input.time.kind == "steady") {
      for (const auto& [name, value] : solver_state()) {
        out.push_back({.name = name, .values = std::vector<double>(n_, value)});
      }
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
    for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_faces()); ++f) {
      surface[static_cast<std::size_t>(mesh.owner()[f])] += g.face_area[f];
      if (f < static_cast<std::size_t>(mesh.n_interior_faces())) {
        surface[static_cast<std::size_t>(mesh.neighbour()[f])] += g.face_area[f];
      }
    }
    double sum = 0.0;
    for (const double a : surface) sum += a * a;
    const double area = std::sqrt(sum / static_cast<double>(n_));
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
    std::vector<io::Field> out;
    const auto f = primitive_fields();
    for (std::size_t k = 0; k < 6; ++k) {
      io::Field field{.name = field_names[k], .values = f[k]};
      if (mean) {
        for (std::size_t c = 0; c < n_; ++c) field.values[c] = mean_[k * n_ + c] / mean_weight_;
      }
      out.push_back(std::move(field));
    }
    if (sa_) {
      if (mean) throw std::invalid_argument("time averages of SA runs are not supported");
      io::Field nt{.name = "TurbulentSANuTilde", .values = std::vector<double>(n_)};
      for (std::size_t c = 0; c < n_; ++c) nt.values[c] = sa_->rnu[c] / u_[c];
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
    const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
    const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
    std::vector<double> p(nb), face_force(3 * nb);
    face_pressure(p, face_force);
    const auto t = face_traction();
    const double p_inf = ctx_.input.freestream.pressure;
    const auto offsets = mesh.boundary_zone_offsets();
    Loads loads;
    for (const auto z : zones) {
      for (auto f = offsets[static_cast<std::size_t>(z)];
           f < offsets[static_cast<std::size_t>(z) + 1]; ++f) {
        const auto i = static_cast<std::size_t>(f);
        const auto b = i - ni;
        const double a =
            std::sqrt(g.face_area_x[i] * g.face_area_x[i] + g.face_area_y[i] * g.face_area_y[i] +
                      g.face_area_z[i] * g.face_area_z[i]);
        const std::array<double, 3> force{
            face_force[b] - p_inf * g.face_area_x[i] + a * t[b],
            face_force[nb + b] - p_inf * g.face_area_y[i] + a * t[nb + b],
            face_force[2 * nb + b] - p_inf * g.face_area_z[i] + a * t[2 * nb + b]};
        const std::array<double, 3> arm{g.face_centroid_x[i] - center[0],
                                        g.face_centroid_y[i] - center[1],
                                        g.face_centroid_z[i] - center[2]};
        for (std::size_t k = 0; k < 3; ++k) loads.force[k] += force[k];
        loads.moment[0] += arm[1] * force[2] - arm[2] * force[1];
        loads.moment[1] += arm[2] * force[0] - arm[0] * force[2];
        loads.moment[2] += arm[0] * force[1] - arm[1] * force[0];
      }
    }
    return loads;
  }

 private:
  // Guaranteed copy elision: Flow is neither copyable nor movable.
  static Flow make_flow(const PhysicsContext& ctx, const Gas& gas) {
    if constexpr (is_viscous) {
      return Flow(ctx.mesh, ctx.geometry, gas, scheme(ctx.input), transport_of(ctx.input));
    } else {
      return Flow(ctx.mesh, ctx.geometry, gas, scheme(ctx.input));
    }
  }

  static Scheme scheme(const runtime::SolverInput& in) {
    const auto& f = in.freestream;
    const double c = std::sqrt(*in.cell_zones.front().medium.gamma * f.pressure / f.density);
    const double speed = std::max(f.speed, c);
    Scheme s;
    s.flux = Flux::roe;
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
      if (ctx_.input.time.kind == "steady") {
        auto state = solver_state();
        for (auto& [name, value] : state) value = find(start.restart_fields, name)->front();
        set_solver_state(state);
      }
      if (sa_) sa_->rnu = *find(start.restart_fields, "DensityNuTilde");
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
    for (std::size_t c = 0; c < n_; ++c) set_cell(c, rho[c], {vx[c], vy[c], vz[c]}, p[c]);
    if (sa_) {
      const auto& nt = *find(start.fields, "TurbulentSANuTilde");
      for (std::size_t c = 0; c < n_; ++c) sa_->rnu[c] = rho[c] * nt[c];
    }
  }

  [[nodiscard]] std::vector<std::pair<std::string, double>> solver_state() const {
    std::vector<std::pair<std::string, double>> v = {
        {"SolverCfl", cfl_},
        {"SolverSecondOrder", second_order_ ? 1.0 : 0.0},
        {"SolverFirstOrderSteps", static_cast<double>(first_order_steps_)}};
    for (std::size_t k = 0; k < 5; ++k) {
      v.emplace_back("SolverReferenceResidual" + std::to_string(k),
                     first_norms_.empty() ? -1.0 : first_norms_[k]);
    }
    v.emplace_back("SolverExemptFloor", exempt_floor_);
    for (std::size_t k = 0; k < 5; ++k) {
      v.emplace_back("SolverExempt" + std::to_string(k), exempt_[k] ? 1.0 : 0.0);
    }
    v.emplace_back("SolverPreviousResidual", previous_norm_);
    v.emplace_back("SolverPreviousStep", previous_omega_);
    v.emplace_back("SolverFreestreamResidual", freestream_norm_);
    return v;
  }

  void set_solver_state(const std::vector<std::pair<std::string, double>>& v) {
    cfl_ = v[0].second;
    second_order_ = v[1].second != 0.0;
    euler_.set_second_order(second_order_);
    first_order_steps_ = static_cast<int>(v[2].second);
    first_norms_.clear();
    if (v[3].second >= 0.0) {
      for (std::size_t k = 0; k < 5; ++k) first_norms_.push_back(v[3 + k].second);
    }
    exempt_floor_ = v[8].second;
    for (std::size_t k = 0; k < 5; ++k) exempt_[k] = v[9 + k].second != 0.0;
    previous_norm_ = v[14].second;
    previous_omega_ = v[15].second;
    freestream_norm_ = v[16].second;
  }

  // SA-neg: rho nt from the zones' initial nt, nt at the boundary faces that give it, the wall
  // distance; a manufactured solution gives its own nt, wall distance and source.
  void init_sa(const Manufactured* mms) {
    const auto& mesh = ctx_.mesh;
    const auto& g = ctx_.geometry;
    const auto& in = ctx_.input;
    if (in.time.kind != "steady") {
      throw std::invalid_argument("the SA physics runs steady cases only");
    }
    sa_ = std::make_unique<SaState<P>>(mesh, g, euler_, transport_of(in),
                                       mms == nullptr || mms->turbulence() != "sa-noft2");
    auto& t = *sa_;
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
      std::vector<index_t> walls;
      for (std::size_t z = 0; z < mesh.face_zones().size(); ++z) {
        const auto it = kinds.find(mesh.face_zones()[z].name);
        if (it != kinds.end() && it->second == "wall") walls.push_back(static_cast<index_t>(z));
      }
      if (walls.empty()) throw std::invalid_argument("the SA physics needs a wall");
      d = wall_distance(mesh, g, walls);
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
    t.rhs.assign(n_, 0.0);
    t.dx.assign(n_, 0.0);
    t.diag.assign(n_, 0.0);
    t.upper.assign(static_cast<std::size_t>(mesh.n_interior_faces()), 0.0);
    t.lower.assign(t.upper.size(), 0.0);
    t.s.assign(n_, typename P::state(0));
    t.res.assign(n_, typename P::residual(0));
  }

  void set_cell(std::size_t c, double rho, const std::array<double, 3>& v, double p) {
    u_[c] = rho;
    for (std::size_t k = 0; k < 3; ++k) u_[(1 + k) * n_ + c] = rho * v[k];
    u_[4 * n_ + c] = p / (gas_.gamma - 1.0) + 0.5 * rho * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }

  void set_boundaries(const Manufactured* mms) {
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
    euler_.residual(state_, res_);
  }

  void record_norms(StepReport& report) const {
    report.residuals.assign(5, 0.0);
    for (std::size_t k = 0; k < 5; ++k) {
      double s = 0.0;
      for (std::size_t c = 0; c < n_; ++c) {
        const double r = static_cast<double>(res_[k * n_ + c]);
        s += r * r;
      }
      report.residuals[k] = std::sqrt(s / static_cast<double>(n_));
    }
  }

  void steady_step(StepReport& report, std::int64_t iteration) {
    euler_.state(u_, state_);
    if (!second_order_) {
      // The history reports the second-order residual from the first step on, so that the
      // runtime never judges convergence on the first-order one.
      euler_.set_second_order(true);
      euler_.residual(state_, res_);
      record_norms(report);
      euler_.set_second_order(false);
    }
    euler_.residual_and_jacobian(state_, res_);
    std::vector<double> norms = report.residuals;
    record_norms(report);
    if (!norms.empty()) std::swap(norms, report.residuals);
    if (sa_) {
      // Loosely coupled: the SA residual at the same state, after the flow's evaluation.
      auto& t = *sa_;
      t.model.residual_and_jacobian(t.stored(t.rnu), t.res);
      double sum = 0.0;
      for (const auto r : t.res) sum += static_cast<double>(r) * static_cast<double>(r);
      report.residuals.push_back(std::sqrt(sum / static_cast<double>(n_)));
    }
    euler_.spectral_radius(lambda_);
    // Largest relative residual over the equations, of the order being solved: each residual
    // relative to its largest value so far in this order, so that the transient of a start from
    // the freestream (a first residual near zero) does not hold the CFL number at its minimum.
    // An equation identically zero at the start of the order (below 1e-12 of the largest, the
    // span momentum of a one-cell-thick mesh) is left out while it stays at that floor, as the
    // runtime's convergence rule does (case-schema.md#time): its round-off would set the CFL.
    const std::vector<double>& current = norms.empty() ? report.residuals : norms;
    if (first_norms_.empty()) {
      first_norms_ = current;
      exempt_floor_ = 1e-12 * *std::max_element(current.begin(), current.end());
      for (std::size_t k = 0; k < 5; ++k) exempt_[k] = current[k] < exempt_floor_;
    }
    double drop = 0.0;
    for (std::size_t k = 0; k < 5; ++k) {
      first_norms_[k] = std::max(first_norms_[k], current[k]);
      if (exempt_[k] && current[k] > exempt_floor_) exempt_[k] = false;
      if (!exempt_[k] && first_norms_[k] > 0.0) {
        drop = std::max(drop, current[k] / first_norms_[k]);
      }
    }
    // Phases (fp64): second order runs ANK with matrix-free products of the residual, the
    // limiter frozen; below newton_drop, Newton-Krylov on the full residual with HANIM's controls.
    static const bool exp_mf = [] {
      const char* e = std::getenv("AUTAN_EXP_MF");
      return e == nullptr || std::string(e) != "0";
    }();
    static const double exp_nk_drop = [] {
      const char* e = std::getenv("AUTAN_EXP_NKDROP");
      return e == nullptr ? newton_drop : std::stod(e);
    }();
    static const bool exp_ser = [] {
      const char* e = std::getenv("AUTAN_EXP_START");
      return e != nullptr && std::string(e).starts_with("ser");
    }();
    static const bool exp_ser0 = [] {
      const char* e = std::getenv("AUTAN_EXP_START");
      return e != nullptr && std::string(e) == "ser0";
    }();
    // main's SER ratio: each residual relative to its first value in the order.
    if (first0_.empty()) first0_ = current;
    double drop0 = 0.0;
    for (std::size_t k = 0; k < 5; ++k) {
      if (first0_[k] > 0.0) drop0 = std::max(drop0, current[k] / first0_[k]);
    }
    constexpr bool fp64 = std::is_same_v<P, kernels::fp64>;
    const bool hanim = fp64 && second_order_ && drop <= exp_nk_drop;
    const bool matrix_free = fp64 && second_order_ && (hanim || (exp_mf && !exp_ser));
    const bool ser = exp_ser && !hanim;
    if (hanim != newton_) {
      newton_ = hanim;
      if (hanim) {
        ctx_.log.write("info", "solver", "Newton-Krylov steps", {},
                       nlohmann::json{{"cfl", cfl_}, {"iteration", iteration}}.dump());
      }
    }
    // The CFL number of this step (ANK eqs. 18-20) from the previous step's length and residual,
    // never below the SER floor (|R_fs| / |R|)^beta, R_fs the residual at the first step.
    const double norm = scaled_norm(res_, {}, 0.0);
    if (freestream_norm_ <= 0.0) freestream_norm_ = norm;
    const double floor =
        norm > 0.0 ? std::pow(freestream_norm_ / norm, cfl_floor_exponent) : cfl_max;
    if (ser) {
      // Switched evolution relaxation as on #205 (Mulder and van Leer 1985).
      penalty_ = shortened_ ? 0.5 * penalty_ : std::min(1.0, 1.5 * penalty_);
      if (exp_ser0) {
        if (drop0 > 0.0) cfl_ = std::clamp(penalty_ * ser_cfl_start / drop0, ser_cfl_min, ser_cfl_max);
      } else {
        const double ratio = drop > 0.0 ? penalty_ * ser_cfl_start / drop : ser_cfl_max;
        cfl_ = std::clamp(std::min(ratio, cfl_cap_), ser_cfl_min, ser_cfl_max);
        cfl_cap_ = ser_cfl_max;
      }
    } else if (!hanim && previous_norm_ > 0.0) {
      if (previous_omega_ > omega_ramp) {
        const double gamma = std::max((previous_norm_ - norm) / previous_norm_, 0.0);
        cfl_ = std::min(cfl_ * std::pow(cfl_ramp, gamma), cfl_max);
      } else if (previous_omega_ > omega_min) {
        cfl_ = std::max(cfl_, floor);
      } else {
        cfl_ = std::max(cfl_cut * cfl_, floor);
      }
    }
    std::vector<double> shift(n_);
    for (std::size_t c = 0; c < n_; ++c) shift[c] = static_cast<double>(lambda_[c]) / cfl_;
    euler_.assemble(shift, *matrix_);
    for (std::size_t i = 0; i < 5 * n_; ++i) rhs_[i] = -static_cast<double>(res_[i]);
    std::fill(du_.begin(), du_.end(), 0.0);
    const double tolerance = hanim || ser ? hanim_tolerance : linear_tolerance;
    solvers::Multigrid::Report linear;
    try {
      multigrid_->set_operator(matrix_->diagonals(), matrix_->uppers(), matrix_->lowers());
      if (matrix_free) {
        euler_.freeze_limiter(!hanim);
        linear = newton_solve(shift, tolerance);
        euler_.freeze_limiter(false);
      } else {
        linear = multigrid_->solve(rhs_, du_, linear_max_cycles, tolerance);
      }
    } catch (const std::domain_error&) {
      euler_.freeze_limiter(false);
      linear = {};  // a singular block
      linear.residual_drop = std::numeric_limits<double>::quiet_NaN();
    }
    const nlohmann::json residual_drop =
        std::isfinite(linear.residual_drop) ? nlohmann::json(linear.residual_drop) : nullptr;
    if (!(linear.residual_drop <= tolerance) && std::isfinite(linear.residual_drop)) {
      ctx_.log.write(
          "info", "linear", "the implicit step's linear solve stopped above its tolerance", {},
          nlohmann::json{{"cycles", linear.cycles}, {"residual_drop", residual_drop}}.dump());
    }
    double omega = 0.0;
    if (hanim) {
      omega = hanim_step(report, iteration, shift, linear, residual_drop);
    } else if (ser) {
      omega = ser_step(report, iteration, linear, residual_drop, exp_ser0 ? drop0 : drop);
    } else {
      // Physicality (ANK eq. 15, density and energy) then backtracking on the unsteady residual
      // |w V/dt du + R(u + w du)| until it is below |R(u)| (ANK eq. 17, algorithm 2).
      if (std::isfinite(linear.residual_drop)) {
        omega = physical_step();
        double unsteady = scaled_norm(trial_residual(omega), shift, omega);
        while (!(unsteady <= norm) && omega > omega_min) {
          omega *= backtrack;
          unsteady = scaled_norm(trial_residual(omega), shift, omega);
        }
        if (!(unsteady <= norm) && (cfl_ > floor || !std::isfinite(unsteady))) {
          if (!std::isfinite(unsteady) && cfl_ <= floor) {
            report.diverged_cell = static_cast<index_t>(worst_cell());
          }
          omega = 0.0;
        }
      }
      if (!(omega > 0.0)) {
        ctx_.log.write("warning", "solver", "implicit step rejected: no decrease of the residual",
                       "W0711",
                       nlohmann::json{{"iteration", iteration},
                                      {"reason", std::isfinite(linear.residual_drop) ? "nonlinear"
                                                                                     : "linear"},
                                      {"cfl", cfl_},
                                      {"residual_drop", residual_drop}}
                           .dump());
      } else if (omega < 1.0) {
        ctx_.log.write("info", "solver", "line search", {},
                       nlohmann::json{{"omega", omega}, {"cfl", cfl_}}.dump());
      }
    }
    if (omega > 0.0) {
      for (std::size_t i = 0; i < 5 * n_; ++i) u_[i] += omega * du_[i];
      if (sa_) {
        // The first-order start has no velocity gradients, hence no production: SA waits for
        // the second order.
        auto& t = *sa_;
        if (second_order_) {
          for (std::size_t c = 0; c < n_; ++c) t.rhs[c] = -static_cast<double>(t.res[c]);
          std::fill(t.dx.begin(), t.dx.end(), 0.0);
          t.model.assemble(shift, t.diag, t.upper, t.lower);
          t.model.solve(t.diag, t.upper, t.lower, t.rhs, t.dx, nt_sweeps);
          for (std::size_t c = 0; c < n_; ++c) {
            const double scale = std::max(std::abs(t.rnu[c]), u_[c] * t.nt_inf);
            t.rnu[c] +=
                omega * std::clamp(t.dx[c], -nt_max_decrease * scale, nt_max_increase * scale);
          }
        }
        t.model.eddy_viscosity(t.stored(t.rnu));
      }
    }
    previous_omega_ = omega;
    previous_norm_ = norm;
    // First-order start: its approximate Jacobian is exact, so the steps are robust far from the
    // solution; second order from the step after the first-order residual has dropped.
    if (!second_order_ &&
        (drop < first_order_drop || ++first_order_steps_ > first_order_max_steps)) {
      second_order_ = true;
      euler_.set_second_order(true);
      first_norms_.clear();
      first0_.clear();
      previous_norm_ = 0.0;  // the norms are scaled per order
    }
  }

  double ser_step(StepReport& report, std::int64_t iteration, const solvers::KrylovReport& linear,
                  const nlohmann::json& residual_drop, double drop) {
    if (!(linear.residual_drop <= linear_failure)) {
      ctx_.log.write("warning", "solver", "implicit step rejected: its linear solve failed",
                     "W0711",
                     nlohmann::json{{"iteration", iteration},
                                    {"reason", "linear"},
                                    {"cfl", cfl_},
                                    {"residual_drop", residual_drop}}
                         .dump());
      if (drop > 0.0) penalty_ = std::min(penalty_, cfl_ * drop / ser_cfl_start);
      cfl_cap_ = 0.5 * cfl_;
      shortened_ = true;
      if (cfl_ <= ser_cfl_min) report.diverged_cell = static_cast<index_t>(worst_cell());
      return 0.0;
    }
    const double alpha = admissible_step();
    shortened_ = alpha < 1.0;
    if (shortened_) {
      if (drop > 0.0) penalty_ = std::min(penalty_, cfl_ * drop / ser_cfl_start);
      cfl_cap_ = 0.5 * cfl_;
      ctx_.log.write("info", "solver", "step shortened", {},
                     nlohmann::json{{"alpha", alpha}, {"cfl", cfl_}}.dump());
    }
    if (alpha < 1e-6) {
      if (cfl_ <= ser_cfl_min) report.diverged_cell = static_cast<index_t>(worst_cell());
      return 0.0;
    }
    return alpha;
  }

  [[nodiscard]] double admissible_step() const {
    double alpha = 1.0;
    for (std::size_t c = 0; c < n_; ++c) {
      const double rho = u_[c], drho = du_[c];
      if (drho < 0.0) alpha = std::min(alpha, max_decrease * rho / -drho);
      const double p = pressure(c, 0.0);
      while (alpha >= 1e-6 && !(pressure(c, alpha) >= (1.0 - max_decrease) * p)) alpha *= 0.5;
    }
    return alpha;
  }

  // HANIM (Wang et al. 2021, sec. II.D) on the Newton-Krylov step: the step length, 0 when
  // rejected; the CFL number doubled after a full step, kept after a line-searched one, divided
  // by 10 after a rejection.
  double hanim_step(StepReport& report, std::int64_t iteration, const std::vector<double>& shift,
                    const solvers::KrylovReport& linear, const nlohmann::json& residual_drop) {
    auto reject = [&](const char* reason, const char* message, nlohmann::json drop_data) {
      ctx_.log.write("warning", "solver", message, "W0711",
                     nlohmann::json{{"iteration", iteration},
                                    {"reason", reason},
                                    {"cfl", cfl_},
                                    {"residual_drop", std::move(drop_data)}}
                         .dump());
      if (cfl_ <= hanim_cfl_min) report.diverged_cell = static_cast<index_t>(worst_cell());
      cfl_ = std::max(hanim_cfl_min, hanim_cfl_decrease * cfl_);
      return 0.0;
    };
    if (!(linear.residual_drop <= linear_failure)) {
      return reject("linear", "implicit step rejected: its linear solve failed", residual_drop);
    }
    const double nonlinear_target = 0.5 * (1.0 + std::max(hanim_tolerance, linear.residual_drop));
    if (!realisable(1.0)) {
      return reject("admissible", "implicit step rejected: not realisable", nullptr);
    }
    const double f0 = active_norm(res_);
    const double g1 = linear.residual_drop * f0;
    const double f1 = pseudo_time_residual(shift, 1.0);
    double omega = 1.0;
    bool accepted = f1 <= nonlinear_target * f0;
    const bool grow = accepted;
    if (!accepted && active_norm(du_) <= round_off_correction<State> * active_norm(u_)) {
      accepted = true;
    }
    if (!accepted && f1 > g1) {
      omega = -(g1 - f0) / (2.0 * (f1 - g1));  // HANIM eq. 22
      if (omega > 0.0 && omega < 1.0 && realisable(omega) &&
          pseudo_time_residual(shift, omega) <= nonlinear_target * f0) {
        accepted = true;
        ctx_.log.write("info", "solver", "line search", {},
                       nlohmann::json{{"omega", omega}, {"cfl", cfl_}}.dump());
      }
    }
    if (!accepted) {
      return reject("nonlinear", "implicit step rejected: too small a residual decrease",
                    residual_drop);
    }
    if (grow) cfl_ = std::min(hanim_cfl_max, hanim_cfl_increase * cfl_);
    return omega;
  }

  [[nodiscard]] double active_norm(std::span<const double> r) const {
    double s = 0.0;
    for (std::size_t k = 0; k < 5; ++k) {
      if (exempt_[k]) continue;
      for (std::size_t c = 0; c < n_; ++c) s += r[k * n_ + c] * r[k * n_ + c];
    }
    return std::sqrt(s / static_cast<double>(n_));
  }

  double pseudo_time_residual(const std::vector<double>& shift, double omega) {
    const auto r = trial_residual(omega);
    double s = 0.0;
    for (std::size_t k = 0; k < 5; ++k) {
      if (exempt_[k]) continue;
      for (std::size_t c = 0; c < n_; ++c) {
        const std::size_t i = k * n_ + c;
        const double v = shift[c] * omega * du_[i] + static_cast<double>(r[i]);
        s += v * v;
      }
    }
    return std::sqrt(s / static_cast<double>(n_));
  }

  [[nodiscard]] bool realisable(double omega) const {
    for (std::size_t c = 0; c < n_; ++c) {
      if (!(u_[c] + omega * du_[c] >= (1.0 - max_decrease) * u_[c])) return false;
      if (!(pressure(c, omega) >= (1.0 - max_decrease) * pressure(c, 0.0))) return false;
    }
    return true;
  }

  // Largest fraction of du (at most 1) that changes no cell's density or total energy by more
  // than physical_fraction of its value (ANK eq. 15).
  [[nodiscard]] double physical_step() const {
    double worst = 1.0;
    for (std::size_t c = 0; c < n_; ++c) {
      for (const std::size_t k : {std::size_t{0}, std::size_t{4}}) {
        const std::size_t i = k * n_ + c;
        worst = std::max(worst, std::abs(du_[i]) / (physical_fraction * std::abs(u_[i])));
      }
    }
    return 1.0 / worst;
  }

  // R(u + w du), the residual of the order being solved, into trial_res_.
  std::span<const Res> trial_residual(double omega) {
    trial_u_.resize(5 * n_);
    trial_state_.resize(5 * n_);
    trial_res_.resize(5 * n_);
    for (std::size_t i = 0; i < 5 * n_; ++i) trial_u_[i] = u_[i] + omega * du_[i];
    euler_.state(trial_u_, trial_state_);
    euler_.residual(trial_state_, trial_res_);
    return trial_res_;
  }

  // Norm of r + w shift du over the equations the ratio follows, each equation scaled by its
  // largest residual so far in the order (ADflow scales nondimensional states instead).
  [[nodiscard]] double scaled_norm(std::span<const Res> r, std::span<const double> shift,
                                   double omega) const {
    double s = 0.0;
    for (std::size_t k = 0; k < 5; ++k) {
      if (exempt_[k] || !(first_norms_[k] > 0.0)) continue;
      for (std::size_t c = 0; c < n_; ++c) {
        const std::size_t i = k * n_ + c;
        double v = static_cast<double>(r[i]);
        if (!shift.empty()) v += omega * shift[c] * du_[i];
        v /= first_norms_[k];
        s += v * v;
      }
    }
    return std::sqrt(s / static_cast<double>(n_));
  }

  // GCR on (V / dt + dR/dU) du = -R from du = 0, dR/dU v = (R(u + eps v) - R(u)) / eps with
  // eps = sqrt((1 + |u|) eps_mach) / |v| (Knoll and Keyes 2004, J. Comput. Phys. 193:357-397,
  // eq. 14), preconditioned by one multigrid cycle on the assembled approximate Jacobian.
  solvers::KrylovReport newton_solve(const std::vector<double>& shift, double tolerance) {
    auto norm = [](std::span<const double> v) {
      double s = 0.0;
      for (const double x : v) s += x * x;
      return std::sqrt(s);
    };
    const double unorm = norm(u_);
    newton_u_.resize(5 * n_);
    newton_state_.resize(5 * n_);
    newton_res_.resize(5 * n_);
    // The iteration runs on the equations the SER ratio follows: an equation identically zero
    // (the span momentum of a one-cell-thick mesh) stays zero, where the round-off of the finite
    // differences would wake it.
    auto restrict_to_active = [&](std::span<double> w) {
      for (std::size_t k = 0; k < 5; ++k) {
        if (exempt_[k]) std::fill_n(w.begin() + static_cast<std::ptrdiff_t>(k * n_), n_, 0.0);
      }
    };
    newton_r_ = rhs_;
    restrict_to_active(newton_r_);
    newton_v_.resize(5 * n_);
    auto apply = [&](std::span<const double> x, std::span<double> y) {
      std::copy(x.begin(), x.end(), newton_v_.begin());
      restrict_to_active(newton_v_);
      const std::vector<double>& v = newton_v_;
      const double vnorm = norm(v);
      if (!(vnorm > 0.0)) {
        std::fill(y.begin(), y.end(), 0.0);
        return;
      }
      const double eps = std::sqrt((1.0 + unorm) * std::numeric_limits<double>::epsilon()) / vnorm;
      for (std::size_t i = 0; i < 5 * n_; ++i) newton_u_[i] = u_[i] + eps * v[i];
      euler_.state(newton_u_, newton_state_);
      euler_.residual(newton_state_, newton_res_);
      for (std::size_t k = 0; k < 5; ++k) {
        for (std::size_t c = 0; c < n_; ++c) {
          const std::size_t i = k * n_ + c;
          y[i] = shift[c] * v[i] +
                 (static_cast<double>(newton_res_[i]) - static_cast<double>(res_[i])) / eps;
        }
      }
      restrict_to_active(y);
    };
    auto precondition = [&](std::span<const double> r, std::span<double> z) {
      multigrid_->cycle(r, z);
      restrict_to_active(z);
    };
    return newton_gcr_.solve(apply, precondition, newton_r_, du_, newton_max_iterations,
                             tolerance);
  }

  // The cell whose density or pressure the full step lowers the most, relatively.
  [[nodiscard]] std::size_t worst_cell() const {
    std::size_t worst = 0;
    double lowest = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < n_; ++c) {
      const double r = std::min((u_[c] + du_[c]) / u_[c], pressure(c, 1.0) / pressure(c, 0.0));
      if (r < lowest) {
        lowest = r;
        worst = c;
      }
    }
    return worst;
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
      for (std::size_t c = 0; c < n_; ++c) {
        dt = std::min(dt, *t.cfl * g.cell_volume[c] / static_cast<double>(lambda_[c]));
      }
    }
    const std::array<std::array<double, 2>, 3> weights{
        {{0.0, 1.0}, {0.75, 0.25}, {1.0 / 3.0, 2.0 / 3.0}}};
    stage_ = u_;
    for (std::size_t s = 0; s < 3; ++s) {
      if (s > 0) residual_of(stage_);
      for (std::size_t k = 0; k < 5; ++k) {
        for (std::size_t c = 0; c < n_; ++c) {
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
    for (std::size_t c = 0; c < n_; ++c) {
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
        return static_cast<index_t>(c);
      }
    }
    for (std::size_t c = 0; c < n_; ++c) {
      for (std::size_t k = 0; k < 5; ++k) {
        if (!std::isfinite(static_cast<double>(res_[k * n_ + c]))) return static_cast<index_t>(c);
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] std::array<std::vector<double>, 6> primitive_fields() const {
    std::array<std::vector<double>, 6> f;
    for (auto& v : f) v.resize(n_);
    for (std::size_t c = 0; c < n_; ++c) {
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

  // Viscous traction -tau n on every boundary face (3 x n_boundary_faces), zero for inviscid flow;
  // after face_pressure(), which evaluates the residual.
  [[nodiscard]] std::vector<double> face_traction() const {
    std::vector<double> t(3 * static_cast<std::size_t>(ctx_.mesh.n_boundary_faces()), 0.0);
    if constexpr (is_viscous) euler_.viscous_traction(t);
    return t;
  }

  void face_pressure(std::span<double> p, std::span<double> force) const {
    auto& self = const_cast<EulerPhysics&>(*this);
    self.residual_of(u_);
    euler_.boundary_pressure(p, force);
  }

  PhysicsContext ctx_;
  Gas gas_;
  std::size_t n_;
  Flow euler_;
  std::unique_ptr<SaState<P>> sa_;   // SA-neg runs only
  std::array<double, 5> reference_;  // the state the cells store differences from
  std::vector<double> u_, stage_, mean_, rhs_, du_, first_norms_;
  std::optional<BlockMatrix> matrix_;  // steady runs only
  std::optional<solvers::Multigrid> multigrid_;
  solvers::Gcr newton_gcr_;
  std::vector<double> newton_u_, newton_r_, newton_v_;
  std::vector<State> newton_state_;
  std::vector<Res> newton_res_;
  bool newton_ = false;
  std::vector<double> first0_;
  double penalty_ = 1.0;
  double cfl_cap_ = 1e6;
  bool shortened_ = false;
  double previous_norm_ = 0.0;
  double previous_omega_ = 1.0;
  double freestream_norm_ = 0.0;
  std::vector<double> trial_u_;
  std::vector<State> trial_state_;
  std::vector<Res> trial_res_;
  double cfl_ = cfl_start;
  bool second_order_ = false;
  int first_order_steps_ = 0;
  // Equations left out of the SER ratio, and their floor (set at the start of each order).
  std::array<bool, 5> exempt_{};
  double exempt_floor_ = 0.0;
  double mean_weight_ = 0.0;
  std::vector<State> state_, lambda_;
  std::vector<Res> res_;
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
// conditions that apply to inviscid flow, and no periodic pairs (it exchanges no ghost cells yet).
std::optional<runtime::PhysicsRefusal> euler_scope(const runtime::SolverInput& in, int /*ranks*/) {
  if (!in.periodic.empty()) {
    return runtime::PhysicsRefusal{
        .message = "the Euler physics does not exchange periodic ghost cells yet",
        .path = "periodic",
        .data_json = nlohmann::json{{"pairs", in.periodic.size()}}.dump()};
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
  }
  return std::nullopt;
}

// The Navier-Stokes physics: one ideal gas as Euler, the Euler conditions and no-slip walls; SA
// runs steady (its loosely coupled step is the steady solver's) and needs a wall to measure
// distances from, unless a manufactured solution gives them.
std::optional<runtime::PhysicsRefusal> navier_stokes_scope(const runtime::SolverInput& in,
                                                           int ranks) {
  bool wall = false;
  for (std::size_t i = 0; i < in.boundaries.size(); ++i) {
    const auto& b = in.boundaries[i];
    if (b.kind == "wall") {
      wall = true;
      continue;
    }
    if (b.kind != "slip" && b.kind != "symmetry" && b.kind != "farfield" && b.kind != "inlet" &&
        b.kind != "outlet" && b.kind != "manufactured") {
      return runtime::PhysicsRefusal{
          .message = "the Navier-Stokes physics has no '" + b.kind +
                     "' condition; boundary zone '" + b.zone +
                     "' needs wall, slip, symmetry, farfield, inlet or outlet",
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
  // The gas checks are the Euler physics' own; its condition check is replaced above.
  runtime::SolverInput gas_only = in;
  gas_only.boundaries.clear();
  return euler_scope(gas_only, ranks);
}

}  // namespace

void register_euler(runtime::PhysicsRegistry& registry) {
  registry.add({"compressible", "inviscid", "fp64", "cpu"},
               factory<kernels::fp64, Euler<kernels::fp64>>(), euler_scope);
  registry.add({"compressible", "inviscid", "fp32+acc64", "cpu"},
               factory<kernels::fp32_acc64, Euler<kernels::fp32_acc64>>(), euler_scope);
}

void register_navier_stokes(runtime::PhysicsRegistry& registry) {
  for (const char* model : {"laminar", "sa"}) {
    registry.add({"compressible", model, "fp64", "cpu"},
                 factory<kernels::fp64, NavierStokes<kernels::fp64>>(), navier_stokes_scope);
    registry.add({"compressible", model, "fp32+acc64", "cpu"},
                 factory<kernels::fp32_acc64, NavierStokes<kernels::fp32_acc64>>(),
                 navier_stokes_scope);
  }
}

}  // namespace autan::physics
