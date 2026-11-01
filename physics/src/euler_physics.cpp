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
#include <vector>

#include <nlohmann/json.hpp>

#include "autan/kernels/runtime.hpp"
#include "autan/physics/euler.hpp"
#include "autan/physics/manufactured.hpp"

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
// relaxation, Mulder and van Leer 1985, J. Comput. Phys. 59:232-246).
constexpr double cfl_start = 10.0;
constexpr double cfl_min = 1.0;
constexpr double first_order_drop = 1e-3;
constexpr int first_order_max_steps = 200;
constexpr double cfl_max = 1e6;
constexpr int gauss_seidel_sweeps = 4;
// No update may lower density or pressure by more than this fraction in one step; a step
// shortened below min_step is rejected.
constexpr double max_decrease = 0.5;
constexpr double min_step = 1e-6;
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

template <class P>
class EulerPhysics final : public runtime::Physics {
 public:
  using State = typename P::state;
  using Res = typename P::residual;

  explicit EulerPhysics(const PhysicsContext& ctx)
      : ctx_(ctx),
        gas_(gas_of(ctx.input)),
        n_(static_cast<std::size_t>(ctx.mesh.n_cells())),
        euler_(ctx.mesh, ctx.geometry, gas_, scheme(ctx.input)),
        reference_(scheme(ctx.input).reference) {
    const auto& mesh = ctx.mesh;
    const auto& in = ctx.input;
    std::optional<Manufactured> mms;
    if (in.verification) mms.emplace(in.verification->text);
    set_boundaries(mms ? &*mms : nullptr);
    if (in.time.kind == "steady") {
      matrix_.emplace(mesh);
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
    if (ctx.start != nullptr) start_from(*ctx.start);
  }

  StepReport step(std::int64_t /*iteration*/) override {
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
      steady_step(report);
    } else {
      unsteady_step(report);
    }
    report.diverged_cell = first_invalid_cell();
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
    // The steady solver's own state, one uniform field per value (cell-centred, as RestartState).
    if (ctx_.input.time.kind == "steady") {
      for (const auto& [name, value] : solver_state()) {
        out.push_back({.name = name, .values = std::vector<double>(n_, value)});
      }
    }
    return out;
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
    return out;
  }

  [[nodiscard]] std::vector<io::Field> wall_fields(std::span<const index_t> faces) const override {
    std::vector<double> p(static_cast<std::size_t>(ctx_.mesh.n_boundary_faces()));
    face_pressure(p, {});
    const auto ni = static_cast<std::size_t>(ctx_.mesh.n_interior_faces());
    std::vector<io::Field> out{{.name = "Pressure", .values = {}},
                               {.name = "WallShearStressX", .values = {}},
                               {.name = "WallShearStressY", .values = {}},
                               {.name = "WallShearStressZ", .values = {}}};
    for (const auto f : faces) {
      out[0].values.push_back(p[static_cast<std::size_t>(f) - ni]);
      for (std::size_t k = 1; k < 4; ++k) out[k].values.push_back(0.0);
    }
    return out;
  }

  [[nodiscard]] Loads loads(std::span<const index_t> zones,
                            const std::array<double, 3>& center) const override {
    // The pressure force exactly as the residual integrates it, minus p_inf S (results.md).
    const auto& mesh = ctx_.mesh;
    const auto& g = ctx_.geometry;
    const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
    const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
    std::vector<double> p(nb), face_force(3 * nb);
    face_pressure(p, face_force);
    const double p_inf = ctx_.input.freestream.pressure;
    const auto offsets = mesh.boundary_zone_offsets();
    Loads loads;
    for (const auto z : zones) {
      for (auto f = offsets[static_cast<std::size_t>(z)];
           f < offsets[static_cast<std::size_t>(z) + 1]; ++f) {
        const auto i = static_cast<std::size_t>(f);
        const auto b = i - ni;
        const std::array<double, 3> force{face_force[b] - p_inf * g.face_area_x[i],
                                          face_force[nb + b] - p_inf * g.face_area_y[i],
                                          face_force[2 * nb + b] - p_inf * g.face_area_z[i]};
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
  }

  [[nodiscard]] std::vector<std::pair<std::string, double>> solver_state() const {
    std::vector<std::pair<std::string, double>> v = {
        {"SolverPenalty", penalty_},
        {"SolverShortened", shortened_ ? 1.0 : 0.0},
        {"SolverSecondOrder", second_order_ ? 1.0 : 0.0},
        {"SolverFirstOrderSteps", static_cast<double>(first_order_steps_)}};
    for (std::size_t k = 0; k < 5; ++k) {
      v.emplace_back("SolverReferenceResidual" + std::to_string(k),
                     first_norms_.empty() ? -1.0 : first_norms_[k]);
    }
    return v;
  }

  void set_solver_state(const std::vector<std::pair<std::string, double>>& v) {
    penalty_ = v[0].second;
    shortened_ = v[1].second != 0.0;
    second_order_ = v[2].second != 0.0;
    euler_.set_second_order(second_order_);
    first_order_steps_ = static_cast<int>(v[3].second);
    first_norms_.clear();
    if (v[4].second >= 0.0) {
      for (std::size_t k = 0; k < 5; ++k) first_norms_.push_back(v[4 + k].second);
    }
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
      } else if (bc.kind == "farfield") {
        put(BoundaryKind::farfield, freestream);
      } else if (bc.kind == "inlet") {
        put(BoundaryKind::inlet, {*bc.total_pressure, *bc.total_temperature, 0.0, 0.0, 0.0});
      } else if (bc.kind == "outlet") {
        put(BoundaryKind::outlet, {0.0, 0.0, 0.0, 0.0, *bc.static_pressure});
      } else if (bc.kind == "manufactured" && mms != nullptr) {
        kind[b] = BoundaryKind::state;
        manufactured_faces.push_back(
            {g.face_centroid_x[face], g.face_centroid_y[face], g.face_centroid_z[face]});
        manufactured_index.push_back(b);
      } else {
        throw std::invalid_argument("boundary zone '" + zone + "': condition '" + bc.kind +
                                    "' does not apply to inviscid flow");
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

  void steady_step(StepReport& report) {
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
    euler_.spectral_radius(lambda_);
    // Largest relative residual over the equations, of the order being solved.
    const std::vector<double>& current = norms.empty() ? report.residuals : norms;
    double drop = 0.0;
    if (first_norms_.empty()) first_norms_ = current;
    for (std::size_t k = 0; k < 5; ++k) {
      if (first_norms_[k] > 0.0) drop = std::max(drop, current[k] / first_norms_[k]);
    }
    // First-order start: its approximate Jacobian is exact, so the steps are robust far from the
    // solution; second order from the step after the first-order residual has dropped.
    if (!second_order_ &&
        (drop < first_order_drop || ++first_order_steps_ > first_order_max_steps)) {
      second_order_ = true;
      euler_.set_second_order(true);
      first_norms_.clear();
    }
    // Switched evolution relaxation, CFL = CFL_0 / (relative residual), times a penalty halved
    // after each step the admissibility check shortened and recovering by half otherwise.
    penalty_ = shortened_ ? 0.5 * penalty_ : std::min(1.0, 1.5 * penalty_);
    if (drop > 0.0) cfl_ = std::clamp(penalty_ * cfl_start / drop, cfl_min, cfl_max);
    std::vector<double> shift(n_);
    for (std::size_t c = 0; c < n_; ++c) shift[c] = static_cast<double>(lambda_[c]) / cfl_;
    euler_.assemble(shift, *matrix_);
    matrix_->factor();
    for (std::size_t i = 0; i < 5 * n_; ++i) rhs_[i] = -static_cast<double>(res_[i]);
    std::fill(du_.begin(), du_.end(), 0.0);
    matrix_->gauss_seidel(rhs_, du_, gauss_seidel_sweeps);
    const double alpha = admissible_step();
    shortened_ = alpha < 1.0;
    if (alpha < min_step) {
      // No admissible step: the state is kept and the CFL number halves; at the smallest CFL
      // number the run has diverged.
      if (cfl_ <= cfl_min) report.diverged_cell = static_cast<index_t>(worst_cell());
      return;
    }
    for (std::size_t i = 0; i < 5 * n_; ++i) u_[i] += alpha * du_[i];
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

  // Largest fraction of du (at most 1) that keeps every cell's density and pressure above
  // (1 - max_decrease) of their current values.
  [[nodiscard]] double admissible_step() const {
    double alpha = 1.0;
    for (std::size_t c = 0; c < n_; ++c) {
      const double rho = u_[c], drho = du_[c];
      if (drho < 0.0) alpha = std::min(alpha, max_decrease * rho / -drho);
      const double p = pressure(c, 0.0);
      while (alpha >= min_step && !(pressure(c, alpha) >= (1.0 - max_decrease) * p)) alpha *= 0.5;
    }
    return alpha;
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

  // The first cell whose state, as the kernels store it (P::state differences from the
  // reference, so FP32 in fp32+acc64), has a non-finite or non-positive density or pressure;
  // else the first cell whose residual is not finite.
  [[nodiscard]] std::optional<index_t> first_invalid_cell() const {
    for (std::size_t c = 0; c < n_; ++c) {
      std::array<double, 5> v{};
      for (std::size_t k = 0; k < 5; ++k) {
        v[k] = static_cast<double>(State(u_[k * n_ + c] - reference_[k])) + reference_[k];
      }
      const double rho = v[0];
      const double p =
          (gas_.gamma - 1.0) * (v[4] - 0.5 * (v[1] * v[1] + v[2] * v[2] + v[3] * v[3]) / rho);
      if (!std::isfinite(rho) || !std::isfinite(p) || rho <= 0.0 || p <= 0.0) {
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

  void face_pressure(std::span<double> p, std::span<double> force) const {
    auto& self = const_cast<EulerPhysics&>(*this);
    self.residual_of(u_);
    euler_.boundary_pressure(p, force);
  }

  PhysicsContext ctx_;
  Gas gas_;
  std::size_t n_;
  Euler<P> euler_;
  std::array<double, 5> reference_;  // the state the cells store differences from
  std::vector<double> u_, stage_, mean_, rhs_, du_, first_norms_;
  std::optional<BlockMatrix> matrix_;  // steady runs only
  double cfl_ = cfl_start;
  double penalty_ = 1.0;
  bool second_order_ = false;
  int first_order_steps_ = 0;
  bool shortened_ = false;
  double mean_weight_ = 0.0;
  std::vector<State> state_, lambda_;
  std::vector<Res> res_;
};

template <class P>
runtime::PhysicsFactory factory() {
  return [](const PhysicsContext& ctx) -> std::unique_ptr<runtime::Physics> {
    return std::make_unique<EulerPhysics<P>>(ctx);
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
std::optional<runtime::PhysicsRefusal> euler_scope(const runtime::SolverInput& in) {
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
        b.kind != "outlet" && b.kind != "manufactured") {
      return runtime::PhysicsRefusal{
          .message = "the Euler physics has no '" + b.kind + "' condition; boundary zone '" +
                     b.zone + "' needs slip, symmetry, farfield, inlet or outlet",
          .path = "boundaries[" + std::to_string(i) + "].kind",
          .data_json = nlohmann::json{{"zone", b.zone}, {"condition", b.kind}}.dump()};
    }
  }
  return std::nullopt;
}

}  // namespace

void register_euler(runtime::PhysicsRegistry& registry) {
  registry.add({"compressible", "inviscid", "fp64", "cpu"}, factory<kernels::fp64>(), euler_scope);
  registry.add({"compressible", "inviscid", "fp32+acc64", "cpu"}, factory<kernels::fp32_acc64>(),
               euler_scope);
}

}  // namespace autan::physics
