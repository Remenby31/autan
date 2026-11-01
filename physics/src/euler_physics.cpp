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

#include "autan/kernels/runtime.hpp"
#include "autan/physics/euler.hpp"
#include "autan/physics/manufactured.hpp"
#include "autan/physics/point_vortex.hpp"
#include "autan/runtime/exact_sum.hpp"
#include "autan/solvers/gcr.hpp"
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
constexpr double first_order_drop = 1e-3;
constexpr int first_order_max_steps = 200;
constexpr double cfl_max = 1e6;
// Each step's linear system is solved to this relative residual (inexact Newton), with at most
// this many multigrid-preconditioned iterations.
constexpr double linear_tolerance = 0.1;
constexpr int linear_max_cycles = 10;
// A solve that leaves more than this fraction of the residual has failed (the multigrid cycle can
// diverge on a system the CFL number made too weakly diagonal): the step is rejected.
constexpr double linear_failure = 0.9;
// No update may lower density or pressure by more than this fraction in one step; a step
// shortened below min_step is rejected.
constexpr double max_decrease = 0.5;
constexpr double min_step = 1e-6;
// Newton-Krylov end phase (fp64): once every active residual is newton_drop below its largest
// value in the order (ADflow's ANK-to-NK switch, Yildirim et al. 2019, J. Comput. Phys.
// 397:108741, sec. 4), each step solves (V / dt + dR/dU) du = -R with dR/dU v by a finite
// difference of the residual, preconditioned by one multigrid cycle on the approximate Jacobian.
// The CFL number and the step follow HANIM (Wang, Diskin, Nielsen and Liu 2021, "Improvements in
// Iterative Convergence of FUN3D Solutions", AIAA SciTech, sec. II.D): a step is rejected and the
// CFL number divided by 10 when its linear solve fails, when it is not realisable, or when it
// reduces the pseudo-time residual too little even after a quadratic line search; an accepted
// full step doubles the CFL number, a line-searched one keeps it.
constexpr double newton_drop = 1e-5;
constexpr int newton_max_iterations = 40;
constexpr double newton_cfl_increase = 2.0;  // HANIM's CFL_incr
constexpr double newton_cfl_decrease = 0.1;  // HANIM's CFL_decr
constexpr double newton_cfl_min = 1e-12;     // HANIM's bounds; the run diverged at the smallest
constexpr double newton_cfl_max = 1e10;
// A correction this small relative to the state is round-off (HANIM: 1e-12).
constexpr double round_off_correction = 1e-12;
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
        owned_(static_cast<std::size_t>(ctx.mesh.n_owned_cells())),
        ghost_fill_(ctx.ghosts),
        euler_(ctx.mesh, ctx.geometry, gas_, scheme(ctx.input)),
        reference_(scheme(ctx.input).reference) {
    // On every rank: the exchange is collective, even for a part without ghosts.
    euler_.set_ghosts(&ghost_fill_);
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
    for (std::size_t c = 0; c < owned_; ++c) {
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
    apply_vortices();
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
    for (auto& v : vortices_) v.circulation = v.next;
    apply_vortices();
    if (!report.diverged_cell) report.diverged_cell = first_invalid_cell();
    return report;
  }

  void accumulate_average(double weight) override {
    const auto f = primitive_fields();
    for (std::size_t k = 0; k < 6; ++k) {
      for (std::size_t c = 0; c < owned_; ++c) mean_[k * n_ + c] += weight * f[k][c];
    }
    mean_weight_ += weight;
  }

  // The conserved variables bit for bit, then the averaging sums and their weight.
  [[nodiscard]] std::vector<io::Field> restart_fields() const override {
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
    // The steady solver's own state, one uniform field per value (cell-centred, as RestartState).
    if (ctx_.input.time.kind == "steady") {
      for (const auto& [name, value] : solver_state()) {
        out.push_back({.name = name, .values = std::vector<double>(owned_, value)});
      }
    }
    for (std::size_t i = 0; i < vortices_.size(); ++i) {
      out.push_back({.name = "VortexCirculation" + std::to_string(i),
                     .values = std::vector<double>(n_, vortices_[i].circulation)});
    }
    return out;
  }

  // Phi_k = f_k sqrt(sum_i A_i^2 / N), A_i the surface of cell i and f_k the face flux magnitude
  // of the freestream with q = |v| + c: rho q, rho q^2 (each momentum) and rho H q
  // (docs/design/euler.md, Precision).
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
    return {mass, momentum, momentum, momentum, mass * h};
  }

  [[nodiscard]] std::vector<io::Field> fields(bool mean) const override {
    std::vector<io::Field> out;
    const auto f = primitive_fields();
    for (std::size_t k = 0; k < 6; ++k) {
      io::Field field{.name = field_names[k], .values = f[k]};
      if (mean) {
        for (std::size_t c = 0; c < owned_; ++c) field.values[c] = mean_[k * n_ + c] / mean_weight_;
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
    const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
    std::vector<double> p(nb), face_force(3 * nb);
    face_pressure(p, face_force);
    const auto offsets = mesh.boundary_zone_offsets();
    // Exact sums over this rank's faces, then over the ranks: the same bits on any number of
    // ranks for the same face values.
    std::array<runtime::ExactSum, 6> sums;
    for (const auto z : zones) {
      for (auto f = offsets[static_cast<std::size_t>(z)];
           f < offsets[static_cast<std::size_t>(z) + 1]; ++f) {
        const auto i = static_cast<std::size_t>(f);
        const auto force = wall_force(face_force, i);
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
    v.emplace_back("SolverCflCap", cfl_cap_);
    v.emplace_back("SolverCfl", cfl_);
    for (std::size_t k = 0; k < 5; ++k) {
      v.emplace_back("SolverPeakResidual" + std::to_string(k), peaks_.empty() ? -1.0 : peaks_[k]);
    }
    v.emplace_back("SolverExemptFloor", exempt_floor_);
    for (std::size_t k = 0; k < 5; ++k) {
      v.emplace_back("SolverExempt" + std::to_string(k), exempt_[k] ? 1.0 : 0.0);
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
    cfl_cap_ = v[9].second;
    cfl_ = v[10].second;
    peaks_.clear();
    if (v[11].second >= 0.0) {
      for (std::size_t k = 0; k < 5; ++k) peaks_.push_back(v[11 + k].second);
    }
    exempt_floor_ = v[16].second;
    for (std::size_t k = 0; k < 5; ++k) exempt_[k] = v[17 + k].second != 0.0;
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

  // The pressure force on boundary face i (global index) minus p_inf S (results.md).
  [[nodiscard]] std::array<double, 3> wall_force(std::span<const double> face_force,
                                                 std::size_t i) const {
    const auto& g = ctx_.geometry;
    const auto nb = static_cast<std::size_t>(ctx_.mesh.n_boundary_faces());
    const auto b = i - static_cast<std::size_t>(ctx_.mesh.n_interior_faces());
    const double p_inf = ctx_.input.freestream.pressure;
    return {face_force[b] - p_inf * g.face_area_x[i], face_force[nb + b] - p_inf * g.face_area_y[i],
            face_force[2 * nb + b] - p_inf * g.face_area_z[i]};
  }

  // Each vortex's circulation from the lift of the residual just evaluated, Kutta-Joukowski
  // L' = rho V Gamma; it takes effect at the end of the step.
  void measure_circulation() {
    if (vortices_.empty()) return;
    const auto& mesh = ctx_.mesh;
    const auto& fs = ctx_.input.freestream;
    const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
    std::vector<double> p(nb), face_force(3 * nb);
    euler_.boundary_pressure(p, face_force);
    const auto off = mesh.boundary_zone_offsets();
    for (auto& v : vortices_) {
      std::array<runtime::ExactSum, 2> force;
      for (const auto z : v.zones) {
        for (auto f = off[static_cast<std::size_t>(z)]; f < off[static_cast<std::size_t>(z) + 1];
             ++f) {
          const auto load = wall_force(face_force, static_cast<std::size_t>(f));
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
    euler_.residual(state_, res_);
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
    measure_circulation();
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
      peaks_.clear();
    }
    // The Newton-Krylov phase follows each residual relative to its largest value so far in the
    // order. An equation identically zero at the start of the order (below 1e-12 of the largest,
    // the span momentum of a one-cell-thick mesh) is left out while it stays at that floor, as
    // the runtime's convergence rule does (case-schema.md#time).
    if (peaks_.empty()) {
      peaks_ = current;
      exempt_floor_ = 1e-12 * *std::max_element(current.begin(), current.begin() + 5);
      for (std::size_t k = 0; k < 5; ++k) exempt_[k] = current[k] < exempt_floor_;
    }
    double peak_drop = 0.0;
    for (std::size_t k = 0; k < 5; ++k) {
      peaks_[k] = std::max(peaks_[k], current[k]);
      if (exempt_[k] && current[k] > exempt_floor_) exempt_[k] = false;
      if (!exempt_[k] && peaks_[k] > 0.0) peak_drop = std::max(peak_drop, current[k] / peaks_[k]);
    }
    const bool newton =
        std::is_same_v<P, kernels::fp64> && second_order_ && peak_drop <= newton_drop;
    if (newton != newton_) {
      newton_ = newton;
      if (newton) ctx_.log.write("info", "solver", "Newton-Krylov steps", {}, "{}");
    }
    if (newton) {
      newton_step(report, iteration);
      return;
    }
    // Switched evolution relaxation, CFL = CFL_0 / (relative residual), times a penalty halved
    // after each step the admissibility check shortened and recovering by half otherwise.
    penalty_ = shortened_ ? 0.5 * penalty_ : std::min(1.0, 1.5 * penalty_);
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
      multigrid_->set_operator(matrix_->diagonals(), matrix_->uppers(), matrix_->lowers());
      linear = multigrid_->solve(rhs_, du_, linear_max_cycles, linear_tolerance);
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
      if (cfl_ <= cfl_min) report.diverged_cell = static_cast<index_t>(worst_cell());
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
      if (cfl_ <= cfl_min) report.diverged_cell = static_cast<index_t>(worst_cell());
      return;
    }
    for (std::size_t i = 0; i < 5 * n_; ++i) u_[i] += alpha * du_[i];
  }

  // One Newton-Krylov step with HANIM's controls (eqs. 16-22): f(w) = |V/dt w du + R(u + w du)|,
  // f(0) = |R(u)|, g(1) the linear residual of the full step; norms over the active equations.
  void newton_step(StepReport& report, std::int64_t iteration) {
    std::vector<double> shift(n_);
    for (std::size_t c = 0; c < n_; ++c) shift[c] = static_cast<double>(lambda_[c]) / cfl_;
    euler_.assemble(shift, *matrix_);
    for (std::size_t i = 0; i < 5 * n_; ++i) rhs_[i] = -static_cast<double>(res_[i]);
    std::fill(du_.begin(), du_.end(), 0.0);
    solvers::KrylovReport linear;
    try {
      multigrid_->set_operator(matrix_->diagonals(), matrix_->uppers(), matrix_->lowers());
      linear = newton_solve(shift);
    } catch (const std::domain_error&) {
      linear = {};  // a singular block: the solve failed
    }
    const nlohmann::json residual_drop =
        std::isfinite(linear.residual_drop) ? nlohmann::json(linear.residual_drop) : nullptr;
    // A rejected step: the state is kept, the CFL number divided by 10; at the smallest CFL
    // number the run has diverged.
    auto reject = [&](const char* reason, const char* message, nlohmann::json drop_data) {
      ctx_.log.write("warning", "solver", message, "W0711",
                     nlohmann::json{{"iteration", iteration},
                                    {"reason", reason},
                                    {"cfl", cfl_},
                                    {"residual_drop", std::move(drop_data)}}
                         .dump());
      if (cfl_ <= newton_cfl_min) report.diverged_cell = static_cast<index_t>(worst_cell());
      cfl_ = std::max(newton_cfl_min, newton_cfl_decrease * cfl_);
    };
    if (!(linear.residual_drop <= linear_failure)) {
      reject("linear", "implicit step rejected: its linear solve failed", residual_drop);
      return;
    }
    if (!(linear.residual_drop <= linear_tolerance)) {
      ctx_.log.write(
          "info", "linear", "the implicit step's linear solve stopped above its tolerance", {},
          nlohmann::json{{"cycles", linear.cycles}, {"residual_drop", residual_drop}}.dump());
    }
    if (!realisable(1.0)) {
      reject("admissible", "implicit step rejected: not realisable", nullptr);
      return;
    }
    const double target = 0.5 * (1.0 + std::max(linear_tolerance, linear.residual_drop));
    const double f0 = active_norm(res_);
    const double g1 = linear.residual_drop * f0;
    const double f1 = pseudo_time_residual(shift, 1.0);
    double omega = 1.0;
    bool accepted = f1 <= target * f0;
    const bool grow = accepted;
    if (!accepted && active_norm(du_) <= round_off_correction * active_norm(u_)) {
      accepted = true;  // a round-off correction: accepted, the CFL number kept
    }
    if (!accepted && f1 > g1) {
      omega = -(g1 - f0) / (2.0 * (f1 - g1));  // HANIM eq. 22
      if (omega > 0.0 && omega < 1.0 && realisable(omega) &&
          pseudo_time_residual(shift, omega) <= target * f0) {
        accepted = true;
        ctx_.log.write("info", "solver", "line search", {},
                       nlohmann::json{{"omega", omega}, {"cfl", cfl_}}.dump());
      }
    }
    if (!accepted) {
      reject("nonlinear", "implicit step rejected: too small a residual decrease", residual_drop);
      return;
    }
    for (std::size_t i = 0; i < 5 * n_; ++i) u_[i] += omega * du_[i];
    if (grow) cfl_ = std::min(newton_cfl_max, newton_cfl_increase * cfl_);
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
    for (std::size_t i = 0; i < 5 * n_; ++i) newton_u_[i] = u_[i] + omega * du_[i];
    euler_.state(newton_u_, newton_state_);
    euler_.residual(newton_state_, newton_res_);
    double s = 0.0;
    for (std::size_t k = 0; k < 5; ++k) {
      if (exempt_[k]) continue;
      for (std::size_t c = 0; c < n_; ++c) {
        const std::size_t i = k * n_ + c;
        const double v = shift[c] * omega * du_[i] + static_cast<double>(newton_res_[i]);
        s += v * v;
      }
    }
    return std::sqrt(s / static_cast<double>(n_));
  }

  // Whether u + w du keeps every cell's density and pressure above (1 - max_decrease) of their
  // current values.
  [[nodiscard]] bool realisable(double omega) const {
    for (std::size_t c = 0; c < n_; ++c) {
      if (!(u_[c] + omega * du_[c] >= (1.0 - max_decrease) * u_[c])) return false;
      if (!(pressure(c, omega) >= (1.0 - max_decrease) * pressure(c, 0.0))) return false;
    }
    return true;
  }

  // GCR on (V / dt + dR/dU) du = -R from du = 0, dR/dU v = (R(u + eps v) - R(u)) / eps with
  // eps = sqrt((1 + |u|) eps_mach) / |v| (Knoll and Keyes 2004, J. Comput. Phys. 193:357-397,
  // eq. 14), preconditioned by one multigrid cycle on the assembled approximate Jacobian.
  solvers::KrylovReport newton_solve(const std::vector<double>& shift) {
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
                             linear_tolerance);
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

  void face_pressure(std::span<double> p, std::span<double> force) const {
    auto& self = const_cast<EulerPhysics&>(*this);
    self.residual_of(u_);
    euler_.boundary_pressure(p, force);
  }

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

   private:
    runtime::GhostExchange& ghosts_;
  };

  // Component k of `v` (laid out k * n_ + c) over the owned cells.
  [[nodiscard]] std::vector<double> owned(const std::vector<double>& v, std::size_t k) const {
    const auto begin = v.begin() + static_cast<std::ptrdiff_t>(k * n_);
    return {begin, begin + static_cast<std::ptrdiff_t>(owned_)};
  }

  PhysicsContext ctx_;
  Gas gas_;
  std::size_t n_;      // cells of the mesh: owned, then ghosts
  std::size_t owned_;  // owned cells: what this rank updates and reports
  HostGhosts ghost_fill_;
  Euler<P> euler_;
  std::array<double, 5> reference_;  // the state the cells store differences from
  std::vector<double> u_, stage_, mean_, rhs_, du_, first_norms_;
  std::optional<BlockMatrix> matrix_;  // steady runs only
  std::optional<solvers::Multigrid> multigrid_;
  solvers::Gcr newton_gcr_;
  std::vector<double> newton_u_, newton_r_, newton_v_, peaks_;
  std::vector<State> newton_state_;
  std::vector<Res> newton_res_;
  std::array<bool, 5> exempt_{};
  double exempt_floor_ = 0.0;
  bool newton_ = false;       // logging only: the phase follows from the restart state
  double cfl_cap_ = cfl_max;  // after a shortened or rejected step: half its CFL number
  double cfl_ = cfl_start;
  double penalty_ = 1.0;
  bool second_order_ = false;
  int first_order_steps_ = 0;
  bool shortened_ = false;
  double mean_weight_ = 0.0;
  std::vector<State> state_, lambda_;
  std::vector<Res> res_;
  std::vector<double> boundary_data_;  // set_boundary's data, before any vortex
  std::vector<Vortex> vortices_;
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
// conditions that apply to inviscid flow, and steady runs on one rank without periodic pairs.
std::optional<runtime::PhysicsRefusal> euler_scope(const runtime::SolverInput& in, int ranks) {
  // The steady solver's block Gauss-Seidel is coloured in the order of one mesh's cells: on a
  // partition, or with periodic image ghosts, it would be another algorithm.
  if (ranks > 1 && in.time.kind == "steady") {
    return runtime::PhysicsRefusal{
        .message = "the steady Euler solver runs on one rank only, not " + std::to_string(ranks),
        .path = "time.kind",
        .data_json = nlohmann::json{{"ranks", ranks}}.dump()};
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
  if (!in.periodic.empty() && in.time.kind == "steady") {
    return runtime::PhysicsRefusal{
        .message = "the steady Euler solver does not run with periodic pairs yet",
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

}  // namespace

void register_euler(runtime::PhysicsRegistry& registry) {
  registry.add({"compressible", "inviscid", "fp64", "cpu"}, factory<kernels::fp64>(), euler_scope,
               /*several_ranks=*/true);
  registry.add({"compressible", "inviscid", "fp32+acc64", "cpu"}, factory<kernels::fp32_acc64>(),
               euler_scope, /*several_ranks=*/true);
}

}  // namespace autan::physics
