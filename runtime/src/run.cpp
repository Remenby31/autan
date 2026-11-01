#include "autan/runtime/run.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <exception>
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include <picosha2.h>

#include "autan/core/error.hpp"
#include "autan/io/cgns.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/runtime/diagnostic.hpp"
#include "run_directory.hpp"

namespace autan::runtime {
namespace {

namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

// Set by SIGINT and SIGTERM; the loop finalises the run as stopped.
volatile std::sig_atomic_t stop_requested = 0;
extern "C" void request_stop(int /*signal*/) { stop_requested = 1; }

class SignalGuard {
 public:
  SignalGuard() {
    struct sigaction action{};
    action.sa_handler = request_stop;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, &previous_int_);
    sigaction(SIGTERM, &action, &previous_term_);
  }
  SignalGuard(const SignalGuard&) = delete;
  SignalGuard& operator=(const SignalGuard&) = delete;
  ~SignalGuard() {
    sigaction(SIGINT, &previous_int_, nullptr);
    sigaction(SIGTERM, &previous_term_, nullptr);
  }

 private:
  struct sigaction previous_int_{};
  struct sigaction previous_term_{};
};

std::string utc(std::chrono::system_clock::time_point moment, const char* format) {
  const std::time_t t = std::chrono::system_clock::to_time_t(moment);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char text[32];
  std::strftime(text, sizeof text, format, &tm);
  return text;
}

std::string sha256_hex(const std::string& text) {
  std::string hash;
  picosha2::hash256_hex_string(text, hash);
  return hash;
}

std::string read_text(const fs::path& path) {
  const std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw InputError(Diagnostic{.code = "E0705",
                                .message = "cannot read '" + path.string() + "'",
                                .path = "",
                                .data_json = "{}"});
  }
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

[[noreturn]] void invalid(const std::string& path, const std::string& message) {
  throw InputError(Diagnostic{.code = "E0705",
                              .message = path + ": " + message,
                              .path = path,
                              .data_json = nlohmann::json{{"path", path}}.dump()});
}

// results.md: BCType of each resolved condition.
std::string_view bc_type(const BoundaryInput& b) {
  if (b.kind == "wall") return b.temperature ? "BCWallViscousIsothermal" : "BCWallViscousHeatFlux";
  if (b.kind == "slip") return "BCWallInviscid";
  if (b.kind == "symmetry") return "BCSymmetryPlane";
  if (b.kind == "farfield") return "BCFarfield";
  if (b.kind == "outlet") return "BCOutflowSubsonic";
  if (b.kind == "manufactured") return "BCTypeUserDefined";
  return "BCInflowSubsonic";  // inlet, velocity-inlet
}

// Everything the run needs, resolved and checked before the directory exists.
struct Prepared {
  SolverInput input;
  Mesh mesh;
  std::string mesh_hash;
  std::map<std::string, index_t> face_zone;  // boundary zone name -> Mesh::face_zones() index
  std::vector<std::vector<index_t>> monitor_zones;
  const PhysicsFactory* factory = nullptr;
};

Prepared prepare(const fs::path& input_file, const PhysicsRegistry& registry) {
  Prepared p{.input = parse_solver_input(read_text(input_file)),
             .mesh = {},
             .mesh_hash = {},
             .face_zone = {},
             .monitor_zones = {},
             .factory = nullptr};
  try {
    p.mesh = io::read_mesh(p.input.mesh_path);
    p.mesh_hash = io::mesh_hash(p.input.mesh_path);
  } catch (const Error& e) {
    throw InputError(Diagnostic{.code = error_code_string(e.code()),
                                .message = e.what(),
                                .path = "mesh.path",
                                .data_json = "{}"});
  }
  if (p.mesh_hash != p.input.mesh_sha256) {
    throw InputError(Diagnostic{
        .code = "E0203",
        .message = "the mesh file's SHA-256 differs from mesh.sha256 of the input",
        .path = "mesh.sha256",
        .data_json =
            nlohmann::json{{"expected", p.input.mesh_sha256}, {"found", p.mesh_hash}}.dump()});
  }
  if (p.input.verification) {
    // The manufactured solution is checked like the mesh: by its hash.
    auto& v = *p.input.verification;
    const std::ifstream in(v.path, std::ios::binary);
    if (!in) {
      throw InputError(Diagnostic{.code = "E0207",
                                  .message = "cannot read '" + v.path + "'",
                                  .path = "verification.path",
                                  .data_json = "{}"});
    }
    std::ostringstream text;
    text << in.rdbuf();
    v.text = text.str();
    if (sha256_hex(v.text) != v.sha256) {
      throw InputError(Diagnostic{
          .code = "E0206",
          .message = "the manufactured solution's SHA-256 differs from verification.sha256",
          .path = "verification.sha256",
          .data_json =
              nlohmann::json{{"expected", v.sha256}, {"found", sha256_hex(v.text)}}.dump()});
    }
  }
  const io::MeshInfoZones zones = io::meshinfo_zones(p.mesh);
  for (const index_t z : zones.boundary) p.face_zone[p.mesh.face_zones()[at(z)].name] = z;
  std::map<std::string, bool> conditioned;
  for (std::size_t i = 0; i < p.input.boundaries.size(); ++i) {
    const auto& name = p.input.boundaries[i].zone;
    if (!p.face_zone.contains(name))
      invalid("boundaries[" + std::to_string(i) + "].zone", "no boundary zone '" + name + "'");
    if (conditioned[name])
      invalid("boundaries[" + std::to_string(i) + "].zone",
              "zone '" + name + "' has two conditions");
    conditioned[name] = true;
  }
  const auto offsets = p.mesh.boundary_zone_offsets();
  for (const auto& [name, z] : p.face_zone) {
    if (offsets[at(z) + 1] > offsets[at(z)] && !conditioned[name]) {
      invalid("boundaries", "boundary zone '" + name + "' has faces and no condition");
    }
  }
  std::map<std::string, bool> described;
  std::map<std::string, bool> in_mesh;
  for (const index_t z : zones.cell) in_mesh[p.mesh.cell_zones()[at(z)].name] = true;
  for (std::size_t i = 0; i < p.input.cell_zones.size(); ++i) {
    const auto& name = p.input.cell_zones[i].zone;
    const std::string path = "cell_zones[" + std::to_string(i) + "].zone";
    if (!in_mesh[name]) invalid(path, "no cell zone '" + name + "' holds cells in the mesh");
    if (described[name]) invalid(path, "cell zone '" + name + "' is described twice");
    described[name] = true;
  }
  for (const index_t z : zones.cell) {
    if (!described[p.mesh.cell_zones()[at(z)].name]) {
      invalid("cell_zones", "cell zone '" + p.mesh.cell_zones()[at(z)].name + "' has no entry");
    }
  }
  for (std::size_t m = 0; m < p.input.monitors.size(); ++m) {
    std::vector<index_t> ids;
    for (const auto& name : p.input.monitors[m].zones) {
      if (!p.face_zone.contains(name))
        invalid("monitors[" + std::to_string(m) + "].zones", "no boundary zone '" + name + "'");
      ids.push_back(p.face_zone.at(name));
    }
    p.monitor_zones.push_back(std::move(ids));
  }
  p.factory = registry.find(p.input);
  if (p.factory == nullptr) {
    const auto& in = p.input;
    throw InputError(Diagnostic{.code = "E0706",
                                .message = "this build of autan-run has no physics for " +
                                           in.physics_kind + " flow, turbulence '" +
                                           in.turbulence.model + "', precision " + in.precision +
                                           " on " + in.device_kind,
                                .path = "physics",
                                .data_json = nlohmann::json{{"kind", in.physics_kind},
                                                            {"turbulence", in.turbulence.model},
                                                            {"precision", in.precision},
                                                            {"device", in.device_kind}}
                                                 .dump()});
  }
  return p;
}

// Coefficients of a forces monitor (results.md, History).
Json forces_record(const Loads& loads, const ForcesMonitor& monitor, const Freestream& fs) {
  Json r{{"fx", loads.force[0]},  {"fy", loads.force[1]},  {"fz", loads.force[2]},
         {"mx", loads.moment[0]}, {"my", loads.moment[1]}, {"mz", loads.moment[2]}};
  if (!monitor.ref_area) return r;
  const double q = 0.5 * fs.density * fs.speed * fs.speed;
  const double a = fs.aoa, b = fs.sideslip;
  const std::array<double, 3> drag{std::cos(a) * std::cos(b), -std::sin(b),
                                   std::sin(a) * std::cos(b)};
  const std::array<double, 3> lift{-std::sin(a), 0.0, std::cos(a)};
  const std::array<double, 3> side{lift[1] * drag[2] - lift[2] * drag[1],
                                   lift[2] * drag[0] - lift[0] * drag[2],
                                   lift[0] * drag[1] - lift[1] * drag[0]};
  const auto dot = [](const std::array<double, 3>& u, const std::array<double, 3>& v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  };
  const double qa = q * *monitor.ref_area;
  r["cd"] = dot(loads.force, drag) / qa;
  r["cs"] = dot(loads.force, side) / qa;
  r["cl"] = dot(loads.force, lift) / qa;
  if (monitor.ref_length) {
    const double qal = qa * *monitor.ref_length;
    r["cmx"] = loads.moment[0] / qal;
    r["cmy"] = loads.moment[1] / qal;
    r["cmz"] = loads.moment[2] / qal;
  }
  return r;
}

class Run {
 public:
  Run(Prepared& p, const SolverBuild& build) : p_(p), geometry_(compute_geometry(p.mesh)) {
    const auto started = std::chrono::system_clock::now();
    std::random_device random;
    char suffix[9];
    std::snprintf(suffix, sizeof suffix, "%08x", static_cast<unsigned>(random()));
    const auto& in = p_.input;
    const std::string backend = in.device_kind;
    const std::string identity = sha256_hex(
        in.case_sha256 + "\n" + p_.mesh_hash + "\n" + build.commit + "\n" + in.precision + "\n" +
        backend + "\n" + in.device_kind + "\n" + std::to_string(in.device_count));
    info_ = Json{
        {"schema", "autan.results/1"},
        {"run_id", "run-" + utc(started, "%Y%m%dT%H%M%SZ") + "-" + suffix},
        {"identity", identity},
        {"status", "running"},
        {"case", {{"name", in.case_name}, {"sha256", in.case_sha256}}},
        {"mesh",
         {{"path", fs::absolute(in.mesh_path).lexically_normal().string()},
          {"sha256", p_.mesh_hash}}},
        {"solver", {{"version", build.version}, {"commit", build.commit}, {"backend", backend}}},
        {"precision", in.precision},
        {"deterministic", in.deterministic},
        {"devices", {{"kind", in.device_kind}, {"count", in.device_count}}},
        {"started", utc(started, "%Y-%m-%dT%H:%M:%SZ")},
        {"finished", nullptr},
        {"iterations", 0},
        {"time", in.time.kind == "unsteady" ? Json(0.0) : Json(nullptr)},
        {"averaging", nullptr},
        {"diagnostics", Json::array()}};
  }

  void open(const fs::path& runs) {
    dir_ = std::make_unique<RunDirectory>(
        runs, info_["run_id"].get<std::string>(), info_.dump(2), p_.input.case_toml,
        io::mesh_info_json(p_.mesh, p_.mesh_hash),
        p_.input.verification ? std::optional<std::string>(p_.input.verification->text)
                              : std::nullopt);
  }
  [[nodiscard]] const fs::path& path() const { return dir_->path(); }

  ExitStatus execute() {
    const PhysicsContext context{.mesh = p_.mesh, .geometry = geometry_, .input = p_.input};
    std::string status = "running";
    std::int64_t done = 0;
    try {
      physics_ = (*p_.factory)(context);
      status = std::string(iterate(done));
    } catch (const std::exception& e) {
      record_failure(e.what());
      return finish("failed", done);
    }
    return finish(status, done);
  }

 private:
  // Thrown by the solution writer when a physics member fails: unlike a
  // write failure, it fails even a diverged run (runtime.md).
  struct PhysicsFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
  };

  void record_failure(const std::string& message) {
    info_["diagnostics"].push_back(Json::parse(
        Diagnostic{.code = "E0704", .message = message, .path = "", .data_json = "{}"}.json()));
  }

  // Returns the final status; every value is a string literal.
  std::string_view iterate(std::int64_t& done) {
    const auto& in = p_.input;
    const bool steady = in.time.kind == "steady";
    const std::int64_t budget = steady ? in.time.max_iterations : in.time.steps;
    const auto equations = residual_names(in);
    std::vector<double> first;
    double time = 0.0, average_start = 0.0;
    for (std::int64_t it = 1; it <= budget; ++it) {
      if (stop_requested != 0) {
        stopped_by_signal_ = true;
        return "stopped";
      }
      const StepReport report = physics_->step(it);
      if (report.residuals.size() != equations.size()) {
        throw std::runtime_error("the physics reported " + std::to_string(report.residuals.size()) +
                                 " residuals for " + std::to_string(equations.size()) +
                                 " equations");
      }
      bool finite = !report.diverged_cell.has_value();
      Json residuals = Json::object();
      for (std::size_t e = 0; e < equations.size(); ++e) {
        finite = finite && std::isfinite(report.residuals[e]);
        residuals[equations[e]] =
            std::isfinite(report.residuals[e]) ? Json(report.residuals[e]) : Json(nullptr);
      }
      if (!finite) {
        diverge(it, report.diverged_cell);
        return "diverged";
      }
      // The step's time and averaging become run metadata only once its
      // history line is recorded, so that run.json never runs ahead of it.
      const double step_time = time + report.time_step;
      Json averaging = info_["averaging"];
      if (!steady && in.time.average_from && it > *in.time.average_from) {
        if (it == *in.time.average_from + 1) average_start = time;
        physics_->accumulate_average(report.time_step);
        averaging = Json{{"from_step", *in.time.average_from + 1},
                         {"to_step", it},
                         {"from_time", average_start},
                         {"to_time", step_time}};
      }
      Json record{{"iteration", it},
                  {"time", steady ? Json(nullptr) : Json(step_time)},
                  {"residuals", residuals},
                  {"monitors", Json::object()}};
      for (std::size_t m = 0; m < in.monitors.size(); ++m) {
        const auto& monitor = in.monitors[m];
        if (it % monitor.every != 0) continue;
        record["monitors"][monitor.name] = forces_record(
            physics_->loads(p_.monitor_zones[m], monitor.moment_center), monitor, in.freestream);
      }
      dir_->append_history(record.dump());
      // `iterations` is the last completed (recorded) iteration: a diverging
      // one is reported by its D diagnostic, not counted.
      done = it;
      time = step_time;
      info_["iterations"] = it;
      if (!steady) {
        info_["time"] = time;
        info_["averaging"] = averaging;
      }
      if (steady) {
        if (it == 1) first = report.residuals;
        const double drop = std::pow(10.0, -in.time.residual_orders);
        bool converged = true;
        for (std::size_t e = 0; e < equations.size(); ++e) {
          converged = converged && report.residuals[e] <= first[e] * drop;
        }
        if (converged) return "converged";
      }
    }
    return "completed";
  }

  void diverge(std::int64_t iteration, std::optional<index_t> cell) {
    nlohmann::json data{{"iteration", iteration}};
    std::string where;
    if (cell) {
      const std::string& zone = p_.mesh.cell_zones()[at(p_.mesh.cell_zone()[at(*cell)])].name;
      data["cell"] = *cell;
      data["zone"] = zone;
      where = " near cell " + std::to_string(*cell) + " (zone '" + zone + "')";
    }
    info_["diagnostics"].push_back(Json::parse(
        Diagnostic{.code = "D0107",
                   .message = "divergence at iteration " + std::to_string(iteration) + where,
                   .path = "",
                   .data_json = data.dump()}
            .json()));
  }

  // Collects the solution from the physics (PhysicsFailure on any error of
  // its members) and writes it; skips a diverged state that is not finite.
  void write_solution(bool diverged) {
    const auto& in = p_.input;
    io::Solution s;
    Json metadata = info_;
    for (const char* key : {"status", "finished", "iterations", "diagnostics"}) metadata.erase(key);
    // ASCII JSON, as readers decode CGNS descriptors.
    s.run_metadata = metadata.dump(-1, ' ', true);
    std::map<std::string, const BoundaryInput*> conditions;
    for (const auto& b : in.boundaries) conditions[b.zone] = &b;
    const auto offsets = p_.mesh.boundary_zone_offsets();
    for (const index_t z : io::meshinfo_zones(p_.mesh).boundary) {
      const auto it = conditions.find(p_.mesh.face_zones()[at(z)].name);
      s.boundary_bc_types.emplace_back(it == conditions.end() ? "" : bc_type(*it->second));
      if (it != conditions.end() && (it->second->kind == "wall" || it->second->kind == "slip")) {
        for (index_t f = offsets[at(z)]; f < offsets[at(z) + 1]; ++f) s.wall_faces.push_back(f);
      }
    }
    std::ranges::sort(s.wall_faces);
    try {
      s.fields = physics_->fields(false);
      if (!info_["averaging"].is_null()) s.mean_fields = physics_->fields(true);
      if (!s.wall_faces.empty()) s.wall_fields = physics_->wall_fields(s.wall_faces);
    } catch (const std::exception& e) {
      throw PhysicsFailure(e.what());
    }
    std::vector<std::string> names;
    names.reserve(s.fields.size());
    for (const auto& f : s.fields) names.push_back(f.name);
    if (names != field_names(in))
      throw PhysicsFailure("the physics returned another set of fields");
    // A diverged state may hold non-finite values; results.md asks for its
    // solution only if one can be written, and readers require finite fields.
    if (diverged) {
      for (const auto& f : s.fields) {
        if (!std::ranges::all_of(f.values, [](double v) { return std::isfinite(v); })) return;
      }
    }
    io::write_solution(dir_->solution_staging(), p_.mesh, s);
    dir_->publish_solution();
  }

  ExitStatus finish(std::string status, std::int64_t done) {
    info_["iterations"] = done;
    // results.md: the solution is required unless the run diverged (written
    // if possible) or failed (optional).
    if (physics_ && status != "failed") {
      try {
        write_solution(status == "diverged");
      } catch (const PhysicsFailure& e) {
        record_failure(e.what());
        status = "failed";
      } catch (const std::exception& e) {
        if (status != "diverged") {
          record_failure(std::string("writing the solution: ") + e.what());
          status = "failed";
        }
      }
    }
    info_["status"] = status;
    info_["finished"] = utc(std::chrono::system_clock::now(), "%Y-%m-%dT%H:%M:%SZ");
    dir_->write_run_json(info_.dump(2));
    if (status == "failed") return ExitStatus::failed;
    return stopped_by_signal_ ? ExitStatus::signalled : ExitStatus::finished;
  }

  Prepared& p_;
  Geometry geometry_;
  Json info_;
  std::unique_ptr<RunDirectory> dir_;
  std::unique_ptr<Physics> physics_;
  bool stopped_by_signal_ = false;
};

}  // namespace

void PhysicsRegistry::add(PhysicsKey key, PhysicsFactory factory) {
  entries_.emplace_back(std::move(key), std::move(factory));
}

const PhysicsFactory* PhysicsRegistry::find(const SolverInput& in) const {
  const auto matches = [](const std::string& pattern, const std::string& value) {
    return pattern == "*" || pattern == value;
  };
  for (const auto& [key, factory] : entries_) {
    if (matches(key.kind, in.physics_kind) && matches(key.turbulence, in.turbulence.model) &&
        matches(key.precision, in.precision) && matches(key.device, in.device_kind)) {
      return &factory;
    }
  }
  return nullptr;
}

ExitStatus run_case(const fs::path& input_file, const fs::path& runs,
                    const PhysicsRegistry& registry, const SolverBuild& build, std::FILE* out,
                    std::FILE* err) {
  std::optional<Prepared> prepared;
  try {
    prepared.emplace(prepare(input_file, registry));
  } catch (const InputError& e) {
    std::fprintf(err, "%s\n", e.diagnostic().json().c_str());
    return ExitStatus::invalid_input;
  }
  stop_requested = 0;
  const SignalGuard signals;
  try {
    Run run(*prepared, build);
    run.open(runs);
    std::fprintf(out, "%s\n", run.path().string().c_str());
    std::fflush(out);
    return run.execute();
  } catch (const std::exception& e) {
    // The run directory could not be created or its run.json not written:
    // nothing records the failure but this diagnostic.
    std::fprintf(err, "%s\n",
                 Diagnostic{.code = "E0704", .message = e.what(), .path = "", .data_json = "{}"}
                     .json()
                     .c_str());
    return ExitStatus::failed;
  }
}

}  // namespace autan::runtime
