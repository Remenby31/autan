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
  if (b.kind == "manufactured") return "UserDefined";
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
  std::optional<StartState> start;  // input.start, read and checked
  Json start_averaging;             // `continue`: the previous run's averaging
};

[[noreturn]] void start_mismatch(const std::string& code, const std::string& message,
                                 const std::string& path, const nlohmann::json& data = {}) {
  throw InputError(Diagnostic{.code = code,
                              .message = message,
                              .path = path,
                              .data_json = data.is_null() ? "{}" : data.dump()});
}

// solver-input.md, `periodic`: the pairs of MeshInfo, which must be the
// mesh's; no physics exchanges across them yet.
void check_periodic(const Prepared& p) {
  const auto pairs = p.mesh.periodic_pairs();
  bool same = pairs.size() == p.input.periodic.size();
  for (std::size_t i = 0; same && i < pairs.size(); ++i) {
    const auto& in = p.input.periodic[i];
    same = in.zones[0] == p.mesh.face_zones()[at(pairs[i].zone_a)].name &&
           in.zones[1] == p.mesh.face_zones()[at(pairs[i].zone_b)].name &&
           in.translation == pairs[i].translation;
  }
  if (!same) invalid("periodic", "differs from the periodic pairs of the mesh");
  if (!pairs.empty()) {
    throw InputError(
        Diagnostic{.code = "E0706",
                   .message = "this build of autan-run does not implement periodic boundaries",
                   .path = "periodic",
                   .data_json = nlohmann::json{{"pairs", pairs.size()}}.dump()});
  }
}

// A private copy of a file and the SHA-256 of the bytes copied: what is
// hashed is what is read, whatever happens to the original meanwhile.
class SnapshotFile {
 public:
  explicit SnapshotFile(const fs::path& from) {
    std::random_device random;
    path_ = fs::temp_directory_path() /
            ("autan-start-" + std::to_string(random()) + "-" + std::to_string(random()) + ".cgns");
    std::ifstream in(from, std::ios::binary);
    std::ofstream out(path_, std::ios::binary | std::ios::trunc);
    if (!in || !out) throw Error(ErrorCode::io_cannot_open, "cannot read '" + from.string() + "'");
    picosha2::hash256_one_by_one hasher;
    std::vector<char> buffer(1 << 20);
    while (in) {
      in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
      const auto n = static_cast<std::size_t>(in.gcount());
      hasher.process(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(n));
      out.write(buffer.data(), static_cast<std::streamsize>(n));
    }
    if (in.bad() || !out.flush()) {
      throw Error(ErrorCode::io_cannot_open, "cannot copy '" + from.string() + "'");
    }
    hasher.finish();
    sha256_ = picosha2::get_hash_hex_string(hasher);
  }
  SnapshotFile(const SnapshotFile&) = delete;
  SnapshotFile& operator=(const SnapshotFile&) = delete;
  ~SnapshotFile() {
    std::error_code ec;
    fs::remove(path_, ec);
  }
  [[nodiscard]] const fs::path& path() const { return path_; }
  [[nodiscard]] const std::string& sha256() const { return sha256_; }

 private:
  fs::path path_;
  std::string sha256_;
};

// results.md, run.json `averaging`: null or the four bounds of the interval.
bool is_averaging(const Json& a) {
  if (a.is_null()) return true;
  return a.is_object() && a.size() == 4 && a.contains("from_step") &&
         a["from_step"].is_number_integer() && a.contains("to_step") &&
         a["to_step"].is_number_integer() && a.contains("from_time") &&
         a["from_time"].is_number() && a.contains("to_time") && a["to_time"].is_number();
}

// solver-input.md, `start`: the runtime checks the previous solution's hash,
// mesh, precision and fields, and reads its state, before any directory exists.
void load_start(Prepared& p) {
  const Start& start = *p.input.start;
  const fs::path file = fs::path(start.run) / "solution.cgns";
  const auto unusable = [&](const std::string& why) {
    start_mismatch("E0707", file.string() + ": " + why, "start.run");
  };
  std::optional<SnapshotFile> snapshot;
  try {
    snapshot.emplace(file);
  } catch (const Error& e) {
    unusable(e.what());
  }
  if (snapshot->sha256() != start.solution_sha256) {
    start_mismatch("E0709", "the previous solution's SHA-256 differs from start.solution_sha256",
                   "start.solution_sha256",
                   {{"expected", start.solution_sha256}, {"found", snapshot->sha256()}});
  }
  io::StoredSolution stored;
  try {
    stored = io::read_solution(snapshot->path());
  } catch (const Error& e) {
    unusable(e.what());
  }
  Json metadata;
  try {
    metadata = Json::parse(stored.run_metadata);
  } catch (const Json::exception&) {
    unusable("RunMetadata is not JSON");
  }
  const auto text = [&](const Json& object, const char* key) {
    if (!object.is_object() || !object.contains(key) || !object[key].is_string())
      unusable(std::string("RunMetadata has no ") + key);
    return object[key].get<std::string>();
  };
  const std::string mesh =
      text(metadata.contains("mesh") ? metadata["mesh"] : Json::object(), "sha256");
  if (mesh != p.mesh_hash) {
    start_mismatch("E0203", "the previous run was computed on another mesh", "start.run",
                   {{"expected", p.mesh_hash}, {"found", mesh}});
  }
  if (stored.cells != p.mesh.n_cells()) {
    unusable(std::to_string(stored.cells) + " cells, the mesh " + std::to_string(p.mesh.n_cells()));
  }
  const std::string precision = text(metadata, "precision");
  if (precision != p.input.precision) {
    start_mismatch("E0708", "the previous run used precision '" + precision + "'", "start.run");
  }
  std::vector<std::string> names;
  for (const auto& f : stored.fields) names.push_back(f.name);
  if (names != field_names(p.input)) {
    start_mismatch("E0708", "the previous run solved other variables", "start.run");
  }
  StartState state{.fields = std::move(stored.fields), .mean_fields = {}, .restart_fields = {}};
  if (start.mode == "continue") {
    if (!metadata.contains("averaging") || !is_averaging(metadata["averaging"]))
      unusable("RunMetadata has no valid averaging");
    p.start_averaging = metadata["averaging"];
    state.mean_fields = std::move(stored.mean_fields);
    state.restart_fields = std::move(stored.restart_fields);
  }
  p.start = std::move(state);
}

Prepared prepare(const fs::path& input_file, const PhysicsRegistry& registry) {
  Prepared p{.input = parse_solver_input(read_text(input_file)),
             .mesh = {},
             .mesh_hash = {},
             .face_zone = {},
             .monitor_zones = {},
             .factory = nullptr,
             .start = std::nullopt,
             .start_averaging = {}};
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
  check_periodic(p);
  if (p.input.start) load_start(p);
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

class Run final : public Log {
 public:
  Run(Prepared& p, const SolverBuild& build) : p_(p), geometry_(compute_geometry(p.mesh)) {
    const auto started = std::chrono::system_clock::now();
    std::random_device random;
    char suffix[9];
    std::snprintf(suffix, sizeof suffix, "%08x", static_cast<unsigned>(random()));
    const auto& in = p_.input;
    const std::string backend = in.device_kind;
    std::string parts = in.case_sha256 + "\n" + p_.mesh_hash + "\n" + build.commit + "\n" +
                        in.precision + "\n" + backend + "\n" + in.device_kind + "\n" +
                        std::to_string(in.device_count);
    Json start = nullptr;
    if (in.start) {
      const Start& s = *in.start;
      parts += "\n" + s.identity + "\n" + s.mode + "\n" + std::to_string(s.iterations) + "\n" +
               s.solution_sha256;
      Json reference = nullptr;
      if (s.residual_reference) {
        const auto equations = residual_names(in);
        reference = Json::object();
        for (std::size_t e = 0; e < equations.size(); ++e)
          reference[equations[e]] = (*s.residual_reference)[e];
      }
      start = Json{{"run_id", s.run_id},
                   {"identity", s.identity},
                   {"mode", s.mode},
                   {"iterations", s.iterations},
                   {"time", s.time ? Json(*s.time) : Json(nullptr)},
                   {"solution_sha256", s.solution_sha256},
                   {"residual_reference", reference}};
    }
    const std::string identity = sha256_hex(parts);
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
        {"start", start},
        {"diagnostics", Json::array()}};
    if (continued()) {
      info_["iterations"] = in.start->iterations;
      if (in.time.kind == "unsteady") info_["time"] = *in.start->time;
      info_["averaging"] = p_.start_averaging;
    }
  }

  void open(const fs::path& runs) {
    dir_ = std::make_unique<RunDirectory>(
        runs, info_["run_id"].get<std::string>(), info_.dump(2), p_.input.case_toml,
        io::mesh_info_json(p_.mesh, p_.mesh_hash),
        log_line("info", "solver", "run " + info_["run_id"].get<std::string>() + " started", "",
                 Json{{"cells", p_.mesh.n_cells()}, {"faces", p_.mesh.n_faces()}}.dump()),
        p_.input.verification ? std::optional<std::string>(p_.input.verification->text)
                              : std::nullopt);
  }
  [[nodiscard]] const fs::path& path() const { return dir_->path(); }

  ExitStatus execute() {
    const PhysicsContext context{.mesh = p_.mesh,
                                 .geometry = geometry_,
                                 .input = p_.input,
                                 .log = *this,
                                 .start = p_.start ? &*p_.start : nullptr};
    std::string status = "running";
    std::int64_t done = continued() ? p_.input.start->iterations : 0;
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
  [[nodiscard]] bool continued() const {
    return p_.input.start && p_.input.start->mode == "continue";
  }

  // Thrown by the solution writer when a physics member fails: unlike a
  // write failure, it fails even a diverged run (runtime.md).
  struct PhysicsFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
  };

  void record_failure(const std::string& message) {
    add_diagnostic(Diagnostic{.code = "E0704", .message = message, .path = "", .data_json = "{}"});
  }

  // A run diagnostic goes to run.json and, with the same code and data, to
  // log.jsonl (results.md).
  void add_diagnostic(const Diagnostic& d) {
    const Json entry = Json::parse(d.json());
    info_["diagnostics"].push_back(entry);
    write("error", "solver", d.message, d.code, entry["data"].dump());
  }

 public:
  void write(std::string_view level, std::string_view source, std::string_view message,
             std::string_view code, std::string_view data_json) override {
    dir_->append_log(log_line(level, source, message, code, data_json));
  }

 private:
  // One log.jsonl line: wall-clock UTC time in milliseconds and the current
  // iteration (null before the first).
  [[nodiscard]] std::string log_line(std::string_view level, std::string_view source,
                                     std::string_view message, std::string_view code,
                                     std::string_view data_json) const {
    const auto now = std::chrono::system_clock::now();
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() %
        1000;
    char fraction[8];
    std::snprintf(fraction, sizeof fraction, ".%03dZ", static_cast<int>(ms));
    return Json{{"time", utc(now, "%Y-%m-%dT%H:%M:%S") + fraction},
                {"iteration", iteration_ > 0 ? Json(iteration_) : Json(nullptr)},
                {"level", level},
                {"source", source},
                {"message", message},
                {"code", code.empty() ? Json(nullptr) : Json(code)},
                {"data", Json::parse(data_json.empty() ? "{}" : data_json)}}
        .dump();
  }

  // Returns the final status; every value is a string literal.
  std::string_view iterate(std::int64_t& done) {
    const auto& in = p_.input;
    const bool steady = in.time.kind == "steady";
    const std::int64_t budget = steady ? in.time.max_iterations : in.time.steps;
    const auto equations = residual_names(in);
    // Steady convergence is measured against iteration 1 of the original run
    // (solver-input.md, `start`); a `continue` goes on counting and timing.
    std::vector<double> first;
    std::vector<char> exempt;
    double floor = 0.0;
    double time = 0.0, average_start = 0.0;
    std::int64_t from = 1;
    if (continued()) {
      const Start& s = *in.start;
      if (s.residual_reference) first = *s.residual_reference;
      if (s.time) time = *s.time;
      from = s.iterations + 1;
      if (!info_["averaging"].is_null()) average_start = info_["averaging"]["from_time"];
    }
    for (std::int64_t it = from; it <= budget; ++it) {
      if (stop_requested != 0) {
        stopped_by_signal_ = true;
        return "stopped";
      }
      iteration_ = it;
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
        if (first.empty()) first = report.residuals;
        // case-schema.md#time: an equation whose residual at iteration 1 is
        // below 1e-12 of the largest is exempt while it stays below that
        // floor; once above, it must drop like the others (W0710, once).
        if (exempt.empty()) {
          floor = 1e-12 * *std::ranges::max_element(first);
          for (const double f : first) exempt.push_back(f < floor ? 1 : 0);
        }
        const double drop = std::pow(10.0, -in.time.residual_orders);
        bool converged = true;
        for (std::size_t e = 0; e < equations.size(); ++e) {
          const double r = report.residuals[e];
          if (exempt[e] != 0 && r > floor) {
            exempt[e] = 0;
            write("warning", "solver",
                  "equation " + equations[e] + " left its floor at iteration " +
                      std::to_string(it) + ": it now has to converge like the others",
                  "W0710",
                  Json{{"equation", equations[e]},
                       {"iteration", it},
                       {"residual", r},
                       {"floor", floor}}
                      .dump());
          }
          converged = converged && (exempt[e] != 0 || r <= first[e] * drop);
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
    add_diagnostic(
        Diagnostic{.code = "D0107",
                   .message = "divergence at iteration " + std::to_string(iteration) + where,
                   .path = "",
                   .data_json = data.dump()});
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
      s.restart_fields = physics_->restart_fields();
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
    write(status == "failed" ? "error" : "info", "solver", "run finished: " + status, "",
          Json{{"status", status}, {"iterations", done}}.dump());
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
  std::int64_t iteration_ = 0;  // the iteration being computed, 0 before the first
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
