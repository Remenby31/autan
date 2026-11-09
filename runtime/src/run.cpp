#include "autan/runtime/run.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <picosha2.h>
#include <sys/resource.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/renumber.hpp"
#include "autan/io/cgns.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/runtime/diagnostic.hpp"
#include "autan/runtime/round_off.hpp"
#include "comm.hpp"
#include "device.hpp"
#include "executable_hash.hpp"
#include "host_memory.hpp"
#include "run_directory.hpp"

namespace autan::runtime {
namespace {

namespace fs = std::filesystem;

#ifdef AUTAN_HAVE_CUDA
constexpr bool gpu_build = true;
#else
constexpr bool gpu_build = false;
#endif
using Json = nlohmann::ordered_json;

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

// case-schema.md, `verification`: the bounds that make a manufactured solution's input space
// finite, checked before the physics builds its expression trees.
constexpr std::uintmax_t manufactured_max_bytes = 1U << 20U;
constexpr int manufactured_max_depth = 1024;

// The depth of the tree the manufactured solution's grammar (physics/src/manufactured.cpp)
// builds for `e`: a number or a name is 1, a unary minus, a parenthesis or a one-argument
// function adds 1, a binary operator (left-associative, * and / before + and -) or pow is 1 more
// than its deeper operand. Evaluated with explicit stacks, never recursively, and stopped as soon
// as it exceeds `limit`. Every operator still pending encloses what follows it, so the depth is at
// least one more than their count: an expression is refused on that count too, finished or not,
// before the parser's recursion could reach it. Otherwise a malformed expression is left to the
// parser, which reports it.
int expression_depth(std::string_view e, int limit) {
  enum Kind : std::uint8_t { open, call, negate, add, multiply };
  std::vector<Kind> ops;
  std::vector<int> depth;          // operands' depths
  std::vector<std::size_t> marks;  // depth.size() at each open parenthesis or call
  int deepest = 0;
  auto keep = [&](int d) {
    depth.push_back(d);
    deepest = std::max(deepest, d);
    return d <= limit;
  };
  // Applies the operator on top of `ops` to the operands it takes.
  auto reduce = [&] {
    const Kind k = ops.back();
    ops.pop_back();
    if (depth.empty()) return true;
    if (k == negate) {
      const int a = depth.back();
      depth.pop_back();
      return keep(a + 1);
    }
    if (depth.size() < 2) return true;
    const int b = depth.back();
    depth.pop_back();
    const int a = depth.back();
    depth.pop_back();
    return keep(1 + std::max(a, b));
  };
  auto binds = [](Kind k) { return k == add ? 1 : k == multiply ? 2 : 3; };
  auto pending = [&] { return static_cast<int>(ops.size()) + 1 > limit; };
  bool operand_expected = true;
  std::size_t i = 0;
  while (i < e.size()) {
    const char c = e[i];
    if (std::isspace(static_cast<unsigned char>(c)) != 0) {
      ++i;
      continue;
    }
    if (operand_expected && c == '-') {
      ops.push_back(negate);
      if (pending()) return limit + 1;
      ++i;
      continue;
    }
    if (operand_expected && c == '(') {
      ops.push_back(open);
      marks.push_back(depth.size());
      if (pending()) return limit + 1;
      ++i;
      continue;
    }
    if (operand_expected) {
      // A number or a name, as the parser reads them.
      std::size_t j = i;
      if (std::isdigit(static_cast<unsigned char>(c)) != 0 || c == '.') {
        while (j < e.size() &&
               (std::isdigit(static_cast<unsigned char>(e[j])) != 0 || e[j] == '.' || e[j] == 'e' ||
                e[j] == 'E' ||
                ((e[j] == '+' || e[j] == '-') && j > i && (e[j - 1] == 'e' || e[j - 1] == 'E'))))
          ++j;
      } else {
        while (j < e.size() && (std::isalnum(static_cast<unsigned char>(e[j])) != 0 || e[j] == '_'))
          ++j;
      }
      if (j == i) return deepest;
      std::size_t k = j;
      while (k < e.size() && std::isspace(static_cast<unsigned char>(e[k])) != 0) ++k;
      if (k < e.size() && e[k] == '(' && std::isalpha(static_cast<unsigned char>(e[i])) != 0) {
        ops.push_back(call);
        marks.push_back(depth.size());
        if (pending()) return limit + 1;
        i = k + 1;
        continue;
      }
      if (!keep(1)) return deepest;
      i = j;
      operand_expected = false;
      continue;
    }
    // After an operand: a binary operator, a comma or a closing parenthesis.
    if (c == ')' || c == ',') {
      while (!ops.empty() && ops.back() != open && ops.back() != call) {
        if (!reduce()) return deepest;
      }
      if (ops.empty()) return deepest;
      if (c == ',') {
        operand_expected = true;
        ++i;
        continue;
      }
      ops.pop_back();
      // A parenthesis or a call is 1 more than its deepest argument (pow has two).
      const std::size_t from = marks.back();
      marks.pop_back();
      if (depth.size() <= from) return deepest;
      const int d =
          *std::max_element(depth.begin() + static_cast<std::ptrdiff_t>(from), depth.end());
      depth.resize(from);
      if (!keep(d + 1)) return deepest;
      ++i;
      continue;
    }
    const Kind k = c == '+' || c == '-' ? add : c == '*' || c == '/' ? multiply : open;
    if (k == open) return deepest;
    while (!ops.empty() && ops.back() != open && ops.back() != call &&
           binds(ops.back()) >= binds(k)) {
      if (!reduce()) return deepest;
    }
    ops.push_back(k);
    if (pending()) return limit + 1;
    operand_expected = true;
    ++i;
  }
  while (!ops.empty()) {
    if (ops.back() == open || ops.back() == call) return deepest;
    if (!reduce()) return deepest;
  }
  return deepest;
}

// E0208 for an expression of the manufactured solution deeper than the contract allows, before
// the physics builds any of them. A document that is not JSON is not an autan.mms/1 document.
void check_manufactured_depth(const std::string& text) {
  auto refuse = [](const std::string& message, const nlohmann::json& data) {
    throw InputError(Diagnostic{.code = "E0208",
                                .message = message,
                                .path = "verification.path",
                                .data_json = data.dump()});
  };
  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(text);
  } catch (const nlohmann::json::exception&) {
    refuse("the manufactured solution is not JSON", nlohmann::json::object());
  }
  if (!doc.is_object()) return;
  auto check = [&](const nlohmann::json& e, const std::string& where) {
    if (!e.is_string()) return;
    const int d = expression_depth(e.get_ref<const std::string&>(), manufactured_max_depth);
    if (d > manufactured_max_depth) {
      refuse("the manufactured solution's expression " + where + " is deeper than " +
                 std::to_string(manufactured_max_depth),
             {{"expression", where}, {"limit", manufactured_max_depth}});
    }
  };
  for (const char* group : {"fields", "gradients", "conserved", "sources", "auxiliary"}) {
    const auto g = doc.find(group);
    if (g == doc.end() || !g->is_object()) continue;
    if (const auto sub = g->find("subexpressions"); sub != g->end() && sub->is_array()) {
      for (std::size_t k = 0; k < sub->size(); ++k) {
        const auto& pair = (*sub)[k];
        if (pair.is_array() && pair.size() == 2) {
          check(pair[1], std::string(group) + ".subexpressions[" + std::to_string(k) + "]");
        }
      }
    }
    if (const auto out = g->find("outputs"); out != g->end() && out->is_object()) {
      for (const auto& [name, e] : out->items()) {
        check(e, std::string(group) + ".outputs." + name);
      }
    }
  }
}

// Set by SIGINT and SIGTERM; the loop finalises the run as stopped.
volatile std::sig_atomic_t stop_requested = 0;

// The process's memory at its peak so far, in bytes: VmHWM on Linux; on macOS the peak physical
// footprint, which counts the pages the memory compressor holds (ru_maxrss counts resident pages
// only, and macOS compresses idle ones even without pressure: #534); ru_maxrss elsewhere.
std::int64_t host_peak_bytes() {
#if defined(__linux__)
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);) {
    if (line.starts_with("VmHWM:")) return std::stoll(line.substr(6)) * 1024;  // kB
  }
  return 0;
#elif defined(__APPLE__)
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
      KERN_SUCCESS) {
    return 0;
  }
  return static_cast<std::int64_t>(info.ledger_phys_footprint_peak);
#else
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return static_cast<std::int64_t>(usage.ru_maxrss);
#endif
}
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

// The bytes of an open file in one string: its one copy in memory.
std::string file_bytes(std::ifstream& in) {
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::string bytes(static_cast<std::size_t>(std::max<std::streamoff>(size, 0)), '\0');
  in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!in) bytes.clear();
  return bytes;
}

std::string read_text(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw InputError(Diagnostic{.code = "E0705",
                                .message = "cannot read '" + path.string() + "'",
                                .path = "",
                                .data_json = "{}"});
  }
  return file_bytes(in);
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
  if (b.kind == "oracle-state") return "BCDirichlet";
  return "BCInflowSubsonic";  // inlet, velocity-inlet
}

// Everything the run needs, resolved and checked before the directory exists.
struct Prepared {
  SolverInput input;
  Mesh mesh;
  std::string mesh_hash;
  std::int64_t mesh_bytes = 0;               // the size of what mesh_hash covers
  std::string mesh_info;                     // MeshInfo of the mesh as read, before mesh.scale
  std::map<std::string, index_t> face_zone;  // boundary zone name -> Mesh::face_zones() index
  std::vector<std::vector<index_t>> monitor_zones;
  const PhysicsFactory* factory = nullptr;
  std::string cell_order;                     // "rcm" (default) or "input", from AUTAN_CELL_ORDER
  const PartitionGroups* groups = nullptr;    // the physics' groups; null or empty: none
  GhostStencil ghosts = GhostStencil::face2;  // the stencil the physics declares
  const RestartScalars* scalars = nullptr;    // its SolverState arrays; null or empty: none
  std::vector<Diagnostic> memory_warnings = {};  // W0712, W0713: the memory check's warnings
  std::optional<StartState> start;               // input.start, read and checked
  Json start_averaging;                          // `continue`: the previous run's averaging
  Json start_binary;                             // `continue`: the previous run's binary_sha256
  // oracle-state: paired face states per face zone, and the files to archive (sha256, bytes).
  std::map<index_t, std::vector<double>> boundary_states;
  std::map<std::string, std::string> state_files;
  // initial_state: Density, VelocityX/Y/Z and Pressure of every cell, NaN in solid cells.
  std::optional<std::vector<io::Field>> initial;
  // Formed on every rank before the ranks agree to start, so that a rank that
  // cannot form them refuses with the others.
  std::optional<Geometry> geometry = std::nullopt;
  CellGroups cell_groups = {};
  std::vector<std::int32_t> group_colours = {};
  std::vector<std::int32_t> parts = {};  // each cell's rank on several ranks; empty on one
};

[[noreturn]] void start_mismatch(const std::string& code, const std::string& message,
                                 const std::string& path, const nlohmann::json& data = {}) {
  throw InputError(Diagnostic{.code = code,
                              .message = message,
                              .path = path,
                              .data_json = data.is_null() ? "{}" : data.dump()});
}

// solver-input.md, `periodic`: the pairs of MeshInfo, which must be the
// mesh's.
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
}

// The ghost exchange of a run without ghosts.
class NoGhosts final : public GhostExchange {
 public:
  void forward(std::span<float* const> /*c*/, MemorySpace /*s*/, void* /*stream*/) override {}
  void forward(std::span<double* const> /*c*/, MemorySpace /*s*/, void* /*stream*/) override {}
};

// A private copy of a file and the SHA-256 of the bytes copied: what is
// hashed is what is read, whatever happens to the original meanwhile.
class SnapshotFile {
 public:
  explicit SnapshotFile(const fs::path& from) {
    std::random_device random;
    path_ = fs::temp_directory_path() /
            ("autan-start-" + std::to_string(random()) + "-" + std::to_string(random()) + ".cgns");
    // A directory opens on some standard libraries and reads nothing without an error: anything
    // but a regular file is refused before it is opened, alike on every platform.
    std::error_code ec;
    if (!fs::is_regular_file(from, ec))
      throw Error(ErrorCode::io_cannot_open, "'" + from.string() + "' is not a regular file");
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

const Geometry& geometry_of(Prepared& p) {
  if (!p.geometry) p.geometry = compute_geometry(p.mesh);
  return *p.geometry;
}

// solver-input.md, `start`: the runtime checks the previous solution's hash,
// mesh, precision and fields, and reads its state, before any directory exists. `entry` is the
// physics that continues it, null when the build has none (E0706 follows).
void load_start(Prepared& p, const PhysicsRegistry::Entry* entry) {
  const Start& start = *p.input.start;
  const bool from_checkpoint = start.file == "checkpoint";
  const fs::path file =
      fs::path(start.run) / (from_checkpoint ? "checkpoint.cgns" : "solution.cgns");
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
    // A warm start takes the fields alone (solver-input.md, `start`).
    // results.md, Sizes a reader accepts: what the writer wrote for this mesh, checked before
    // the CGNS library loads the file.
    if (const std::int64_t cells = io::solution_cells(snapshot->path());
        cells != p.mesh.n_cells()) {
      start_mismatch("E0203", "the previous run was computed on another mesh", "start.run",
                     {{"expected_cells", p.mesh.n_cells()}, {"found_cells", cells}});
    }
    const io::MeshInfoZones zones = io::meshinfo_zones(p.mesh);
    io::check_start_solution(
        snapshot->path(),
        {.zones = static_cast<std::int64_t>(zones.boundary.size() + zones.cell.size()),
         .periodic_pairs = static_cast<std::int64_t>(p.mesh.periodic_pairs().size())});
    // results.md, Sizes a reader accepts: the arrays the run writes, checked by name before
    // any is read, cells doubles each.
    const auto arrays = io::solution_field_names(snapshot->path());
    const auto named = [&](const char* solution) {
      const auto it = arrays.find(solution);
      return it == arrays.end() ? std::vector<std::string>{} : it->second;
    };
    if (named("FlowSolution") != field_names(p.input)) {
      start_mismatch("E0708", "the previous run solved other variables", "start.run");
    }
    if (start.mode == "continue") {
      if (const auto mean = named("FlowSolutionMean");
          !mean.empty() && mean != field_names(p.input)) {
        unusable("FlowSolutionMean holds other fields than the run averages");
      }
      const bool averaged = !named("FlowSolutionMean").empty();
      const auto restart = entry != nullptr && entry->restart ? entry->restart(p.input, averaged)
                                                              : std::vector<std::string>{};
      if (named("RestartState") != restart) {
        unusable("RestartState holds other arrays than the physics writes");
      }
      const auto scalars = entry != nullptr && entry->scalars
                               ? entry->scalars(p.input, averaged)
                               : std::vector<std::pair<std::string, std::int64_t>>{};
      if (io::solver_state_arrays(snapshot->path()) != scalars) {
        unusable("SolverState holds other arrays or lengths than the physics writes");
      }
    }
    stored = io::read_solution(snapshot->path(), start.mode == "continue");
  } catch (const Error& e) {
    if (e.code() == ErrorCode::io_external_reference) {
      throw InputError(Diagnostic{
          .code = "E0134", .message = e.what(), .path = "start.run", .data_json = e.data_json()});
    }
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
  // A checkpoint names its iteration (results.md#checkpointcgns); a final solution does not.
  if (from_checkpoint) {
    const Json* c = metadata.contains("checkpoint") ? &metadata["checkpoint"] : nullptr;
    if (c == nullptr || !c->is_object() || !c->contains("iteration") ||
        !(*c)["iteration"].is_number_integer())
      unusable("RunMetadata has no checkpoint iteration");
    if ((*c)["iteration"].get<std::int64_t>() != start.iterations)
      unusable("the checkpoint is of iteration " + (*c)["iteration"].dump() +
               ", start.iterations " + std::to_string(start.iterations));
  } else if (metadata.contains("checkpoint")) {
    unusable("RunMetadata of a final solution has a checkpoint");
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
  StartState state{
      .fields = std::move(stored.fields), .mean_fields = {}, .restart_fields = {}, .scalars = {}};
  if (start.mode == "continue") {
    // A continuation goes on in the order the run computed in (results.md,
    // cell_order): another order would change its bits from the restart on.
    const std::string order = text(metadata, "cell_order");
    if (order != p.cell_order) {
      start_mismatch("E0708", "the previous run computed in cell order '" + order + "'",
                     "start.run", {{"expected", p.cell_order}, {"found", order}});
    }
    if (!metadata.contains("averaging") || !is_averaging(metadata["averaging"]))
      unusable("RunMetadata has no valid averaging");
    p.start_averaging = metadata["averaging"];
    const Json solver = metadata.contains("solver") ? metadata["solver"] : Json::object();
    if (!solver.is_object() || !solver.contains("binary_sha256") ||
        !(solver["binary_sha256"].is_string() || solver["binary_sha256"].is_null()))
      unusable("RunMetadata has no solver.binary_sha256");
    p.start_binary = solver["binary_sha256"];
    state.mean_fields = std::move(stored.mean_fields);
    state.restart_fields = std::move(stored.restart_fields);
    state.scalars = std::move(stored.solver_state);
  }
  p.start = std::move(state);
}

// One zone of an autan.face-states/1 document: rows of centroids (3) and values (5).
struct ZoneStates {
  std::size_t faces = 0;
  std::vector<double> centroids, values;
};

// The rows of a member "centroids" (3 numbers each) or "values" (5) of a state document, taken
// out of the parse as they come: `numbers` row after row, and `ok` while every row is an array of
// that many finite numbers.
struct Rows {
  std::vector<double> numbers;
  std::size_t rows = 0;
  bool ok = true;
};
// Rows by the ordinals of the members leading to them (each member's position in its object,
// in file order): no key is copied, so a long name costs its text and its one document key.
using RowMap = std::map<std::vector<std::size_t>, Rows>;
using StateDocument = nlohmann::ordered_json;

// The position of member `key` in `object`, in file order; npos when absent.
std::size_t ordinal(const StateDocument& object, std::string_view key) {
  std::size_t i = 0;
  for (const auto& [k, v] : object.items()) {
    if (k == key) return i;
    ++i;
  }
  return std::string_view::npos;
}

// A free text of a state file, the oracle and every member name: at most 4096 bytes in UTF-8
// (case-schema.md#free-texts, face-states.md, initial-state.md); a number, at most 64 characters.
constexpr std::size_t max_text_bytes = 4096;
constexpr std::size_t max_number_chars = 64;
constexpr std::size_t max_state_zones = 16384;

// What a scan of a state file's text finds before it is parsed: the longest run of characters a
// JSON number is made of, outside strings, which bounds every number token before the lexer,
// holding a token twice, accumulates one; and the deepest nesting of brackets, which bounds the
// parser's stacks before it pushes a level.
struct TextScan {
  std::size_t number = 0;
  std::size_t depth = 0;
  std::size_t string = 0;  // the longest string's raw bytes, escapes included
  std::size_t rows =
      0;  // members "centroids" and "values" holding an array: what parse_rows streams
};

TextScan scan_text(std::string_view text) {
  TextScan out;
  std::size_t run = 0, depth = 0, start = 0;
  bool in_string = false, escaped = false;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char ch = text[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == '"') {
        in_string = false;
        const std::string_view s = text.substr(start, i - start);
        out.string = std::max(out.string, s.size());
        if (s == "centroids" || s == "values") {
          // A member name followed by an array, past the white space JSON allows.
          const auto next = [&](std::size_t from) {
            return text.find_first_not_of(" \t\n\r", from);
          };
          const std::size_t colon = next(i + 1);
          if (colon != std::string_view::npos && text[colon] == ':') {
            const std::size_t value = next(colon + 1);
            if (value != std::string_view::npos && text[value] == '[') ++out.rows;
          }
        }
      }
      continue;
    }
    start = i + 1;
    const bool number =
        (ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E';
    run = number ? run + 1 : 0;
    out.number = std::max(out.number, run);
    if (ch == '[' || ch == '{') out.depth = std::max(out.depth, ++depth);
    if ((ch == ']' || ch == '}') && depth > 0) --depth;
    in_string = ch == '"';
  }
  return out;
}

// Why `text` exceeds a state file's bounds: numbers of 64 characters, `depth` levels of nesting.
std::string beyond_bounds(std::string_view text, std::size_t depth) {
  const TextScan scan = scan_text(text);
  if (scan.number > max_number_chars) {
    return "a number of " + std::to_string(scan.number) + " characters (at most 64)";
  }
  if (scan.depth > depth) return "nested deeper than its structure";
  // A string is at most 4096 bytes, each written in at most 6 (\\uXXXX).
  if (scan.string > 6 * max_text_bytes) {
    return "a string of " + std::to_string(scan.string) + " bytes (at most 4096)";
  }
  // A zone's centroids and values each, of at most 16 384 zones (mesh.md, Readers).
  if (scan.rows > 2 * max_state_zones) {
    return "more than " + std::to_string(max_state_zones) + " zones";
  }
  return "";
}

// `text` parsed (discarded if not JSON), every "centroids" and "values" array member streamed into
// `streamed` under the ordinals of the members leading to it and left empty in the document:
// what a state file holds in memory is then its text, 8 bytes per number of its rows, and its
// other members once. `duplicate` is set when an object has a member twice; `longest` is the
// longest member name.
StateDocument parse_rows(const std::string& text, RowMap& streamed, bool& duplicate,
                         std::size_t& longest) {
  enum class Kind : std::uint8_t { plain, rows, row };
  enum class Member : std::uint8_t { other, centroids, values };
  constexpr std::size_t none = std::string_view::npos;
  struct Frame {
    bool array;
    std::size_t ordinal;  // its position in the object holding it; none in an array or at the top
    Kind kind;
    std::size_t count;    // a row's numbers so far
    std::size_t members;  // an object's keys so far
  };
  std::vector<Frame> stack;
  Member pending = Member::other;
  Rows* current = nullptr;
  std::size_t width = 0;
  using Event = StateDocument::parse_event_t;
  const auto callback = [&](int /*depth*/, Event event, StateDocument& parsed) {
    switch (event) {
      case Event::key: {
        const auto& key = parsed.get_ref<const std::string&>();
        longest = std::max(longest, key.size());
        pending = key == "centroids" ? Member::centroids
                  : key == "values"  ? Member::values
                                     : Member::other;
        if (!stack.empty()) ++stack.back().members;
        return true;
      }
      case Event::object_start:
      case Event::array_start: {
        const bool member = !stack.empty() && !stack.back().array;
        const Kind parent = stack.empty() ? Kind::plain : stack.back().kind;
        Frame f{.array = event == Event::array_start,
                .ordinal = member ? stack.back().members - 1 : none,
                .kind = Kind::plain,
                .count = 0,
                .members = 0};
        if (parent == Kind::rows && f.array) {
          f.kind = Kind::row;
        } else if (parent != Kind::plain) {
          current->ok = false;  // an object in the rows, or anything nested in a row
        } else if (f.array && member && current == nullptr && pending != Member::other) {
          f.kind = Kind::rows;  // not inside another rows member, which this one would replace
          width = pending == Member::centroids ? 3 : 5;
        }
        pending = Member::other;
        stack.push_back(f);
        if (f.kind == Kind::rows) {
          std::vector<std::size_t> path;
          for (const auto& g : stack) {
            if (g.ordinal != none) path.push_back(g.ordinal);
          }
          current = &streamed[path];
        }
        return true;
      }
      case Event::value: {
        if (stack.empty()) return true;
        Frame& top = stack.back();
        if (top.kind == Kind::row) {
          const bool number = parsed.is_number() && std::isfinite(parsed.get<double>());
          if (number) current->numbers.push_back(parsed.get<double>());
          current->ok = current->ok && number;
          ++top.count;
          return false;
        }
        if (top.kind == Kind::rows) {
          current->ok = false;
          return false;
        }
        if (!top.array) pending = Member::other;
        return true;
      }
      case Event::object_end:
      case Event::array_end: {
        const Frame f = stack.back();
        stack.pop_back();
        if (f.kind == Kind::row) {
          current->ok = current->ok && f.count == width;
          ++current->rows;
          return false;
        }
        if (f.kind == Kind::rows) {
          current = nullptr;
          return true;
        }
        // A member given twice leaves the object with fewer members than keys were read.
        if (!f.array && f.kind == Kind::plain && parsed.is_object() && parsed.size() < f.members)
          duplicate = true;
        return stack.empty() || stack.back().kind == Kind::plain;
      }
    }
    return true;
  };
  return StateDocument::parse(text, callback, false);
}

// The zones of a face-state document, or why it is not one (face-states.md, Document): every
// key, every zone and every row are checked, whatever the case uses.
// The zones are views of the document's keys, which must outlive them.
std::map<std::string_view, ZoneStates> face_state_zones(const StateDocument& doc, RowMap& rows,
                                                        std::string& why) {
  std::map<std::string_view, ZoneStates> zones;
  const auto fail = [&](std::string what) {
    why = std::move(what);
    return std::map<std::string_view, ZoneStates>{};
  };
  if (!doc.is_object()) return fail("not a JSON object");
  for (const auto& [key, value] : doc.items()) {
    if (key != "schema" && key != "oracle" && key != "mesh_sha256" && key != "variables" &&
        key != "zones")
      return fail("unknown key '" + key + "'");
  }
  const auto string_at = [&](const char* key) {
    return doc.contains(key) && doc[key].is_string() ? doc[key].get<std::string>() : std::string();
  };
  if (string_at("schema") != "autan.face-states/1") return fail("schema");
  if (!doc.contains("oracle") || !doc["oracle"].is_string() ||
      doc["oracle"].get_ref<const std::string&>().size() > max_text_bytes)
    return fail("oracle");
  const std::string mesh = string_at("mesh_sha256");
  if (mesh.size() != 64 || mesh.find_first_not_of("0123456789abcdef") != std::string::npos)
    return fail("mesh_sha256");
  if (!doc.contains("variables") ||
      doc["variables"] !=
          StateDocument{"density", "velocity-x", "velocity-y", "velocity-z", "pressure"})
    return fail("variables");
  if (!doc.contains("zones") || !doc["zones"].is_object()) return fail("zones");
  const std::size_t zones_at = ordinal(doc, "zones");
  std::size_t index = 0;
  for (const auto& item : doc["zones"].items()) {
    const std::string& name = item.key();
    const StateDocument& z = item.value();
    const std::size_t zone_at = index++;
    const auto at_zone = [&] { return "zones." + name; };
    if (!z.is_object() || z.size() != 3 || !z.contains("faces") ||
        !z["faces"].is_number_integer() || z["faces"].get<std::int64_t>() < 0)
      return fail(at_zone() + ".faces");
    ZoneStates out;
    out.faces = z["faces"].get<std::size_t>();
    for (const char* array : {"centroids", "values"}) {
      const auto it = rows.find({zones_at, zone_at, ordinal(z, array)});
      if (!z.contains(array) || !z[array].is_array() || it == rows.end() || !it->second.ok ||
          it->second.rows != out.faces)
        return fail(at_zone() + "." + array);
      (std::string(array) == "centroids" ? out.centroids : out.values) =
          std::move(it->second.numbers);
    }
    for (std::size_t r = 0; r < out.faces; ++r) {
      if (!(out.values[5 * r] > 0.0) || !(out.values[5 * r + 4] > 0.0))
        return fail(at_zone() + ".values: density and pressure must be > 0");
    }
    zones.emplace(std::string_view(name), std::move(out));
  }
  return zones;
}

// face-states.md: the file of each oracle-state condition, checked by its hash and as a
// document, then against the mesh and the zone, then each face of the zone paired with exactly
// one row by centroid.
void load_states(Prepared& p) {
  const auto offsets = p.mesh.boundary_zone_offsets();
  for (std::size_t i = 0; i < p.input.boundaries.size(); ++i) {
    const auto& b = p.input.boundaries[i];
    if (!b.states) continue;
    const std::string key = "boundaries[" + std::to_string(i) + "].states";
    const auto fail = [&](const char* code, const std::string& message, const std::string& sub,
                          const nlohmann::json& data = nlohmann::json::object()) {
      throw InputError(Diagnostic{
          .code = code, .message = message, .path = key + sub, .data_json = data.dump()});
    };
    std::error_code ec;
    if (!fs::is_regular_file(b.states->path, ec)) {
      fail("E0420", "'" + b.states->path + "' is not a readable file", ".path");
    }
    std::ifstream in(b.states->path, std::ios::binary);
    if (!in) fail("E0420", "cannot read '" + b.states->path + "'", ".path");
    std::string bytes = file_bytes(in);
    const std::string found = sha256_hex(bytes);
    if (found != b.states->sha256) {
      fail("E0420", "the face-state file's SHA-256 differs from states.sha256", ".sha256",
           {{"expected", b.states->sha256}, {"found", found}});
    }
    RowMap rows;
    bool duplicate = false;
    std::size_t longest = 0;
    // face-states.md: object, zones, zone, rows, row.
    std::string why = beyond_bounds(bytes, 5);
    StateDocument doc = why.empty() ? parse_rows(bytes, rows, duplicate, longest) : StateDocument{};
    if (why.empty() && doc.is_discarded()) why = "not JSON";
    if (why.empty() && duplicate) why = "an object has a member twice";
    if (why.empty() && longest > max_text_bytes) {
      why = "a member name of " + std::to_string(longest) + " bytes (at most 4096)";
    }
    auto zones =
        why.empty() ? face_state_zones(doc, rows, why) : std::map<std::string_view, ZoneStates>{};
    if (!why.empty()) {
      fail("E0420", "'" + b.states->path + "' is not an autan.face-states/1 document: " + why,
           ".path", {{"problem", why}});
    }
    if (doc["mesh_sha256"].get<std::string>() != p.mesh_hash) {
      fail("E0422", "the face states were exported for another mesh", ".path",
           {{"expected", p.mesh_hash}, {"found", doc["mesh_sha256"]}});
    }
    const auto it = zones.find(b.zone);
    if (it == zones.end()) {
      fail("E0423", "the face states do not cover zone '" + b.zone + "'", ".path",
           {{"zone", b.zone}});
    }
    const ZoneStates z = std::move(it->second);
    zones.clear();
    doc = StateDocument();  // the zone's numbers are in z: the pairing needs no more
    const index_t zone = p.face_zone.at(b.zone);
    const auto first = static_cast<std::size_t>(offsets[at(zone)]);
    const auto n = static_cast<std::size_t>(offsets[at(zone) + 1]) - first;
    if (z.faces != n) {
      fail("E0423", "the face states give zone '" + b.zone + "' another face count", ".path",
           {{"zone", b.zone}, {"expected", n}, {"found", z.faces}});
    }
    const Geometry& g = geometry_of(p);
    // Tolerance: 1e-9 times the largest extent of the bounding box of the zone's nodes.
    std::array<double, 3> lo{}, hi{};
    lo.fill(std::numeric_limits<double>::infinity());
    hi.fill(-std::numeric_limits<double>::infinity());
    const std::array<std::span<const double>, 3> xyz{p.mesh.x(), p.mesh.y(), p.mesh.z()};
    for (std::size_t f = first; f < first + n; ++f) {
      for (const index_t v : p.mesh.face_nodes(static_cast<index_t>(f))) {
        for (std::size_t d = 0; d < 3; ++d) {
          lo[d] = std::min(lo[d], xyz[d][at(v)]);
          hi[d] = std::max(hi[d], xyz[d][at(v)]);
        }
      }
    }
    std::size_t axis = 0;
    for (std::size_t d = 1; d < 3; ++d) {
      if (hi[d] - lo[d] > hi[axis] - lo[axis]) axis = d;
    }
    const double tol = 1e-9 * (hi[axis] - lo[axis]);
    // Rows sorted along the zone's widest axis: a face looks only at the rows within tol on it.
    std::vector<std::size_t> sorted(n);
    for (std::size_t r = 0; r < n; ++r) sorted[r] = r;
    const auto coordinate = [&](std::size_t r) { return z.centroids[3 * r + axis]; };
    std::ranges::sort(sorted, {}, coordinate);
    std::vector<char> used(n, 0);
    std::vector<double> paired(5 * n);
    for (std::size_t j = 0; j < n; ++j) {
      const std::size_t f = first + j;
      const std::array<double, 3> c{g.face_centroid_x[f], g.face_centroid_y[f],
                                    g.face_centroid_z[f]};
      std::size_t match = n, count = 0;
      for (auto r = std::ranges::lower_bound(sorted, c[axis] - tol, {}, coordinate);
           r != sorted.end() && coordinate(*r) <= c[axis] + tol; ++r) {
        const double* q = &z.centroids[3 * *r];
        if (std::hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]) <= tol) {
          match = *r;
          ++count;
        }
      }
      if (count != 1 || used[match] != 0) {
        fail("E0421",
             "face " + std::to_string(j) + " of zone '" + b.zone + "' matches " +
                 std::to_string(count) + " rows of the face states",
             ".path", {{"zone", b.zone}, {"face", j}, {"centroid", c}, {"matches", count}});
      }
      used[match] = 1;
      for (std::size_t k = 0; k < 5; ++k) paired[k * n + j] = z.values[5 * match + k];
    }
    p.boundary_states[zone] = std::move(paired);
    p.state_files.emplace(found, std::move(bytes));
  }
}

// initial-state.md: the file read once and checked (E0508), against the mesh (E0509) and its
// fluid cells (E0510), then every fluid cell paired with exactly one row by centroid, and every
// row with exactly one cell (E0511), on the whole mesh.
void load_initial(Prepared& p) {
  const InitialStateFile& file = *p.input.initial_state;
  const auto fail = [&](const char* code, const std::string& message,
                        const nlohmann::json& data = nlohmann::json::object()) {
    throw InputError(Diagnostic{
        .code = code, .message = message, .path = "initial_state.path", .data_json = data.dump()});
  };
  std::error_code ec;
  if (!fs::is_regular_file(file.path, ec)) fail("E0508", "'" + file.path + "' is not a file");
  std::ifstream in(file.path, std::ios::binary);
  if (!in) fail("E0508", "cannot read '" + file.path + "'");
  std::string bytes = file_bytes(in);
  const std::string found = sha256_hex(bytes);
  if (found != file.sha256) {
    fail("E0508", "the initial-state file's SHA-256 differs from initial_state.sha256",
         {{"expected", file.sha256}, {"found", found}});
  }
  RowMap streamed;
  bool duplicate = false;
  std::size_t longest = 0;
  const auto invalid_document = [&](const std::string& why) {
    fail("E0508", "'" + file.path + "' is not an autan.initial-state/1 document: " + why,
         {{"problem", why}});
  };
  // initial-state.md: object, rows, row.
  if (const std::string why = beyond_bounds(bytes, 3); !why.empty()) invalid_document(why);
  StateDocument doc = parse_rows(bytes, streamed, duplicate, longest);
  if (doc.is_discarded() || !doc.is_object()) invalid_document("not a JSON object");
  if (duplicate) invalid_document("an object has a member twice");
  if (longest > max_text_bytes) {
    invalid_document("a member name of " + std::to_string(longest) + " bytes (at most 4096)");
  }
  for (const auto& [key, value] : doc.items()) {
    if (key != "schema" && key != "oracle" && key != "mesh_sha256" && key != "variables" &&
        key != "cells" && key != "centroids" && key != "values")
      invalid_document("unknown key '" + key + "'");
  }
  if (!doc.contains("schema") || doc["schema"] != "autan.initial-state/1")
    invalid_document("schema");
  if (!doc.contains("oracle") || !doc["oracle"].is_string() ||
      doc["oracle"].get_ref<const std::string&>().size() > max_text_bytes)
    invalid_document("oracle");
  if (!doc.contains("mesh_sha256") || !doc["mesh_sha256"].is_string() ||
      doc["mesh_sha256"].get<std::string>().size() != 64 ||
      doc["mesh_sha256"].get<std::string>().find_first_not_of("0123456789abcdef") !=
          std::string::npos)
    invalid_document("mesh_sha256");
  if (!doc.contains("variables") ||
      doc["variables"] !=
          StateDocument{"density", "velocity-x", "velocity-y", "velocity-z", "pressure"})
    invalid_document("variables");
  if (!doc.contains("cells") || !doc["cells"].is_number_integer() ||
      doc["cells"].get<std::int64_t>() < 1)
    invalid_document("cells");
  const auto rows = doc["cells"].get<std::size_t>();
  std::vector<double> centroids, values;
  for (const char* array : {"centroids", "values"}) {
    const auto it = streamed.find({ordinal(doc, array)});
    if (!doc.contains(array) || !doc[array].is_array() || it == streamed.end() || !it->second.ok ||
        it->second.rows != rows)
      invalid_document(array);
    (std::string(array) == "centroids" ? centroids : values) = std::move(it->second.numbers);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    if (!(values[5 * r] > 0.0) || !(values[5 * r + 4] > 0.0))
      invalid_document("values: density and pressure must be > 0");
  }
  if (doc["mesh_sha256"].get<std::string>() != p.mesh_hash) {
    fail("E0509", "the initial state was written for another mesh",
         {{"expected", p.mesh_hash}, {"found", doc["mesh_sha256"]}});
  }
  doc = StateDocument();  // its numbers are in centroids and values: the pairing needs no more
  // Fluid cells: those of zones whose initial state is a fluid's.
  std::map<std::string, bool> fluid_zone;
  for (const auto& z : p.input.cell_zones) fluid_zone[z.zone] = z.initial.kind == "fluid";
  const Mesh& mesh = p.mesh;
  std::vector<index_t> fluid;
  for (index_t c = 0; c < mesh.n_cells(); ++c) {
    if (fluid_zone[mesh.cell_zones()[at(mesh.cell_zone()[at(c)])].name]) fluid.push_back(c);
  }
  if (rows != fluid.size()) {
    fail("E0510",
         "the initial state has " + std::to_string(rows) + " rows for " +
             std::to_string(fluid.size()) + " fluid cells",
         {{"expected", fluid.size()}, {"found", rows}});
  }
  // Tolerance: 1e-9 times the largest extent of the mesh's bounding box.
  const std::array<std::span<const double>, 3> xyz{mesh.x(), mesh.y(), mesh.z()};
  double extent = 0.0;
  for (const auto& coordinate : xyz) {
    const auto [lo, hi] = std::ranges::minmax_element(coordinate);
    extent = std::max(extent, *hi - *lo);
  }
  const double tol = 1e-9 * extent;
  const Geometry& g = geometry_of(p);
  const auto centroid = [&](index_t c) {
    const auto i = at(c);
    return std::array<double, 3>{g.cell_centroid_x[i], g.cell_centroid_y[i], g.cell_centroid_z[i]};
  };
  // Points binned in cubes of side tol: two points within tol lie in
  // neighbouring bins, whatever their alignment, so pairing is linear.
  using Bin = std::array<std::int64_t, 3>;
  struct BinHash {
    std::size_t operator()(const Bin& b) const noexcept {
      return std::hash<std::int64_t>{}(b[0] * 73856093 ^ b[1] * 19349663 ^ b[2] * 83492791);
    }
  };
  const auto bin = [&](const std::array<double, 3>& x) {
    return Bin{static_cast<std::int64_t>(std::floor(x[0] / tol)),
               static_cast<std::int64_t>(std::floor(x[1] / tol)),
               static_cast<std::int64_t>(std::floor(x[2] / tol))};
  };
  const auto near = [&](const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return std::hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) <= tol;
  };
  // The 27 bins around b, each visited with `visit`.
  const auto around = [](const Bin& b, const auto& visit) {
    for (std::int64_t dx = -1; dx <= 1; ++dx) {
      for (std::int64_t dy = -1; dy <= 1; ++dy) {
        for (std::int64_t dz = -1; dz <= 1; ++dz) visit(Bin{b[0] + dx, b[1] + dy, b[2] + dz});
      }
    }
  };
  // Within tol of each other, two fluid cells cannot be told apart.
  std::unordered_map<Bin, std::vector<index_t>, BinHash> cell_bins;
  for (const index_t c : fluid) cell_bins[bin(centroid(c))].push_back(c);
  for (const index_t c : fluid) {
    around(bin(centroid(c)), [&](const Bin& b) {
      const auto it = cell_bins.find(b);
      if (it == cell_bins.end()) return;
      for (const index_t other : it->second) {
        if (other > c && near(centroid(c), centroid(other))) {
          fail("E0511",
               "cells " + std::to_string(c) + " and " + std::to_string(other) +
                   " lie within the pairing tolerance of each other",
               {{"cells", {c, other}}, {"tolerance", tol}});
        }
      }
    });
  }
  std::unordered_map<Bin, std::vector<std::size_t>, BinHash> row_bins;
  for (std::size_t r = 0; r < rows; ++r)
    row_bins[bin({centroids[3 * r], centroids[3 * r + 1], centroids[3 * r + 2]})].push_back(r);
  std::vector<std::int64_t> owner(rows, -1);  // the cell each row is paired with
  constexpr double unset = std::numeric_limits<double>::quiet_NaN();
  std::vector<io::Field> fields{{.name = "Density", .values = {}},
                                {.name = "VelocityX", .values = {}},
                                {.name = "VelocityY", .values = {}},
                                {.name = "VelocityZ", .values = {}},
                                {.name = "Pressure", .values = {}}};
  for (auto& f : fields) f.values.assign(at(mesh.n_cells()), unset);
  for (const index_t c : fluid) {
    const auto x = centroid(c);
    std::size_t match = rows, count = 0;
    around(bin(x), [&](const Bin& b) {
      const auto it = row_bins.find(b);
      if (it == row_bins.end()) return;
      for (const std::size_t r : it->second) {
        if (near(x, {centroids[3 * r], centroids[3 * r + 1], centroids[3 * r + 2]})) {
          match = r;
          ++count;
        }
      }
    });
    if (count != 1) {
      fail("E0511",
           "cell " + std::to_string(c) + " matches " + std::to_string(count) +
               " rows of the initial state",
           {{"cells", {c}}, {"centroid", x}, {"matches", count}});
    }
    if (owner[match] >= 0) {
      fail("E0511",
           "cells " + std::to_string(owner[match]) + " and " + std::to_string(c) +
               " match one row of the initial state",
           {{"cells", {owner[match], c}}, {"centroid", x}});
    }
    owner[match] = c;
    for (std::size_t k = 0; k < 5; ++k) fields[k].values[at(c)] = values[5 * match + k];
  }
  p.initial = std::move(fields);
  p.state_files.emplace(found, std::move(bytes));
}

// solver-input.md, `memory`: the pools of this rank's host, measured before any rank reads the
// mesh. A device whose memory is the host's (GB10) is one pool with it, and gets no context here.
MemoryPools memory_pools(const SolverInput& in, const Comm& comm) {
  HostMemory host = host_memory();
  MemoryPools pools{.host_available = host.available,
                    .device_free = {},
                    .devices = 1,
                    .unseen_cgroups = std::move(host.unseen_cgroups),
                    .cgroup_scope = std::move(host.cgroup_scope)};
  if (in.device_kind == "cuda" && visible_gpus() > 0 && !device_integrated(comm.local_rank())) {
    pools.device_free = device_free_bytes(comm.local_rank());
    pools.devices = visible_gpus();
  }
  return pools;
}

// The data of E0711 and W0712: what the check compared, and whether a cgroup limit above the
// process could not be read (W0713) on this rank or, for W0712, on any (`unverifiable`), so that
// "available" may overstate what the host gives.
Diagnostic memory_diagnostic(const char* code, const SolverInput& in, std::int64_t required,
                             std::int64_t available, std::int64_t margin,
                             const std::vector<std::string>& unseen, bool unverifiable,
                             const std::string& scope) {
  // solver-input.md, memory: in a cgroup namespace, the limits above its root are not observable.
  const std::string within = scope == "namespace_root"
                                 ? " (within the limits visible from the cgroup namespace root)"
                                 : "";
  return Diagnostic{.code = code,
                    .message = "the run needs " + std::to_string(required) + " bytes; " +
                               std::to_string(available) + " are available" + within +
                               ", less a margin of " + std::to_string(margin),
                    .path = "memory",
                    .data_json = Json{
                        {"required_bytes", required},
                        {"available_bytes", available},
                        {"margin_bytes", margin},
                        {"device_kind", in.device_kind},
                        {"cgroup_limit", unverifiable ? "unknown" : "known"},
                        {"cgroup_scope", scope},
                        {"unseen_cgroups",
                         unseen}}.dump()};
}

// W0713: a memory cgroup above the process that no mount shows, whose limit the check could not
// read; the check rests on MemAvailable and the limits it could read.
Diagnostic unverifiable_memory(const std::vector<std::string>& unseen, const std::string& scope) {
  std::string names;
  for (const auto& u : unseen) names += (names.empty() ? "" : ", ") + u;
  return Diagnostic{.code = "W0713",
                    .message = "the memory limit of cgroup " +
                               (names.empty() ? std::string("(on another rank)") : names) +
                               " cannot be read: the memory check rests on MemAvailable",
                    .path = "memory",
                    .data_json = Json{
                        {"cgroup_limit", "unknown"},
                        {"cgroup_scope", scope},
                        {"unseen_cgroups",
                         unseen}}.dump()};
}

// What a rank's memory check found: W0712 when it starts beyond its memory, and the cgroups whose
// limits it could not read.
struct MemoryCheck {
  std::optional<Diagnostic> beyond;
  std::vector<std::string> unseen_cgroups;
  std::string cgroup_scope = "host";
};

// The memory check of every rank, before any reads the mesh: E0711 when the ranks of this host
// need more than it can give less the margin, or with oversubscription allowed, W0712.
MemoryCheck check_memory(const fs::path& input_file, const Comm& comm, const MemoryPolicy& memory) {
  const SolverInput in = parse_solver_input(read_text(input_file));
  const MemoryPools pools = memory_pools(in, comm);
  const auto s = memory_shortfall(in.memory_bytes_host, in.memory_bytes_device, comm.local_size(),
                                  pools, memory.margin_bytes);
  if (!s) {
    return {.beyond = std::nullopt,
            .unseen_cgroups = pools.unseen_cgroups,
            .cgroup_scope = pools.cgroup_scope};
  }
  const char* code = memory.allow_oversubscription ? "W0712" : "E0711";
  Diagnostic d =
      memory_diagnostic(code, in, s->required, s->available, memory.margin_bytes,
                        pools.unseen_cgroups, !pools.unseen_cgroups.empty(), pools.cgroup_scope);
  if (!memory.allow_oversubscription) throw InputError(std::move(d));
  return {.beyond = std::move(d),
          .unseen_cgroups = pools.unseen_cgroups,
          .cgroup_scope = pools.cgroup_scope};
}

Prepared prepare(const fs::path& input_file, const PhysicsRegistry& registry, int ranks) {
  Prepared p{.input = parse_solver_input(read_text(input_file)),
             .mesh = {},
             .mesh_hash = {},
             .mesh_bytes = 0,
             .mesh_info = {},
             .face_zone = {},
             .monitor_zones = {},
             .factory = nullptr,
             .cell_order = {},
             .start = std::nullopt,
             .start_averaging = {},
             .start_binary = {},
             .boundary_states = {},
             .state_files = {},
             .initial = std::nullopt,
             .geometry = std::nullopt,
             .cell_groups = {},
             .group_colours = {},
             .parts = {}};
  try {
    p.mesh = io::read_mesh(p.input.mesh_path);
    p.mesh_hash = io::mesh_hash(p.input.mesh_path);
    p.mesh_bytes = io::mesh_bytes(p.input.mesh_path);
  } catch (const Error& e) {
    throw InputError(Diagnostic{.code = error_code_string(e.code()),
                                .message = e.what(),
                                .path = "mesh.path",
                                .data_json = e.data_json()});
  }
  if (p.mesh_hash != p.input.mesh_sha256) {
    throw InputError(Diagnostic{
        .code = "E0203",
        .message = "the mesh file's SHA-256 differs from mesh.sha256 of the input",
        .path = "mesh.sha256",
        .data_json =
            nlohmann::json{{"expected", p.input.mesh_sha256}, {"found", p.mesh_hash}}.dump()});
  }
  // solver-input.md, mesh_info: its source.bytes is the size of the mesh read.
  if (p.mesh_bytes != p.input.mesh_info_bytes) {
    throw InputError(Diagnostic{
        .code = "E0203",
        .message = "mesh_info's source.bytes differs from the size of the mesh read",
        .path = "mesh_info.source.bytes",
        .data_json =
            nlohmann::json{{"expected", p.input.mesh_info_bytes}, {"found", p.mesh_bytes}}.dump()});
  }
  check_periodic(p);
  const char* order = std::getenv("AUTAN_CELL_ORDER");
  p.cell_order = order == nullptr ? "rcm" : order;
  if (p.cell_order != "rcm" && p.cell_order != "input") {
    throw std::runtime_error("AUTAN_CELL_ORDER is '" + p.cell_order + "', not rcm or input");
  }
  // solver-input.md, `devices`: one rank per device, as many as the launcher started.
  if (p.input.device_count != ranks) {
    invalid("devices.count", "is " + std::to_string(p.input.device_count) + ", but " +
                                 std::to_string(ranks) + " ranks run the case");
  }
  // The run archives the zone groups the case was resolved with: the sidecar
  // must be the one the SDK hashed (solver-input.md, mesh.groups_sha256).
  // A sidecar the input does not name is refused unread: its plan counted no
  // memory for reading it (plan.md, zone groups).
  if (!p.input.mesh_groups_sha256) {
    const fs::path sidecar = io::zone_groups_path(p.input.mesh_path);
    std::error_code ec;
    if (fs::exists(fs::symlink_status(sidecar, ec)) ||
        (ec && ec != std::errc::no_such_file_or_directory)) {
      throw InputError(Diagnostic{
          .code = "E0137",
          .message = "the mesh has a zone-groups file '" + sidecar.string() +
                     "' that mesh.groups_sha256 does not name",
          .path = "mesh.groups_sha256",
          .data_json = nlohmann::json{{"expected", nullptr}, {"path", sidecar.string()}}.dump()});
    }
  }
  std::optional<io::ZoneGroups> groups;
  try {
    groups = io::read_zone_groups(p.input.mesh_path, p.mesh, p.mesh_hash);
  } catch (const Error& e) {
    throw InputError(Diagnostic{.code = error_code_string(e.code()),
                                .message = e.what(),
                                .path = "mesh.groups_sha256",
                                .data_json = "{}"});
  }
  const std::optional<std::string> found =
      groups ? std::optional<std::string>(groups->sha256) : std::nullopt;
  if (found != p.input.mesh_groups_sha256) {
    const auto text = [](const std::optional<std::string>& s) {
      return s ? nlohmann::json(*s) : nlohmann::json(nullptr);
    };
    throw InputError(Diagnostic{
        .code = "E0137",
        .message = "the mesh's zone-groups file is not the one mesh.groups_sha256 names",
        .path = "mesh.groups_sha256",
        .data_json =
            nlohmann::json{{"expected", text(p.input.mesh_groups_sha256)}, {"found", text(found)}}
                .dump()});
  }
  // solver-input.md, mesh_info: planned with the lines of the order this run computes in.
  if (p.input.mesh_info_order != p.cell_order) {
    invalid("mesh_info.quality.lines.order", "is '" + p.input.mesh_info_order +
                                                 "', but the run computes in cell order '" +
                                                 p.cell_order + "'");
  }
  if (p.input.mesh_info_scale != p.input.mesh_scale) {
    invalid("mesh_info.quality.lines.scale",
            "counted with " + std::to_string(p.input.mesh_info_scale) +
                " metres per mesh unit, but mesh.scale is " + std::to_string(p.input.mesh_scale));
  }
  p.mesh_info = p.input.mesh_info;
  // solver-input.md, mesh.scale: the coordinates and periodic translations in
  // metres once checked against the file, before any geometry or pairing.
  if (p.input.mesh_scale != 1.0) p.mesh = scaled(p.mesh, p.input.mesh_scale);
  if (p.input.start) load_start(p, registry.find(p.input));
  if (p.input.verification) {
    // The manufactured solution is checked like the mesh: by its hash.
    auto& v = *p.input.verification;
    std::ifstream in(v.path, std::ios::binary);
    if (!in) {
      throw InputError(Diagnostic{.code = "E0207",
                                  .message = "cannot read '" + v.path + "'",
                                  .path = "verification.path",
                                  .data_json = "{}"});
    }
    auto too_large = [&](const std::string& bytes) {
      throw InputError(
          Diagnostic{.code = "E0208",
                     .message = "the manufactured solution has " + bytes + " bytes, more than " +
                                std::to_string(manufactured_max_bytes),
                     .path = "verification.path",
                     .data_json = nlohmann::json{{"limit", manufactured_max_bytes}}.dump()});
    };
    // A regular file's size is known before reading; anything else (a pipe) is read up to one
    // byte past the limit.
    std::error_code size_error;
    const auto size = fs::file_size(v.path, size_error);
    if (!size_error && size > manufactured_max_bytes) too_large(std::to_string(size));
    v.text.assign(manufactured_max_bytes + 1, '\0');
    in.read(v.text.data(), static_cast<std::streamsize>(v.text.size()));
    v.text.resize(static_cast<std::size_t>(in.gcount()));
    if (v.text.size() > manufactured_max_bytes)
      too_large("at least " + std::to_string(v.text.size()));
    if (sha256_hex(v.text) != v.sha256) {
      throw InputError(Diagnostic{
          .code = "E0206",
          .message = "the manufactured solution's SHA-256 differs from verification.sha256",
          .path = "verification.sha256",
          .data_json =
              nlohmann::json{{"expected", v.sha256}, {"found", sha256_hex(v.text)}}.dump()});
    }
    check_manufactured_depth(v.text);
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
    // A periodic zone joins its partner; it takes no condition.
    if (p.mesh.face_zones()[at(p.face_zone.at(name))].kind == BcKind::periodic)
      invalid("boundaries[" + std::to_string(i) + "].zone", "zone '" + name + "' is periodic");
    conditioned[name] = true;
  }
  const auto offsets = p.mesh.boundary_zone_offsets();
  for (const auto& [name, z] : p.face_zone) {
    if (offsets[at(z) + 1] > offsets[at(z)] && !conditioned[name] &&
        p.mesh.face_zones()[at(z)].kind != BcKind::periodic) {
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
      // A periodic zone is no surface the fluid exerts a load on.
      if (p.mesh.face_zones()[at(p.face_zone.at(name))].kind == BcKind::periodic)
        invalid("monitors[" + std::to_string(m) + "].zones", "zone '" + name + "' is periodic");
      ids.push_back(p.face_zone.at(name));
    }
    p.monitor_zones.push_back(std::move(ids));
  }
  // A point vortex takes its circulation from the lift on walls (slip walls in inviscid flow).
  std::map<std::string, std::string> condition;
  for (const auto& b : p.input.boundaries) condition[b.zone] = b.kind;
  for (std::size_t i = 0; i < p.input.boundaries.size(); ++i) {
    const auto& vortex = p.input.boundaries[i].point_vortex;
    if (!vortex) continue;
    const std::string path = "boundaries[" + std::to_string(i) + "].point_vortex.zones";
    for (const auto& name : vortex->zones) {
      if (!p.face_zone.contains(name)) invalid(path, "no boundary zone '" + name + "'");
      const auto it = condition.find(name);
      if (it == condition.end() || (it->second != "wall" && it->second != "slip"))
        invalid(path, "zone '" + name + "' is not a wall");
    }
  }
  load_states(p);
  if (p.input.initial_state) load_initial(p);
  // A device run needs a build that drives that device and a device to
  // drive; neither falls back to the CPU.
  if (p.input.device_kind != "cpu") {
    const bool built = p.input.device_kind == "cuda" && gpu_build;
    if (!built) {
      throw InputError(Diagnostic{
          .code = "E0706",
          .message = "this build of autan-run does not drive " + p.input.device_kind + " devices",
          .path = "devices.kind",
          .data_json = nlohmann::json{{"path", "devices.kind"}}.dump()});
    }
    if (visible_gpus() == 0) {
      throw InputError(Diagnostic{
          .code = "E0807",
          .message = "devices.kind is cuda, but this process sees no CUDA device",
          .path = "devices.kind",
          .data_json =
              nlohmann::json{{"visible", 0}, {"needed", 1}, {"path", "devices.kind"}}.dump()});
    }
  }
  const auto* entry = registry.find(p.input);
  p.factory = entry != nullptr ? &entry->factory : nullptr;
  p.groups = entry != nullptr ? &entry->groups : nullptr;
  p.scalars = entry != nullptr ? &entry->scalars : nullptr;
  if (entry != nullptr && entry->scope) {
    if (auto refusal = entry->scope(p.input, ranks)) {
      auto data = nlohmann::json::parse(refusal->data_json);
      data["path"] = refusal->path;
      throw InputError(Diagnostic{.code = refusal->code,
                                  .message = std::move(refusal->message),
                                  .path = std::move(refusal->path),
                                  .data_json = data.dump()});
    }
  }
  if (entry != nullptr && entry->ghosts) {
    const std::optional<GhostStencil> ghosts = entry->ghosts(p.input);
    // A part with ghosts: on several ranks or with periodic pairs. Under reproducibility "ranks" on
    // one rank without pairs, the one part has none, and every stencil is the mesh's.
    const bool partitioned = ranks > 1 || !p.input.periodic.empty();
    if (!ghosts && partitioned) {
      throw InputError(Diagnostic{
          .code = "E0148",
          .message = "the physics' stencil is wider than any partition's ghosts, and this run "
                     "computes on a partition",
          .path = "physics",
          .data_json = nlohmann::json{{"path", "physics"}, {"ranks", ranks}}.dump()});
    }
    if (ghosts) p.ghosts = *ghosts;
  }
  if (entry != nullptr && ranks > 1 && !entry->several_ranks) {
    throw InputError(Diagnostic{
        .code = "E0706",
        .message =
            "this build's physics for the input runs on one rank, not " + std::to_string(ranks),
        .path = "physics",
        .data_json = nlohmann::json{{"path", "physics"}, {"ranks", ranks}}.dump()});
  }
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
  const Geometry& geometry = geometry_of(p);
  if (p.groups != nullptr && *p.groups) {
    p.cell_groups = (*p.groups)(p.input, p.mesh, geometry);
    p.group_colours = colour_cells(p.mesh, p.cell_groups.offsets, p.cell_groups.cells);  // E0140
  }
  // Every rank partitions the same mesh with the same seed (E0140 to E0143).
  if (ranks > 1) {
    p.parts = partition_cells(p.mesh, ranks,
                              {.cell_weights = cell_cost_weights(p.mesh),
                               .group_offsets = p.cell_groups.offsets,
                               .group_cells = p.cell_groups.cells});
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
  Run(Prepared& p, const SolverBuild& build, Comm& comm,
      std::chrono::steady_clock::time_point launched)
      : p_(p),
        comm_(comm),
        launched_(launched),
        ordered_(make_ordered()),
        groups_(std::move(p.cell_groups)),
        geometry_(compute_physics_geometry()),
        local_(make_part()) {
    if (p_.input.device_kind == "cuda") {
      gpu_.emplace(comm_.local_rank());
    }
    ghosts_ = local_ ? comm_.ghost_exchange(*local_) : std::make_unique<NoGhosts>();
    // The physics' cells and faces differ from the mesh's on a partition or
    // a renumbered mesh: their shares of the start, the initial state and
    // the face states, in its order.
    // Each global copy goes once its share is taken.
    if (local_ || ordered_) {
      if (p_.start) {
        local_start_ = owned_start(*p_.start);
        p_.start.reset();
      }
      if (p_.initial) {
        local_initial_ = owned_fields(*p_.initial);
        p_.initial.reset();
      }
    }
    local_states_ = physics_boundary_states();
    p_.boundary_states.clear();
    colour_cells_of_physics();
    face_keys();
    // Rank 0 names the run: its clock and its random suffix.
    const auto started = std::chrono::system_clock::now();
    std::random_device random;
    char suffix[9];
    std::snprintf(suffix, sizeof suffix, "%08x", static_cast<unsigned>(random()));
    const std::string run_id =
        comm_.broadcast("run-" + utc(started, "%Y%m%dT%H%M%SZ") + "-" + suffix);
    const std::string started_at = comm_.broadcast(utc(started, "%Y-%m-%dT%H:%M:%SZ"));
    const auto& in = p_.input;
    const std::string backend = in.device_kind;
    std::string parts = in.case_sha256 + "\n" + p_.mesh_hash + "\n" +
                        in.mesh_groups_sha256.value_or("") + "\n" + build.commit + "\n" +
                        in.precision + "\n" + backend + "\n" + in.device_kind + "\n" +
                        std::to_string(in.device_count) + "\n" + cell_order();
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
                   {"file", s.file},
                   {"residual_reference", reference},
                   {"woken", s.woken ? Json(*s.woken) : Json(nullptr)}};
    }
    const std::string identity = sha256_hex(parts);
    // Rank 0 hashes its executable; a failure ends every rank.
    std::string binary;
    if (comm_.rank() == 0) {
      try {
        binary = executable_hash().sha256;
        hash_cache_problem_ = executable_hash().cache_problem;
      } catch (const std::exception& e) {
        binary = std::string("!") + e.what();
      }
    }
    binary = comm_.broadcast(binary);
    if (binary.starts_with('!')) throw std::runtime_error(binary.substr(1));
    info_ = Json{
        {"schema", "autan.results/1"},
        {"run_id", run_id},
        {"identity", identity},
        {"status", "running"},
        {"case", {{"name", in.case_name}, {"sha256", in.case_sha256}}},
        {"mesh",
         {{"path", fs::absolute(in.mesh_path).lexically_normal().string()},
          {"sha256", p_.mesh_hash},
          {"groups_sha256", in.mesh_groups_sha256 ? Json(*in.mesh_groups_sha256) : Json(nullptr)}}},
        {"solver",
         {{"version", build.version},
          {"commit", build.commit},
          {"dirty", build.dirty},
          {"backend", backend},
          {"binary_sha256", binary}}},
        {"precision", in.precision},
        {"devices", {{"kind", in.device_kind}, {"count", in.device_count}}},
        {"cell_order", cell_order()},
        {"started", started_at},
        {"finished", nullptr},
        {"iterations", 0},
        {"time", in.time.kind == "unsteady" ? Json(0.0) : Json(nullptr)},
        {"averaging", nullptr},
        {"convergence", nullptr},
        {"checkpoint", nullptr},
        {"start", start},
        {"monitors", nullptr},
        {"timing", nullptr},
        {"diagnostics", Json::array()}};
    // W0712 and W0713 are in the run.json the directory is published with.
    for (const auto& w : p_.memory_warnings) info_["diagnostics"].push_back(Json::parse(w.json()));
    if (continued()) {
      info_["iterations"] = in.start->iterations;
      if (in.time.kind == "unsteady") info_["time"] = *in.start->time;
      info_["averaging"] = p_.start_averaging;
    }
  }

  // Rank 0 creates the directory; the others append to it once it exists.
  void open(const fs::path& runs) {
    std::string path;
    if (comm_.rank() == 0) {
      try {
        // The log is published with the diagnostics run.json names from the start.
        std::string first_lines = log_line(
            "info", "solver", "run " + info_["run_id"].get<std::string>() + " started", "",
            Json{
                {"cells", p_.mesh.n_cells()}, {"faces", p_.mesh.n_faces()}, {"ranks", comm_.size()}}
                .dump());
        for (const auto& w : p_.memory_warnings) {
          first_lines += "\n" + log_line("warning", "solver", w.message, w.code,
                                         Json::parse(w.data_json).dump());
        }
        dir_ = std::make_unique<RunDirectory>(
            runs, info_["run_id"].get<std::string>(), info_.dump(2), p_.input.case_toml,
            p_.mesh_info, first_lines,
            p_.input.verification ? &p_.input.verification->text : nullptr, p_.state_files);
        path = dir_->path().string();
      } catch (...) {
        (void)comm_.broadcast({});
        throw;
      }
    }
    path = comm_.broadcast(path);
    if (path.empty()) throw std::runtime_error("rank 0 could not create the run directory");
    if (comm_.rank() != 0) dir_ = std::make_unique<RunDirectory>(fs::path(path));
    p_.state_files.clear();  // archived in the run directory
    if (ordered_) {
      note("info", "solver", "cells renumbered in reverse Cuthill-McKee order", "",
           Json{{"order", "rcm"}, {"seconds", renumber_seconds_}}.dump());
    } else {
      note("warning", "solver", "the mesh's own cell order is kept (AUTAN_CELL_ORDER=input)", "",
           Json{{"order", "input"}}.dump());
    }
    if (!hash_cache_problem_.empty()) {
      // Rank 0's: the only one that hashes. Every process then hashes its executable whole.
      note("warning", "solver", "the executable's hash cache is not used: " + hash_cache_problem_,
           "", Json{{"cache", std::getenv("AUTAN_BINARY_HASH_CACHE")}}.dump());
    }
    if (comm_.size() > 1) log_partition();
    if (const PeriodicSnap& snap = p_.mesh.periodic_snap(); snap.nodes > 0) {
      // mesh.md, Periodic pairs: the run computes on the snapped coordinates (#426).
      add_diagnostic(
          Diagnostic{.code = "W0150",
                     .message = std::to_string(snap.nodes) +
                                " periodic nodes moved to make their pairs exact translations",
                     .path = "mesh.path",
                     .data_json = Json{{"nodes", snap.nodes},
                                       {"max_distance", snap.max_distance},
                                       {"max_relative", snap.max_relative},
                                       {"max_volume_change", snap.max_volume_change}}
                                      .dump()});
    }
    if (continued() && p_.start_binary != info_["solver"]["binary_sha256"]) {
      // results.md, run.json `solver.binary_sha256`: bit-for-bit continuation needs the binary.
      add_diagnostic(Diagnostic{.code = "W0715",
                                .message = "this run continues a run of another solver binary",
                                .path = "start.run",
                                .data_json = Json{{"expected", p_.start_binary},
                                                  {"found", info_["solver"]["binary_sha256"]}}
                                                 .dump()});
    }
    if (comm_.size() > 1 && p_.input.reproducibility != "ranks") {
      note("info", "solver",
           "results depend on the number of ranks: reproducibility is " + p_.input.reproducibility +
               ", not ranks",
           "", Json{{"mode", p_.input.reproducibility}, {"ranks", comm_.size()}}.dump());
    }
    if (gpu_) {
      note(
          "info", "cuda",
          "rank 0 runs on CUDA device " + std::to_string(gpu_->index()) + " (" + gpu_->name() + ")",
          "",
          Json{{"device", gpu_->index()}, {"name", gpu_->name()}, {"visible", visible_gpus()}}
              .dump());
    }
  }
  [[nodiscard]] const fs::path& path() const { return dir_->path(); }
  void stop_on(fs::path file) { stop_file_ = std::move(file); }

  ExitStatus execute() {
    const StartState* start = local_start_ ? &*local_start_ : p_.start ? &*p_.start : nullptr;
    const std::vector<io::Field>* initial = local_initial_ ? &*local_initial_
                                            : p_.initial   ? &*p_.initial
                                                           : nullptr;
    const PhysicsContext context{.mesh = physics_mesh(),
                                 .geometry = physics_geometry(),
                                 .input = p_.input,
                                 .log = *this,
                                 .start = start,
                                 .ghosts = *ghosts_,
                                 .ranks = comm_,
                                 .boundary_states = local_states_,
                                 .initial = initial,
                                 .device = gpu_ ? gpu_->index() : -1,
                                 .stream = gpu_ ? gpu_->stream() : nullptr,
                                 .face_global_id = face_global_id_,
                                 .face_reversed = face_reversed_,
                                 .cell_colours = colours_,
                                 .n_colours = n_colours_,
                                 .group_offsets = group_offsets_,
                                 .group_cells = group_cells_,
                                 .cell_group_colours = group_colours_,
                                 .n_group_colours = n_group_colours_,
                                 .ghost_stencil = local_ ? std::optional(p_.ghosts) : std::nullopt};
    std::string status = "running";
    std::int64_t done = continued() ? p_.input.start->iterations : 0;
    try {
      physics_ = (*p_.factory)(context);
      // The physics has read the start and the initial state (runtime.md, PhysicsContext).
      local_start_.reset();
      p_.start.reset();
      local_initial_.reset();
      p_.initial.reset();
      host_after_setup_ = host_peak_bytes();
      if (gpu_) device_after_setup_ = gpu_->sample();
      set_floor();
      status = std::string(iterate(done));
    } catch (const std::exception& e) {
      return fail(e.what(), done);
    }
    return finish(status, done);
  }

 private:
  // case-schema.md#time: in fp32+acc64, a steady run also counts an equation converged on a
  // round-off plateau at most round_off_band floors high, floor_k = round_off_floor * Phi_k,
  // Phi_k the physics' constant flux scale; run.json records the floors and, once converged,
  // the equations that converged on a plateau.
  void set_floor() {
    const auto& in = p_.input;
    const auto equations = residual_names(in);
    const std::vector<double> scales = physics_->residual_scales();
    if (scales.empty()) return;
    if (scales.size() != equations.size()) {
      throw std::runtime_error("the physics gave " + std::to_string(scales.size()) +
                               " residual scales for " + std::to_string(equations.size()) +
                               " equations");
    }
    for (const double s : scales) {
      if (!std::isfinite(s) || s < 0.0)
        throw std::runtime_error("the physics gave a residual scale that is not finite and >= 0");
    }
    if (in.time.kind != "steady" || in.precision != "fp32+acc64") return;
    Json floors = Json::object();
    for (std::size_t e = 0; e < equations.size(); ++e) {
      floor_.push_back(round_off_floor * scales[e]);
      floors[equations[e]] = floor_.back();
    }
    info_["convergence"] = Json{{"floor", floors}, {"plateau", Json::array()}};
    if (comm_.rank() == 0) dir_->write_run_json(info_.dump(2));
  }

  // On several ranks, each runs its part of the mesh, as prepare() cut it
  // (p_.parts: METIS, the face-count cost of each cell, the physics' groups
  // kept whole; every rank cuts the same mesh with the same seed, so all
  // agree on the parts, partitioning.md).
  // On one rank, a mesh with periodic pairs runs on its one-part partition,
  // whose image ghosts make each periodic face an interior face; its owned
  // cells are the mesh's cells, in order. Otherwise, the mesh itself.
  std::optional<LocalPartition> make_part() {
    const Mesh& mesh = compute_mesh();
    const std::int32_t n = comm_.size();
    // Reproducibility "ranks" computes on a partition, one part on one rank,
    // so that a physics finds the same global keys on any number of ranks.
    if (n == 1 && mesh.periodic_pairs().empty() && p_.input.reproducibility != "ranks")
      return std::nullopt;
    // Each cell keeps its part through the renumbering.
    std::vector<std::int32_t> parts =
        n > 1 ? std::move(p_.parts) : std::vector<std::int32_t>(at(mesh.n_cells()), 0);
    if (n > 1 && ordered_) {
      std::vector<std::int32_t> ordered(parts.size());
      for (std::size_t c = 0; c < ordered.size(); ++c)
        ordered[c] = parts[at(ordered_->new_to_old[c])];
      parts = std::move(ordered);
    }
    LocalPartition part = extract_partition(mesh, geometry_, parts, n, comm_.rank(), p_.ghosts);
    comm_.fill_ghosts(part);
    return part;
  }

  // This rank's owned cells' share of cell fields of the mesh.
  [[nodiscard]] std::vector<io::Field> owned_fields(const std::vector<io::Field>& fields) const {
    const auto ids = owned_ids();
    std::vector<io::Field> out;
    for (const auto& f : fields) {
      io::Field part{.name = f.name, .values = {}};
      part.values.reserve(ids.size());
      for (const auto id : ids) part.values.push_back(f.values.at(static_cast<std::size_t>(id)));
      out.push_back(std::move(part));
    }
    return out;
  }

  // The start state of this rank's owned cells, from that of the mesh.
  [[nodiscard]] StartState owned_start(const StartState& global) const {
    return StartState{.fields = owned_fields(global.fields),
                      .mean_fields = owned_fields(global.mean_fields),
                      .restart_fields = owned_fields(global.restart_fields),
                      .scalars = global.scalars};
  }

  // The mesh index of each owned cell of the physics' mesh.
  [[nodiscard]] std::vector<std::int64_t> owned_ids() const {
    const Mesh& mesh = physics_mesh();
    const auto global = mesh.cell_global_id();
    return {global.begin(), global.begin() + mesh.n_owned_cells()};
  }

  // By default (AUTAN_CELL_ORDER unset or "rcm"), the mesh's cells in reverse
  // Cuthill-McKee order (mesh.md, Renumbering), for memory locality; the
  // outputs keep the mesh's numbering through cell_global_id and face_source.
  // "input": the mesh's order.
  struct Ordered {
    Mesh mesh;
    std::vector<index_t> new_to_old;  // the mesh cell of each cell of `mesh`
    AssemblyMap map;                  // the mesh face of each face of `mesh`, flips
  };
  std::optional<Ordered> make_ordered() {
    // Rank 0's order decides for every rank: ranks partitioning differently
    // numbered meshes would exchange unrelated cells.
    if (comm_.broadcast(p_.cell_order) == "input") return std::nullopt;
    const auto start = std::chrono::steady_clock::now();
    AssemblyMap map;
    std::vector<index_t> new_to_old = rcm_order(p_.mesh);
    Mesh mesh = renumber_cells(p_.mesh, new_to_old, &map);
    renumber_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return Ordered{
        .mesh = std::move(mesh), .new_to_old = std::move(new_to_old), .map = std::move(map)};
  }
  // The geometry of the compute mesh, computed on the mesh the run read and
  // carried through the renumbering: the same bits in either cell order.
  [[nodiscard]] Geometry compute_physics_geometry() {
    if (!ordered_) return std::move(*p_.geometry);
    Geometry g = renumber_geometry(*p_.geometry, ordered_->new_to_old, ordered_->map);
    p_.geometry.reset();
    return g;
  }
  // The cell order the run computes in, as run.json records it.
  [[nodiscard]] const char* cell_order() const { return ordered_ ? "rcm" : "input"; }
  // The mesh the runtime partitions and the physics computes on: the mesh,
  // or its renumbered copy.
  [[nodiscard]] const Mesh& compute_mesh() const { return ordered_ ? ordered_->mesh : p_.mesh; }
  // Each face of the physics' mesh, keyed in the mesh the run read: its face
  // in the compute mesh (the partition's, or itself), then in the mesh (the
  // renumbering's), reversed if one of the two flipped it.
  void face_keys() {
    const auto n = at(physics_mesh().n_faces());
    face_global_id_.resize(n);
    face_reversed_.resize(n);
    for (std::size_t f = 0; f < n; ++f) {
      const auto g =
          local_ ? static_cast<index_t>(local_->face_global_id[f]) : static_cast<index_t>(f);
      std::uint8_t reversed = local_ ? local_->face_reversed[f] : 0;
      if (ordered_) reversed ^= ordered_->map.face_reversed[at(g)];
      face_global_id_[f] = mesh_face(static_cast<index_t>(f));
      face_reversed_[f] = reversed;
    }
  }
  // The mesh face of face f of the physics' mesh.
  [[nodiscard]] index_t mesh_face(index_t f) const {
    const auto g = local_ ? static_cast<index_t>(local_->face_global_id[at(f)]) : f;
    return ordered_ ? ordered_->map.face_source[at(g)] : g;
  }

  // The oracle-state face states in the order of the physics' owned
  // boundary faces of each zone (PhysicsContext::boundary_states), from
  // those in the mesh's zone order.
  [[nodiscard]] std::map<index_t, std::vector<double>> physics_boundary_states() const {
    std::map<index_t, std::vector<double>> out;
    const Mesh& mesh = physics_mesh();
    const auto offsets = mesh.boundary_zone_offsets();
    const auto mesh_offsets = p_.mesh.boundary_zone_offsets();
    for (const auto& [zone, states] : p_.boundary_states) {
      const index_t first = mesh_offsets[at(zone)];
      const auto total = at(mesh_offsets[at(zone) + 1] - first);
      const auto n = at(offsets[at(zone) + 1] - offsets[at(zone)]);
      std::vector<double> paired(5 * n);
      for (std::size_t j = 0; j < n; ++j) {
        const auto row = at(mesh_face(offsets[at(zone)] + static_cast<index_t>(j)) - first);
        for (std::size_t k = 0; k < 5; ++k) paired[k * n + j] = states[k * total + row];
      }
      out.emplace(zone, std::move(paired));
    }
    return out;
  }

  // Owned and ghost cells of each rank, in log.jsonl.
  void log_partition() {
    const Mesh& mesh = physics_mesh();
    std::vector<ExactSum> counts(2 * at(comm_.size()));
    counts[2 * at(comm_.rank())].add(static_cast<double>(mesh.n_owned_cells()));
    counts[2 * at(comm_.rank()) + 1].add(
        static_cast<double>(mesh.n_cells() - mesh.n_owned_cells()));
    comm_.sum(counts);
    Json owned = Json::array(), ghosts = Json::array();
    for (std::size_t r = 0; r < at(comm_.size()); ++r) {
      owned.push_back(static_cast<std::int64_t>(counts[2 * r].value()));
      ghosts.push_back(static_cast<std::int64_t>(counts[2 * r + 1].value()));
    }
    write("info", "mpi", "mesh partitioned over " + std::to_string(comm_.size()) + " ranks", "",
          Json{{"ranks", comm_.size()}, {"owned_cells", owned}, {"ghost_cells", ghosts}}.dump());
  }
  // The global colouring of the mesh (colour_cells), for each cell of the
  // physics' mesh through its global id: every rank computes the same.
  // With the physics' groups, their colouring and the groups this rank owns
  // (by increasing least global id, each in the physics' order) likewise.
  void colour_cells_of_physics() {
    const Mesh& mesh = p_.mesh;
    const auto z = [](std::int64_t i) { return static_cast<std::size_t>(i); };
    const auto global_ids = mesh.cell_global_id();
    const auto ids = physics_mesh().cell_global_id();
    const auto of_physics = [&](const std::vector<std::int32_t>& colours,
                                std::vector<std::int32_t>& out, std::int32_t& n_out) {
      std::vector<std::int32_t> by_id(colours.size());
      for (std::size_t c = 0; c < colours.size(); ++c) by_id[z(global_ids[c])] = colours[c];
      out.resize(ids.size());
      for (std::size_t c = 0; c < ids.size(); ++c) out[c] = by_id[z(ids[c])];
      for (const std::int32_t k : colours) n_out = std::max(n_out, k + 1);
    };
    of_physics(colour_cells(mesh), colours_, n_colours_);
    if (p_.groups == nullptr || !*p_.groups) return;
    of_physics(p_.group_colours, group_colours_, n_group_colours_);
    p_.group_colours = {};
    std::vector<index_t> owned_at(at(mesh.n_cells()), -1);  // by global id
    for (index_t c = 0; c < physics_mesh().n_owned_cells(); ++c) owned_at[z(ids[at(c)])] = c;
    std::vector<std::pair<std::int64_t, std::size_t>> order;  // (least global id, group)
    for (std::size_t g = 0; g + 1 < groups_.offsets.size(); ++g) {
      std::int64_t least = std::numeric_limits<std::int64_t>::max();
      for (auto k = groups_.offsets[g]; k < groups_.offsets[g + 1]; ++k)
        least = std::min(least, global_ids[at(groups_.cells[z(k)])]);
      order.emplace_back(least, g);
    }
    std::ranges::sort(order);
    group_offsets_.assign(1, 0);
    for (const auto& [least, g] : order) {
      if (owned_at[z(least)] < 0) continue;
      for (auto k = groups_.offsets[g]; k < groups_.offsets[g + 1]; ++k)
        group_cells_.push_back(owned_at[z(global_ids[at(groups_.cells[z(k)])])]);
      group_offsets_.push_back(static_cast<offset_t>(group_cells_.size()));
    }
  }

  [[nodiscard]] const Mesh& physics_mesh() const { return local_ ? local_->mesh : compute_mesh(); }
  [[nodiscard]] const Geometry& physics_geometry() const {
    return local_ ? local_->geometry : geometry_;
  }

  [[nodiscard]] bool continued() const {
    return p_.input.start && p_.input.start->mode == "continue";
  }

  // Thrown by the solution writer when a physics member fails: unlike a
  // write failure, it fails even a diverged run (runtime.md).
  struct PhysicsFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
  };

  void record_failure(const std::string& message) {
    add_diagnostic(Diagnostic{
        .code = "E0704",
        .message = message,
        .path = "",
        .data_json = comm_.size() > 1 ? Json{{"rank", comm_.rank()}}.dump() : std::string("{}")});
  }

  // A run diagnostic goes to run.json and, with the same code and data, to
  // log.jsonl (results.md). Every rank records it; rank 0 logs it.
  void add_diagnostic(const Diagnostic& d) {
    const Json entry = Json::parse(d.json());
    info_["diagnostics"].push_back(entry);
    // The level follows the code's letter (results.md, Log).
    note(d.code.starts_with('W') ? "warning" : "error", "solver", d.message, d.code,
         entry["data"].dump());
  }

  // An event of the run, which every rank knows: rank 0 logs it.
  void note(std::string_view level, std::string_view source, std::string_view message,
            std::string_view code, std::string_view data_json) {
    if (comm_.rank() == 0) dir_->append_log(log_line(level, source, message, code, data_json));
  }

  // On several ranks, a physics failure on one rank leaves the others
  // waiting in a collective call: that rank records E0704 and the failed
  // run.json, and ends every process with exit status 1.
  ExitStatus fail(const std::string& message, std::int64_t done) {
    if (comm_.size() == 1) {
      record_failure(message);
      return finish("failed", done);
    }
    // The abort follows whatever these writes meet.
    try {
      record_failure(message);
      if (comm_.rank() != 0) {
        const Json& entry = info_["diagnostics"].back();
        dir_->append_log(log_line("error", "solver", message, "E0704", entry["data"].dump()));
      }
      info_["iterations"] = done;
      info_["status"] = "failed";
      if (info_["convergence"].is_object()) info_["convergence"]["plateau"] = Json::array();
      info_["finished"] = utc(std::chrono::system_clock::now(), "%Y-%m-%dT%H:%M:%SZ");
      dir_->write_run_json(info_.dump(2));
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s\n",
                   Diagnostic{.code = "E0704",
                              .message = message + "; recording it: " + e.what(),
                              .path = "",
                              .data_json = Json{{"rank", comm_.rank()}}.dump()}
                       .json()
                       .c_str());
    }
    comm_.abort(static_cast<int>(ExitStatus::failed));
    return ExitStatus::failed;
  }

 public:
  // A physics' event: rank 0 logs all, the other ranks their warnings and
  // errors, with their rank in `data`.
  void write(std::string_view level, std::string_view source, std::string_view message,
             std::string_view code, std::string_view data_json) override {
    // Time spent here is the log's, not the solve's (timing.solve_seconds).
    const auto start = std::chrono::steady_clock::now();
    if (comm_.rank() == 0) {
      dir_->append_log(log_line(level, source, message, code, data_json));
    } else if (level != "info") {
      Json data = Json::parse(data_json.empty() ? "{}" : data_json);
      data["rank"] = comm_.rank();
      dir_->append_log(log_line(level, source, message, code, data.dump()));
    }
    log_seconds_ += seconds_since(start);
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
    // Steady convergence is measured against the run's reference (case-schema.md#time): its own
    // iteration 1, or the one `start` carries for a `continue` or a steady `warm` from a steady
    // run; only a `continue` goes on counting and timing.
    const bool inherited = in.start && in.start->residual_reference;
    std::vector<double> first;
    if (inherited) first = *in.start->residual_reference;
    // The equations that left their floor, in the chain so far: a checkpoint carries them.
    std::set<std::string> woken;
    if (inherited && in.start->woken)
      woken.insert(in.start->woken->begin(), in.start->woken->end());
    RoundOffExemption exempt(
        equations, inherited && in.start->woken ? *in.start->woken : std::vector<std::string>{});
    // One round-off plateau per equation, counted from this run's first step.
    std::vector<ResidualPlateau> plateaus(equations.size());
    double time = 0.0, average_start = 0.0;
    std::int64_t from = 1;
    if (continued()) {
      const Start& s = *in.start;
      if (s.time) time = *s.time;
      from = s.iterations + 1;
      if (!info_["averaging"].is_null()) average_start = info_["averaging"]["from_time"];
    }
    last_checkpoint_ = std::chrono::steady_clock::now();
    for (std::int64_t it = from; it <= budget; ++it) {
      // A launcher forwards a signal to every rank, not at the same instant,
      // and may abort the job on one: the stop file asks for the same stop
      // without signals. Rank 0 looks for it; the ranks stop together once
      // any has a reason to.
      std::int64_t stop = stop_requested;
      if (comm_.rank() == 0 && !stop_file_.empty()) {
        std::error_code ec;
        if (fs::exists(stop_file_, ec)) stop = 1;
      }
      if (comm_.max(stop) != 0) {
        stopped_by_signal_ = true;
        return "stopped";
      }
      iteration_ = it;
      // The solve: the physics' own work, without the history and log writes.
      const auto stepping = std::chrono::steady_clock::now();
      const double logged = log_seconds_;
      StepReport report = physics_->step(it);
      solve_seconds_ += seconds_since(stepping) - (log_seconds_ - logged);
      report.diverged_cell = first_diverged(report.diverged_cell);
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
        const auto averaging_start = std::chrono::steady_clock::now();
        const double logged_before = log_seconds_;
        physics_->accumulate_average(report.time_step);
        solve_seconds_ += seconds_since(averaging_start) - (log_seconds_ - logged_before);
        averaging = Json{{"from_step", *in.time.average_from + 1},
                         {"to_step", it},
                         {"from_time", average_start},
                         {"to_time", step_time}};
      }
      Json record{{"iteration", it},
                  {"time", steady ? Json(nullptr) : Json(step_time)},
                  {"residuals", residuals},
                  {"monitors", Json::object()}};
      if (steady) record["provisional"] = report.provisional;
      for (std::size_t m = 0; m < in.monitors.size(); ++m) {
        const auto& monitor = in.monitors[m];
        if (it % monitor.every != 0) continue;
        record["monitors"][monitor.name] = monitor_record(m, "at iteration " + std::to_string(it));
      }
      if (comm_.rank() == 0) dir_->append_history(record.dump());
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
        exempt.reference(first);
        // case-schema.md#time: equations at the round-off floor are exempt while they stay there
        // (runtime/round_off.hpp); one leaving it reports W0710, once.
        for (const std::size_t e : exempt.update(report.residuals)) {
          woken.insert(equations[e]);
          add_diagnostic(
              Diagnostic{.code = "W0710",
                         .message = "equation " + equations[e] + " left its floor at iteration " +
                                    std::to_string(it) + ": it now has to converge like the others",
                         .path = "",
                         .data_json = Json{{"equation", equations[e]},
                                           {"iteration", it},
                                           {"residual", report.residuals[e]},
                                           {"floor", exempt.floor()}}
                                          .dump()});
        }
        const double drop = std::pow(10.0, -in.time.residual_orders);
        bool converged = true;
        Json at_plateau = Json::array();
        for (std::size_t e = 0; e < equations.size(); ++e) {
          const double r = report.residuals[e];
          // A provisional step's residuals start every plateau anew.
          if (report.provisional) plateaus[e].reset();
          const bool flat = !report.provisional && plateaus[e].add(r);
          const bool met = exempt.exempt(e) ||
                           r <= (exempt.woken(e) ? exempt.woken_reference(e) : first[e]) * drop;
          const bool plateau = !floor_.empty() && flat && r <= round_off_band * floor_[e];
          if (!met && plateau) at_plateau.push_back(equations[e]);
          converged = converged && (met || plateau);
        }
        if (converged) {
          plateau_ = std::move(at_plateau);
          return "converged";
        }
      }
      if (it < budget && checkpoint_due(it)) {
        Json state = Json{{"iteration", it}, {"residual_reference", nullptr}, {"woken", nullptr}};
        if (steady) {
          Json reference = Json::object();
          for (std::size_t e = 0; e < equations.size(); ++e) reference[equations[e]] = first[e];
          state["residual_reference"] = std::move(reference);
          state["woken"] = Json(std::vector<std::string>(woken.begin(), woken.end()));
        }
        write_checkpoint(it, state);
      }
    }
    return "completed";
  }

  // Rank 0, after the final run.json: a failure is recorded in run.json again.
  void remove_checkpoint() {
    try {
      dir_->remove_checkpoint();
    } catch (const std::exception& e) {
      try {
        add_diagnostic(
            Diagnostic{.code = "W0714",
                       .message = std::string("checkpoint.cgns was not removed: ") + e.what(),
                       .path = "",
                       .data_json = "{}"});
        dir_->write_run_json(info_.dump(2));
      } catch (const std::exception&) {
        // Nothing more can be recorded; the stale checkpoint stays beside the final solution.
      }
    }
  }

  // Whether a checkpoint follows iteration `it` (case-schema.md#time): every `iterations`, or once
  // `seconds` of wall-clock time have passed since the last one, as rank 0 measures it; every
  // rank decides alike.
  bool checkpoint_due(std::int64_t it) {
    const auto& t = p_.input.time;
    if (t.checkpoint_iterations) return it % *t.checkpoint_iterations == 0;
    const std::int64_t due =
        comm_.rank() == 0 && seconds_since(last_checkpoint_) >= *t.checkpoint_seconds ? 1 : 0;
    return comm_.max(due) != 0;
  }

  // The state after iteration `it` to checkpoint.cgns, with `state` (its iteration and steady
  // convergence state) as RunMetadata `checkpoint`; then run.json `checkpoint` and a log line
  // (results.md#checkpointcgns). Collective. A failure of the physics fails the run; any other
  // failure, of the write or of its records, leaves the previous checkpoint and goes on with
  // W0714, which every rank records alike.
  void write_checkpoint(std::int64_t it, const Json& state) {
    const auto start = std::chrono::steady_clock::now();
    std::string failure;
    try {
      write_solution(false, state);
    } catch (const PhysicsFailure&) {
      throw;
    } catch (const std::exception& e) {
      failure = e.what();
    }
    // Only rank 0 writes: its outcome is every rank's.
    failure = comm_.broadcast(failure);
    const double seconds =
        Json::parse(comm_.broadcast(Json(seconds_since(start)).dump())).get<double>();
    last_checkpoint_ = std::chrono::steady_clock::now();
    if (failure.empty()) {
      info_["checkpoint"] =
          Json{{"iteration", it}, {"time", info_["time"]}, {"write_seconds", seconds}};
      try {
        if (comm_.rank() == 0) {
          std::error_code ec;
          const auto bytes = fs::file_size(dir_->path() / "checkpoint.cgns", ec);
          dir_->write_run_json(info_.dump(2));
          note("info", "writer", "checkpoint", "",
               Json{{"iteration", it},
                    {"write_seconds", seconds},
                    {"bytes", ec ? Json(nullptr) : Json(static_cast<std::int64_t>(bytes))}}
                   .dump());
        }
      } catch (const std::exception& e) {
        failure = std::string("recording it: ") + e.what();
      }
      failure = comm_.broadcast(failure);
    }
    if (failure.empty()) return;
    // One W0714 in run.json counts the failed writes (results.md#checkpointcgns): it does not
    // grow with the run's length. The log has one line per failure, each with the data of
    // that moment.
    ++write_failures_;
    if (write_failures_ == 1) first_failure_ = Json{{"iteration", it}, {"time", info_["time"]}};
    const Diagnostic warning{
        .code = "W0714",
        .message = "checkpoint writes failed: " + std::to_string(write_failures_) +
                   ", the last at iteration " + std::to_string(it) + ": " + failure,
        .path = "",
        .data_json = Json{{"failures", write_failures_},
                          {"first_iteration", first_failure_["iteration"]},
                          {"first_time", first_failure_["time"]},
                          {"last_iteration", it},
                          {"last_time", info_["time"]}}
                         .dump()};
    std::string unlogged;
    try {
      note("warning", "solver", warning.message, warning.code, warning.data_json);
    } catch (const std::exception& e) {
      unlogged = e.what();
    }
    // run.json names only diagnostics the log holds (results.md, Log): a warning the log
    // cannot take leaves run.json's as it was.
    if (!comm_.broadcast(unlogged).empty()) return;
    auto& diagnostics = info_["diagnostics"];
    if (!write_warning_) {
      write_warning_ = diagnostics.size();
      diagnostics.push_back(nullptr);
    }
    diagnostics[*write_warning_] = Json::parse(warning.json());
  }

  // The mesh index of the least diverged cell over all ranks, from the cell
  // of this rank's physics mesh (on a partition, a ghost names its owner).
  std::optional<index_t> first_diverged(std::optional<index_t> local) {
    constexpr std::int64_t none = std::numeric_limits<std::int64_t>::min();
    std::int64_t key = none;
    if (local) key = -1 - physics_mesh().cell_global_id()[at(*local)];
    key = comm_.max(key);
    if (key == none) return std::nullopt;
    return static_cast<index_t>(-1 - key);
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
  // With `checkpoint`, the state after that iteration goes to checkpoint.cgns
  // (results.md#checkpointcgns), which names the iteration.
  void write_solution(bool diverged, const Json& checkpoint = nullptr) {
    io::Solution s;
    // A failure before the gathers end leaves other ranks inside them: it
    // fails the run like a physics failure, which ends every rank.
    try {
      s = gathered_solution(checkpoint);
    } catch (const PhysicsFailure&) {
      throw;
    } catch (const std::exception& e) {
      throw PhysicsFailure(std::string("gathering the solution: ") + e.what());
    }
    if (comm_.rank() != 0) return;
    // A diverged state may hold non-finite values; results.md asks for its
    // solution only if one can be written, and readers require finite fields.
    if (diverged) {
      for (const auto& f : s.fields) {
        if (!std::ranges::all_of(f.values, [](double v) { return std::isfinite(v); })) return;
      }
    }
    if (!checkpoint.is_null()) {
      io::write_solution(dir_->checkpoint_staging(), p_.mesh, s);
      dir_->publish_checkpoint();
      return;
    }
    io::write_solution(dir_->solution_staging(), p_.mesh, s);
    dir_->publish_solution();
  }

  // The current state with `checkpoint` as RunMetadata `checkpoint`, every field gathered on
  // rank 0. Collective.
  io::Solution gathered_solution(const Json& checkpoint) {
    const auto& in = p_.input;
    io::Solution s;
    // run.json without what only the end of the run knows, and without its diagnostics, which
    // are not copied.
    static const std::set<std::string> left_out{"status", "finished",    "iterations", "monitors",
                                                "timing", "diagnostics", "checkpoint"};
    Json metadata = Json::object();
    for (const auto& [key, value] : info_.items()) {
      if (!left_out.contains(key)) metadata[key] = value;
    }
    if (!checkpoint.is_null()) metadata["checkpoint"] = checkpoint;
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
    // The physics names faces in its own mesh. On a partition, this rank's
    // wall faces are its owned boundary faces whose mesh face is a wall
    // face, ascending, and `slots` their positions in s.wall_faces.
    std::vector<index_t> physics_faces;
    std::vector<std::int64_t> slots;
    const Mesh& own = physics_mesh();
    for (index_t f = own.n_interior_faces(); f < own.n_owned_faces(); ++f) {
      const index_t global = mesh_face(f);
      const auto it = std::ranges::lower_bound(s.wall_faces, global);
      if (it == s.wall_faces.end() || *it != global) continue;
      physics_faces.push_back(f);
      slots.push_back(it - s.wall_faces.begin());
    }
    try {
      s.fields = physics_->fields(false);
      if (!info_["averaging"].is_null()) s.mean_fields = physics_->fields(true);
      s.restart_fields = physics_->restart_fields();
      s.solver_state = physics_->restart_scalars();
      if (!s.wall_faces.empty()) s.wall_fields = physics_->wall_fields(physics_faces);
    } catch (const std::exception& e) {
      throw PhysicsFailure(e.what());
    }
    std::vector<std::string> names;
    names.reserve(s.fields.size());
    for (const auto& f : s.fields) names.push_back(f.name);
    if (names != field_names(in))
      throw PhysicsFailure("the physics returned another set of fields");
    // The SolverState a continue checks names and lengths of (E0707): what the entry registers.
    std::vector<std::pair<std::string, std::int64_t>> scalars;
    for (const auto& f : s.solver_state)
      scalars.emplace_back(f.name, static_cast<std::int64_t>(f.values.size()));
    const bool averaged = !s.mean_fields.empty();
    if (scalars != (p_.scalars != nullptr && *p_.scalars
                        ? (*p_.scalars)(in, averaged)
                        : std::vector<std::pair<std::string, std::int64_t>>{})) {
      throw PhysicsFailure("the physics returned other solver state arrays than it registers");
    }
    // Rank 0 gathers every field, cells by mesh index, wall faces by slot.
    const auto ids = owned_ids();
    const auto gather = [&](std::vector<io::Field>& fields, std::span<const std::int64_t> where,
                            std::int64_t n) {
      for (auto& f : fields) {
        if (f.values.size() != where.size()) {
          throw PhysicsFailure("the physics returned " + std::to_string(f.values.size()) +
                               " values of " + f.name + " for " + std::to_string(where.size()));
        }
        f.values = comm_.gather(where, f.values, n);
      }
    };
    gather(s.fields, ids, p_.mesh.n_cells());
    gather(s.mean_fields, ids, p_.mesh.n_cells());
    gather(s.restart_fields, ids, p_.mesh.n_cells());
    gather(s.wall_fields, slots, static_cast<std::int64_t>(s.wall_faces.size()));
    return s;
  }

  // The forces of every monitor on the final state, whether or not its last
  // iteration was a multiple of `every` (results.md, run.json `monitors`).
  // The forces record of monitor m on the current state. A value that is not
  // finite fails the run (E0704) rather than reaching history.jsonl or
  // run.json as null (results.md, History).
  [[nodiscard]] Json monitor_record(std::size_t m, const std::string& when) const {
    const auto& in = p_.input;
    Json record = forces_record(physics_->loads(p_.monitor_zones[m], in.monitors[m].moment_center),
                                in.monitors[m], in.freestream);
    for (const auto& [key, value] : record.items()) {
      if (!std::isfinite(value.get<double>())) {
        throw std::runtime_error("monitor '" + in.monitors[m].name + "': " + key +
                                 " is not finite " + when);
      }
    }
    return record;
  }

  void record_final_monitors(std::int64_t done) {
    const auto& in = p_.input;
    Json monitors = Json::object();
    try {
      for (std::size_t m = 0; m < in.monitors.size(); ++m) {
        Json record = monitor_record(m, "on the final state");
        record["iteration"] = done;
        monitors[in.monitors[m].name] = std::move(record);
      }
    } catch (const std::exception& e) {
      throw PhysicsFailure(e.what());
    }
    info_["monitors"] = std::move(monitors);
  }

  static double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
  }

  // results.md, run.json `timing`: the process's wall-clock seconds and those
  // of the physics' steps and averaging (the slowest rank's), for this run's
  // iterations; null without any. Collective.
  // The memory every rank used (#325): its host high-water mark and the
  // device memory its allocators hold, at their peak and once the physics was
  // built, in bytes. Collective.
  void log_memory() {
    const auto n = at(comm_.size());
    const auto r = at(comm_.rank());
    std::vector<ExactSum> slots(4 * n);
    slots[4 * r].add(static_cast<double>(host_peak_bytes()));
    slots[4 * r + 1].add(static_cast<double>(gpu_ ? gpu_->peak() : 0));
    slots[4 * r + 2].add(static_cast<double>(host_after_setup_));
    slots[4 * r + 3].add(static_cast<double>(device_after_setup_));
    comm_.sum(slots);
    Json host = Json::array(), device = Json::array();
    std::int64_t host_setup = 0, device_setup = 0;
    for (std::size_t k = 0; k < n; ++k) {
      host.push_back(static_cast<std::int64_t>(slots[4 * k].value()));
      device.push_back(static_cast<std::int64_t>(slots[4 * k + 1].value()));
      host_setup = std::max(host_setup, static_cast<std::int64_t>(slots[4 * k + 2].value()));
      device_setup = std::max(device_setup, static_cast<std::int64_t>(slots[4 * k + 3].value()));
    }
    const auto most = [](const Json& values) {
      std::int64_t m = 0;
      for (const auto& v : values) m = std::max(m, v.get<std::int64_t>());
      return m;
    };
    note("info", "solver", "memory", "",
         Json{{"host_peak_bytes", most(host)},
              {"device_peak_bytes", most(device)},
              {"host_peak_bytes_by_rank", host},
              {"device_peak_bytes_by_rank", device},
              {"after_setup", {{"host", host_setup}, {"device", device_setup}}}}
             .dump());
  }

  Json timing(std::int64_t done) {
    const double solve = -comm_.min(-solve_seconds_);
    const std::int64_t iterations = done - (continued() ? p_.input.start->iterations : 0);
    if (iterations <= 0) return nullptr;
    return Json{
        {"wall_seconds", seconds_since(launched_)},
        {"solve_seconds", solve},
        {"iterations", iterations},
        {"ns_per_cell_iteration",
         solve * 1e9 / (static_cast<double>(p_.mesh.n_cells()) * static_cast<double>(iterations))}};
  }

  ExitStatus finish(std::string status, std::int64_t done) {
    info_["iterations"] = done;
    if (physics_ && status != "failed" && status != "diverged") {
      try {
        record_final_monitors(done);
      } catch (const PhysicsFailure& e) {
        // A throw on one rank leaves the others inside the collective sum of
        // loads(): abort them all, as for write_solution below.
        if (comm_.size() > 1) return fail(e.what(), done);
        record_failure(e.what());
        status = "failed";
      }
    }
    // The plateau belongs to a run that ends converged (results.md), in run.json and in the
    // solution's RunMetadata alike: set before the solution is written.
    if (info_["convergence"].is_object())
      info_["convergence"]["plateau"] = status == "converged" ? plateau_ : Json::array();
    // results.md: the solution is required unless the run diverged (written
    // if possible) or failed (optional).
    if (physics_ && status != "failed") {
      try {
        write_solution(status == "diverged");
      } catch (const PhysicsFailure& e) {
        if (comm_.size() > 1) return fail(e.what(), done);
        record_failure(e.what());
        status = "failed";
      } catch (const std::exception& e) {
        if (status != "diverged") {
          record_failure(std::string("writing the solution: ") + e.what());
          status = "failed";
        }
      }
    }
    // Only rank 0 writes the solution: its status is the run's.
    status = comm_.broadcast(status);
    log_memory();
    // A failed run is not timed (results.md).
    info_["timing"] = status == "failed" ? Json(nullptr) : timing(done);
    if (status == "failed") info_["monitors"] = nullptr;
    // A run that fails to write its solution lists no plateau.
    if (info_["convergence"].is_object() && status != "converged")
      info_["convergence"]["plateau"] = Json::array();
    info_["status"] = status;
    info_["finished"] = utc(std::chrono::system_clock::now(), "%Y-%m-%dT%H:%M:%SZ");
    note(status == "failed" ? "error" : "info", "solver", "run finished: " + status, "",
         Json{{"status", status}, {"iterations", done}}.dump());
    if (comm_.rank() == 0) dir_->write_run_json(info_.dump(2));
    // The final solution supersedes the checkpoint once run.json says so; a diverged or failed
    // run keeps it, its last resumable state (results.md#checkpointcgns).
    if (comm_.rank() == 0 && status != "failed" && status != "diverged") remove_checkpoint();
    if (status == "failed") return ExitStatus::failed;
    return stopped_by_signal_ ? ExitStatus::signalled : ExitStatus::finished;
  }

  Prepared& p_;
  Comm& comm_;
  std::chrono::steady_clock::time_point launched_;  // run_case's entry
  // The end of the last checkpoint, or the start of the iterations.
  std::chrono::steady_clock::time_point last_checkpoint_ = std::chrono::steady_clock::now();
  // The failed checkpoint writes: their count, the first one's iteration and time, and their
  // W0714's index in run.json `diagnostics` once logged.
  std::int64_t write_failures_ = 0;
  Json first_failure_ = nullptr;
  std::optional<std::size_t> write_warning_;
  double log_seconds_ = 0.0;    // spent in write()
  fs::path stop_file_;          // --stop-file; empty: none
  double solve_seconds_ = 0.0;  // this rank's step() and accumulate_average()
  // Declared before the physics, so destroyed after it: its stream outlives the physics' use.
  std::optional<Gpu> gpu_;
  double renumber_seconds_ = 0.0;
  std::string hash_cache_problem_;  // rank 0: why the executable hash cache was not used
  std::optional<Ordered> ordered_;
  CellGroups groups_;  // the physics' groups, over p_.mesh's cells
  Geometry geometry_;
  std::optional<LocalPartition> local_;
  std::map<index_t, std::vector<double>> local_states_;  // boundary_states, physics order
  // PhysicsContext's global keys of the physics' faces: through the partition
  // and the renumbering, in the mesh the run read.
  std::vector<std::int64_t> face_global_id_;
  std::vector<std::uint8_t> face_reversed_;
  std::vector<std::int32_t> colours_;  // global colouring, the physics mesh's cells
  std::int32_t n_colours_ = 0;
  // The groups this rank owns, over the physics mesh's owned cells, and their colouring.
  std::vector<offset_t> group_offsets_;
  std::vector<index_t> group_cells_;
  std::vector<std::int32_t> group_colours_;
  std::int32_t n_group_colours_ = 0;
  std::unique_ptr<GhostExchange> ghosts_;
  std::optional<StartState> local_start_;                // the owned cells' share of p_.start
  std::optional<std::vector<io::Field>> local_initial_;  // and of p_.initial
  Json info_;
  std::unique_ptr<RunDirectory> dir_;
  std::unique_ptr<Physics> physics_;
  std::int64_t host_after_setup_ = 0;    // host_peak_bytes() once the physics exists
  std::int64_t device_after_setup_ = 0;  // the device's sample() then
  bool stopped_by_signal_ = false;
  std::int64_t iteration_ = 0;
  Json plateau_ = Json::array();  // the equations the run converged on a plateau
  std::vector<double> floor_;     // fp32+acc64 convergence floor per equation; empty: none  // the
                                  // iteration being computed, 0 before the first
};

}  // namespace

void PhysicsRegistry::add(PhysicsKey key, PhysicsFactory factory, PhysicsScope scope,
                          bool several_ranks, PartitionGroups groups, RestartNames restart,
                          GhostStencils ghosts, RestartScalars scalars) {
  entries_.emplace_back(
      std::move(key), Entry{std::move(factory), std::move(scope), several_ranks, std::move(groups),
                            std::move(restart), std::move(ghosts), std::move(scalars)});
}

const PhysicsRegistry::Entry* PhysicsRegistry::find(const SolverInput& in) const {
  const auto matches = [](const std::string& pattern, const std::string& value) {
    return pattern == "*" || pattern == value;
  };
  for (const auto& [key, entry] : entries_) {
    if (matches(key.kind, in.physics_kind) && matches(key.turbulence, in.turbulence.model) &&
        matches(key.precision, in.precision) && matches(key.device, in.device_kind)) {
      return &entry;
    }
  }
  return nullptr;
}

ExitStatus run_case(const fs::path& input_file, const fs::path& runs,
                    const PhysicsRegistry& registry, const SolverBuild& build, std::FILE* out,
                    std::FILE* err, const fs::path& stop_file, const MemoryPolicy& memory) {
  const auto launched = std::chrono::steady_clock::now();
  // Every rank reads and checks the same input, so all reach the same
  // verdict; rank 0 prints it.
  std::unique_ptr<Comm> comm;
  try {
    comm = make_comm();
  } catch (const Error& e) {
    std::fprintf(
        err, "%s\n",
        Diagnostic{
            .code = error_code_string(e.code()), .message = e.what(), .path = "", .data_json = "{}"}
            .json()
            .c_str());
    return ExitStatus::invalid_input;
  }
  // The ranks agree on whether the input can run before any collective
  // call; the first rank that cannot prints why: a refused input (exit 2)
  // or any other failure (E0704, exit 1). They agree twice: on the memory,
  // measured on every rank before any reads the mesh, then on the input.
  const auto refused = [&](const std::function<void()>& check) -> std::optional<ExitStatus> {
    std::optional<Diagnostic> refusal;
    bool failed = false;  // a failure (exit 1) rather than a refused input (exit 2)
    try {
      check();
    } catch (const InputError& e) {
      refusal = e.diagnostic();
    } catch (const Error& e) {
      // A coded error that is not the input's (invalid groups of the physics: E0140).
      failed = true;
      refusal = Diagnostic{.code = error_code_string(e.code()),
                           .message = e.what(),
                           .path = "",
                           .data_json = Json{{"rank", comm->rank()}}.dump()};
    } catch (const std::exception& e) {
      failed = true;
      refusal = Diagnostic{.code = "E0704", .message = e.what(), .path = "", .data_json = "{}"};
    }
    // The key of the first refusing rank: -(2 rank + 1 if it failed).
    constexpr std::int64_t none = std::numeric_limits<std::int64_t>::min();
    const std::int64_t key = refusal ? -(2 * std::int64_t{comm->rank()} + (failed ? 1 : 0)) : none;
    const std::int64_t first = comm->max(key);
    if (first == none) return std::nullopt;
    if (first == key) {
      const std::string line = refusal->json() + "\n";
      std::fwrite(line.data(), 1, line.size(), err);
      std::fflush(err);
    }
    return -first % 2 == 1 ? ExitStatus::failed : ExitStatus::invalid_input;
  };
  MemoryCheck checked;
  if (const auto status = refused([&] { checked = check_memory(input_file, *comm, memory); }))
    return *status;
  // The memory check's warnings on every rank, written by rank 0 before any rank reads the mesh,
  // which can be what exhausts the host: W0712 when any rank starts beyond its memory (the
  // largest need and the least memory available of those ranks), W0713 when any rank could not
  // read a cgroup limit above it (rank 0's unseen cgroups).
  std::vector<Diagnostic> warnings;
  {
    const auto& warning = checked.beyond;
    const Json data = warning ? Json::parse(warning->data_json) : Json::object();
    const std::int64_t required =
        comm->max(warning ? data["required_bytes"].get<std::int64_t>() : std::int64_t{-1});
    const std::int64_t least = comm->max(warning ? -data["available_bytes"].get<std::int64_t>()
                                                 : std::numeric_limits<std::int64_t>::min());
    const bool unverifiable = comm->max(std::int64_t{checked.unseen_cgroups.empty() ? 0 : 1}) != 0;
    const std::string scope =
        comm->max(std::int64_t{checked.cgroup_scope == "namespace_root" ? 1 : 0}) != 0
            ? "namespace_root"
            : "host";
    if (required >= 0) {
      warnings.push_back(memory_diagnostic("W0712", parse_solver_input(read_text(input_file)),
                                           required, -least, memory.margin_bytes,
                                           checked.unseen_cgroups, unverifiable, scope));
    }
    if (unverifiable) warnings.push_back(unverifiable_memory(checked.unseen_cgroups, scope));
    if (comm->rank() == 0) {
      for (const auto& w : warnings) {
        const std::string line = w.json() + "\n";
        std::fwrite(line.data(), 1, line.size(), err);
      }
      std::fflush(err);
    }
    // No rank reads the mesh before rank 0 has written them: a collective it joins once flushed.
    if (!warnings.empty()) (void)comm->max(std::int64_t{0});
  }
  std::optional<Prepared> prepared;
  if (const auto status =
          refused([&] { prepared.emplace(prepare(input_file, registry, comm->size())); }))
    return *status;
  prepared->memory_warnings = std::move(warnings);
  stop_requested = 0;
  const SignalGuard signals;
  try {
    Run run(*prepared, build, *comm, launched);
    run.stop_on(stop_file);
    run.open(runs);
    if (comm->rank() == 0) {
      std::fprintf(out, "%s\n", run.path().string().c_str());
      std::fflush(out);
    }
    return run.execute();
  } catch (const std::exception& e) {
    // The run could not be set up after the ranks agreed it could (a CUDA
    // device, the ghost exchange), its directory created or its run.json
    // written: nothing records the failure but this diagnostic, flushed
    // before the others, which may wait in a collective call, are ended.
    // Every rank may fail here at once: each line goes out in one write, so
    // that the launcher's merged stderr keeps lines whole.
    const auto* coded = dynamic_cast<const Error*>(&e);
    const std::string line =
        Diagnostic{.code = coded != nullptr ? error_code_string(coded->code()) : "E0704",
                   .message = e.what(),
                   .path = "",
                   .data_json = Json{{"rank", comm->rank()}}.dump()}
            .json() +
        "\n";
    std::fwrite(line.data(), 1, line.size(), err);
    std::fflush(err);
    if (comm->size() > 1) comm->abort(static_cast<int>(ExitStatus::failed));
    return ExitStatus::failed;
  }
}

}  // namespace autan::runtime
