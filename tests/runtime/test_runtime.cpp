#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cgnslib.h>
#include <fcntl.h>
#include <hdf5.h>
#include <nlohmann/json.hpp>
#include <picosha2.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__linux__)
#include <malloc.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include "../io/outside_data.hpp"
#include "autan/core/geometry.hpp"
#include "autan/io/cgns.hpp"
#include "autan/io/mesh_counts.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/partition/partition.hpp"
#include "autan/physics/euler_physics.hpp"
#include "autan/runtime/input.hpp"
#include "autan/runtime/round_off.hpp"
#include "autan/runtime/run.hpp"
#include "planned.hpp"
#include "relaxation.hpp"

extern char** environ;

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;
using autan::runtime::ExitStatus;
using Catch::Matchers::WithinRel;

const fs::path data_dir{AUTAN_TEST_DATA_DIR};
const fs::path mesh_path = data_dir / "mesh" / "hybrid" / "polyMesh";
// The Euler manufactured solution autan-vv exported for the SDK's tests.
const fs::path mms_path = data_dir / ".." / ".." / "py" / "tests" / "fixtures" / "euler.mms.json";
const autan::runtime::SolverBuild build{
    .version = "0.0.0", .commit = std::string(40, 'a'), .dirty = true};

std::string read(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

fs::path fresh_dir() {
  std::random_device random;
  const fs::path dir = fs::temp_directory_path() / ("autan_runtime-" + std::to_string(random()));
  fs::create_directories(dir);
  return dir;
}

// The solver input `autan prepare` wrote for tests/data/mesh/hybrid (steady
// SA, wall ymin, symmetry zmax, farfield elsewhere, forces on ymin every 5),
// pointed at this checkout's mesh and edited by `edit`.
Json input(const std::function<void(Json&)>& edit = {},
           const char* file = "hybrid.solver-input.json") {
  Json doc = Json::parse(read(data_dir / "runtime" / file));
  doc["mesh"]["path"] = fs::absolute(mesh_path).string();
  if (doc["verification"].is_object())
    doc["verification"]["path"] = fs::absolute(mms_path).string();
  if (edit) edit(doc);
  return autan_test::planned(doc, mesh_path);
}

struct Outcome {
  ExitStatus status;
  fs::path runs;
  fs::path run;  // empty when no directory was created
  std::string err;
};

Outcome run_in(const fs::path& dir, const Json& doc,
               const autan::runtime::PhysicsRegistry& registry, const fs::path& stop_file = {},
               const autan::runtime::MemoryPolicy& memory = {}) {
  const fs::path input_file = dir / "solver-input.json";
  std::ofstream(input_file) << doc.dump();
  std::FILE* out = std::tmpfile();
  std::FILE* err = std::tmpfile();
  const ExitStatus status = autan::runtime::run_case(input_file, dir / "runs", registry, build, out,
                                                     err, stop_file, memory);
  auto text = [](std::FILE* f) {
    std::rewind(f);
    std::string s;
    for (int c = std::fgetc(f); c != EOF; c = std::fgetc(f)) s.push_back(static_cast<char>(c));
    std::fclose(f);
    return s;
  };
  std::string printed = text(out);
  if (!printed.empty() && printed.back() == '\n') printed.pop_back();
  return Outcome{.status = status, .runs = dir / "runs", .run = printed, .err = text(err)};
}

Outcome run(const Json& doc, const autan::runtime::PhysicsRegistry& registry,
            const fs::path& stop_file = {}, const autan::runtime::MemoryPolicy& memory = {}) {
  return run_in(fresh_dir(), doc, registry, stop_file, memory);
}

Outcome run(const Json& doc, autan_test::RelaxationOptions options = {},
            const fs::path& stop_file = {}) {
  return run(doc, autan_test::relaxation_registry(options), stop_file);
}

// The hybrid input on periodic-box (Gmsh): xmin -> xmax by (tx, 0, 0), ymin
// -> ymax by (0, 1, 0), zmin a wall, zmax a symmetry plane.
Json periodic_input(double tx, const std::function<void(Json&)>& edit = {}) {
  const fs::path box = data_dir / "mesh" / "periodic-box" / "periodic-box.ascii.msh";
  return input([&](Json& d) {
    d["mesh"]["path"] = fs::absolute(box).string();
    d["mesh"]["sha256"] = autan::io::mesh_hash(box);
    d["periodic"] =
        Json::array({Json{{"zones", {"xmin", "xmax"}},
                          {"transform", {{"kind", "translation"}, {"vector", {tx, 0.0, 0.0}}}}},
                     Json{{"zones", {"ymin", "ymax"}},
                          {"transform", {{"kind", "translation"}, {"vector", {0.0, 1.0, 0.0}}}}}});
    // zmin takes the wall condition of ymin (now periodic), zmax keeps its own.
    Json kept = Json::array();
    for (const auto& b : d["boundaries"]) {
      if (b["zone"] == "ymin") {
        Json wall = b;
        wall["zone"] = "zmin";
        kept.push_back(wall);
      }
      if (b["zone"] == "zmax") kept.push_back(b);
    }
    d["boundaries"] = kept;
    Json zone = d["cell_zones"][0];
    zone["zone"] = "fluid";
    d["cell_zones"] = Json::array({zone});
    d["monitors"][0]["zones"] = Json::array({"zmin"});
    if (edit) edit(d);
  });
}

// The data of the run's `memory` event (#325); null without one.
Json memory_event(const fs::path& run) {
  std::ifstream in(run / "log.jsonl");
  Json found = nullptr;
  for (std::string line; std::getline(in, line);) {
    const Json event = Json::parse(line);
    if (event["message"] == "memory") found = event["data"];
  }
  return found;
}

std::vector<Json> history(const fs::path& run) {
  std::vector<Json> lines;
  std::ifstream in(run / "history.jsonl");
  for (std::string line; std::getline(in, line);) lines.push_back(Json::parse(line));
  return lines;
}

std::vector<double> solution_field(const fs::path& file, const char* solution, const char* name) {
  int fn = 0;
  REQUIRE(cg_open(file.string().c_str(), CG_MODE_READ, &fn) == CG_OK);
  int count = 0;
  REQUIRE(cg_nsols(fn, 1, 1, &count) == CG_OK);
  std::vector<double> values;
  for (int s = 1; s <= count; ++s) {
    char sol_name[33];
    CGNS_ENUMT(GridLocation_t) location{};
    REQUIRE(cg_sol_info(fn, 1, 1, s, sol_name, &location) == CG_OK);
    if (std::string(sol_name) != solution) continue;
    char zone_name[33];
    cgsize_t size[9];
    REQUIRE(cg_zone_read(fn, 1, 1, zone_name, size) == CG_OK);
    values.resize(static_cast<std::size_t>(size[1]));
    const cgsize_t first = 1;
    REQUIRE(cg_field_read(fn, 1, 1, s, name, CGNS_ENUMV(RealDouble), &first, &size[1],
                          values.data()) == CG_OK);
  }
  cg_close(fn);
  return values;
}

// The node coordinates `axis` (CoordinateX, Y or Z) of a solution.cgns.
std::vector<double> coordinates(const fs::path& file, const char* axis) {
  int fn = 0;
  REQUIRE(cg_open(file.string().c_str(), CG_MODE_READ, &fn) == CG_OK);
  char zone_name[33];
  cgsize_t size[9];
  REQUIRE(cg_zone_read(fn, 1, 1, zone_name, size) == CG_OK);
  std::vector<double> values(static_cast<std::size_t>(size[0]));
  const cgsize_t first = 1;
  REQUIRE(cg_coord_read(fn, 1, 1, axis, CGNS_ENUMV(RealDouble), &first, &size[0], values.data()) ==
          CG_OK);
  cg_close(fn);
  return values;
}

std::vector<std::string> field_list(const fs::path& file) {
  int fn = 0;
  REQUIRE(cg_open(file.string().c_str(), CG_MODE_READ, &fn) == CG_OK);
  int count = 0;
  REQUIRE(cg_nfields(fn, 1, 1, 1, &count) == CG_OK);
  std::vector<std::string> names;
  for (int f = 1; f <= count; ++f) {
    char name[33];
    CGNS_ENUMT(DataType_t) type{};
    REQUIRE(cg_field_info(fn, 1, 1, 1, f, &type, name) == CG_OK);
    names.emplace_back(name);
  }
  cg_close(fn);
  return names;
}

// The PointList of solution.cgns's WallSolution subregion; empty without one.
std::vector<cgsize_t> wall_point_list(const fs::path& file) {
  int fn = 0;
  REQUIRE(cg_open(file.string().c_str(), CG_MODE_READ, &fn) == CG_OK);
  struct Closer {
    explicit Closer(int f) : fn(f) {}
    Closer(const Closer&) = delete;
    Closer& operator=(const Closer&) = delete;
    ~Closer() { cg_close(fn); }
    int fn;
  };
  const Closer closer(fn);
  int regions = 0;
  REQUIRE(cg_nsubregs(fn, 1, 1, &regions) == CG_OK);
  std::vector<cgsize_t> points;
  for (int r = 1; r <= regions; ++r) {
    char name[33];
    int dim = 0, bc_len = 0, gc_len = 0;
    CGNS_ENUMT(GridLocation_t) location{};
    CGNS_ENUMT(PointSetType_t) ptset{};
    cgsize_t npnts = 0;
    REQUIRE(cg_subreg_info(fn, 1, 1, r, name, &dim, &location, &ptset, &npnts, &bc_len, &gc_len) ==
            CG_OK);
    if (std::string(name) != "WallSolution") continue;
    points.resize(static_cast<std::size_t>(npnts));
    REQUIRE(cg_subreg_ptset_read(fn, 1, 1, r, points.data()) == CG_OK);
  }
  return points;
}

void require_error(const Outcome& o, const std::string& code) {
  CHECK(o.status == ExitStatus::invalid_input);
  CHECK(o.run.empty());
  CHECK((!fs::exists(o.runs) || fs::is_empty(o.runs)));
  REQUIRE(!o.err.empty());
  CHECK(Json::parse(o.err)["code"] == code);
}

std::string file_sha256(const fs::path& file) {
  const std::string bytes = read(file);
  std::string hash;
  picosha2::hash256_hex_string(bytes, hash);
  return hash;
}

// An autan.initial-state/1 document for the fluid cells of `mesh_file`
// (every cell of the hybrid box), rows in reverse cell order: density
// 1 + 1e-3 c, velocity (c, -c, 0.5), pressure 1e5 + c for cell c.
Json initial_state_document(const fs::path& mesh_file) {
  const autan::Mesh mesh = autan::io::read_mesh(mesh_file);
  const autan::Geometry g = autan::compute_geometry(mesh);
  Json centroids = Json::array(), values = Json::array();
  for (autan::index_t c = mesh.n_cells() - 1; c >= 0; --c) {
    const auto i = static_cast<std::size_t>(c);
    const auto x = static_cast<double>(c);
    centroids.push_back({g.cell_centroid_x[i], g.cell_centroid_y[i], g.cell_centroid_z[i]});
    values.push_back({1.0 + 1e-3 * x, x, -x, 0.5, 1e5 + x});
  }
  return Json{{"schema", "autan.initial-state/1"},
              {"oracle", "test"},
              {"mesh_sha256", autan::io::mesh_hash(mesh_file)},
              {"variables", {"density", "velocity-x", "velocity-y", "velocity-z", "pressure"}},
              {"cells", mesh.n_cells()},
              {"centroids", centroids},
              {"values", values}};
}

// The hybrid input made inviscid: an initial-state file has no turbulence
// variables (initial-state.md).
void inviscid(Json& d) {
  d["physics"]["turbulence"]["model"] = "inviscid";
  d["physics"]["turbulence"]["nu_tilde_ratio"] = nullptr;
  d["freestream"]["nu_tilde"] = nullptr;
  for (auto& zone : d["cell_zones"]) zone["initial"]["nu_tilde"] = nullptr;
}

// `doc` written to a fresh file; the solver input's initial_state names it.
Json with_initial_state(Json input_doc, const std::string& text) {
  const fs::path file = fresh_dir() / "initial.json";
  std::ofstream(file) << text;
  input_doc["initial_state"] = {{"path", file.string()}, {"sha256", file_sha256(file)}};
  return input_doc;
}

Json with_initial_state(Json input_doc, const Json& doc) {
  return with_initial_state(std::move(input_doc), doc.dump());
}

// `text` (compact JSON) with an empty `member` array added after the first full one found from
// `from` on: a member twice, the last empty, which a reader keeping the last would take.
std::string emptied_twice(std::string text, const std::string& member, std::size_t from = 0) {
  const std::string key = "\"" + member + "\":[";
  std::size_t i = text.find(key, from);
  REQUIRE(i != std::string::npos);
  int depth = 0;
  for (i += key.size() - 1; i < text.size(); ++i) {
    depth += text[i] == '[' ? 1 : (text[i] == ']' ? -1 : 0);
    if (depth == 0) break;
  }
  text.insert(i + 1, "," + key + "]");
  return text;
}

// `text` (compact JSON) with the first number of its first `member` row written in `length`
// characters, zeros added to its fraction: the same value, a longer token.
std::string padded_number(std::string text, const std::string& member, std::size_t length) {
  const std::size_t key = text.find("\"" + member + "\":[[");
  REQUIRE(key != std::string::npos);
  const std::size_t first = key + member.size() + 5;
  const std::size_t last = text.find_first_not_of("0123456789+-.eE", first);
  std::string number = text.substr(first, last - first);
  const std::size_t e = number.find_first_of("eE");
  std::string mantissa = number.substr(0, e),
              exponent = e == std::string::npos ? "" : number.substr(e);
  if (mantissa.find('.') == std::string::npos) mantissa += '.';
  REQUIRE(mantissa.size() + exponent.size() <= length);
  mantissa.append(length - mantissa.size() - exponent.size(), '0');
  return text.replace(first, last - first, mantissa + exponent);
}

#if defined(__linux__)
// The peak resident memory `run` adds while it runs (the runtime runs in this process).
template <class Run>
std::int64_t peak_of(const Run& run) {
  const auto status = [](const std::string& key) -> std::int64_t {
    std::ifstream in("/proc/self/status");
    for (std::string line; std::getline(in, line);) {
      if (line.rfind(key + ":", 0) == 0) return std::stoll(line.substr(key.size() + 1)) * 1024;
    }
    return -1;
  };
  malloc_trim(0);
  const std::int64_t before = status("VmRSS");
  std::ofstream("/proc/self/clear_refs") << "5";  // VmHWM := VmRSS
  run();
  return status("VmHWM") - before;
}
#endif

// The `start` the SDK writes for a run that starts from `previous`
// (solver-input.md): its run.json, the hash of its solution and, for a
// steady `continue` or a `warm` start from a steady run, the convergence
// reference of the chain and the equations exempt from it that went above
// their floor in the chain, none for a warm start (case-schema.md#time).
Json start_from(const fs::path& previous, const std::string& mode) {
  const Json info = Json::parse(read(previous / "run.json"));
  Json reference = nullptr;
  Json woken = nullptr;
  if (info["time"].is_null()) {
    const bool chained = !info["start"].is_null() && !info["start"]["residual_reference"].is_null();
    reference = chained ? info["start"]["residual_reference"] : history(previous)[0]["residuals"];
    double largest = 0.0;
    for (const auto& [name, f] : reference.items()) largest = std::max(largest, f.get<double>());
    const double floor = 1e-12 * largest;
    std::set<std::string> names;
    if (chained) {
      for (const auto& n : info["start"]["woken"]) names.insert(n.get<std::string>());
    }
    for (const auto& line : history(previous)) {
      for (const auto& [name, f] : reference.items()) {
        if (f.get<double>() < floor && line["residuals"][name].get<double>() > floor)
          names.insert(name);
      }
    }
    woken = Json(std::vector<std::string>(names.begin(), names.end()));
    if (mode == "warm") woken = Json::array();
  }
  return Json{{"run", previous.string()},
              {"run_id", info["run_id"]},
              {"identity", info["identity"]},
              {"mode", mode},
              {"iterations", info["iterations"]},
              {"time", info["time"]},
              {"solution_sha256", file_sha256(previous / "solution.cgns")},
              {"file", "solution"},
              {"residual_reference", reference},
              {"woken", woken}};
}

std::vector<std::string> lines_of(const fs::path& file) {
  std::vector<std::string> lines;
  std::ifstream in(file);
  for (std::string line; std::getline(in, line);) lines.push_back(line);
  return lines;
}

// autan-run (the relaxation build) on `doc` as a process, with `env` added to its environment:
// its pid and, once `ready(run directory)` holds, its run directory. The caller reaps it.
struct Spawned {
  pid_t pid = 0;
  fs::path run;
};

Spawned spawn_until(const Json& doc, const std::vector<std::string>& env,
                    const std::function<bool(const fs::path&)>& ready, int poll_us = 2000,
                    const std::string& executable = AUTAN_RUN_RELAXATION) {
  const fs::path dir = fresh_dir();
  const fs::path input_file = dir / "solver-input.json";
  std::ofstream(input_file) << doc.dump();
  const std::string runs = (dir / "runs").string();
  std::vector<std::string> args{executable, "--input", input_file.string(), "--runs", runs};
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  std::vector<std::string> all = env;
  for (char** e = environ; *e != nullptr; ++e) all.emplace_back(*e);
  std::vector<char*> envp;
  for (auto& e : all) envp.push_back(e.data());
  envp.push_back(nullptr);
  Spawned s;
  REQUIRE(posix_spawn(&s.pid, executable.c_str(), nullptr, nullptr, argv.data(), envp.data()) == 0);
  for (int attempt = 0; attempt < 10'000'000 / poll_us; ++attempt) {
    if (fs::exists(runs)) {
      for (const auto& entry : fs::directory_iterator(runs)) {
        if (entry.path().filename().string()[0] != '.' && ready(entry.path())) {
          s.run = entry.path();
          return s;
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(poll_us));
  }
  kill(s.pid, SIGKILL);
  waitpid(s.pid, nullptr, 0);
  FAIL("the run never got ready");
  return s;
}

void kill_and_reap(pid_t pid) {
  REQUIRE(kill(pid, SIGKILL) == 0);
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFSIGNALED(status));
}

// The iteration a checkpoint holds (results.md#checkpointcgns).
std::int64_t checkpoint_iteration(const fs::path& file) {
  return Json::parse(autan::io::read_solution(file).run_metadata)["checkpoint"]["iteration"]
      .get<std::int64_t>();
}

// The `start` the SDK writes to continue a run that did not end from its checkpoint
// (solver-input.md): the checkpoint's iteration, time and steady convergence state, and its
// hash.
Json start_from_checkpoint(const fs::path& previous) {
  const fs::path file = previous / "checkpoint.cgns";
  const Json metadata = Json::parse(autan::io::read_solution(file).run_metadata);
  const Json info = Json::parse(read(previous / "run.json"));
  return Json{{"run", previous.string()},
              {"run_id", info["run_id"]},
              {"identity", info["identity"]},
              {"mode", "continue"},
              {"iterations", metadata["checkpoint"]["iteration"]},
              {"time", metadata["time"]},
              {"solution_sha256", file_sha256(file)},
              {"file", "checkpoint"},
              {"residual_reference", metadata["checkpoint"]["residual_reference"]},
              {"woken", metadata["checkpoint"]["woken"]}};
}

}  // namespace

TEST_CASE("A run killed between two checkpoints continues from the last one bit for bit",
          "[runtime]") {
  // results.md#checkpointcgns: SIGKILL loses at most the iterations since the last checkpoint;
  // the continuation from it equals the uninterrupted run, history line by line and solution
  // bit for bit. The uninterrupted run has the default cadence: checkpoints change nothing.
  const bool steady = GENERATE(true, false);
  CAPTURE(steady);
  const std::int64_t total = 40;
  const auto doc = [&](const Json& checkpoint) {
    const auto edit = [&](Json& d) {
      d["reproducibility"] = "run";
      d["monitors"][0]["every"] = 1;
      d["time"]["checkpoint"] = checkpoint;
      d["time"][steady ? "max_iterations" : "steps"] = total;
    };
    return steady ? input(edit) : input(edit, "hybrid-unsteady.solver-input.json");
  };
  // Residuals decay slowly enough that the steady run does not converge within the budget.
  const autan_test::RelaxationOptions options{.factor = 0.95, .growth = 1e-3};
  const Outcome whole = run(doc(Json{{"seconds", 1800.0}}), options);
  REQUIRE(whole.status == ExitStatus::finished);
  CHECK(!fs::exists(whole.run / "checkpoint.cgns"));
  CHECK(Json::parse(read(whole.run / "run.json"))["checkpoint"].is_null());

  const Json every4 = doc(Json{{"iterations", 4}});
  const auto killed = spawn_until(
      every4, {"AUTAN_TEST_FACTOR=0.95", "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_PAUSE_MS=40"},
      [](const fs::path& r) {
        const Json info = Json::parse(read(r / "run.json"), nullptr, false);
        return !info.is_discarded() && info["checkpoint"].is_object() &&
               info["checkpoint"]["iteration"].get<std::int64_t>() >= 8;
      });
  kill_and_reap(killed.pid);
  const Json dead = Json::parse(read(killed.run / "run.json"));
  CHECK(dead["status"] == "running");
  REQUIRE(fs::exists(killed.run / "checkpoint.cgns"));
  const std::int64_t k = checkpoint_iteration(killed.run / "checkpoint.cgns");
  CHECK(k % 4 == 0);
  CHECK(k >= 8);
  CHECK(dead["checkpoint"]["iteration"] == k);
  CHECK(dead["checkpoint"]["write_seconds"].get<double>() >= 0.0);

  Json next = doc(Json{{"seconds", 1800.0}});
  next["start"] = start_from_checkpoint(killed.run);
  const Outcome cont = run(next, options);
  REQUIRE(cont.status == ExitStatus::finished);
  const Json info = Json::parse(read(cont.run / "run.json"));
  CHECK(info["start"]["file"] == "checkpoint");
  CHECK(info["start"]["iterations"] == k);

  const auto all = lines_of(whole.run / "history.jsonl");
  auto chain = lines_of(killed.run / "history.jsonl");
  REQUIRE(chain.size() >= static_cast<std::size_t>(k));
  chain.resize(static_cast<std::size_t>(k));
  const auto more = lines_of(cont.run / "history.jsonl");
  chain.insert(chain.end(), more.begin(), more.end());
  CHECK(chain == all);
  for (const char* field : {"Density", "Pressure", "Temperature"}) {
    CAPTURE(field);
    CHECK(solution_field(cont.run / "solution.cgns", "FlowSolution", field) ==
          solution_field(whole.run / "solution.cgns", "FlowSolution", field));
    if (!steady) {
      CHECK(solution_field(cont.run / "solution.cgns", "FlowSolutionMean", field) ==
            solution_field(whole.run / "solution.cgns", "FlowSolutionMean", field));
    }
  }
}

TEST_CASE("A run killed while it writes a checkpoint keeps the previous one whole", "[runtime]") {
  // The checkpoint is written under .checkpoint.cgns.tmp and renamed over checkpoint.cgns: a
  // kill while the temporary file exists leaves the previous checkpoint, readable and resumable.
  const Json doc = input([](Json& d) {
    d["time"]["max_iterations"] = 100000;
    d["time"]["checkpoint"] = Json{{"iterations", 1}};
  });
  // Retried until the kill lands while the temporary file still exists, after a first
  // checkpoint: a kill after the rename proves nothing.
  Spawned killed;
  bool during_write = false;
  for (int attempt = 0; attempt < 20 && !during_write; ++attempt) {
    killed = spawn_until(
        doc, {"AUTAN_TEST_FACTOR=0.999"},
        [](const fs::path& r) {
          return fs::exists(r / "checkpoint.cgns") && fs::exists(r / ".checkpoint.cgns.tmp");
        },
        100);
    kill_and_reap(killed.pid);
    during_write = fs::exists(killed.run / ".checkpoint.cgns.tmp");
  }
  REQUIRE(during_write);
  const fs::path file = killed.run / "checkpoint.cgns";
  REQUIRE(fs::exists(file));
  // Whole: readable, and its iteration has its history line.
  const std::int64_t k = checkpoint_iteration(file);
  CHECK(k >= 1);
  CHECK(static_cast<std::int64_t>(lines_of(killed.run / "history.jsonl").size()) >= k);
  Json next = input([](Json& d) {
    d["time"]["max_iterations"] = 100000;
    d["time"]["checkpoint"] = Json{{"iterations", 1}};
  });
  next["time"]["max_iterations"] = k + 3;
  next["start"] = start_from_checkpoint(killed.run);
  const Outcome cont = run(next, autan_test::RelaxationOptions{.factor = 0.999});
  INFO(cont.err);
  REQUIRE(cont.status == ExitStatus::finished);
  CHECK(Json::parse(read(cont.run / "run.json"))["iterations"] == k + 3);
}

TEST_CASE("Failed checkpoint writes take one W0714 in run.json however many", "[runtime]") {
  // results.md#checkpointcgns: run.json and the RunMetadata do not grow with the failures; the
  // log has one line each. Every write fails: a directory stands at the staging path.
  const auto failing = [](std::int64_t steps) {
    const fs::path dir = fresh_dir();
    const Json doc = input(
        [&](Json& d) {
          d["time"]["steps"] = steps;
          d["time"]["checkpoint"] = Json{{"iterations", 1}};
        },
        "hybrid-unsteady.solver-input.json");
    autan_test::RelaxationOptions options;
    options.block_checkpoints_in = (dir / "runs").string();
    const Outcome o = run_in(dir, doc, autan_test::relaxation_registry(options));
    INFO(o.err);
    REQUIRE(o.status == ExitStatus::finished);
    return o;
  };
  std::vector<std::size_t> run_json, metadata, entry;
  for (const std::int64_t steps : {5, 60}) {
    CAPTURE(steps);
    const Outcome o = failing(steps);
    const Json info = Json::parse(read(o.run / "run.json"));
    const auto lines = history(o.run);
    std::vector<Json> warnings;
    for (const auto& d : info["diagnostics"]) {
      if (d["code"] == "W0714") warnings.push_back(d);
    }
    REQUIRE(warnings.size() == 1);
    entry.push_back(warnings[0].dump().size());
    // A checkpoint follows every iteration but the last.
    CHECK(warnings[0]["data"] ==
          Json{{"failures", steps - 1},
               {"first_iteration", 1},
               {"first_time", lines[0]["time"]},
               {"last_iteration", steps - 1},
               {"last_time", lines[static_cast<std::size_t>(steps - 2)]["time"]}});
    std::int64_t logged = 0;
    for (const auto& line : lines_of(o.run / "log.jsonl")) {
      if (Json::parse(line)["code"] == "W0714") ++logged;
    }
    CHECK(logged == steps - 1);
    CHECK(!fs::exists(o.run / "checkpoint.cgns"));
    run_json.push_back(read(o.run / "run.json").size());
    const std::string stored = autan::io::read_solution(o.run / "solution.cgns").run_metadata;
    CHECK(!Json::parse(stored).contains("diagnostics"));
    metadata.push_back(stored.size());
  }
  // Twelve times the failures: less than one more W0714 (timings and paths vary by a few digits).
  CHECK(run_json[1] < run_json[0] + entry[0]);
  CHECK(metadata[1] < metadata[0] + entry[0]);
}

TEST_CASE("A checkpoint names its iteration, and a start checks it", "[runtime]") {
  // A start from a checkpoint whose iteration is not start.iterations exits 2 (E0707), as does
  // a final solution read as a checkpoint.
  const Json doc = input([](Json& d) {
    d["time"]["max_iterations"] = 10;
    d["time"]["checkpoint"] = Json{{"iterations", 3}};
  });
  const auto stopped = spawn_until(
      doc, {"AUTAN_TEST_FACTOR=0.999", "AUTAN_TEST_PAUSE_MS=40"}, [](const fs::path& r) {
        const Json i = Json::parse(read(r / "run.json"), nullptr, false);
        return !i.is_discarded() && i["checkpoint"].is_object();
      });
  kill_and_reap(stopped.pid);
  Json next = input([](Json& d) {
    d["time"]["max_iterations"] = 20;
    d["time"]["checkpoint"] = Json{{"iterations", 3}};
  });
  next["start"] = start_from_checkpoint(stopped.run);
  next["start"]["iterations"] = next["start"]["iterations"].get<std::int64_t>() + 1;
  const Outcome off = run(next, autan_test::RelaxationOptions{.factor = 0.999});
  CHECK(off.status == ExitStatus::invalid_input);
  CHECK(off.run.empty());
  CHECK(off.err.find("E0707") != std::string::npos);
}

// `s` quoted for the shell.
std::string quoted(const std::string& s) {
  std::string q = "'";
  for (const char c : s) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
  return q + "'";
}

// The SHA-256 of `file` by CMake's implementation, independent of the runtime's.
std::string sha256_of(const std::string& file) {
  const std::string command = quoted(AUTAN_TEST_CMAKE) + " -E sha256sum " + quoted(file);
  std::FILE* pipe = popen(command.c_str(), "r");
  REQUIRE(pipe != nullptr);
  std::array<char, 65> hex{};
  const std::size_t got = std::fread(hex.data(), 1, 64, pipe);
  REQUIRE(pclose(pipe) == 0);
  REQUIRE(got == 64);
  return std::string(hex.data(), 64);
}

// This test executable's.
std::string own_sha256() {
#ifdef __APPLE__
  std::uint32_t size = 0;
  (void)_NSGetExecutablePath(nullptr, &size);
  std::string self(size, '\0');
  REQUIRE(_NSGetExecutablePath(self.data(), &size) == 0);
  self.resize(std::char_traits<char>::length(self.c_str()));
  return sha256_of(self);
#else
  // Opens even if the file was replaced since; /proc/self would be cmake's own.
  return sha256_of("/proc/" + std::to_string(getpid()) + "/exe");
#endif
}

TEST_CASE("An executable's cached hash serves only that unchanged file", "[runtime]") {
  // AUTAN_BINARY_HASH_CACHE keeps run.json's solver.binary_sha256 across processes, keyed by the
  // file's device, inode, size, mtime and ctime. A file changed less than 2 s ago is not stored:
  // a coarse timestamp could miss a second change within the same tick.
  const fs::path dir = fresh_dir();
  const fs::path exe = dir / "autan-run";
  fs::copy_file(AUTAN_RUN_RELAXATION, exe);
  fs::permissions(exe, fs::status(AUTAN_RUN_RELAXATION).permissions());
  const fs::path cache = dir / "cache";
  fs::path last_run;
  const auto recorded = [&](const fs::path& cache_dir) {
    const Spawned s = spawn_until(
        input(), {"AUTAN_BINARY_HASH_CACHE=" + cache_dir.string()},
        [](const fs::path&) { return true; }, 2000, exe.string());
    int status = 0;
    REQUIRE(waitpid(s.pid, &status, 0) == s.pid);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
    last_run = s.run;
    return Json::parse(read(s.run / "run.json"))["solver"]["binary_sha256"].get<std::string>();
  };
  const auto entries = [&] {
    std::vector<fs::path> files;
    if (fs::exists(cache)) {
      for (const auto& e : fs::directory_iterator(cache)) files.push_back(e.path());
    }
    return files;
  };

  std::this_thread::sleep_for(std::chrono::milliseconds(2100));
  const std::string hash = sha256_of(exe.string());
  CHECK(recorded(cache) == hash);
  REQUIRE(entries().size() == 1);
  CHECK(read(entries()[0]) == hash + "\n");

  // The entry is what a later process reads: a forged one is believed.
  const std::string forged(64, 'f');
  std::ofstream(entries()[0], std::ios::trunc) << forged << '\n';
  CHECK(recorded(cache) == forged);

  // A changed file has another key: its own hash is recorded, and any entry stored for it (once
  // its ctime is 2 s old) holds that hash.
  const fs::path forged_entry = entries()[0];
  std::ofstream(exe, std::ios::binary | std::ios::app) << "autan";
  const std::string changed = sha256_of(exe.string());
  CHECK(changed != hash);
  CHECK(recorded(cache) == changed);
  for (const fs::path& e : entries()) {
    CAPTURE(e);
    CHECK(read(e) == (e == forged_entry ? forged : changed) + "\n");
  }

  // A cache it cannot use (a file where the directory should be) is logged, never silent.
  const fs::path not_a_dir = dir / "file";
  std::ofstream(not_a_dir) << "x";
  CHECK(recorded(not_a_dir) == changed);
  bool logged = false;
  for (const std::string& line : lines_of(last_run / "log.jsonl")) {
    const Json l = Json::parse(line);
    logged =
        logged || (l["level"] == "warning" && l["data"].value("cache", "") == not_a_dir.string());
  }
  CHECK(logged);
}

// The exit status and standard output of `command` run by the shell.
std::pair<int, std::string> shell(const std::string& command) {
  std::FILE* pipe = popen(command.c_str(), "r");
  REQUIRE(pipe != nullptr);
  std::string out;
  std::array<char, 4096> chunk{};
  std::size_t got = 0;
  while ((got = std::fread(chunk.data(), 1, chunk.size(), pipe)) > 0) out.append(chunk.data(), got);
  const int status = pclose(pipe);
  return {WIFEXITED(status) ? WEXITSTATUS(status) : -1, out};
}

TEST_CASE("solver.dirty is the SDK's rule in a checkout and the build's own flag in an export",
          "[runtime]") {
  const fs::path dir = fresh_dir();
  const fs::path out = dir / "autan_build_info.hpp";
  const auto build_info = [&](const fs::path& source, const std::string& defines) {
    fs::remove(out);
    return shell(quoted(AUTAN_TEST_CMAKE) + " -DSOURCE=" + quoted(source.string()) +
                 " -DOUT=" + quoted(out.string()) + " -DVERSION=0.0.0 " + defines + " -P " +
                 quoted(AUTAN_TEST_BUILD_INFO) + " >/dev/null 2>&1")
        .first;
  };
  const auto recorded = [&] {
    std::smatch m;
    const std::string text = read(out);
    REQUIRE(std::regex_search(text, m,
                              std::regex("AUTAN_BUILD_COMMIT \"([0-9a-f]*)\"\n#define "
                                         "AUTAN_BUILD_DIRTY (true|false)\n")));
    return std::pair<std::string, bool>{m[1].str(), m[2].str() == "true"};
  };
  const auto git = [](const fs::path& repo, const std::string& args) {
    const auto [status, text] =
        shell("git -C " + quoted(repo.string()) +
              " -c user.name=t -c user.email=t@t -c commit.gpgsign=false " + args + " 2>/dev/null");
    REQUIRE(status == 0);
    return text;
  };
  // A checkout, unpacked inside another repository: its own HEAD, dirty when a tracked file
  // differs from it (results.md). py/build_hook.py reads the same `git status` for the SDK.
  const fs::path outer = dir / "outer";
  const fs::path source = outer / "autan";
  fs::create_directories(source);
  git(outer, "init -q");
  git(outer, "commit -q --allow-empty -m outer");
  std::ofstream(source / "f") << "a";
  git(source, "init -q");
  git(source, "add f");
  git(source, "commit -q -m c");
  const std::string head = git(source, "rev-parse HEAD").substr(0, 40);
  const std::string other(40, 'b');
  struct Step {
    std::string what;
    std::function<void()> change;
    bool dirty;
  };
  const std::vector<Step> steps{
      {"clean", [] {}, false},
      {"untracked file only", [&] { std::ofstream(source / "new") << "x"; }, false},
      {"tracked file edited", [&] { std::ofstream(source / "f") << "b"; }, true},
      {"edit staged", [&] { git(source, "add f"); }, true},
  };
  for (const Step& step : steps) {
    INFO(step.what);
    step.change();
    REQUIRE(build_info(source, "-DAUTAN_COMMIT=" + other + " -DAUTAN_DIRTY=OFF") == 0);
    CHECK(recorded() == std::pair<std::string, bool>{head, step.dirty});
    CHECK(step.dirty == !git(source, "status --porcelain --untracked-files=no").empty());
  }
  // An export (no .git) states both, even inside another repository; missing or malformed, the
  // build stops.
  const fs::path exported = outer / "export";
  fs::create_directories(exported);
  REQUIRE(build_info(exported, "-DAUTAN_COMMIT=" + other + " -DAUTAN_DIRTY=ON") == 0);
  CHECK(recorded() == std::pair<std::string, bool>{other, true});
  REQUIRE(build_info(exported, "-DAUTAN_COMMIT=" + other + " -DAUTAN_DIRTY=OFF") == 0);
  CHECK(recorded() == std::pair<std::string, bool>{other, false});
  const std::vector<std::string> refused{
      "",
      "-DAUTAN_DIRTY=OFF",
      "-DAUTAN_COMMIT=" + other,
      "-DAUTAN_COMMIT=" + other + " -DAUTAN_DIRTY=",
      "-DAUTAN_COMMIT=" + other + " -DAUTAN_DIRTY=yes",
      "-DAUTAN_COMMIT=deadbeef -DAUTAN_DIRTY=OFF",
  };
  for (const std::string& defines : refused) {
    INFO(defines);
    CHECK(build_info(exported, defines) != 0);
    CHECK_FALSE(fs::exists(out));
  }
  fs::remove_all(dir);
}

TEST_CASE("A converging run writes the run directory of results.md", "[runtime]") {
  // Residuals 0.5^(k-1) drop 8 orders at k = 28 (0.5^27 < 1e-8 < 0.5^26).
  const Json doc = input();
  const Outcome o = run(doc);
  REQUIRE(o.status == ExitStatus::finished);
  REQUIRE(fs::is_directory(o.run));
  for (const auto& entry : fs::directory_iterator(o.runs)) {
    CHECK(entry.path().filename().string()[0] != '.');  // the staging directory is gone
  }
  std::vector<std::string> files;
  for (const auto& entry : fs::directory_iterator(o.run))
    files.push_back(entry.path().filename().string());
  std::ranges::sort(files);
  CHECK(files == std::vector<std::string>{"case.toml", "history.jsonl", "log.jsonl",
                                          "mesh-info.json", "run.json", "solution.cgns"});

  const Json info = Json::parse(read(o.run / "run.json"));
  CHECK(info["schema"] == "autan.results/1");
  CHECK(info["status"] == "converged");
  CHECK(info["iterations"] == 28);
  // results.md, timing: this run's 28 iterations, the loop within the
  // process, and ns per cell and iteration by its definition.
  const Json& t = info["timing"];
  REQUIRE(t.is_object());
  CHECK(t["iterations"] == 28);
  CHECK(t["solve_seconds"].get<double>() > 0.0);
  CHECK(t["solve_seconds"].get<double>() <= t["wall_seconds"].get<double>());
  const double cells = static_cast<double>(autan::io::read_mesh(mesh_path).n_cells());
  CHECK_THAT(
      t["ns_per_cell_iteration"].get<double>(),
      Catch::Matchers::WithinRel(t["solve_seconds"].get<double>() * 1e9 / (cells * 28), 1e-12));
  CHECK(std::regex_match(info["run_id"].get<std::string>(),
                         std::regex("run-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}")));
  CHECK(o.run.filename() == info["run_id"].get<std::string>());
  CHECK(std::regex_match(info["finished"].get<std::string>(),
                         std::regex("[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z")));
  CHECK(info["mesh"]["path"] == fs::absolute(mesh_path).lexically_normal().string());
  CHECK(info["solver"] == Json{{"version", "0.0.0"},
                               {"commit", build.commit},
                               {"dirty", build.dirty},
                               {"backend", "cpu"},
                               {"binary_sha256", info["solver"]["binary_sha256"]}});
  // The executable that ran, hashed once per process: a second run records the same.
  CHECK(info["solver"]["binary_sha256"] == own_sha256());
  const Outcome again = run(doc);
  REQUIRE(again.status == ExitStatus::finished);
  CHECK(Json::parse(read(again.run / "run.json"))["solver"]["binary_sha256"] ==
        info["solver"]["binary_sha256"]);
  // identity: SHA-256 of the nine fields joined by "\n" (results.md), the zone groups' hash
  // empty without a sidecar; the cell order is RCM by default.
  CHECK(info["cell_order"] == "rcm");
  std::string expected;
  picosha2::hash256_hex_string(doc["case"]["sha256"].get<std::string>() + "\n" +
                                   doc["mesh"]["sha256"].get<std::string>() + "\n\n" +
                                   build.commit + "\nfp32+acc64\ncpu\ncpu\n1\nrcm",
                               expected);
  CHECK(info["identity"] == expected);
  CHECK(read(o.run / "case.toml") == doc["case"]["toml"].get<std::string>());
  // solver-input.md, mesh_info: written as planned, not computed again.
  CHECK(read(o.run / "mesh-info.json") == doc["mesh_info"].dump(2));

  const auto lines = history(o.run);
  REQUIRE(lines.size() == 28);
  const std::vector<std::string> equations{"density",    "momentum-x", "momentum-y",
                                           "momentum-z", "energy",     "nu-tilde"};
  for (std::size_t k = 0; k < lines.size(); ++k) {
    CHECK(lines[k]["iteration"] == k + 1);
    CHECK(lines[k]["time"].is_null());
    for (std::size_t e = 0; e < equations.size(); ++e) {
      CHECK(lines[k]["residuals"][equations[e]] ==
            static_cast<double>(e + 1) * std::pow(0.5, static_cast<double>(k)));
    }
    // Forces on ymin every 5 iterations; uniform p = p_inf gives no load.
    const bool recorded = (k + 1) % 5 == 0;
    CHECK(lines[k]["monitors"].contains("forces") == recorded);
    if (recorded) {
      const Json& f = lines[k]["monitors"]["forces"];
      CHECK(f.size() == 12);  // fx..mz, cd cs cl (ref_area), cmx..cmz (ref_length)
      for (const auto& [key, value] : f.items()) CHECK(value == 0.0);
    }
  }
  // Convergence between two monitor iterations: run.json still holds the
  // final forces, at the last iteration.
  const Json& final_forces = info["monitors"]["forces"];
  CHECK(final_forces.size() == 13);
  CHECK(final_forces["iteration"] == 28);
  for (const auto& [key, value] : final_forces.items()) {
    if (key != "iteration") CHECK(value == 0.0);
  }

  const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
  const autan::Mesh back = autan::io::read_mesh(o.run / "solution.cgns");
  CHECK(back.n_cells() == mesh.n_cells());
  const auto density = solution_field(o.run / "solution.cgns", "FlowSolution", "Density");
  REQUIRE(density.size() == static_cast<std::size_t>(mesh.n_cells()));
  for (const double rho : density)
    CHECK_THAT(rho, WithinRel(doc["freestream"]["density"].get<double>(), 1e-15));
  // WallSolution covers every wall and slip face, face j as element 1 + j: ymin's, the one wall.
  std::vector<cgsize_t> wall;
  const auto offsets = mesh.boundary_zone_offsets();
  for (std::size_t z = 0; z < mesh.face_zones().size(); ++z) {
    if (mesh.face_zones()[z].name != "ymin") continue;
    for (auto f = offsets[z]; f < offsets[z + 1]; ++f) wall.push_back(static_cast<cgsize_t>(f + 1));
  }
  CHECK(wall.size() == 20);
  CHECK(wall_point_list(o.run / "solution.cgns") == wall);
}

TEST_CASE("Every final status is recorded as results.md defines it", "[runtime]") {
  SECTION("completed: budget reached before convergence") {
    const Outcome o =
        run(input([](Json& d) { d["time"]["max_iterations"] = 10; }), {.factor = 0.9});
    CHECK(o.status == ExitStatus::finished);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "completed");
    CHECK(info["iterations"] == 10);
    CHECK(fs::exists(o.run / "solution.cgns"));
  }
  SECTION("final monitors: the last state, equal to its history line when it has one") {
    // The pressure grows at each step, so each line's forces differ.
    const Outcome o = run(input([](Json& d) { d["time"]["max_iterations"] = 10; }),
                          {.factor = 0.9, .pressure_offset = 1000.0, .growth = 1e-3});
    const Json info = Json::parse(read(o.run / "run.json"));
    Json expected = history(o.run)[9]["monitors"]["forces"];
    CHECK(expected != history(o.run)[4]["monitors"]["forces"]);
    expected["iteration"] = 10;
    CHECK(info["monitors"] == Json{{"forces", expected}});

    // Between two monitor iterations: ymin, the unit square y = 0 with
    // outward normal -y, carries fy = -(p - p_inf), p = (p_inf + dp) 1.001^11.
    const Json doc = input([](Json& d) { d["time"]["max_iterations"] = 11; });
    const Outcome off = run(doc, {.factor = 0.9, .pressure_offset = 1000.0, .growth = 1e-3});
    const Json f = Json::parse(read(off.run / "run.json"))["monitors"]["forces"];
    CHECK(f["iteration"] == 11);
    const double p_inf = doc["freestream"]["pressure"].get<double>();
    CHECK_THAT(f["fy"].get<double>(),
               WithinRel(-((p_inf + 1000.0) * std::pow(1.001, 11) - p_inf), 1e-12));
  }
  SECTION("stopped before iteration 1: the initial state at iteration 0") {
    // ymin, the unit square y = 0 with outward normal -y, under a pressure
    // excess dp carries F = (0, -dp, 0).
    const Outcome o =
        run(input(), {.factor = 0.9, .pressure_offset = 1000.0, .signal_on_start = true});
    CHECK(o.status == ExitStatus::signalled);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "stopped");
    CHECK(info["iterations"] == 0);
    CHECK(info["timing"].is_null());  // no iteration to time
    CHECK_FALSE(fs::exists(o.run / "history.jsonl"));
    const Json& f = info["monitors"]["forces"];
    CHECK(f["iteration"] == 0);
    CHECK_THAT(f["fy"].get<double>(), Catch::Matchers::WithinAbs(-1000.0, 1e-9));
    CHECK(fs::exists(o.run / "solution.cgns"));
  }
  SECTION("failed in the final loads: E0704, no monitors") {
    const Outcome o = run(input([](Json& d) { d["monitors"][0]["every"] = 1000; }),
                          {.factor = 0.5, .throw_in_loads = true});
    CHECK(o.status == ExitStatus::failed);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "failed");
    CHECK(info["iterations"] == 28);
    CHECK(info["diagnostics"][0]["code"] == "E0704");
    CHECK(info["monitors"].is_null());
  }
  SECTION("failed on final loads that overflow: E0704, no monitors") {
    // A pressure excess on ymin about a centre 1e308 m away: moments beyond
    // the double range.
    const Outcome o = run(input([](Json& d) {
                            d["monitors"][0]["every"] = 1000;
                            d["monitors"][0]["moment_center"] = {1e308, 1e308, 1e308};
                          }),
                          {.factor = 0.5, .pressure_offset = 1000.0});
    CHECK(o.status == ExitStatus::failed);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "failed");
    CHECK(info["diagnostics"][0]["code"] == "E0704");
    CHECK(info["monitors"].is_null());
  }
  SECTION("failed on a recorded load that is not finite: E0704, history up to the line before") {
    // results.md, History: a record holds numbers, so a non-finite force
    // at iteration 5 fails the run there instead of writing null.
    const Outcome o = run(input([](Json& d) { d["monitors"][0]["every"] = 1; }),
                          {.factor = 0.9, .nan_loads_from = 5});
    CHECK(o.status == ExitStatus::failed);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "failed");
    CHECK(info["iterations"] == 4);
    CHECK(info["diagnostics"][0]["code"] == "E0704");
    CHECK(info["monitors"].is_null());
    const auto lines = history(o.run);
    REQUIRE(lines.size() == 4);
    CHECK(lines.back()["iteration"] == 4);
    CHECK(std::isfinite(lines.back()["monitors"]["forces"]["fy"].get<double>()));
  }
  SECTION("failed writing the solution: E0704, no monitors") {
    const Outcome o = run(input(), {.factor = 0.5, .throw_in_fields = true});
    CHECK(o.status == ExitStatus::failed);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "failed");
    CHECK(info["diagnostics"][0]["code"] == "E0704");
    CHECK(info["monitors"].is_null());
    CHECK(info["timing"].is_null());
  }
  SECTION("failed writing the solution file: E0704, no solution") {
    // A directory stands at the solution's staging path.
    const fs::path dir = fresh_dir();
    autan_test::RelaxationOptions options;
    options.factor = 0.5;
    options.block_solution_in = (dir / "runs").string();
    const Outcome o = run_in(dir, input(), autan_test::relaxation_registry(options));
    CHECK(o.status == ExitStatus::failed);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "failed");
    CHECK(info["diagnostics"][0]["code"] == "E0704");
    CHECK_FALSE(fs::exists(o.run / "solution.cgns"));
  }
  SECTION("diverged: D0107 with the iteration, cell and zone") {
    const Outcome o = run(input(), {.factor = 0.9, .diverge_at = 7, .diverge_cell = 3});
    CHECK(o.status == ExitStatus::finished);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "diverged");
    CHECK(info["iterations"] == 6);
    REQUIRE(info["diagnostics"].size() == 1);
    const Json& d = info["diagnostics"][0];
    CHECK(d["code"] == "D0107");
    CHECK(d["severity"] == "divergence");
    CHECK(d["data"]["iteration"] == 7);
    CHECK(d["data"]["cell"] == 3);
    const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
    CHECK(d["data"]["zone"] ==
          mesh.cell_zones()[static_cast<std::size_t>(mesh.cell_zone()[3])].name);
    CHECK(history(o.run).size() == 6);
    CHECK(info["monitors"].is_null());
    // The diverged state holds a NaN: no solution can satisfy the readers.
    CHECK_FALSE(fs::exists(o.run / "solution.cgns"));
  }
  SECTION("failed: E0704, history kept, exit 1") {
    const Outcome o = run(input(), {.factor = 0.9, .throw_at = 4});
    CHECK(o.status == ExitStatus::failed);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "failed");
    CHECK(info["diagnostics"][0]["code"] == "E0704");
    CHECK(info["diagnostics"][0]["data"] == Json::object());  // one rank: no data.rank
    CHECK(history(o.run).size() == 3);
    CHECK(info["monitors"].is_null());
    CHECK(info["timing"].is_null());  // a failed run is not timed
    const auto log = lines_of(o.run / "log.jsonl");
    REQUIRE_FALSE(log.empty());
    CHECK(Json::parse(log.back())["level"] == "error");
  }
  SECTION("failed creating the run directory: E0704 on stderr, exit 1") {
    const fs::path dir = fresh_dir();
    std::ofstream(dir / "runs") << "x";
    const Outcome o = run_in(dir, input(), autan_test::relaxation_registry({}));
    CHECK(o.status == ExitStatus::failed);
    CHECK(o.run.empty());
    // Warnings (W0713) may come first; the last diagnostic is the failure.
    std::vector<Json> diagnostics;
    std::istringstream err_lines(o.err);
    for (std::string line; std::getline(err_lines, line);)
      if (line.starts_with("{")) diagnostics.push_back(Json::parse(line));
    REQUIRE(!diagnostics.empty());
    for (std::size_t k = 0; k + 1 < diagnostics.size(); ++k)
      CHECK(diagnostics[k]["severity"] == "warning");
    CHECK(diagnostics.back()["code"] == "E0704");
  }
  SECTION("unsteady with averaging: time, averaging and FlowSolutionMean") {
    // Written by `autan prepare` with time = Unsteady(steps=6, dt=0.01 s,
    // average_from=3); fields grow by 10 % per step, so the mean is
    // (1.1^4 + 1.1^5 + 1.1^6) / 3 times the initial value.
    const Outcome o = run(input({}, "hybrid-unsteady.solver-input.json"), {.growth = 0.1});
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "completed");
    CHECK_THAT(info["time"].get<double>(), WithinRel(0.06, 1e-12));
    CHECK(info["averaging"]["from_step"] == 4);
    CHECK(info["averaging"]["to_step"] == 6);
    CHECK_THAT(info["averaging"]["from_time"].get<double>(), WithinRel(0.03, 1e-12));
    CHECK_THAT(info["averaging"]["to_time"].get<double>(), WithinRel(0.06, 1e-12));
    const auto lines = history(o.run);
    REQUIRE(lines.size() == 6);
    CHECK_THAT(lines[1]["time"].get<double>(), WithinRel(0.02, 1e-12));
    const auto mean = solution_field(o.run / "solution.cgns", "FlowSolutionMean", "Density");
    const auto last = solution_field(o.run / "solution.cgns", "FlowSolution", "Density");
    const double rho =
        input({}, "hybrid-unsteady.solver-input.json")["freestream"]["density"].get<double>();
    REQUIRE(mean.size() == last.size());
    for (std::size_t c = 0; c < mean.size(); ++c) {
      CHECK_THAT(
          mean[c],
          WithinRel(rho * (std::pow(1.1, 4) + std::pow(1.1, 5) + std::pow(1.1, 6)) / 3.0, 1e-12));
      CHECK_THAT(last[c], WithinRel(rho * std::pow(1.1, 6), 1e-12));
    }
  }
}

TEST_CASE("Invalid inputs exit 2 with a coded diagnostic and no directory", "[runtime]") {
  SECTION("E0705 unknown key") {
    require_error(run(input([](Json& d) { d["physics"]["extra"] = 1; })), "E0705");
  }
  SECTION("E0705 missing key") {
    require_error(run(input([](Json& d) { d["freestream"].erase("speed"); })), "E0705");
    // The keys that choose an object's shape.
    require_error(
        run(input([](Json& d) { d["cell_zones"][0]["medium"]["viscosity"].erase("law"); })),
        "E0705");
    require_error(run(input([](Json& d) { d["time"].erase("kind"); })), "E0705");
    require_error(run(input([](Json& d) { d["time"].erase("start_order"); })), "E0705");
  }
  SECTION("E0705 start_order other than first or second") {
    require_error(run(input([](Json& d) { d["time"]["start_order"] = "third"; })), "E0705");
  }
  SECTION("E0705 value a kind forbids") {
    require_error(run(input([](Json& d) { d["boundaries"][0]["static_pressure"] = 1e5; })),
                  "E0705");
    // Only a wall takes a temperature: here on a farfield.
    require_error(run(input([](Json& d) { d["boundaries"][0]["temperature"] = 300.0; })), "E0705");
  }
  SECTION("E0705 states on a condition that takes none") {
    require_error(run(input([](Json& d) {
                    d["boundaries"][0]["states"] = {{"path", "/s.json"},
                                                    {"sha256", std::string(64, 'a')}};
                  })),
                  "E0705");
  }
  SECTION("E0705 oracle-state without an absolute states path") {
    require_error(run(input([](Json& d) {
                    d["boundaries"][0]["kind"] = "oracle-state";
                    d["boundaries"][0]["states"] = {{"path", "s.json"},
                                                    {"sha256", std::string(64, 'a')}};
                  })),
                  "E0705");
  }
  SECTION("E0705 oracle-state with a turbulence model") {
    require_error(run(input([](Json& d) {
                    d["boundaries"][0]["kind"] = "oracle-state";
                    d["boundaries"][0]["states"] = {{"path", "/s.json"},
                                                    {"sha256", std::string(64, 'a')}};
                  })),
                  "E0705");
  }
  SECTION("E0705 NUL in a states path") {
    require_error(run(input([](Json& d) {
                    d["physics"]["turbulence"]["model"] = "inviscid";
                    d["physics"]["turbulence"]["nu_tilde_ratio"] = nullptr;
                    d["freestream"]["nu_tilde"] = nullptr;
                    for (auto& z : d["cell_zones"]) z["initial"]["nu_tilde"] = nullptr;
                    d["boundaries"][0]["kind"] = "oracle-state";
                    d["boundaries"][0]["states"] = {{"path", std::string("/s\0.json", 8)},
                                                    {"sha256", std::string(64, 'a')}};
                  })),
                  "E0705");
  }
  SECTION("E0705 point vortex on a condition other than farfield") {
    require_error(run(input([](Json& d) {
                    d["boundaries"][2]["point_vortex"] = {{"zones", {"ymin"}},
                                                          {"center", {0.0, 0.0, 0.0}}};
                  })),
                  "E0705");
  }
  SECTION("E0705 point vortex over a zone that is not a wall") {
    require_error(run(input([](Json& d) {
                    d["boundaries"][0]["point_vortex"] = {{"zones", {"xmax"}},
                                                          {"center", {0.0, 0.0, 0.0}}};
                  })),
                  "E0705");
  }
  SECTION("E0705 point vortex zones that are not a list") {
    require_error(run(input([](Json& d) {
                    d["boundaries"][0]["point_vortex"] = {{"zones", "ymin"},
                                                          {"center", {0.0, 0.0, 0.0}}};
                  })),
                  "E0705");
  }
  SECTION("E0705 zone absent from the mesh") {
    require_error(run(input([](Json& d) { d["boundaries"][0]["zone"] = "nowhere"; })), "E0705");
  }
  SECTION("E0705 a zone with faces and no condition; none needed without faces") {
    require_error(run(input([](Json& d) { d["boundaries"].erase(std::size_t{0}); })), "E0705");
    // The hybrid mesh with one more patch, declared without faces.
    const fs::path copy = fresh_dir() / "polyMesh";
    fs::copy(mesh_path, copy);
    std::string boundary = read(copy / "boundary");
    const std::string count = "6 // entry0";
    REQUIRE(boundary.find(count) != std::string::npos);
    boundary.replace(boundary.find(count), count.size(), "7");
    boundary.insert(boundary.rfind("\n)"),
                    "\n    unused\n    {\n        type            patch;\n        nFaces          "
                    "0;\n        startFace       564;\n    }");
    std::ofstream(copy / "boundary") << boundary;
    const Outcome o = run(input([&](Json& d) {
      d["mesh"]["path"] = copy.string();
      d["mesh"]["sha256"] = autan::io::mesh_hash(copy);
    }));
    INFO(o.err);
    CHECK(o.status == ExitStatus::finished);
  }
  SECTION("E0203 mesh hash") {
    // The file read is not the one the input and its MeshInfo name.
    Json other = input();
    other["mesh"]["sha256"] = std::string(64, '0');
    other["mesh_info"]["source"]["sha256"] = std::string(64, '0');
    require_error(run(other), "E0203");
  }
  SECTION("mesh_info: planned for this mesh, in the run's cell order") {
    // solver-input.md, mesh_info: written as the run's mesh-info.json, so it must describe the
    // mesh read (E0203) and give the lines of the order the run computes in (E0705).
    Json other = input();
    other["mesh_info"]["source"]["sha256"] = std::string(64, '0');
    require_error(run(other), "E0203");
    // Its source.bytes is the size of the mesh read: another is another mesh, 0 none.
    Json other_size = input();
    other_size["mesh_info"]["source"]["bytes"] =
        other_size["mesh_info"]["source"]["bytes"].get<std::int64_t>() + 1;
    require_error(run(other_size), "E0203");
    other_size["mesh_info"]["source"]["bytes"] = 0;
    require_error(run(other_size), "E0705");
    require_error(run(autan_test::planned(input(), mesh_path, "input")), "E0705");
    for (const Json& broken :
         {Json(nullptr), Json::object(), Json{{"schema", "autan.mesh-info/2"}}}) {
      CAPTURE(broken.dump());
      Json d = input();
      d["mesh_info"] = broken;
      require_error(run(d), "E0705");
    }
    Json other_scale = input();
    other_scale["mesh_info"]["quality"]["lines"]["scale"] = 0.0254;
    require_error(run(other_scale), "E0705");
#ifdef AUTAN_HAVE_METIS
    // mesh-info.md, partition.groups: a run on several ranks, this steady CPU multigrid on two,
    // is planned with a partition that keeps its multigrid's finest lines whole; one formed with
    // the line smoother's lines or none is another run's. Here one process runs it, so the input
    // that passes this check is refused for its devices.count instead.
    {
      const autan::Mesh read = autan::io::read_mesh(mesh_path);
      using autan::io::RankGroups;
      for (const auto& [groups, name] :
           {std::pair{RankGroups::none, "none"}, std::pair{RankGroups::lines, "lines"},
            std::pair{RankGroups::multigrid, "multigrid"}}) {
        CAPTURE(name);
        const auto counts =
            autan::io::run_counts(read, 1.0, autan::io::CellOrder::rcm,
                                  autan::io::RanksRequest{.ranks = 2, .groups = groups});
        Json d = input();
        d["devices"]["count"] = 2;
        d["mesh_info"]["partition"] = Json::parse(autan::io::mesh_info_json(
            read, autan::io::mesh_hash(mesh_path), autan::io::mesh_bytes(mesh_path),
            autan::io::mesh_logical_bytes(mesh_path), counts))["partition"];
        REQUIRE(d["mesh_info"]["partition"]["groups"] == name);
        const Outcome o = run(d);
        require_error(o, "E0705");
        const bool groups_refused =
            Json::parse(o.err)["location"]["path"] == "mesh_info.partition.groups";
        CHECK(groups_refused == (groups != RankGroups::multigrid));
      }
    }
#endif
    // The whole contract of mesh-info.md, which Results.open checks on the archived document:
    // each of these breaks one of its members or rules.
    const std::vector<std::function<void(Json&)>> breaks{
        [](Json& m) { m["quality"].erase("lines"); },
        [](Json& m) { m.erase("counts"); },
        [](Json& m) { m["unknown"] = 1; },
        [](Json& m) { m["counts"]["stencil_entries"] = -1; },
        [](Json& m) { m["counts"].erase("band_entries"); },
        [](Json& m) {
          m["counts"]["cells_by_type"]["hexa"] =
              m["counts"]["cells_by_type"]["hexa"].get<int>() + 1;
        },
        [](Json& m) { m["cell_zones"][0]["cells"] = m["cell_zones"][0]["cells"].get<int>() + 1; },
        [](Json& m) { m["boundary_zones"][0]["type"] = "outflow"; },
        [](Json& m) { m["boundary_zones"].push_back(m["boundary_zones"][0]); },
        // first_cell_height (mesh-info.md): null exactly without faces, 0 <= min <= median <=
        // max, three equal numbers for one face.
        [](Json& m) { m["boundary_zones"][0].erase("first_cell_height"); },
        [](Json& m) { m["boundary_zones"][0]["first_cell_height"] = nullptr; },
        [](Json& m) { m["boundary_zones"][0]["first_cell_height"]["min"] = -1.0; },
        [](Json& m) {
          auto& h = m["boundary_zones"][0]["first_cell_height"];
          h["median"] = h["max"].get<double>() * 2;
        },
        [](Json& m) { m["bounds"]["min"][0] = 1e9; },
        [](Json& m) { m["quality"]["lines"]["long_cells"] = m["counts"]["cells"].get<int>() + 1; },
        [](Json& m) {
          auto& edges = m["quality"]["multigrid"]["edges_by_level"][0];
          edges = edges.get<int>() + 1;
        },
        [](Json& m) { m["quality"]["multigrid"]["order"] = "metis"; },
        [](Json& m) {
          const int cells = m["counts"]["cells"].get<int>();
          m["partition"] = {{"ranks", 2},
                            {"groups", "none"},
                            {"ghost_stencil", "face2"},
                            {"cells_by_rank", {cells, 0}},
                            {"ghosts_by_rank", {0}},
                            {"faces_by_rank", {0, 0}},
                            {"interior_faces_by_rank", {0, 0}},
                            {"stencil_entries_by_rank", {0, 0}},
                            {"vertex_stencil_entries_by_rank", {0, 0}},
                            {"band_entries_by_rank", {0, 0}},
                            {"boundary_fit_entries_by_rank", {0, 0}},
                            {"nodes_by_rank", {0, 0}},
                            {"face_nodes_by_rank", {0, 0}},
                            {"sent_cells_by_rank", {0, 0}}};
        },
    };
    for (std::size_t k = 0; k < breaks.size(); ++k) {
      CAPTURE(k);
      Json d = input();
      breaks[k](d["mesh_info"]);
      require_error(run(d), "E0705");
    }
  }
  SECTION("E0706 no physics in the build") {
    require_error(run(input(), autan::runtime::PhysicsRegistry{}), "E0706");
  }
  SECTION("E0146 a launcher's ranks without MPI started") {
    // As a launcher would start one of four ranks, without MPI initialized.
    REQUIRE(setenv("OMPI_COMM_WORLD_SIZE", "4", 1) == 0);
    const Outcome o = run(input());
    unsetenv("OMPI_COMM_WORLD_SIZE");
    require_error(o, "E0146");
    // PMIx may give the rank alone: without a count, even rank 0 cannot
    // tell it runs alone.
    REQUIRE(setenv("PMIX_RANK", "0", 1) == 0);
    const Outcome pmix = run(input());
    unsetenv("PMIX_RANK");
    require_error(pmix, "E0146");
  }
}

TEST_CASE("A device run needs a build and a device that drive it", "[runtime]") {
  // Never a silent CPU run: a kind this build does not drive is E0706, and
  // a CUDA build that sees no device is E0807, both before any directory,
  // whatever the physics registered: here, one for the CPU only.
  autan::runtime::PhysicsRegistry cpu_only;
  cpu_only.add({.kind = "*", .turbulence = "*", .precision = "*", .device = "cpu"},
               [](const autan::runtime::PhysicsContext& context) {
                 return std::make_unique<autan_test::Relaxation>(context,
                                                                 autan_test::RelaxationOptions{});
               });
  const Outcome hip = run(input([](Json& d) { d["devices"]["kind"] = "hip"; }), cpu_only);
  require_error(hip, "E0706");
  CHECK(Json::parse(hip.err)["data"]["path"] == "devices.kind");
#ifdef AUTAN_HAVE_CUDA
  // As a host without a GPU: CUDA reads this before its first call, and
  // this test runs in a process of its own.
  REQUIRE(setenv("CUDA_VISIBLE_DEVICES", "-1", 1) == 0);
  const Outcome o = run(input([](Json& d) { d["devices"]["kind"] = "cuda"; }), cpu_only);
  require_error(o, "E0807");
  CHECK(Json::parse(o.err)["data"] ==
        Json{{"visible", 0}, {"needed", 1}, {"path", "devices.kind"}});
#else
  const Outcome o = run(input([](Json& d) { d["devices"]["kind"] = "cuda"; }), cpu_only);
  require_error(o, "E0706");
  CHECK(Json::parse(o.err)["data"]["path"] == "devices.kind");
#endif
}

#ifdef AUTAN_HAVE_CUDA
TEST_CASE("A CUDA run gives the physics its device and stream", "[runtime][gpu]") {
  // On a periodic mesh too, where the relaxation physics checks that the
  // device exchange fills each image ghost on the run's stream.
  const bool periodic = GENERATE(false, true);
  CAPTURE(periodic);
  const auto on_cuda = [](Json& d) { d["devices"]["kind"] = "cuda"; };
  // The physics holds 64 MiB of device memory from its construction on: the run's device
  // figures count exactly that, and the periodic exchange's buffers on the periodic mesh.
  autan_test::RelaxationOptions holding;
  holding.hold_device_bytes = std::int64_t{64} << 20;
  const Outcome o = run(periodic ? periodic_input(1.0, on_cuda) : input(on_cuda), holding);
  REQUIRE(o.status == ExitStatus::finished);
  const Json memory = memory_event(o.run);
  const auto peak = memory["device_peak_bytes"].get<std::int64_t>();
  const auto setup = memory["after_setup"]["device"].get<std::int64_t>();
  if (periodic) {
    CHECK(setup >= holding.hold_device_bytes);
    CHECK(peak >= setup);
  } else {
    CHECK(setup == holding.hold_device_bytes);
    CHECK(peak == setup);
  }
  CHECK(memory["device_peak_bytes_by_rank"] == Json::array({peak}));
  std::ifstream log_in(o.run / "log.jsonl");
  Json ready, cuda;
  for (std::string line; std::getline(log_in, line);) {
    const Json event = Json::parse(line);
    if (event["message"] == "relaxation physics ready") ready = event["data"];
    if (event["source"] == "cuda") cuda = event["data"];
  }
  CHECK(ready["device"] == 0);
  CHECK(ready["stream"] == true);
  CHECK(cuda["device"] == 0);
  CHECK(!cuda["name"].get<std::string>().empty());
  CHECK(cuda["visible"].get<int>() >= 1);
}
#endif

TEST_CASE("A scope's refusal carries its own code", "[runtime]") {
  // E0706 unless the physics says more (E1109 for a reproducibility it cannot honour): any
  // registered code goes through, here E0705.
  autan_test::RelaxationOptions refusing;
  refusing.refuse_code = "E0705";
  const Outcome o = run(input(), refusing);
  require_error(o, "E0705");
  CHECK(Json::parse(o.err)["data"]["path"] == "physics");
}

TEST_CASE("A physics' scope refuses an input before any directory", "[runtime]") {
  const bool refuse = GENERATE(true, false);
  autan::runtime::PhysicsRegistry registry;
  registry.add(
      {.kind = "*", .turbulence = "*", .precision = "*", .device = "*"},
      [](const autan::runtime::PhysicsContext& context) {
        return std::make_unique<autan_test::Relaxation>(context, autan_test::RelaxationOptions{});
      },
      [refuse](const autan::runtime::SolverInput& in,
               int ranks) -> std::optional<autan::runtime::PhysicsRefusal> {
        // The scope sees the run's rank count: one, here.
        if (!refuse || ranks != 1) return std::nullopt;
        return autan::runtime::PhysicsRefusal{
            .message = "refused",
            .path = "cell_zones[1].medium",
            .data_json = Json{{"zone", in.cell_zones[1].zone}}.dump()};
      });
  const Json doc = input();
  const Outcome o = run(doc, registry);
  if (!refuse) {
    CHECK(o.status == ExitStatus::finished);
    return;
  }
  require_error(o, "E0706");
  const Json d = Json::parse(o.err);
  CHECK(d["location"]["path"] == "cell_zones[1].medium");
  CHECK(d["data"] ==
        Json{{"zone", doc["cell_zones"][1]["zone"]}, {"path", "cell_zones[1].medium"}});
}

TEST_CASE("Deterministic runs produce identical history and solution arrays", "[runtime]") {
  const Json doc = input([](Json& d) { d["reproducibility"] = "run"; });
  const Outcome a = run(doc);
  const Outcome b = run(doc);
  CHECK(read(a.run / "history.jsonl") == read(b.run / "history.jsonl"));
  for (const char* field : {"Density", "Pressure", "TurbulentSANuTilde"}) {
    CHECK(solution_field(a.run / "solution.cgns", "FlowSolution", field) ==
          solution_field(b.run / "solution.cgns", "FlowSolution", field));
  }
}

TEST_CASE("A stop file stops the run as SIGINT does", "[runtime]") {
  // The relaxation physics creates the file during iteration 5: the run
  // stops before iteration 6, with its solution, and exits 130.
  const fs::path stop = fresh_dir() / "stop";
  autan_test::RelaxationOptions options{.factor = 0.9};
  options.touch_at = 5;
  options.touch_file = stop.string();
  const Outcome o = run(input([](Json& d) { d["time"]["max_iterations"] = 1000; }), options, stop);
  CHECK(o.status == ExitStatus::signalled);
  const Json info = Json::parse(read(o.run / "run.json"));
  CHECK(info["status"] == "stopped");
  CHECK(info["iterations"] == 5);
  CHECK(fs::exists(o.run / "solution.cgns"));
}

TEST_CASE("solve_seconds leaves out the physics' log writes", "[runtime]") {
  // Every step logs 10 MB: the log takes most of the run, which the solve
  // time does not count (results.md, timing).
  autan_test::RelaxationOptions options{.factor = 0.9};
  options.log_bytes = 10'000'000;
  const Outcome o = run(input([](Json& d) { d["time"]["max_iterations"] = 4; }), options);
  REQUIRE(o.status == ExitStatus::finished);
  const Json t = Json::parse(read(o.run / "run.json"))["timing"];
  CHECK(t["solve_seconds"].get<double>() < 0.25 * t["wall_seconds"].get<double>());
}

TEST_CASE("SIGTERM stops a running process: stopped, solution written, exit 130", "[runtime]") {
  const fs::path dir = fresh_dir();
  const fs::path input_file = dir / "solver-input.json";
  std::ofstream(input_file) << input([](Json& d) { d["time"]["max_iterations"] = 100000; }).dump();
  const std::string runs = (dir / "runs").string();
  std::vector<std::string> args{AUTAN_RUN_RELAXATION, "--input", input_file.string(), "--runs",
                                runs};
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  std::vector<std::string> env{"AUTAN_TEST_FACTOR=0.999", "AUTAN_TEST_PAUSE_MS=20"};
  for (char** e = environ; *e != nullptr; ++e) env.emplace_back(*e);
  std::vector<char*> envp;
  for (auto& e : env) envp.push_back(e.data());
  envp.push_back(nullptr);
  pid_t pid = 0;
  REQUIRE(posix_spawn(&pid, AUTAN_RUN_RELAXATION, nullptr, nullptr, argv.data(), envp.data()) == 0);

  // Wait for a few recorded iterations, then ask the process to stop.
  fs::path run_dir;
  for (int attempt = 0; attempt < 500 && run_dir.empty(); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (!fs::exists(runs)) continue;
    for (const auto& entry : fs::directory_iterator(runs)) {
      if (entry.path().filename().string()[0] != '.' &&
          fs::exists(entry.path() / "history.jsonl") && history(entry.path()).size() >= 3) {
        run_dir = entry.path();
      }
    }
  }
  REQUIRE(!run_dir.empty());
  REQUIRE(kill(pid, SIGTERM) == 0);
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 130);
  const Json info = Json::parse(read(run_dir / "run.json"));
  CHECK(info["status"] == "stopped");
  CHECK(info["iterations"].get<int>() >= 3);
  CHECK(info["iterations"].get<std::size_t>() == history(run_dir).size());
  CHECK(fs::exists(run_dir / "solution.cgns"));
}

TEST_CASE("Forces monitors report the loads and their coefficients", "[runtime]") {
  // ymin is the unit square y = 0 with outward normal -y: a pressure excess
  // dp gives F = (0, -dp, 0) and, about the origin, M = c x F with c the
  // square's centroid (1/2, 0, 1/2). Coefficients follow results.md with the
  // drag, lift and side directions of the incidence and sideslip.
  constexpr double dp = 1000.0, alpha = 0.3, beta = 0.2;
  const Json doc = input([&](Json& d) {
    d["freestream"]["aoa"] = alpha;
    d["freestream"]["sideslip"] = beta;
  });
  const Outcome o = run(doc, {.factor = 0.5, .pressure_offset = dp});
  REQUIRE(o.status == ExitStatus::finished);
  const auto lines = history(o.run);
  REQUIRE(lines.size() > 4);
  const Json f = lines[4]["monitors"]["forces"];
  const double q = 0.5 * doc["freestream"]["density"].get<double>() *
                   std::pow(doc["freestream"]["speed"].get<double>(), 2);
  const std::array<double, 3> force{0.0, -dp, 0.0};
  const std::array<double, 3> moment{0.5 * dp, 0.0, -0.5 * dp};
  const std::array<double, 3> drag{std::cos(alpha) * std::cos(beta), -std::sin(beta),
                                   std::sin(alpha) * std::cos(beta)};
  const std::array<double, 3> lift{-std::sin(alpha), 0.0, std::cos(alpha)};
  const std::array<double, 3> side{lift[1] * drag[2] - lift[2] * drag[1],
                                   lift[2] * drag[0] - lift[0] * drag[2],
                                   lift[0] * drag[1] - lift[1] * drag[0]};
  auto dot = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  };
  const std::array<std::string, 3> axes{"x", "y", "z"};
  // A history line, and run.json's final record at iteration 28, between two
  // monitor iterations.
  const Json final_forces = Json::parse(read(o.run / "run.json"))["monitors"]["forces"];
  CHECK(final_forces["iteration"] == 28);
  for (const Json& r : {f, final_forces}) {
    for (std::size_t k = 0; k < 3; ++k) {
      CHECK_THAT(r["f" + axes[k]].get<double>(), Catch::Matchers::WithinAbs(force[k], 1e-9));
      CHECK_THAT(r["m" + axes[k]].get<double>(), Catch::Matchers::WithinAbs(moment[k], 1e-9));
      CHECK_THAT(r["cm" + axes[k]].get<double>(), Catch::Matchers::WithinAbs(moment[k] / q, 1e-9));
    }
    CHECK_THAT(r["cd"].get<double>(), Catch::Matchers::WithinAbs(dot(force, drag) / q, 1e-12));
    CHECK_THAT(r["cl"].get<double>(), Catch::Matchers::WithinAbs(dot(force, lift) / q, 1e-12));
    CHECK_THAT(r["cs"].get<double>(), Catch::Matchers::WithinAbs(dot(force, side) / q, 1e-12));
  }
  CHECK(std::abs(f["cd"].get<double>()) > 1e-3);
}

TEST_CASE("An incompressible input from the SDK runs", "[runtime]") {
  // `autan prepare` for water through the hybrid box: velocity inlet,
  // outlet, walls; no temperature anywhere.
  const Outcome o = run(input({}, "hybrid-water.solver-input.json"));
  REQUIRE(o.status == ExitStatus::finished);
  const auto lines = history(o.run);
  CHECK(lines.front()["residuals"].contains("pressure"));
  CHECK_FALSE(lines.front()["residuals"].contains("energy"));
  CHECK(field_list(o.run / "solution.cgns") ==
        std::vector<std::string>{"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"});
}

TEST_CASE("The input reader enforces the solver-input rules", "[runtime]") {
  SECTION("WMLES needs its wall model") {
    require_error(run(input([](Json& d) {
                    d["physics"]["turbulence"] =
                        Json{{"model", "wmles"},      {"base", nullptr},
                             {"wall_model", nullptr}, {"nu_tilde_ratio", nullptr},
                             {"intensity", nullptr},  {"viscosity_ratio", nullptr}};
                  })),
                  "E0705");
  }
  SECTION("DDES needs an unsteady time") {
    require_error(run(input([](Json& d) {
                    d["physics"]["turbulence"]["model"] = "ddes";
                    d["physics"]["turbulence"]["base"] = "sa";
                  })),
                  "E0705");
  }
  SECTION("a boundary's rule is an index or null") {
    for (const Json& rule : {Json(-1), Json(1.5), Json("0")}) {
      CAPTURE(rule);
      require_error(run(input([&](Json& d) { d["boundaries"][0]["rule"] = rule; })), "E0705");
    }
  }
  SECTION("the mesh scale is finite and positive") {
    for (const double scale : {0.0, -0.0254}) {
      CAPTURE(scale);
      require_error(run(input([&](Json& d) { d["mesh"]["scale"] = scale; })), "E0705");
    }
    require_error(run(input([](Json& d) { d["mesh"].erase("scale"); })), "E0705");
  }
  SECTION("monitor names are unique") {
    require_error(run(input([](Json& d) { d["monitors"].push_back(d["monitors"][0]); })), "E0705");
  }
  SECTION("cell zones are described once") {
    require_error(run(input([](Json& d) { d["cell_zones"].push_back(d["cell_zones"][0]); })),
                  "E0705");
  }
  SECTION("periodic pairs are the mesh's, well formed") {
    const Json pair{{"zones", {"xmin", "xmax"}},
                    {"transform", {{"kind", "translation"}, {"vector", {1.0, 0.0, 0.0}}}}};
    // The hybrid mesh has no periodic pair.
    require_error(run(input([&](Json& d) { d["periodic"] = Json::array({pair}); })), "E0705");
    require_error(run(input([&](Json& d) {
                    Json bad = pair;
                    bad["transform"]["kind"] = "rotation";
                    d["periodic"] = Json::array({bad});
                  })),
                  "E0705");
  }
  SECTION("one rank per device: devices.count is the rank count") {
    const Outcome o = run(input([](Json& d) { d["devices"]["count"] = 2; }));
    require_error(o, "E0705");
    CHECK(Json::parse(o.err)["location"]["path"] == "devices.count");
  }
  SECTION("mesh path is absolute") {
    require_error(run(input([](Json& d) { d["mesh"]["path"] = "relative/polyMesh"; })), "E0705");
  }
  SECTION("hashes are 64 lowercase hexadecimal characters") {
    require_error(run(input([](Json& d) { d["case"]["sha256"] = "ABC"; })), "E0705");
  }
  SECTION("a number is a JSON number, alone or in a vector") {
    require_error(run(input([](Json& d) { d["freestream"]["speed"] = "34"; })), "E0705");
    require_error(run(input([](Json& d) { d["freestream"]["velocity"][0] = "1"; })), "E0705");
  }
  SECTION("an ideal gas has gamma above 1") {
    require_error(run(input([](Json& d) { d["cell_zones"][0]["medium"]["gamma"] = 1.0; })),
                  "E0705");
  }
  SECTION("a checkpoint every 60 s at the shortest") {
    const Outcome o =
        run(input([](Json& d) { d["time"]["checkpoint"] = Json{{"seconds", 60.0}}; }));
    INFO(o.err);
    CHECK(o.status == ExitStatus::finished);
  }
  SECTION("averaging starts before the last step") {
    require_error(run(input([](Json& d) { d["time"]["average_from"] = d["time"]["steps"]; },
                            "hybrid-unsteady.solver-input.json")),
                  "E0705");
  }
  SECTION("a number beyond double range") {
    std::string text = input().dump();
    const std::string key = "\"speed\":";
    const auto at = text.find(key) + key.size();
    text.replace(at, text.find(',', at) - at, "1e400");
    const fs::path dir = fresh_dir();
    std::ofstream(dir / "in.json") << text;
    std::FILE* out = std::tmpfile();
    std::FILE* err = std::tmpfile();
    CHECK(autan::runtime::run_case(dir / "in.json", dir / "runs",
                                   autan_test::relaxation_registry({}), build, out,
                                   err) == ExitStatus::invalid_input);
    std::fclose(out);
    std::fclose(err);
  }
}

TEST_CASE("A verification case archives its manufactured solution", "[runtime]") {
  // `autan prepare` with verification = Manufactured("euler.mms.json"): every
  // boundary is manufactured, BCType BCTypeUserDefined in solution.cgns.
  const Outcome o = run(input({}, "hybrid-verification.solver-input.json"));
  REQUIRE(o.status == ExitStatus::finished);
  CHECK(read(o.run / "manufactured.json") == read(mms_path));
  const autan::Mesh back = autan::io::read_mesh(o.run / "solution.cgns");
  for (const auto& zone : back.face_zones()) {
    if (zone.boundary) CHECK(zone.bc_type == "UserDefined");  // SIDS string of BCTypeUserDefined
  }
  SECTION("E0206: the file differs from its hash") {
    require_error(run(input([](Json& d) { d["verification"]["sha256"] = std::string(64, 'a'); },
                            "hybrid-verification.solver-input.json")),
                  "E0206");
  }
  SECTION("E0207: the file is missing") {
    require_error(
        run(input([](Json& d) { d["verification"]["path"] = "/nonexistent/euler.mms.json"; },
                  "hybrid-verification.solver-input.json")),
        "E0207");
  }
  // case-schema.md, `verification`: at most 1 MiB, and no expression whose tree is deeper than
  // 1024, refused with E0208 before the physics reads the document.
  auto with_mms = [](const std::string& text) {
    const fs::path file = fresh_dir() / "solution.mms.json";
    std::ofstream(file, std::ios::binary) << text;
    return input(
        [&](Json& d) {
          d["verification"]["path"] = file.string();
          d["verification"]["sha256"] = file_sha256(file);
        },
        "hybrid-verification.solver-input.json");
  };
  SECTION("E0208: the bounds on the manufactured solution") {
    const std::string fixture = read(mms_path);
    REQUIRE(fixture.size() < (std::size_t{1} << 20U));
    const std::string padded = fixture + std::string((std::size_t{1} << 20U) - fixture.size(), ' ');
    CHECK(run(with_mms(padded)).status == ExitStatus::finished);
    require_error(run(with_mms(padded + " ")), "E0208");
    auto repeat = [](const std::string& s, std::size_t n) {
      std::string out;
      for (std::size_t i = 0; i < n; ++i) out += s;
      return out;
    };
    // An unused subexpression of the given depth.
    auto deep = [&](const std::string& expr) {
      Json doc = Json::parse(fixture);
      doc["fields"]["subexpressions"].push_back(Json::array({"deep", expr}));
      return doc.dump();
    };
    for (const std::size_t n : {std::size_t{1024}, std::size_t{1025}}) {
      INFO("depth " << n);
      const std::array<std::string, 3> shapes = {
          "x" + repeat(" + x", n - 1),                    // left-deep chain
          repeat("(", n - 1) + "x" + repeat(")", n - 1),  // parentheses
          repeat("-", n - 1) + "x",                       // unary minus
      };
      for (const auto& expr : shapes) {
        const Outcome bounded = run(with_mms(deep(expr)));
        if (n == 1024) {
          CHECK(bounded.status == ExitStatus::finished);
        } else {
          require_error(bounded, "E0208");
        }
      }
    }
    // * before +, left to right, and a parenthesis one level above its content: each 1025 deep.
    for (const std::string& expr :
         {"x" + repeat(" * x", 1023) + " + x", "(x" + repeat(" + x", 1023) + ")",
          repeat("(", 1021) + "x" + repeat(")", 1021) + " + x + x + x"}) {
      require_error(run(with_mms(deep(expr))), "E0208");
    }
    // An output is bounded as a subexpression is.
    Json wrapped = Json::parse(fixture);
    Json& rho = wrapped["fields"]["outputs"]["rho"];
    rho = repeat("(", 1024) + rho.get<std::string>() + repeat(")", 1024);
    require_error(run(with_mms(wrapped.dump())), "E0208");
    // Unfinished, the parser would recurse through all of it before reporting the syntax.
    require_error(run(with_mms(deep(repeat("(", 100000) + "x"))), "E0208");
    require_error(run(with_mms(deep(repeat("-", 100000)))), "E0208");
    // A number with a leading dot and an exponent is one leaf, as the parser reads it.
    CHECK(run(with_mms(deep(repeat("(", 1023) + ".1e-2" + repeat(")", 1023)))).status ==
          ExitStatus::finished);
    require_error(run(with_mms(deep(repeat("x*", 512) + ".1e-2" + repeat("*x", 512)))), "E0208");
    require_error(run(with_mms("{not json")), "E0208");
    // A file whose size is not known before reading (a pipe) is bounded by what is read.
    const fs::path pipe = fresh_dir() / "solution.mms.json";
    REQUIRE(mkfifo(pipe.c_str(), 0600) == 0);
    const std::string over = padded + " ";
    std::string over_sha;
    picosha2::hash256_hex_string(over, over_sha);
    std::thread writer([&] { std::ofstream(pipe, std::ios::binary) << over; });
    const Outcome piped = run(input(
        [&](Json& d) {
          d["verification"]["path"] = pipe.string();
          d["verification"]["sha256"] = over_sha;
        },
        "hybrid-verification.solver-input.json"));
    writer.join();
    require_error(piped, "E0208");
  }
  SECTION("E0705: manufactured boundaries without verification") {
    require_error(run(input([](Json& d) { d["verification"] = nullptr; },
                            "hybrid-verification.solver-input.json")),
                  "E0705");
  }
}

TEST_CASE("log.jsonl records the run's events in the results.md shape", "[runtime]") {
  const Outcome o = run(input(), {.factor = 0.9, .diverge_at = 4});
  std::vector<Json> lines;
  std::ifstream in(o.run / "log.jsonl");
  for (std::string line; std::getline(in, line);) lines.push_back(Json::parse(line));
  REQUIRE(lines.size() >= 4);
  const std::regex time("[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]{3}Z");
  const std::regex source("[a-z][a-z0-9-]*");
  for (const Json& l : lines) {
    CHECK(l.size() == 7);
    CHECK(std::regex_match(l["time"].get<std::string>(), time));
    CHECK((l["iteration"].is_null() || l["iteration"].is_number_integer()));
    CHECK((l["level"] == "info" || l["level"] == "warning" || l["level"] == "error"));
    CHECK(std::regex_match(l["source"].get<std::string>(), source));
    CHECK(l["data"].is_object());
  }
  CHECK(lines.front()["iteration"].is_null());
  // The physics' own line, after the runtime's (the cell order).
  CHECK(std::ranges::any_of(
      lines, [](const Json& l) { return l["message"] == "relaxation physics ready"; }));
  // runtime.md#ranks: one rank logs no partition and, in "run", no rank-count event.
  CHECK(std::ranges::none_of(lines, [](const Json& l) { return l["source"] == "mpi"; }));
  CHECK(std::ranges::none_of(lines, [](const Json& l) { return l["data"].contains("mode"); }));
  // The run.json diagnostic appears with its code and data.
  const Json info = Json::parse(read(o.run / "run.json"));
  const auto d0107 =
      std::ranges::find_if(lines, [](const Json& l) { return l["code"] == "D0107"; });
  REQUIRE(d0107 != lines.end());
  CHECK((*d0107)["data"] == info["diagnostics"][0]["data"]);
  CHECK((*d0107)["iteration"] == 4);
  CHECK(lines.back()["data"]["status"] == "diverged");
  CHECK(lines.back()["level"] == "info");
}

TEST_CASE("A continued run goes on bit for bit from the run it continues", "[runtime]") {
  // The state grows at each step, at a rate that rises with the physics' step count kept in its
  // SolverState, so that the history (forces) and the solution depend on the state and the
  // solver state handed over; the uninterrupted run is the reference (results.md: same identity
  // parts, same iterations).
  const bool steady = GENERATE(true, false);
  CAPTURE(steady);
  const autan_test::RelaxationOptions options{.growth = 1e-3, .scalars = 3};
  const auto budget = [&](std::int64_t n) {
    return [steady, n](Json& d) {
      d["reproducibility"] = "run";
      d["monitors"][0]["every"] = 1;
      if (steady)
        d["time"]["max_iterations"] = n;
      else
        d["time"]["steps"] = n;
    };
  };
  const auto doc = [&](std::int64_t n) {
    return steady ? input(budget(n)) : input(budget(n), "hybrid-unsteady.solver-input.json");
  };
  // A chain of three runs. Steady: converges at 28 (see above); measuring
  // from the first iteration of either continuation instead would converge at
  // 38 or later. Unsteady: averaging starts after step 3, in the first run,
  // and goes on through the other two.
  const std::int64_t total = steady ? 10000 : 8;
  const std::int64_t first = steady ? 10 : 4;
  const std::int64_t middle = steady ? 20 : 6;
  const Outcome whole = run(doc(total), options);
  const Outcome a = run(doc(first), options);
  REQUIRE(Json::parse(read(a.run / "run.json"))["status"] == "completed");
  Json next = doc(middle);
  next["start"] = start_from(a.run, "continue");
  const Outcome m = run(next, options);
  REQUIRE(Json::parse(read(m.run / "run.json"))["status"] == "completed");
  Json second = doc(total);
  second["start"] = start_from(m.run, "continue");
  const Outcome b = run(second, options);
  REQUIRE(b.status == ExitStatus::finished);
  // In another cell order than the run it continues, whose bits it would
  // change from the restart on, a continuation is refused.
  REQUIRE(setenv("AUTAN_CELL_ORDER", "input", 1) == 0);
  const Outcome other_order = run(autan_test::planned(second, mesh_path), options);
  unsetenv("AUTAN_CELL_ORDER");
  require_error(other_order, "E0708");
  // A SolverState of other lengths than the physics registers is refused (E0707); a physics that
  // returns other arrays than it registers fails the run (E0704).
  autan_test::RelaxationOptions longer = options;
  longer.scalars = 4;
  require_error(run(second, longer), "E0707");
  autan_test::RelaxationOptions wrong = options;
  wrong.scalars_wrong_rank = 0;
  const Outcome failed = run(doc(first), wrong);
  const Json failed_info = Json::parse(read(failed.run / "run.json"));
  CHECK(failed_info["status"] == "failed");
  CHECK(std::ranges::count_if(failed_info["diagnostics"],
                              [](const Json& d) { return d["code"] == "E0704"; }) == 1);
  // The solution holds the physics' SolverState as it was at the end, rank 0's.
  const auto stored = autan::io::read_solution(b.run / "solution.cgns");
  REQUIRE(stored.solver_state.size() == 2);
  CHECK(stored.solver_state[0].name == "RelaxationSteps");
  CHECK(stored.solver_state[0].values ==
        std::vector<double>{static_cast<double>(
            Json::parse(read(b.run / "run.json"))["iterations"].get<std::int64_t>())});

  const auto all = lines_of(whole.run / "history.jsonl");
  auto chain = lines_of(a.run / "history.jsonl");
  for (const auto& run_dir : {m.run, b.run}) {
    const auto more = lines_of(run_dir / "history.jsonl");
    chain.insert(chain.end(), more.begin(), more.end());
  }
  CHECK(chain == all);
  const Json w = Json::parse(read(whole.run / "run.json"));
  const Json info = Json::parse(read(b.run / "run.json"));
  for (const char* key : {"status", "iterations", "time", "averaging"}) {
    CAPTURE(key);
    CHECK(info[key] == w[key]);
  }
  Json expected_start = second["start"];
  expected_start.erase("run");
  CHECK(info["start"] == expected_start);
  // timing counts this run's iterations only.
  CHECK(info["timing"]["iterations"] ==
        info["iterations"].get<std::int64_t>() - second["start"]["iterations"].get<std::int64_t>());
  // identity: the nine parts, then the start's identity, mode, iterations
  // and solution hash, joined by "\n" (results.md).
  const auto& s = second["start"];
  std::string identity;
  picosha2::hash256_hex_string(second["case"]["sha256"].get<std::string>() + "\n" +
                                   second["mesh"]["sha256"].get<std::string>() + "\n\n" +
                                   build.commit + "\nfp32+acc64\ncpu\ncpu\n1\nrcm\n" +
                                   s["identity"].get<std::string>() + "\ncontinue\n" +
                                   std::to_string(s["iterations"].get<int>()) + "\n" +
                                   s["solution_sha256"].get<std::string>(),
                               identity);
  CHECK(info["identity"] == identity);
  for (const char* field : {"Density", "Pressure", "Temperature"}) {
    CAPTURE(field);
    CHECK(solution_field(b.run / "solution.cgns", "FlowSolution", field) ==
          solution_field(whole.run / "solution.cgns", "FlowSolution", field));
    if (!steady) {
      CHECK(solution_field(b.run / "solution.cgns", "FlowSolutionMean", field) ==
            solution_field(whole.run / "solution.cgns", "FlowSolutionMean", field));
    }
  }
}

TEST_CASE("An initial-state file gives each fluid cell its state", "[runtime]") {
  // initial-state.md: rows paired with cells by centroid, whatever their
  // order; the relaxation physics keeps its state (factor 0.5, no growth),
  // so the solution is the file's values cell for cell.
  const Json state = initial_state_document(mesh_path);
  const Json doc = with_initial_state(input(inviscid), state);
  const Outcome o = run(doc);
  REQUIRE(o.status == ExitStatus::finished);
  const auto n = state["cells"].get<std::size_t>();
  const std::array<std::pair<const char*, std::function<double(double)>>, 5> expected{{
      {"Density", [](double c) { return 1.0 + 1e-3 * c; }},
      {"VelocityX", [](double c) { return c; }},
      {"VelocityY", [](double c) { return -c; }},
      {"VelocityZ", [](double) { return 0.5; }},
      {"Pressure", [](double c) { return 1e5 + c; }},
  }};
  for (const auto& [field, value] : expected) {
    CAPTURE(field);
    const auto got = solution_field(o.run / "solution.cgns", "FlowSolution", field);
    REQUIRE(got.size() == n);
    for (std::size_t c = 0; c < n; ++c) CHECK(got[c] == value(static_cast<double>(c)));
  }
  // Archived byte for byte under its hash.
  const std::string sha = doc["initial_state"]["sha256"];
  CHECK(read(o.run / "states" / (sha + ".json")) ==
        read(doc["initial_state"]["path"].get<std::string>()));

  SECTION("E0508: the file differs from its hash") {
    Json bad = doc;
    bad["initial_state"]["sha256"] = std::string(64, '0');
    require_error(run(bad), "E0508");
  }
  SECTION("E0508: not an initial-state document") {
    Json other = state;
    other["variables"] = {"density", "velocity-x", "velocity-y", "velocity-z", "temperature"};
    require_error(run(with_initial_state(input(inviscid), other)), "E0508");
    // Density and pressure > 0.
    Json empty = state;
    empty["values"][0][0] = 0.0;
    require_error(run(with_initial_state(input(inviscid), empty)), "E0508");
  }
  SECTION("E0508: an unknown key, or a mesh hash that is no SHA-256") {
    Json extra = state;
    extra["note"] = "unknown";
    require_error(run(with_initial_state(input(inviscid), extra)), "E0508");
    Json malformed = state;
    malformed["mesh_sha256"] = "not-a-sha256";
    require_error(run(with_initial_state(input(inviscid), malformed)), "E0508");
    // A rows member inside a row: no second member opens there.
    Json nested = state;
    nested["centroids"][0] = Json::array({Json{{"values", Json::array()}}});
    require_error(run(with_initial_state(input(inviscid), nested)), "E0508");
  }
  SECTION("E0508: a number over 64 characters") {
    // initial-state.md: refused before the parser reads it; the same value in 64 accepted.
    require_error(
        run(with_initial_state(input(inviscid), padded_number(state.dump(), "values", 65))),
        "E0508");
    const Outcome at_limit =
        run(with_initial_state(input(inviscid), padded_number(state.dump(), "values", 64)));
    INFO(at_limit.err);
    CHECK(at_limit.status == ExitStatus::finished);
  }
  SECTION("E0508: an oracle over 4096 bytes") {
    // initial-state.md: the oracle, as every member name, at most 4096 bytes.
    Json long_oracle = state;
    long_oracle["oracle"] = std::string(4097, 'o');
    require_error(run(with_initial_state(input(inviscid), long_oracle)), "E0508");
    Json longest = state;
    longest["oracle"] = std::string(4096, 'o');
    const Outcome at_limit = run(with_initial_state(input(inviscid), longest));
    INFO(at_limit.err);
    CHECK(at_limit.status == ExitStatus::finished);
  }
  SECTION("E0508: a member twice, the last empty") {
    for (const char* member : {"centroids", "values"}) {
      CAPTURE(member);
      require_error(run(with_initial_state(input(inviscid), emptied_twice(state.dump(), member))),
                    "E0508");
    }
  }
  SECTION("E0705: a turbulence model, whose variables the file lacks") {
    require_error(run(with_initial_state(input(), state)), "E0705");
  }
  SECTION("E0509: written for another mesh") {
    Json other = state;
    other["mesh_sha256"] = std::string(64, 'a');
    require_error(run(with_initial_state(input(inviscid), other)), "E0509");
  }
  SECTION("E0510: a row short") {
    Json other = state;
    other["centroids"].erase(other["centroids"].size() - 1);
    other["values"].erase(other["values"].size() - 1);
    other["cells"] = n - 1;
    require_error(run(with_initial_state(input(inviscid), other)), "E0510");
    // One row, the fewest a document may hold.
    Json one = state;
    one["centroids"] = Json::array({state["centroids"][0]});
    one["values"] = Json::array({state["values"][0]});
    one["cells"] = 1;
    require_error(run(with_initial_state(input(inviscid), one)), "E0510");
  }
  SECTION("rows anywhere within the tolerance of their cells") {
    // Each row moved by 0.999 times the tolerance, 1e-9 times the box's 1 m extent, along x, y or
    // z in turn: still its cell's row.
    Json moved = state;
    for (std::size_t r = 0; r < n; ++r) {
      Json& x = moved["centroids"][r][r % 3];
      x = x.get<double>() + 0.999e-9;
    }
    const Outcome near = run(with_initial_state(input(inviscid), moved));
    INFO(near.err);
    REQUIRE(near.status == ExitStatus::finished);
    const auto density = solution_field(near.run / "solution.cgns", "FlowSolution", "Density");
    REQUIRE(density.size() == n);
    for (std::size_t c = 0; c < n; ++c) CHECK(density[c] == 1.0 + 1e-3 * static_cast<double>(c));
  }
  SECTION("E0511: a row far from every cell, or two rows on one cell") {
    Json far = state;
    far["centroids"][3][0] = far["centroids"][3][0].get<double>() + 1e-3;
    require_error(run(with_initial_state(input(inviscid), far)), "E0511");
    Json twice = state;
    twice["centroids"][3] = twice["centroids"][4];
    require_error(run(with_initial_state(input(inviscid), twice)), "E0511");
  }
  SECTION("E0511: one row within the tolerance of two cells, the first of them cell 0") {
    // The hex box flattened to 3 layers 1.5e-9 m thick: the tolerance is 1e-9 times its 1 m
    // extent, so a row half-way between cell 0 and the next cell of its column lies within it of
    // both, and each cell within it of its own row only. That next cell's row is moved away.
    const fs::path hex = data_dir / "mesh" / "hex" / "polyMesh";
    const fs::path flat = fresh_dir() / "polyMesh";
    fs::copy(hex, flat);
    {
      std::ifstream in(hex / "points");
      std::ostringstream out;
      const std::regex point(R"(^\(([^ ]+) ([^ ]+) ([^ ]+)\)$)");
      char buffer[96];
      for (std::string line; std::getline(in, line);) {
        std::smatch m;
        if (std::regex_match(line, m, point)) {
          std::snprintf(buffer, sizeof buffer, "(%s %s %.17g)", m[1].str().c_str(),
                        m[2].str().c_str(), std::stod(m[3]) * 4.5e-9);
          line = buffer;
        }
        out << line << '\n';
      }
      std::ofstream(flat / "points") << out.str();
    }
    Json doc_flat = input([&](Json& d) {
      d["mesh"]["path"] = flat.string();
      d["mesh"]["sha256"] = autan::io::mesh_hash(flat);
      Json zone = d["cell_zones"][0];
      zone["zone"] = "fluid";
      d["cell_zones"] = Json::array({zone});
    });
    inviscid(doc_flat);
    Json shared = initial_state_document(flat);
    const auto rows = shared["cells"].get<std::size_t>();
    const auto centroid = [&](std::size_t c) { return shared["centroids"][rows - 1 - c]; };
    std::size_t above = 1;
    const auto distance = [&](std::size_t c) {
      return std::hypot(centroid(c)[0].get<double>() - centroid(0)[0].get<double>(),
                        centroid(c)[1].get<double>() - centroid(0)[1].get<double>(),
                        centroid(c)[2].get<double>() - centroid(0)[2].get<double>());
    };
    for (std::size_t c = 2; c < rows; ++c) {
      if (distance(c) < distance(above)) above = c;
    }
    REQUIRE(distance(above) > 1e-9);
    REQUIRE(distance(above) < 2e-9);
    Json midpoint = Json::array();
    for (std::size_t k = 0; k < 3; ++k) {
      midpoint.push_back(0.5 * (centroid(0)[k].get<double>() + centroid(above)[k].get<double>()));
    }
    shared["centroids"][rows - 1] = midpoint;
    shared["centroids"][rows - 1 - above] = Json::array({-1.0, -1.0, -1.0});
    require_error(run(with_initial_state(doc_flat, shared)), "E0511");
  }
  SECTION("E0705: initial_state with start") {
    Json both = doc;
    both["start"] = start_from(o.run, "warm");
    require_error(run(both), "E0705");
  }
}

TEST_CASE("A warm start takes the previous state and counts from 1", "[runtime]") {
  const Outcome a = run(input([](Json& d) { d["time"]["max_iterations"] = 5; }),
                        autan_test::RelaxationOptions{.growth = 0.25});
  Json doc = input([](Json& d) { d["time"]["max_iterations"] = 3; });
  doc["start"] = start_from(a.run, "warm");
  const Outcome b = run(doc);  // growth 0: the state stays the one handed over
  REQUIRE(b.status == ExitStatus::finished);
  const auto lines = history(b.run);
  REQUIRE(lines.size() == 3);
  CHECK(lines[0]["iteration"] == 1);
  const Json info = Json::parse(read(b.run / "run.json"));
  CHECK(info["iterations"] == 3);
  CHECK(info["start"]["mode"] == "warm");
  CHECK(info["start"]["residual_reference"] == history(a.run)[0]["residuals"]);
  CHECK(info["start"]["woken"] == Json::array());
  const auto previous = solution_field(a.run / "solution.cgns", "FlowSolution", "Density");
  CHECK(solution_field(b.run / "solution.cgns", "FlowSolution", "Density") == previous);
  // 1.25^5 times the initial density: not the freestream state.
  CHECK_THAT(previous[0],
             WithinRel(doc["freestream"]["density"].get<double>() * std::pow(1.25, 5.0), 1e-14));
}

TEST_CASE("A warm start converges against the reference of the run it starts from", "[runtime]") {
  // case-schema.md#time: a steady warm start from a steady run stops at the first iteration where
  // every residual has dropped residual_orders below the source's reference, not below its own
  // first residual.
  const auto doc = [](std::int64_t n) {
    return input([n](Json& d) {
      d["reproducibility"] = "run";
      d["time"]["max_iterations"] = n;
    });
  };
  const Outcome a = run(doc(10));
  REQUIRE(a.status == ExitStatus::finished);
  Json next = doc(1000);
  next["start"] = start_from(a.run, "warm");
  const Outcome b = run(next);
  INFO(b.err);
  REQUIRE(b.status == ExitStatus::finished);
  const Json info = Json::parse(read(b.run / "run.json"));
  CHECK(info["status"] == "converged");
  const Json reference = history(a.run)[0]["residuals"];
  CHECK(info["start"]["residual_reference"] == reference);
  const double drop = std::pow(10.0, -next["time"]["residual_orders"].get<double>());
  double largest = 0.0;
  for (const auto& [name, r] : reference.items()) largest = std::max(largest, r.get<double>());
  const auto met = [&](const Json& line) {
    for (const auto& [name, r] : reference.items()) {
      const double v = line["residuals"][name].get<double>();
      // An identically zero equation (below 1e-12 of the largest) passes at its floor.
      if (r.get<double>() < 1e-12 * largest ? v > 1e-12 * largest : v > r.get<double>() * drop)
        return false;
    }
    return true;
  };
  const auto lines = history(b.run);
  REQUIRE(lines.size() >= 2);
  CHECK(met(lines.back()));
  CHECK_FALSE(met(lines[lines.size() - 2]));
}

TEST_CASE("A warm start begins with no equation woken and wakes against its reference's floor",
          "[runtime]") {
  // case-schema.md#time: momentum-x is silent in the reference, so exempt; the source woke it
  // (iterations 10-14), but a warm start begins with none woken and reports its own W0710 when
  // momentum-x rises above the floor of the reference it takes.
  autan_test::RelaxationOptions source;
  source.silent_equation = 1;
  source.wake_at = 10;
  source.sleep_at = 15;
  const auto doc = [](std::int64_t n) {
    return input([n](Json& d) {
      d["reproducibility"] = "run";
      d["time"]["max_iterations"] = n;
    });
  };
  const Outcome a = run(doc(20), source);
  REQUIRE(a.status == ExitStatus::finished);
  Json next = doc(40);
  next["start"] = start_from(a.run, "warm");
  CHECK(next["start"]["woken"] == Json::array());
  autan_test::RelaxationOptions warm;
  warm.silent_equation = 1;
  warm.wake_at = 3;
  warm.sleep_at = 6;
  const Outcome b = run(next, warm);
  INFO(b.err);
  REQUIRE(b.status == ExitStatus::finished);
  const Json info = Json::parse(read(b.run / "run.json"));
  CHECK(info["start"]["residual_reference"] == history(a.run)[0]["residuals"]);
  std::vector<Json> wakes;
  for (const auto& d : info["diagnostics"]) {
    if (d["code"] == "W0710") wakes.push_back(d["data"]);
  }
  REQUIRE(wakes.size() == 1);
  CHECK(wakes[0]["equation"] == "momentum-x");
  CHECK(wakes[0]["iteration"] == 3);
}

TEST_CASE("A start from a run with the longest case name and mesh path is honoured", "[runtime]") {
  // results.md, RunMetadata: a case name of 4096 control bytes, which ensure_ascii writes as
  // \u00XX (6 bytes each), and a mesh path as long as the system allows (about 4000 bytes on
  // Linux; PATH_MAX is 1024 on macOS) stay under its 64 KiB.
  const std::size_t longest = std::min<std::size_t>(3800, PATH_MAX - 300);
  fs::path deep = fresh_dir();
  while (deep.string().size() < longest) deep /= std::string(200, 'd');
  fs::create_directories(deep);
  const fs::path mesh = deep / "polyMesh";
  fs::copy(data_dir / "mesh" / "hybrid" / "polyMesh", mesh, fs::copy_options::recursive);
  const auto edit = [&](Json& d) {
    d["case"]["name"] = std::string(4096, '\x01');
    d["mesh"]["path"] = fs::absolute(mesh).string();
    d["mesh"]["sha256"] = autan::io::mesh_hash(mesh);
    d["time"]["max_iterations"] = 2;
  };
  const Outcome a = run(input(edit));
  REQUIRE(a.status == ExitStatus::finished);
  Json doc = input(edit);
  doc["start"] = start_from(a.run, "warm");
  CHECK(run(doc).status == ExitStatus::finished);
}

TEST_CASE("A start the runtime cannot honour exits 2 before any directory", "[runtime]") {
  const Outcome a = run(input([](Json& d) { d["time"]["max_iterations"] = 4; }));
  const auto from = [&](const std::function<void(Json&)>& edit) {
    Json doc = input();
    doc["start"] = start_from(a.run, "continue");
    edit(doc);
    return run(doc);
  };
  SECTION("the solution changed since the SDK hashed it") {
    require_error(from([](Json& d) { d["start"]["solution_sha256"] = std::string(64, '0'); }),
                  "E0709");
  }
  SECTION("the previous run has no solution") {
    require_error(from([&](Json& d) { d["start"]["run"] = fresh_dir().string(); }), "E0707");
  }
  SECTION("the previous run's solution is not a regular file") {
    // A directory reads nothing without an error on libc++; a named pipe blocks its reader.
    const fs::path directory = fresh_dir();
    fs::create_directory(directory / "solution.cgns");
    require_error(from([&](Json& d) { d["start"]["run"] = directory.string(); }), "E0707");
    const fs::path pipe = fresh_dir();
    REQUIRE(::mkfifo((pipe / "solution.cgns").c_str(), 0600) == 0);
    require_error(from([&](Json& d) { d["start"]["run"] = pipe.string(); }), "E0707");
  }
  SECTION("the previous run used another precision") {
    require_error(from([](Json& d) { d["precision"] = "fp64"; }), "E0708");
  }
  SECTION("a solution holding what the writer does not write") {
    // results.md, Sizes a reader accepts: a descriptor beside RunMetadata and ZoneNames, one
    // family more than the mesh has zones, or a RestartState holding one array more than the
    // physics writes is refused before the CGNS library loads the file or any array.
    const fs::path solution = a.run / "solution.cgns";
    const int kind = GENERATE(0, 1, 2);
    CAPTURE(kind);
    int fn = 0;
    REQUIRE(cg_open(solution.string().c_str(), CG_MODE_MODIFY, &fn) == CG_OK);
    if (kind == 0) {
      REQUIRE(cg_goto(fn, 1, "end") == CG_OK);
      REQUIRE(cg_descriptor_write("About", "a note") == CG_OK);
    } else if (kind == 1) {
      int family = 0;
      REQUIRE(cg_family_write(fn, 1, "Extra", &family) == CG_OK);
    } else {
      char name[33];
      cgsize_t size[3] = {};
      REQUIRE(cg_zone_read(fn, 1, 1, name, size) == CG_OK);
      int solutions = 0, restart = 0;
      REQUIRE(cg_nsols(fn, 1, 1, &solutions) == CG_OK);
      for (int s = 1; s <= solutions; ++s) {
        CGNS_ENUMT(GridLocation_t) location{};
        REQUIRE(cg_sol_info(fn, 1, 1, s, name, &location) == CG_OK);
        if (std::string_view(name) == "RestartState") restart = s;
      }
      REQUIRE(restart > 0);
      const std::vector<double> zeros(static_cast<std::size_t>(size[1]), 0.0);
      int field = 0;
      REQUIRE(cg_field_write(fn, 1, 1, restart, CGNS_ENUMV(RealDouble), "Extra", zeros.data(),
                             &field) == CG_OK);
    }
    REQUIRE(cg_close(fn) == CG_OK);
    require_error(from([](Json&) {}), "E0707");
  }
  SECTION("RunMetadata of 64 KiB, not 64 KiB + 1") {
    // results.md, RunMetadata: at most 64 KiB, sized before it is read. The previous run's own
    // metadata, padded with spaces after its opening brace: the same document.
    const fs::path solution = a.run / "solution.cgns";
    for (const std::size_t bytes : {std::size_t{64} << 10, (std::size_t{64} << 10) + 1}) {
      CAPTURE(bytes);
      int fn = 0;
      REQUIRE(cg_open(solution.string().c_str(), CG_MODE_MODIFY, &fn) == CG_OK);
      REQUIRE(cg_goto(fn, 1, "end") == CG_OK);
      int count = 0;
      REQUIRE(cg_ndescriptors(&count) == CG_OK);
      std::string metadata;
      for (int d = 1; d <= count; ++d) {
        char name[33];
        char* text = nullptr;
        REQUIRE(cg_descriptor_read(d, name, &text) == CG_OK);
        if (std::string_view(name) == "RunMetadata") metadata = Json::parse(text).dump();
        cg_free(text);
      }
      REQUIRE(!metadata.empty());
      REQUIRE(metadata.size() < bytes);
      const std::string padded =
          "{" + std::string(bytes - metadata.size(), ' ') + metadata.substr(1);
      REQUIRE(cg_descriptor_write("RunMetadata", padded.c_str()) == CG_OK);
      REQUIRE(cg_close(fn) == CG_OK);
      const Outcome o = from([](Json&) {});
      if (bytes == std::size_t{64} << 10) {
        CHECK(o.status == ExitStatus::finished);
      } else {
        require_error(o, "E0707");
      }
    }
  }
  SECTION("the previous run solved other variables") {
    require_error(from([](Json& d) {
                    d["start"]["mode"] = "warm";
                    d["start"]["residual_reference"].erase("nu-tilde");
                    d["start"]["woken"] = Json::array();
                    d["physics"]["turbulence"]["model"] = "laminar";
                    d["physics"]["turbulence"]["nu_tilde_ratio"] = nullptr;
                    for (auto& z : d["cell_zones"]) z["initial"]["nu_tilde"] = nullptr;
                    d["freestream"]["nu_tilde"] = nullptr;
                  }),
                  "E0708");
  }
  SECTION("the previous run was computed on a mesh of other cells") {
    const fs::path hex = data_dir / "mesh" / "hex" / "polyMesh";
    Json doc = input([&](Json& d) {
      d["mesh"]["path"] = fs::absolute(hex).string();
      d["mesh"]["sha256"] = autan::io::mesh_hash(hex);
    });
    doc["start"] = start_from(a.run, "continue");
    require_error(run(doc), "E0203");
  }
  SECTION("the previous solution lacks run metadata the continuation needs") {
    // results.md, RunMetadata: `averaging` is null or the four bounds of the interval. The
    // previous run's own solution, its RunMetadata without `averaging` or with one bound only.
    const bool absent = GENERATE(true, false);
    CAPTURE(absent);
    const fs::path solution = a.run / "solution.cgns";
    int fn = 0;
    REQUIRE(cg_open(solution.string().c_str(), CG_MODE_MODIFY, &fn) == CG_OK);
    REQUIRE(cg_goto(fn, 1, "end") == CG_OK);
    int count = 0;
    REQUIRE(cg_ndescriptors(&count) == CG_OK);
    Json metadata;
    for (int d = 1; d <= count; ++d) {
      char name[33];
      char* text = nullptr;
      REQUIRE(cg_descriptor_read(d, name, &text) == CG_OK);
      if (std::string_view(name) == "RunMetadata") metadata = Json::parse(text);
      cg_free(text);
    }
    REQUIRE(metadata.contains("averaging"));
    if (absent) {
      metadata.erase("averaging");
    } else {
      metadata["averaging"] = Json{{"from_step", 1}};
    }
    REQUIRE(cg_descriptor_write("RunMetadata", metadata.dump().c_str()) == CG_OK);
    REQUIRE(cg_close(fn) == CG_OK);
    require_error(from([](Json&) {}), "E0707");
  }
  SECTION("the previous solution keeps data in another file") {
    // mesh.md, E0134, in each form (tests/io/outside_data.hpp), refused from the file's
    // structure alone: the donor is gone when the run starts.
    const auto form =
        GENERATE(autan_test::Outside::cgns_link, autan_test::Outside::external_link,
                 autan_test::Outside::external_storage, autan_test::Outside::virtual_dataset);
    CAPTURE(autan_test::outside_name(form));
    const fs::path previous = fresh_dir();
    fs::copy_file(a.run / "solution.cgns", previous / "solution.cgns");
    std::string control;
    const std::string node = autan_test::move_outside(previous / "solution.cgns", form,
                                                      previous / "donor", 1.5, control);
    for (const double v : autan_test::read_doubles(previous / "solution.cgns", control))
      CHECK(v == 1.5);
    fs::remove(previous / "donor");
    const Outcome o = from([&](Json& d) {
      d["start"]["run"] = previous.string();
      d["start"]["solution_sha256"] = file_sha256(previous / "solution.cgns");
    });
    require_error(o, "E0134");
    const Json refusal = Json::parse(o.err.substr(0, o.err.find('\n')));
    CHECK(refusal["data"]["node"] == node);
    CHECK(refusal["location"]["path"] == "start.run");
  }
  SECTION("the previous run was computed on another mesh") {
    // The same cells, but another file: its hash differs.
    const fs::path copy = fresh_dir() / "polyMesh";
    fs::copy(mesh_path, copy);
    std::ofstream(copy / "boundary", std::ios::app) << "\n// copy\n";
    Json doc = input([&](Json& d) {
      d["mesh"]["path"] = copy.string();
      d["mesh"]["sha256"] = autan::io::mesh_hash(copy);
    });
    doc["start"] = start_from(a.run, "continue");
    require_error(run(doc), "E0203");
  }
  SECTION("the solver-input rules of start") {
    require_error(from([](Json& d) { d["start"]["residual_reference"] = nullptr; }), "E0705");
    // A steady warm start from a steady run carries the reference; from an unsteady one, none.
    require_error(from([](Json& d) {
                    d["start"]["mode"] = "warm";
                    d["start"]["residual_reference"] = nullptr;
                    d["start"]["woken"] = nullptr;
                  }),
                  "E0705");
    require_error(from([](Json& d) {
                    d["start"]["mode"] = "warm";
                    d["start"]["time"] = 0.5;
                  }),
                  "E0705");
    require_error(from([](Json& d) {
                    d["start"]["mode"] = "warm";
                    d["start"]["woken"] = Json::array({"momentum-y"});
                  }),
                  "E0705");
    require_error(from([](Json& d) { d["time"]["max_iterations"] = 4; }), "E0705");
    require_error(from([](Json& d) { d["start"]["run"] = "relative/run"; }), "E0705");
    require_error(from([](Json& d) { d["start"]["residual_reference"].erase("energy"); }), "E0705");
    require_error(from([](Json& d) { d["start"]["woken"] = nullptr; }), "E0705");
    require_error(from([](Json& d) { d["start"]["woken"] = Json::array({"vorticity"}); }), "E0705");
    require_error(from([](Json& d) { d["start"]["woken"] = Json::array({"energy", "density"}); }),
                  "E0705");
    require_error(from([](Json& d) { d["start"]["time"] = 0.0; }), "E0705");
    const Outcome u =
        run(input([](Json& d) { d["time"]["steps"] = 4; }, "hybrid-unsteady.solver-input.json"));
    REQUIRE(u.status == ExitStatus::finished);
    const auto unsteady = [&](double time) {
      Json doc =
          input([](Json& d) { d["time"]["steps"] = 6; }, "hybrid-unsteady.solver-input.json");
      doc["start"] = start_from(u.run, "continue");
      doc["start"]["time"] = time;
      return run(doc);
    };
    require_error(unsteady(-1.0), "E0705");
    CHECK(unsteady(0.0).status == ExitStatus::finished);
    require_error(from([](Json& d) { d["start"]["woken"] = Json::array({"density", "density"}); }),
                  "E0705");
  }
}

TEST_CASE("A run reads nothing on its standard input and runs with it closed", "[runtime]") {
  // solver-input.md, Invocation: the run's own files may then take descriptor 0.
  const fs::path dir = fresh_dir();
  const fs::path input_file = dir / "solver-input.json";
  std::ofstream(input_file) << input().dump();
  std::FILE* out = std::tmpfile();
  std::FILE* err = std::tmpfile();
  const int saved = ::dup(0);
  REQUIRE(saved >= 0);
  ::close(0);
  const ExitStatus status = autan::runtime::run_case(
      input_file, dir / "runs", autan_test::relaxation_registry({}), build, out, err, {}, {});
  REQUIRE(::dup2(saved, 0) == 0);
  ::close(saved);
  std::fclose(out);
  std::fclose(err);
  CHECK(status == ExitStatus::finished);
}

TEST_CASE("A periodic mesh runs through its image ghosts", "[runtime][periodic]") {
  // periodic_input(1.0). The relaxation physics
  // checks at every step that the exchange gives each image ghost its
  // owner's value; the run converges as on any mesh (k = 28) and its
  // solution.cgns gives the pairs back.
  const Outcome o = run(periodic_input(1.0));
  REQUIRE(o.status == ExitStatus::finished);
  const Json info = Json::parse(read(o.run / "run.json"));
  CHECK(info["status"] == "converged");
  CHECK(info["iterations"] == 28);
  // The physics ran on the partition: image ghosts, as many as the
  // relaxation physics reported in its first log line.
  std::ifstream log_in(o.run / "log.jsonl");
  std::int64_t ghosts = -1;
  for (std::string line; std::getline(log_in, line);) {
    const Json event = Json::parse(line);
    if (event["message"] == "relaxation physics ready") ghosts = event["data"]["ghosts"];
  }
  CHECK(ghosts > 0);
  const autan::Mesh back = autan::io::read_mesh(o.run / "solution.cgns");
  CHECK(back.periodic_pairs().size() == 2);
  CHECK(!solution_field(o.run / "solution.cgns", "FlowSolution", "Density").empty());

  require_error(run(periodic_input(2.0)), "E0705");
  // A periodic zone takes no condition and carries no forces monitor.
  require_error(
      run(periodic_input(1.0, [](Json& d) { d["monitors"][0]["zones"] = Json::array({"ymin"}); })),
      "E0705");
  // A periodic zone takes no condition.
  require_error(run(periodic_input(1.0,
                                   [](Json& d) {
                                     Json b = d["boundaries"][0];
                                     b["zone"] = "xmin";
                                     d["boundaries"].push_back(b);
                                   })),
                "E0705");
}

TEST_CASE("A run moves periodic nodes matched within the tolerance and reports it",
          "[runtime][periodic]") {
  // mesh.md, Periodic pairs: periodic-box with every node of xmax (x = 1) moved by 5e-10 in z,
  // far within the matching tolerance (the reader checks the $Periodic pairs to 1e-9 of the
  // translation). The run moves one side of each matched class back, as many
  // nodes as xmax holds, and reports W0150 with them.
  const fs::path box = data_dir / "mesh" / "periodic-box" / "periodic-box.ascii.msh";
  std::istringstream in(read(box));
  std::string out;
  std::int64_t on_xmax = 0;
  bool nodes = false;
  for (std::string line; std::getline(in, line);) {
    if (line == "$Nodes") nodes = true;
    if (line == "$EndNodes") nodes = false;
    std::istringstream fields(line);
    double x = 0.0, y = 0.0, z = 0.0;
    std::string rest;
    if (nodes && (fields >> x >> y >> z) && !(fields >> rest) && x == 1.0) {
      ++on_xmax;
      std::array<char, 96> buffer{};
      std::snprintf(buffer.data(), buffer.size(), "%.17g %.17g %.17g", x, y, z + 5e-10);
      line = buffer.data();
    }
    out += line + "\n";
  }
  REQUIRE(on_xmax > 0);
  const fs::path moved = fresh_dir() / "periodic-box.ascii.msh";
  std::ofstream(moved, std::ios::binary) << out;
  const Outcome o = run(periodic_input(1.0, [&](Json& d) {
    d["mesh"]["path"] = fs::absolute(moved).string();
    d["mesh"]["sha256"] = autan::io::mesh_hash(moved);
  }));
  INFO(o.err);
  REQUIRE(o.status == ExitStatus::finished);
  const Json info = Json::parse(read(o.run / "run.json"));
  Json reported;
  for (const auto& d : info["diagnostics"]) {
    if (d["code"] == "W0150") reported = d["data"];
  }
  REQUIRE(reported.is_object());
  CHECK(reported["nodes"] == on_xmax);
  CHECK(reported["max_relative"].get<double>() > 0.0);
  CHECK(reported["max_relative"].get<double>() <= 1.0);
}

// face-states.md: tests/data/mesh/ringleb, its states written by autan-vv, on the four in-plane
// sides; `edit` changes the document, then `rewrite` its compact text, before it is written
// beside the input.
Json ringleb_input(const std::function<void(Json&)>& edit = {},
                   const std::function<std::string(std::string)>& rewrite = {}) {
  const fs::path mesh = data_dir / "mesh" / "ringleb" / "ringleb-n004.msh";
  Json states = Json::parse(read(data_dir / "mesh" / "ringleb" / "ringleb-n004.states.json"));
  std::string bytes = read(data_dir / "mesh" / "ringleb" / "ringleb-n004.states.json");
  if (edit) edit(states);
  if (edit || rewrite) bytes = states.dump();
  if (rewrite) bytes = rewrite(bytes);
  const fs::path file = fresh_dir() / "ringleb.states.json";
  std::ofstream(file, std::ios::binary) << bytes;
  std::string sha;
  picosha2::hash256_hex_string(bytes, sha);
  return input([&](Json& d) {
    d["mesh"]["path"] = fs::absolute(mesh).string();
    d["mesh"]["sha256"] = autan::io::mesh_hash(mesh);
    d["physics"]["turbulence"]["model"] = "inviscid";
    d["physics"]["turbulence"]["nu_tilde_ratio"] = nullptr;
    d["freestream"]["nu_tilde"] = nullptr;
    Json zone = d["cell_zones"][0];
    zone["zone"] = "fluid";
    zone["initial"]["nu_tilde"] = nullptr;
    d["cell_zones"] = Json::array({zone});
    Json boundaries = Json::array();
    for (const char* name : {"k-min", "k-max", "inflow", "outflow", "back", "front"}) {
      Json b = d["boundaries"][0];
      for (auto& [key, value] : b.items()) {
        if (key != "zone" && key != "kind") value = nullptr;
      }
      b["zone"] = name;
      const bool sides = std::string(name) != "back" && std::string(name) != "front";
      b["kind"] = sides ? "oracle-state" : "symmetry";
      if (sides) b["states"] = Json{{"path", fs::absolute(file).string()}, {"sha256", sha}};
      boundaries.push_back(b);
    }
    d["boundaries"] = boundaries;
    d["monitors"][0]["zones"] = {"k-min"};
  });
}

TEST_CASE("Face states pair with their zone's faces and are archived with the run", "[runtime]") {
  const Json doc = ringleb_input();
  const Outcome o = run(doc);
  INFO(o.err);
  REQUIRE(o.status == ExitStatus::finished);
  const std::string sha = doc["boundaries"][0]["states"]["sha256"];
  CHECK(read(o.run / "states" / (sha + ".json")) ==
        read(data_dir / "mesh" / "ringleb" / "ringleb-n004.states.json"));
  // results.md, BCType: an oracle-state zone is BCDirichlet in solution.cgns.
  const autan::Mesh back = autan::io::read_mesh(o.run / "solution.cgns");
  int oracle_zones = 0;
  for (const auto& zone : back.face_zones()) {
    if (!zone.boundary || zone.name == "back" || zone.name == "front") continue;
    CHECK(zone.bc_type == "BCDirichlet");
    ++oracle_zones;
  }
  CHECK(oracle_zones == 4);
}

TEST_CASE("Face states that do not fit the case exit 2 before any directory", "[runtime]") {
  SECTION("E0420 another file than states.sha256") {
    Json doc = ringleb_input();
    doc["boundaries"][0]["states"]["sha256"] = std::string(64, '0');
    require_error(run(doc), "E0420");
  }
  SECTION("E0420 not a face-state document, whatever zone or key is wrong") {
    require_error(run(ringleb_input([](Json& s) { s["variables"][4] = "temperature"; })), "E0420");
    require_error(run(ringleb_input([](Json& s) { s["schema"] = nullptr; })), "E0420");
    require_error(run(ringleb_input([](Json& s) { s["extra"] = 1; })), "E0420");
    require_error(run(ringleb_input([](Json& s) { s.erase("oracle"); })), "E0420");
    require_error(run(ringleb_input([](Json& s) { s["mesh_sha256"] = "abc"; })), "E0420");
    require_error(run(ringleb_input([](Json& s) { s["zones"] = Json::array(); })), "E0420");
    // face-states.md: the oracle and every member name, zone names included, at most 4096 bytes;
    // a longer zone name is refused as text before it is matched with the mesh's zones (E0421).
    require_error(run(ringleb_input([](Json& s) { s["oracle"] = std::string(4097, 'o'); })),
                  "E0420");
    require_error(run(ringleb_input(
                      [](Json& s) { s["zones"][std::string(4097, 'z')] = s["zones"]["inflow"]; })),
                  "E0420");
    const Outcome longest =
        run(ringleb_input([](Json& s) { s["oracle"] = std::string(4096, 'o'); }));
    INFO(longest.err);
    CHECK(longest.status == ExitStatus::finished);
    // A number of at most 64 characters: refused before the parser reads it, the same value in
    // 64 characters accepted.
    require_error(
        run(ringleb_input({}, [](std::string b) { return padded_number(b, "values", 65); })),
        "E0420");
    const Outcome padded =
        run(ringleb_input({}, [](std::string b) { return padded_number(b, "values", 64); }));
    INFO(padded.err);
    CHECK(padded.status == ExitStatus::finished);
#if defined(__linux__)
    // Nested deeper than its structure (5 levels): refused before the parser keeps a level, so
    // 2^22 levels cost what as long a number does, which the scan refuses as well, not the
    // parser's stacks.
    constexpr std::size_t levels = std::size_t{1} << 22;
    const auto oracle_as = [&](const std::string& value) {
      return ringleb_input({}, [&](std::string b) {
        const std::size_t at = b.find("\"oracle\":");
        REQUIRE(at != std::string::npos);
        const std::size_t first = at + 9, end = b.find(',', first);
        return b.replace(first, end - first, value);
      });
    };
    const Json deep = oracle_as(std::string(levels, '[') + std::string(levels, ']'));
    const Json digits = oracle_as(std::string(2 * levels, '1'));
    const std::int64_t baseline = peak_of([&] { require_error(run(digits), "E0420"); });
    const std::int64_t peak = peak_of([&] { require_error(run(deep), "E0420"); });
    CAPTURE(baseline, peak);
    CHECK(peak - baseline <= std::int64_t{1} << 20);
    // A string longer than 4096 bytes written in 6 each: refused before the lexer, which holds
    // a token twice, reads it; it costs what as long a number does.
    const Json string = oracle_as("\"" + std::string(2 * levels - 2, 'o') + "\"");
    const std::int64_t text = peak_of([&] { require_error(run(string), "E0420"); });
    CAPTURE(text);
    CHECK(text - baseline <= std::int64_t{1} << 20);
    // More than 16 384 zones, each a valid empty zone the run does not use: refused as text
    // (E0420), before any zone is read.
    require_error(run(ringleb_input([](Json& s) {
                    for (int z = 0; z <= 16384; ++z) {
                      s["zones"]["z" + std::to_string(z)] = {
                          {"faces", 0}, {"centroids", Json::array()}, {"values", Json::array()}};
                    }
                  })),
                  "E0420");
    // 16 384 zones, one named "values", and the oracle named "values": past the scan, which
    // counts rows members, not strings, to the mesh hash (E0422).
    require_error(run(ringleb_input([](Json& s) {
                    s["oracle"] = "values";
                    s["mesh_sha256"] = std::string(64, '0');
                    for (auto z = s["zones"].size(); z < 16384; ++z) {
                      const std::string name = z == 16383 ? "values" : "z" + std::to_string(z);
                      s["zones"][name] = {
                          {"faces", 0}, {"centroids", Json::array()}, {"values", Json::array()}};
                    }
                    REQUIRE(s["zones"].size() == 16384);
                  })),
                  "E0422");
#endif
    // Rows, read as they stream: each an array of 3 (centroids) or 5 (values) finite numbers,
    // one per face of the zone.
    for (const auto& edit : std::vector<std::function<void(Json&)>>{
             [](Json& rows) { rows["centroids"][0].push_back(0.0); },
             [](Json& rows) { rows["values"][1][2] = "fast"; },
             [](Json& rows) { rows["values"][0][0] = Json::array({1.0}); },
             [](Json& rows) { rows["centroids"].erase(rows["centroids"].size() - 1); },
             [](Json& rows) { rows["values"] = Json::object(); },
             // A rows member inside a row: no second member opens there.
             [](Json& rows) {
               rows["centroids"][0] = Json::array({Json{{"values", Json::array()}}});
             }}) {
      require_error(run(ringleb_input([&](Json& s) { edit(s["zones"]["inflow"]); })), "E0420");
    }
    // Density and pressure > 0.
    require_error(run(ringleb_input([](Json& s) { s["zones"]["inflow"]["values"][0][0] = 0.0; })),
                  "E0420");
    require_error(run(ringleb_input([](Json& s) {
                    s["zones"]["unused"] =
                        Json{{"faces", "two"}, {"centroids", {}}, {"values", {}}};
                  })),
                  "E0420");
  }
  SECTION("E0420 a zone's member twice, the last empty") {
    for (const char* member : {"centroids", "values"}) {
      CAPTURE(member);
      require_error(run(ringleb_input({},
                                      [&](std::string text) {
                                        return emptied_twice(text, member, text.find("\"inflow\""));
                                      })),
                    "E0420");
    }
  }
  SECTION("E0420 a path that is not a regular file") {
    Json doc = ringleb_input();
    doc["boundaries"][0]["states"]["path"] = fresh_dir().string();
    require_error(run(doc), "E0420");
  }
  SECTION("E0422 exported for another mesh") {
    require_error(run(ringleb_input([](Json& s) { s["mesh_sha256"] = std::string(64, 'a'); })),
                  "E0422");
  }
  SECTION("E0423 a zone missing or with another face count") {
    require_error(run(ringleb_input([](Json& s) { s["zones"].erase("inflow"); })), "E0423");
    require_error(run(ringleb_input([](Json& s) {
                    auto& z = s["zones"]["inflow"];
                    z["faces"] = z["faces"].get<int>() - 1;
                    z["centroids"].erase(z["centroids"].size() - 1);
                    z["values"].erase(z["values"].size() - 1);
                  })),
                  "E0423");
  }
  SECTION("E0421 a centroid away from every face") {
    // The zone spans about 1 m: 1e-6 m is far beyond the 1e-9 x extent tolerance.
    require_error(run(ringleb_input([](Json& s) {
                    auto& c = s["zones"]["k-max"]["centroids"][1];
                    c[0] = c[0].get<double>() + 1e-6;
                  })),
                  "E0421");
  }
}

TEST_CASE("Face states pair on a plane normal to an axis, in any row order", "[runtime]") {
  // The xmin side of the hybrid mesh: every centroid has x = 0. The rows, written here from the
  // mesh's geometry in reverse face order, pair with the faces by centroid (face-states.md).
  const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
  const autan::Geometry g = autan::compute_geometry(mesh);
  autan::index_t zone = -1;
  for (std::size_t z = 0; z < mesh.face_zones().size(); ++z) {
    if (mesh.face_zones()[z].name == "xmin") zone = static_cast<autan::index_t>(z);
  }
  REQUIRE(zone >= 0);
  const auto off = mesh.boundary_zone_offsets();
  Json centroids = Json::array(), values = Json::array();
  for (auto f = off[static_cast<std::size_t>(zone) + 1];
       f-- > off[static_cast<std::size_t>(zone)];) {
    const auto i = static_cast<std::size_t>(f);
    centroids.push_back({g.face_centroid_x[i], g.face_centroid_y[i], g.face_centroid_z[i]});
    values.push_back({1.2, 100.0, 0.0, 0.0, 1e5});
  }
  const Json states = {
      {"schema", "autan.face-states/1"},
      {"oracle", "uniform"},
      {"mesh_sha256", autan::io::mesh_hash(mesh_path)},
      {"variables", {"density", "velocity-x", "velocity-y", "velocity-z", "pressure"}},
      {"zones",
       {{"xmin", {{"faces", centroids.size()}, {"centroids", centroids}, {"values", values}}}}}};
  const std::string bytes = states.dump();
  const fs::path file = fresh_dir() / "xmin.states.json";
  std::ofstream(file, std::ios::binary) << bytes;
  std::string sha;
  picosha2::hash256_hex_string(bytes, sha);
  const Json doc = input([&](Json& d) {
    d["physics"]["turbulence"]["model"] = "inviscid";
    d["physics"]["turbulence"]["nu_tilde_ratio"] = nullptr;
    d["freestream"]["nu_tilde"] = nullptr;
    for (auto& z : d["cell_zones"]) z["initial"]["nu_tilde"] = nullptr;
    for (auto& b : d["boundaries"]) {
      if (b["kind"] == "wall") b["kind"] = "slip";
      if (b["zone"] == "xmin") {
        b["kind"] = "oracle-state";
        b["states"] = Json{{"path", file.string()}, {"sha256", sha}};
      }
    }
  });
  const Outcome o = run(doc);
  INFO(o.err);
  CHECK(o.status == ExitStatus::finished);
}

TEST_CASE("An equation at round-off from iteration 1 does not hold back convergence", "[runtime]") {
  // Residuals (e + 1) 0.5^(k-1) drop 8 orders at k = 28; momentum-x stays at
  // 1e-15, below the floor 1e-12 x 6 (case-schema.md#time), so it is exempt.
  autan_test::RelaxationOptions quiet;
  quiet.silent_equation = 1;
  const Outcome o = run(input([](Json& d) { d["time"]["max_iterations"] = 100; }), quiet);
  REQUIRE(o.status == ExitStatus::finished);
  const Json info = Json::parse(read(o.run / "run.json"));
  CHECK(info["status"] == "converged");
  CHECK(info["iterations"] == 28);

  // From iteration 10 it is 1e-3, above the floor: W0710 once. It then drops from the largest
  // momentum residual at iteration 1, momentum-z's 4 (#432): 1e-3 is above 4e-8, so the run
  // cannot converge.
  autan_test::RelaxationOptions waking = quiet;
  waking.wake_at = 10;
  const Outcome w = run(input([](Json& d) { d["time"]["max_iterations"] = 100; }), waking);
  REQUIRE(w.status == ExitStatus::finished);
  CHECK(Json::parse(read(w.run / "run.json"))["status"] == "completed");
  const Json info_w = Json::parse(read(w.run / "run.json"));
  REQUIRE(info_w["diagnostics"].size() == 1);
  CHECK(info_w["diagnostics"][0]["code"] == "W0710");
  std::vector<Json> warnings;
  std::ifstream in(w.run / "log.jsonl");
  for (std::string line; std::getline(in, line);) {
    const Json event = Json::parse(line);
    if (event["code"] == "W0710") warnings.push_back(event);
  }
  REQUIRE(warnings.size() == 1);
  CHECK(warnings[0]["level"] == "warning");
  CHECK(warnings[0]["iteration"] == 10);
  CHECK(warnings[0]["data"]["equation"] == "momentum-x");
  CHECK(warnings[0]["data"]["iteration"] == 10);
  CHECK(warnings[0]["data"]["residual"] == 1e-3);
  CHECK(warnings[0]["data"]["floor"] == 1e-12 * 6.0);

  // Just below that target the run converges with the other equations; just above it does not.
  waking.awake_value = 3.9e-8;
  const Outcome below = run(input([](Json& d) { d["time"]["max_iterations"] = 100; }), waking);
  REQUIRE(below.status == ExitStatus::finished);
  const Json info_below = Json::parse(read(below.run / "run.json"));
  CHECK(info_below["status"] == "converged");
  CHECK(info_below["iterations"] == 28);
  waking.awake_value = 4.1e-8;
  const Outcome above = run(input([](Json& d) { d["time"]["max_iterations"] = 100; }), waking);
  REQUIRE(above.status == ExitStatus::finished);
  CHECK(Json::parse(read(above.run / "run.json"))["status"] == "completed");
}

TEST_CASE("A continuation keeps the equations that left their floor", "[runtime]") {
  // momentum-x leaves its floor at iterations 10-14, then falls back to round-off or to exactly
  // zero, which its reference then holds (results.md, `start`): woken, it drops from momentum-z's
  // 4 at iteration 1, and the run converges with the others at k = 28. A run continued from
  // iteration 12, while it is still above its floor, must do the same through start.woken:
  // without it, iteration 13 would report a second W0710.
  const double silent = GENERATE(1e-15, 0.0);
  CAPTURE(silent);
  autan_test::RelaxationOptions options;
  options.silent_equation = 1;
  options.silent_residual = silent;
  options.wake_at = 10;
  options.sleep_at = 15;
  const auto doc = [](std::int64_t n) {
    return input([n](Json& d) {
      d["reproducibility"] = "run";
      d["time"]["max_iterations"] = n;
    });
  };
  const Outcome whole = run(doc(40), options);
  const Outcome a = run(doc(12), options);
  Json next = doc(40);
  next["start"] = start_from(a.run, "continue");
  CHECK(next["start"]["woken"] == Json::array({"momentum-x"}));
  CHECK(next["start"]["residual_reference"]["momentum-x"] == silent);
  const Outcome b = run(next, options);
  INFO(b.err);
  REQUIRE(b.status == ExitStatus::finished);
  const Json info = Json::parse(read(b.run / "run.json"));
  CHECK(info["status"] == "converged");
  CHECK(info["iterations"] == 28);
  CHECK(info["start"]["woken"] == Json::array({"momentum-x"}));
  // The equation woke in the first run: no new W0710 in the continuation.
  CHECK(info["diagnostics"].empty());
  auto chain = lines_of(a.run / "history.jsonl");
  const auto rest = lines_of(b.run / "history.jsonl");
  chain.insert(chain.end(), rest.begin(), rest.end());
  CHECK(chain == lines_of(whole.run / "history.jsonl"));
}

TEST_CASE("A residual's plateau follows its level, not its lowest value", "[runtime]") {
  // case-schema.md#time: the level is the median of the last 100 counted residuals, its first
  // value (counted step 100) the reference; with no level below 0.9 times it, the residual is on a
  // plateau at counted step 200. Stationary round-off with a coefficient of variation of 0.2
  // (#557's largest), uniform on 0.65 to 1.35 times p along the golden-ratio sequence, with one
  // draw at 0.85 times its lowest value at step 150 that does not move the level; then the same
  // noise falling 15 % every 100 steps, whose level sets a new reference within every 100 steps.
  const double p = 1e-6;
  const auto noise = [p](int k) {
    return p * (1.0 + 0.35 * (2.0 * std::fmod(k * 0.6180339887498949, 1.0) - 1.0));
  };
  for (const bool low_draw : {false, true}) {
    CAPTURE(low_draw);
    autan::runtime::ResidualPlateau plateau;
    double lowest = std::numeric_limits<double>::infinity();
    int at = 0;
    for (int k = 1; k <= 1000 && at == 0; ++k) {
      const double r = low_draw && k == 150 ? 0.85 * lowest : noise(k);
      lowest = std::min(lowest, r);
      if (plateau.add(r)) at = k;
    }
    CHECK(at == 200);
  }
  autan::runtime::ResidualPlateau falling;
  bool flat = false;
  for (int k = 1; k <= 1000; ++k) flat = flat || falling.add(noise(k) * std::pow(0.85, k / 100.0));
  CHECK_FALSE(flat);
}

TEST_CASE("In fp32+acc64 an equation converges on a round-off plateau near its floor",
          "[runtime]") {
  // case-schema.md#time. Residuals (e + 1) max(0.5^(k-1), p) and scales 1e4 (e + 1) give floors
  // f_e = 0.5 2^-23 1e4 (e + 1) = 5.96e-4 (e + 1). A plateau is 100 counted steps without a level
  // (median of the last 100) below 0.9 times the reference; an equation converges on one at most
  // 10 f_e high.
  const std::vector<std::string> equations{"density",    "momentum-x", "momentum-y",
                                           "momentum-z", "energy",     "nu-tilde"};
  autan_test::RelaxationOptions scaled;
  for (int e = 0; e < 6; ++e) scaled.scales.push_back(1e4 * (e + 1));
  const double floor = std::ldexp(0.5, -23) * 1e4;
  auto run_json = [&](const autan_test::RelaxationOptions& options, const std::string& precision,
                      std::int64_t budget = 10000) {
    const Outcome o = run(input([&](Json& d) {
                            d["precision"] = precision;
                            d["time"]["max_iterations"] = budget;
                          }),
                          options);
    REQUIRE(o.status == ExitStatus::finished);
    return Json::parse(read(o.run / "run.json"));
  };

  SECTION("a residual that crosses its floor while still falling converges by its drop") {
    // The floors are crossed at k = 12 (0.5^11 = 4.9e-4 < 5.96e-4); the residuals go on halving
    // to the 8-order drop at k = 28 (0.5^27 = 7.5e-9 <= 1e-8 < 0.5^26), in either precision.
    for (const std::string precision : {"fp32+acc64", "fp64"}) {
      CAPTURE(precision);
      const Json info = run_json(scaled, precision);
      CHECK(info["status"] == "converged");
      CHECK(info["iterations"] == 28);
      if (precision == "fp64") {
        CHECK(info["convergence"].is_null());
        continue;
      }
      REQUIRE(info["convergence"]["floor"].size() == equations.size());
      for (std::size_t e = 0; e < equations.size(); ++e) {
        CHECK(info["convergence"]["floor"][equations[e]] == floor * static_cast<double>(e + 1));
      }
      CHECK(info["convergence"]["plateau"] == Json::array());
    }
  }

  SECTION("a plateau above the floor and within the band converges, every equation on it") {
    // p = 1.5 f_0 (the NS manufactured solution's plateau of #467): 0.5^10 = 9.8e-4 > p >
    // 0.5^11, so the values fall to k = 11 and are p from k = 12. The level, the median of the
    // last 100, is p from its first value at k = 100, the reference: the plateau holds at k = 200.
    autan_test::RelaxationOptions flat = scaled;
    flat.plateau = 1.5 * floor;
    const Outcome converged = run(input([&](Json& d) {
                                    d["precision"] = "fp32+acc64";
                                    d["time"]["max_iterations"] = 10000;
                                  }),
                                  flat);
    REQUIRE(converged.status == ExitStatus::finished);
    const Json info = Json::parse(read(converged.run / "run.json"));
    CHECK(info["status"] == "converged");
    CHECK(info["iterations"] == 200);
    CHECK(info["convergence"]["plateau"] == Json(equations));
    // results.md, RunMetadata: the solution carries run.json's convergence, plateau included.
    int fn = 0;
    REQUIRE(cg_open((converged.run / "solution.cgns").string().c_str(), CG_MODE_READ, &fn) ==
            CG_OK);
    REQUIRE(cg_goto(fn, 1, "end") == CG_OK);
    int count = 0;
    REQUIRE(cg_ndescriptors(&count) == CG_OK);
    Json metadata;
    for (int d = 1; d <= count; ++d) {
      char name[33];
      char* text = nullptr;
      REQUIRE(cg_descriptor_read(d, name, &text) == CG_OK);
      if (std::string_view(name) == "RunMetadata") metadata = Json::parse(text);
      cg_free(text);
    }
    cg_close(fn);
    REQUIRE(metadata.is_object());
    CHECK(metadata["convergence"] == info["convergence"]);

    // Provisional steps (a first-order start) do not count: counted from k = 50, the first level
    // is at k = 149 and the plateau holds 100 counted steps later, at k = 249.
    flat.provisional_until = 50;
    CHECK(run_json(flat, "fp32+acc64")["iterations"] == 249);

    // A run that converges on its plateau and then fails to write its solution lists none.
    flat.provisional_until = 0;
    flat.throw_in_fields = true;
    const Outcome failing = run(input(), flat);
    CHECK(failing.status == ExitStatus::failed);
    const Json failed = Json::parse(read(failing.run / "run.json"));
    CHECK(failed["status"] == "failed");
    CHECK(failed["convergence"]["plateau"] == Json::array());
  }

  SECTION("a plateau above the band is not convergence") {
    autan_test::RelaxationOptions high = scaled;
    high.plateau = 20.0 * floor;
    const Json info = run_json(high, "fp32+acc64", 300);
    CHECK(info["status"] == "completed");
    CHECK(info["iterations"] == 300);
    CHECK(info["convergence"]["plateau"] == Json::array());
  }

  SECTION("a physics without scales has no floor; one with the wrong count fails") {
    const Outcome plain = run(input());
    CHECK(Json::parse(read(plain.run / "run.json"))["convergence"].is_null());
    autan_test::RelaxationOptions short_scales;
    short_scales.scales = {1.0};
    const Outcome bad = run(input(), short_scales);
    CHECK(bad.status == ExitStatus::failed);
    const Json failed = Json::parse(read(bad.run / "run.json"));
    CHECK(failed["status"] == "failed");
    CHECK(failed["diagnostics"][0]["code"] == "E0704");
  }
}

TEST_CASE("A run that needs more memory than its host can give is refused before the mesh",
          "[runtime]") {
  // solver-input.md, `memory`: on the CPU, the host and device shares of a rank together against
  // the memory its host can still give less the margin (E0711). The mesh path does not exist: the
  // refusal comes before the mesh is read.
  constexpr std::int64_t huge = std::int64_t{1} << 60;
  constexpr std::int64_t margin = std::int64_t{2} << 30;
  const auto registry = autan_test::relaxation_registry({});
  const auto refusal = [](const Outcome& o) {
    CHECK(o.status == ExitStatus::invalid_input);
    CHECK_FALSE(fs::exists(o.runs));
    return Json::parse(o.err.substr(0, o.err.find('\n')));
  };
  const Json heavy = refusal(run(input([](Json& d) {
                                   d["memory"]["bytes_host"] = huge / 2;
                                   d["memory"]["bytes_device"] = huge / 2;
                                   d["mesh"]["path"] = "/nonexistent/polyMesh";
                                 }),
                                 registry));
  CHECK(heavy["code"] == "E0711");
  CHECK(heavy["location"]["path"] == "memory");
  CHECK(heavy["data"]["required_bytes"] == huge);
  CHECK(heavy["data"]["margin_bytes"] == margin);
  CHECK(heavy["data"]["device_kind"] == "cpu");
  const auto available = heavy["data"]["available_bytes"].get<std::int64_t>();
  CHECK(available >= 0);
  CHECK(available < huge);
  // The margin is kept free: nothing more than it fits.
  const Json kept = refusal(run(input(), registry, {}, {.margin_bytes = huge}));
  CHECK(kept["code"] == "E0711");
  CHECK(kept["data"]["margin_bytes"] == huge);
  // Started anyway on request, with W0712 and the same data in run.json and the log, which is
  // published with it: its second line, after the start (results.md).
  const Outcome over = run(input([&](Json& d) { d["memory"]["bytes_host"] = huge; }), registry, {},
                           {.allow_oversubscription = true});
  REQUIRE(over.status == ExitStatus::finished);
  const Json info = Json::parse(read(over.run / "run.json"));
  REQUIRE(info["diagnostics"].size() == 1);
  CHECK(info["diagnostics"][0]["code"] == "W0712");
  CHECK(info["diagnostics"][0]["data"]["required_bytes"] == huge);
  const auto logged = lines_of(over.run / "log.jsonl");
  REQUIRE(logged.size() >= 2);
  const Json second = Json::parse(logged[1]);
  CHECK(second["code"] == "W0712");
  CHECK(second["level"] == "warning");
  CHECK(second["data"] == info["diagnostics"][0]["data"]);
  std::size_t w0712 = 0;
  for (const auto& line : logged) w0712 += Json::parse(line)["code"] == "W0712" ? 1 : 0;
  CHECK(w0712 == 1);
  // W0712 reaches stderr before the mesh is read, so before what may exhaust the host: here the
  // mesh does not exist, and its refusal follows the warning.
  const Outcome early = run(input([&](Json& d) {
                              d["memory"]["bytes_host"] = huge;
                              d["mesh"]["path"] = "/nonexistent/polyMesh";
                            }),
                            registry, {}, {.allow_oversubscription = true});
  CHECK(early.status == ExitStatus::invalid_input);
  std::vector<std::string> codes;
  std::istringstream err_lines(early.err);
  for (std::string line; std::getline(err_lines, line);)
    if (line.starts_with("{")) codes.push_back(Json::parse(line)["code"]);
  REQUIRE(codes.size() == 2);
  CHECK(codes[0] == "W0712");
  CHECK(codes[1] != "W0712");
  // The estimate is required.
  const Json missing = refusal(run(input([](Json& d) { d.erase("memory"); }), registry));
  CHECK(missing["code"] == "E0705");
  CHECK(missing["data"]["path"] == "memory");
}

TEST_CASE("A physics' groups reach it whole on one rank", "[runtime][groups]") {
  // runtime.md#registration: the groups a physics registers are formed and given to it on any
  // number of ranks. The relaxation physics checks that it was asked for them once and was given
  // each group whole, in its order (check_groups). In the mesh's own order, cell 0, the least of
  // group 0, is the physics' cell 0.
  struct InputOrder {
    InputOrder() {
      if (const char* given = std::getenv("AUTAN_CELL_ORDER")) saved = given;
      REQUIRE(setenv("AUTAN_CELL_ORDER", "input", 1) == 0);
    }
    ~InputOrder() {
      if (saved) {
        setenv("AUTAN_CELL_ORDER", saved->c_str(), 1);
      } else {
        unsetenv("AUTAN_CELL_ORDER");
      }
    }
    InputOrder(const InputOrder&) = delete;
    InputOrder& operator=(const InputOrder&) = delete;
    std::optional<std::string> saved;
  };
  autan_test::RelaxationOptions lining;
  lining.groups = 2;
  Outcome o;
  {
    const InputOrder input_order;
    o = run(input(), lining);
  }
  INFO(o.err);
  CHECK(o.status == ExitStatus::finished);
}

#ifdef AUTAN_MPIEXEC
namespace {

// The arrays of solution.cgns's WallSolution subregion, by name.
std::map<std::string, std::vector<double>> wall_solution(const fs::path& file) {
  int fn = 0;
  REQUIRE(cg_open(file.string().c_str(), CG_MODE_READ, &fn) == CG_OK);
  int regions = 0;
  REQUIRE(cg_nsubregs(fn, 1, 1, &regions) == CG_OK);
  std::map<std::string, std::vector<double>> out;
  for (int r = 1; r <= regions; ++r) {
    char name[33];
    int dim = 0, bc_len = 0, gc_len = 0;
    CGNS_ENUMT(GridLocation_t) location{};
    CGNS_ENUMT(PointSetType_t) ptset{};
    cgsize_t npnts = 0;
    REQUIRE(cg_subreg_info(fn, 1, 1, r, name, &dim, &location, &ptset, &npnts, &bc_len, &gc_len) ==
            CG_OK);
    if (std::string(name) != "WallSolution") continue;
    REQUIRE(cg_goto(fn, 1, "Zone_t", 1, "ZoneSubRegion_t", r, "end") == CG_OK);
    int arrays = 0;
    REQUIRE(cg_narrays(&arrays) == CG_OK);
    for (int a = 1; a <= arrays; ++a) {
      char array[33];
      CGNS_ENUMT(DataType_t) type{};
      int rank = 0;
      cgsize_t size[3];
      REQUIRE(cg_array_info(a, array, &type, &rank, size) == CG_OK);
      if (std::string(array) == "PointList") continue;
      std::vector<double> values(static_cast<std::size_t>(size[0]));
      REQUIRE(cg_array_read_as(a, CGNS_ENUMV(RealDouble), values.data()) == CG_OK);
      out[array] = std::move(values);
    }
  }
  cg_close(fn);
  return out;
}

// The relaxation physics' colour sweeps redone on the whole mesh, from the
// global colouring: each step, `sweeps` symmetric sweeps over the colours,
// each cell of a colour taking the mean of itself and its face neighbours,
// summed in increasing global id; the field starts at global id mod 7. With
// `groups` (RelaxationOptions::groups), a line smoother: a group is one node
// of the colouring, greedy by increasing least global id, and its cells are
// swept in decreasing id; this colouring is computed here, not by the code
// under test.
std::vector<double> swept(const autan::Mesh& mesh, int steps, int sweeps, int groups = 0) {
  const auto ids = mesh.cell_global_id();
  const auto z = [](auto i) { return static_cast<std::size_t>(i); };
  const std::int64_t period = 2 * std::int64_t{groups};
  const auto node = [&](autan::index_t c) -> std::int64_t {
    return groups > 0 && ids[z(c)] % period < groups ? ids[z(c)] % period
                                                     : std::int64_t{groups} + ids[z(c)];
  };
  std::map<std::int64_t, std::vector<autan::index_t>> members;  // node -> cells, by id
  std::map<std::int64_t, std::set<std::int64_t>> around_node;
  for (autan::index_t c = 0; c < mesh.n_cells(); ++c) members[node(c)].push_back(c);
  for (autan::index_t f = 0; f < mesh.n_interior_faces(); ++f) {
    const auto a = node(mesh.owner()[z(f)]), b = node(mesh.neighbour()[z(f)]);
    if (a != b) {
      around_node[a].insert(b);
      around_node[b].insert(a);
    }
  }
  std::vector<std::pair<std::int64_t, std::int64_t>> by_least;  // (least id, node)
  for (const auto& [n, cells] : members) {
    std::int64_t least = ids[z(cells.front())];
    for (const auto c : cells) least = std::min(least, ids[z(c)]);
    by_least.emplace_back(least, n);
  }
  std::ranges::sort(by_least);
  std::map<std::int64_t, std::int32_t> node_colour;
  for (const auto& [least, n] : by_least) {
    std::set<std::int32_t> taken;
    for (const auto m : around_node[n]) {
      if (node_colour.contains(m)) taken.insert(node_colour[m]);
    }
    std::int32_t k = 0;
    while (taken.contains(k)) ++k;
    node_colour[n] = k;
  }
  std::int32_t n_colours = 0;
  for (const auto& [n, k] : node_colour) n_colours = std::max(n_colours, k + 1);
  std::vector<double> w(z(mesh.n_cells()));
  for (std::size_t c = 0; c < w.size(); ++c) w[c] = static_cast<double>(ids[c] % 7);
  const auto colour_sweep = [&](std::int32_t k) {
    std::vector<autan::index_t> cells;
    for (const auto& [least, n] : by_least) {
      if (node_colour[n] != k || n >= groups) continue;
      auto line = members[n];
      std::ranges::sort(line, std::greater{}, [&](autan::index_t c) { return ids[z(c)]; });
      cells.insert(cells.end(), line.begin(), line.end());
    }
    for (autan::index_t c = 0; c < mesh.n_cells(); ++c) {
      if (node(c) >= groups && node_colour[node(c)] == k) cells.push_back(c);
    }
    for (const autan::index_t c : cells) {
      std::vector<std::pair<std::int64_t, double>> around{
          {ids[static_cast<std::size_t>(c)], w[static_cast<std::size_t>(c)]}};
      for (const autan::index_t f : mesh.cell_faces(c)) {
        if (f >= mesh.n_interior_faces()) continue;
        const auto i = static_cast<std::size_t>(f);
        const autan::index_t o = mesh.owner()[i] == c ? mesh.neighbour()[i] : mesh.owner()[i];
        around.emplace_back(ids[static_cast<std::size_t>(o)], w[static_cast<std::size_t>(o)]);
      }
      std::ranges::sort(around);
      double sum = 0.0;
      for (const auto& [id, value] : around) sum += value;
      w[static_cast<std::size_t>(c)] = sum / static_cast<double>(around.size());
    }
  };
  for (int step = 0; step < steps; ++step) {
    for (int s = 0; s < sweeps; ++s) {
      for (std::int32_t k = 0; k < n_colours; ++k) colour_sweep(k);
      for (std::int32_t k = n_colours - 1; k >= 0; --k) colour_sweep(k);
    }
  }
  return w;
}

// The ranks of a run, by the markers relaxation_main.cpp writes once
// run_main returns ("<status> <pid>\n" in exit-<rank>): those whose marker is
// missing or malformed, holds another status, or whose process still exists.
// A process that has ended: gone, or a zombie its parent has not reaped (proc(5),
// /proc/<pid>/stat: the state follows the last ')' of the command name). A rank reparented to a
// parent that does not reap, such as a container's first process, stays a zombie.
bool ended(pid_t pid) {
  const auto gone = [pid] { return kill(pid, 0) != 0 && errno == ESRCH; };
  if (gone()) return true;
  std::string stat;
  {
    std::ifstream in("/proc/" + std::to_string(pid) + "/stat");
    std::getline(in, stat);
  }
  const auto close = stat.rfind(')');
  if (close != std::string::npos && close + 2 < stat.size() && stat[close + 2] == 'Z') return true;
  // Reaped between the two looks: its /proc entry went with it.
  return gone();
}

std::vector<std::string> ranks_not_returned(const fs::path& exits, int ranks, int expected) {
  std::vector<std::string> wrong;
  const std::regex marker(R"(^(\d+) (\d+)\n$)");
  for (int r = 0; r < ranks; ++r) {
    std::ifstream in(exits / ("exit-" + std::to_string(r)));
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::smatch m;
    const bool ok = std::regex_match(text, m, marker) && std::stoi(m[1]) == expected &&
                    ended(static_cast<pid_t>(std::stol(m[2])));
    if (!ok) wrong.push_back(std::to_string(r));
  }
  return wrong;
}

// autan-run-relaxation under the launcher on `ranks` ranks: its run
// directory (none for an invalid input), after checking its exit status;
// `diagnostic`, when given, receives the JSON diagnostic line of stderr.
fs::path run_on_ranks(const Json& doc, int ranks, const std::vector<std::string>& options,
                      int expected_status = 0, Json* diagnostic = nullptr,
                      const std::vector<std::string>& extra_args = {}, bool directory = true) {
  const fs::path dir = fresh_dir();
  const fs::path input_file = dir / "solver-input.json";
  // One device per rank (solver-input.md, `devices`), unless a test sets it.
  Json with_count = doc;
  if (with_count["devices"]["count"] == 1) with_count["devices"]["count"] = ranks;
  std::ofstream(input_file) << with_count.dump();
  std::vector<std::string> args{AUTAN_MPIEXEC,
                                AUTAN_MPIEXEC_NUMPROC_FLAG,
                                std::to_string(ranks),
                                AUTAN_RUN_RELAXATION,
                                "--input",
                                input_file.string(),
                                "--runs",
                                (dir / "runs").string()};
  args.insert(args.end(), extra_args.begin(), extra_args.end());
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  // Each rank writes its exit status to exit-<rank> (relaxation_main.cpp), in
  // the directory the test names or in exits/.
  fs::path exits = dir / "exits";
  for (const auto& o : options) {
    if (o.starts_with("AUTAN_TEST_EXIT_DIR=")) exits = o.substr(20);
  }
  fs::create_directories(exits);
  REQUIRE(fs::is_empty(exits));  // no marker of another launch
  std::vector<std::string> env = options;
  env.push_back("AUTAN_TEST_EXIT_DIR=" + exits.string());
  for (char** e = environ; *e != nullptr; ++e) env.emplace_back(*e);
  std::vector<char*> envp;
  for (auto& e : env) envp.push_back(e.data());
  envp.push_back(nullptr);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  const std::string err_file = (dir / "stderr.txt").string();
  posix_spawn_file_actions_addopen(&actions, 2, err_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
                                   0644);
  pid_t pid = 0;
  REQUIRE(posix_spawn(&pid, AUTAN_MPIEXEC, &actions, nullptr, argv.data(), envp.data()) == 0);
  posix_spawn_file_actions_destroy(&actions);
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  if (WIFSIGNALED(status)) {
    // #304: the launcher can hang in its cleanup once every rank has ended,
    // and the CI watchdog then kills it. The run is judged by the ranks
    // themselves: every one must have returned with the expected status.
    const auto wrong = ranks_not_returned(exits, ranks, expected_status);
    INFO("launcher ended by signal " << WTERMSIG(status)
                                     << "; ranks not returned as expected: " << wrong.size());
    REQUIRE(wrong.empty());
    WARN("launcher killed after every rank ended (#304)");
  } else {
    REQUIRE(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == expected_status);
    // A run every rank finishes leaves each rank's status, as the watchdog's
    // judgement above reads it (a failing rank may end the others first).
    if (expected_status == 0) CHECK(ranks_not_returned(exits, ranks, 0).empty());
  }
  if (diagnostic != nullptr) {
    std::ifstream in(err_file);
    for (std::string line; std::getline(in, line);) {
      if (line.starts_with("{")) *diagnostic = Json::parse(line);
    }
  }
  fs::path run;
  if (fs::exists(dir / "runs")) {
    for (const auto& entry : fs::directory_iterator(dir / "runs")) run = entry.path();
  }
  CHECK(run.empty() == (expected_status == 2 || !directory));
  return run;
}

}  // namespace

TEST_CASE("A launcher killed after its ranks is judged by their markers", "[runtime]") {
  // #304: a rank counts as returned only by a well-formed "<status> <pid>\n"
  // marker of the expected status whose process has ended: gone, or a zombie.
  const fs::path exits = fresh_dir();
  const pid_t gone = fork();
  if (gone == 0) _exit(0);
  int waited = 0;
  REQUIRE(waitpid(gone, &waited, 0) == gone);
  const std::string finished = std::to_string(gone);
  const auto marker = [&](int r, const std::string& text) {
    std::ofstream(exits / ("exit-" + std::to_string(r))) << text;
  };
  marker(0, "2 " + finished + "\n");
  marker(1, "2 " + finished + "\n");
  CHECK(ranks_not_returned(exits, 2, 2).empty());
  CHECK(ranks_not_returned(exits, 2, 0) == std::vector<std::string>{"0", "1"});
  CHECK(ranks_not_returned(exits, 3, 2) == std::vector<std::string>{"2"});
  for (const std::string& text :
       std::vector<std::string>{"garbage\n", "2\n", "2 " + finished + "\n1\n", "2 " + finished}) {
    CAPTURE(text);
    marker(1, text);
    CHECK(ranks_not_returned(exits, 2, 2) == std::vector<std::string>{"1"});
  }
  marker(1, "2 " + std::to_string(getpid()) + "\n");  // a process that still runs
  CHECK(ranks_not_returned(exits, 2, 2) == std::vector<std::string>{"1"});
  // A rank that exited and that nobody reaped yet: ended (siginfo_t tells it exited, unreaped).
  const pid_t zombie = fork();
  if (zombie == 0) _exit(0);
  siginfo_t info{};
  REQUIRE(waitid(P_PID, static_cast<id_t>(zombie), &info, WEXITED | WNOWAIT) == 0);
  marker(1, "2 " + std::to_string(zombie) + "\n");
  CHECK(ranks_not_returned(exits, 2, 2).empty());
  REQUIRE(waitpid(zombie, &waited, 0) == zombie);
}

TEST_CASE("The ranks of a host share its memory", "[runtime][mpi]") {
  // solver-input.md, `memory` is one rank's: two ranks on one host need twice it.
  constexpr std::int64_t huge = std::int64_t{1} << 60;
  constexpr int ranks = 2;
  const Json heavy = input([&](Json& d) { d["memory"]["bytes_host"] = huge; });
  Json refusal;
  CHECK(run_on_ranks(heavy, ranks, {}, 2, &refusal, {}, false).empty());
  CHECK(refusal["code"] == "E0711");
  CHECK(refusal["data"]["required_bytes"] == ranks * huge);
  // The command line sets the margin, and starts a run beyond its memory with W0712.
  Json margined;
  CHECK(run_on_ranks(input(), ranks, {}, 2, &margined, {"--memory-margin", std::to_string(huge)},
                     false)
            .empty());
  CHECK(margined["code"] == "E0711");
  CHECK(margined["data"]["margin_bytes"] == huge);
  const fs::path over = run_on_ranks(heavy, ranks, {}, 0, nullptr, {"--allow-oversubscription"});
  const Json info = Json::parse(read(over / "run.json"));
  REQUIRE(info["diagnostics"].size() == 1);
  CHECK(info["diagnostics"][0]["code"] == "W0712");
  CHECK(info["diagnostics"][0]["data"]["required_bytes"] == ranks * huge);
}

TEST_CASE("Several ranks give the bits of one", "[runtime][mpi][gpu]") {
  // Fields and residuals vary from cell to cell, so that a cell gathered to
  // the wrong place or a norm summed in another order would change bits;
  // forces are recorded at every iteration, and the state grows.
  const bool periodic = GENERATE(false, true);
  const int ranks = GENERATE(2, 4);
  CAPTURE(periodic, ranks);
  const auto edit = [](Json& d) {
    d["reproducibility"] = "run";
    d["monitors"][0]["every"] = 1;
    d["time"]["max_iterations"] = 12;
  };
  const Json doc = periodic ? periodic_input(1.0, edit) : input(edit);
  const autan_test::RelaxationOptions options{
      .factor = 0.9, .pressure_offset = 1000.0, .growth = 1e-3, .vary = true};
  const Outcome one = run(doc, options);
  REQUIRE(one.status == ExitStatus::finished);
  const fs::path many = run_on_ranks(doc, ranks,
                                     {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_PRESSURE_OFFSET=1000",
                                      "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1"});

  CHECK(lines_of(many / "history.jsonl") == lines_of(one.run / "history.jsonl"));
  // In the mesh's own order too (AUTAN_CELL_ORDER=input; RCM is the
  // default): partitioned on other cells, the same outputs.
  const fs::path rcm = run_on_ranks(
      autan_test::planned(doc, mesh_path, "input"), ranks,
      {"AUTAN_CELL_ORDER=input", "AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_PRESSURE_OFFSET=1000",
       "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1"});
  CHECK(lines_of(rcm / "history.jsonl") == lines_of(one.run / "history.jsonl"));
  CHECK(solution_field(rcm / "solution.cgns", "FlowSolution", "Pressure") ==
        solution_field(one.run / "solution.cgns", "FlowSolution", "Pressure"));
  CHECK(wall_solution(rcm / "solution.cgns") == wall_solution(one.run / "solution.cgns"));
  // A smoother sweeping the global colours (PhysicsContext::cell_colours),
  // the ghosts exchanged after each colour, and an argmin over the ranks:
  // the history of one rank.
  autan_test::RelaxationOptions sweeping = options;
  sweeping.colour_sweeps = 2;
  const Outcome one_swept = run(doc, sweeping);
  const fs::path many_swept =
      run_on_ranks(doc, ranks,
                   {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_PRESSURE_OFFSET=1000",
                    "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1", "AUTAN_TEST_COLOUR_SWEEPS=2"});
  CHECK(lines_of(one_swept.run / "history.jsonl") != lines_of(one.run / "history.jsonl"));
  CHECK(lines_of(many_swept / "history.jsonl") == lines_of(one_swept.run / "history.jsonl"));
  // The swept field, cell for cell: on one rank, as an independent sweep of
  // the whole mesh gives it (12 steps of 2 symmetric sweeps); on several,
  // the same bits.
  const auto swept_one =
      solution_field(one_swept.run / "solution.cgns", "FlowSolution", "TurbulentSANuTilde");
  CHECK(solution_field(many_swept / "solution.cgns", "FlowSolution", "TurbulentSANuTilde") ==
        swept_one);
  if (!periodic) CHECK(swept_one == swept(autan::io::read_mesh(mesh_path), 12, 2));
  // Reproducibility "ranks": a partition even on one rank, and face keys that
  // match the mesh file on 1 and N ranks (the physics checks them); a run on
  // several ranks in another mode says its results depend on the count.
  Json keyed = doc;
  keyed["reproducibility"] = "ranks";
  autan_test::RelaxationOptions checking = options;
  checking.check_keys = true;
  const Outcome one_keyed = run(keyed, checking);
  REQUIRE(one_keyed.status == ExitStatus::finished);
  const fs::path many_keyed =
      run_on_ranks(keyed, ranks,
                   {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_PRESSURE_OFFSET=1000",
                    "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1", "AUTAN_TEST_CHECK_KEYS=1"});
  CHECK(Json::parse(read(many_keyed / "run.json"))["status"] == "completed");
  CHECK(lines_of(many_keyed / "history.jsonl") == lines_of(one_keyed.run / "history.jsonl"));
  CHECK(solution_field(many_keyed / "solution.cgns", "FlowSolution", "Pressure") ==
        solution_field(one_keyed.run / "solution.cgns", "FlowSolution", "Pressure"));
  // The keys name the mesh file's faces through the RCM renumbering (the
  // default) and its flipped faces: in the mesh's own order the physics
  // finds the same keys, so "ranks" gives the same bits in either order.
  const fs::path input_keyed = run_on_ranks(
      autan_test::planned(keyed, mesh_path, "input"), ranks,
      {"AUTAN_CELL_ORDER=input", "AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_PRESSURE_OFFSET=1000",
       "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1", "AUTAN_TEST_CHECK_KEYS=1"});
  CHECK(lines_of(input_keyed / "history.jsonl") == lines_of(one_keyed.run / "history.jsonl"));
  CHECK(solution_field(input_keyed / "solution.cgns", "FlowSolution", "Pressure") ==
        solution_field(one_keyed.run / "solution.cgns", "FlowSolution", "Pressure"));
  // On one rank in "run", the renumbered mesh without a partition: its keys too.
  if (!periodic) CHECK(run(doc, checking).status == ExitStatus::finished);
  const auto has_event = [](const fs::path& run_dir) {
    std::ifstream in(run_dir / "log.jsonl");
    for (std::string line; std::getline(in, line);) {
      if (Json::parse(line)["data"].contains("mode")) return true;
    }
    return false;
  };
  CHECK(has_event(many));
  CHECK_FALSE(has_event(many_keyed));
  // A line smoother: the physics' groups (each spanning the mesh) stay whole
  // on one rank, and it sweeps the colours of its groups, each in its order;
  // on several ranks, the bits of one, and on one, those of an independent
  // sweep of the whole mesh.
  autan_test::RelaxationOptions lining = sweeping;
  lining.groups = ranks;
  const Outcome one_lined = run(doc, lining);
  REQUIRE(one_lined.status == ExitStatus::finished);
  const fs::path many_lined =
      run_on_ranks(doc, ranks,
                   {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_PRESSURE_OFFSET=1000",
                    "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1", "AUTAN_TEST_COLOUR_SWEEPS=2",
                    "AUTAN_TEST_GROUPS=" + std::to_string(ranks)});
  CHECK(lines_of(many_lined / "history.jsonl") == lines_of(one_lined.run / "history.jsonl"));
  const auto lined_one =
      solution_field(one_lined.run / "solution.cgns", "FlowSolution", "TurbulentSANuTilde");
  CHECK(lined_one != swept_one);
  CHECK(solution_field(many_lined / "solution.cgns", "FlowSolution", "TurbulentSANuTilde") ==
        lined_one);
  if (!periodic) CHECK(lined_one == swept(autan::io::read_mesh(mesh_path), 12, 2, ranks));
  // Groups that cannot be formed or partitioned, on every rank or on rank 1
  // alone, are refused by the ranks together before any directory: the
  // refusing rank's code reaches stderr, and no rank ends in an abort. The
  // launcher may end the other ranks once one has exited non-zero, so at
  // least one rank, not every rank, is seen returning.
  const auto refused_together = [&](const char* groups, Json& diagnostic) {
    const fs::path exits = fresh_dir();
    CHECK(run_on_ranks(
              doc, ranks,
              {std::string("AUTAN_TEST_GROUPS=") + groups, "AUTAN_TEST_EXIT_DIR=" + exits.string()},
              1, &diagnostic, {}, false)
              .empty());
    int returned = 0;
    for (int r = 0; r < ranks; ++r) returned += fs::exists(exits / ("exit-" + std::to_string(r)));
    CAPTURE(groups);
    CHECK(returned >= 1);
  };
  Json malformed;
  refused_together("-1", malformed);
  CHECK(malformed["code"] == "E0140");
  Json alone;
  refused_together("-2", alone);
  CHECK(alone["code"] == "E0140");
  CHECK(alone["data"]["rank"] == 1);
#ifdef AUTAN_HAVE_CUDA
  // On CUDA devices, the ranks sharing the host's GPUs: the relaxation
  // physics checks at every step that the exchange fills its ghosts in
  // device memory, through a CUDA-aware MPI; the history stays that of one rank.
  Json on_gpu = doc;
  on_gpu["devices"]["kind"] = "cuda";
  const fs::path gpu = run_on_ranks(on_gpu, ranks,
                                    {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_PRESSURE_OFFSET=1000",
                                     "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1"});
  CHECK(lines_of(gpu / "history.jsonl") == lines_of(one.run / "history.jsonl"));
#endif
  const Json a = Json::parse(read(one.run / "run.json"));
  const Json b = Json::parse(read(many / "run.json"));
  for (const char* key : {"status", "iterations", "time", "diagnostics"}) {
    CAPTURE(key);
    CHECK(a[key] == b[key]);
  }
  for (const char* field : {"Density", "VelocityX", "Pressure", "TurbulentSANuTilde"}) {
    CAPTURE(field);
    const auto expected = solution_field(one.run / "solution.cgns", "FlowSolution", field);
    CHECK(!expected.empty());
    CHECK(solution_field(many / "solution.cgns", "FlowSolution", field) == expected);
  }
  // An initial-state file, split among the ranks by their owned cells: the
  // solution of one rank, cell for cell.
  if (!periodic) {
    Json plain = doc;
    inviscid(plain);
    const Json from_file = with_initial_state(plain, initial_state_document(mesh_path));
    const Outcome one_file = run(from_file, options);
    const fs::path many_file =
        run_on_ranks(from_file, ranks,
                     {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_PRESSURE_OFFSET=1000",
                      "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1"});
    for (const char* field : {"Density", "VelocityY", "Pressure"}) {
      CAPTURE(field);
      CHECK(solution_field(many_file / "solution.cgns", "FlowSolution", field) ==
            solution_field(one_file.run / "solution.cgns", "FlowSolution", field));
    }
  }
  const auto wall = wall_solution(one.run / "solution.cgns");
  CHECK(wall.contains("Pressure"));
  CHECK(wall_solution(many / "solution.cgns") == wall);
  // The memory event names every rank's peak: the last rank alone holds
  // 256 MiB, which its slot and the maximum show, not rank 0's.
  {
    constexpr std::int64_t held = std::int64_t{256} << 20;
    const Json memory =
        memory_event(run_on_ranks(doc, ranks,
                                  {"AUTAN_TEST_HOLD_BYTES=" + std::to_string(held),
                                   "AUTAN_TEST_HOLD_RANK=" + std::to_string(ranks - 1)}));
    const auto& by_rank = memory["host_peak_bytes_by_rank"];
    REQUIRE(by_rank.size() == static_cast<std::size_t>(ranks));
    CHECK(by_rank.back().get<std::int64_t>() >= held);
    CHECK(by_rank.front().get<std::int64_t>() < held);
    CHECK(memory["host_peak_bytes"] == by_rank.back());
    CHECK(memory["device_peak_bytes_by_rank"].size() == static_cast<std::size_t>(ranks));
  }
  // Rank 0 logged the partition: every cell owned once, and each rank's owned and ghost cells
  // those autan-mesh-info --ranks gives for the mesh (mesh-info.md, partition).
  Json partitioned;
  std::ifstream log_in(many / "log.jsonl");
  for (std::string line; std::getline(log_in, line);) {
    const Json event = Json::parse(line);
    if (event["source"] == "mpi") partitioned = event["data"];
  }
  const autan::Mesh mesh = autan::io::read_mesh(doc["mesh"]["path"].get<std::string>());
  std::int64_t owned = 0;
  for (const auto& n : partitioned["owned_cells"]) owned += n.get<std::int64_t>();
  CHECK(owned == mesh.n_cells());
  const auto counts = autan::io::run_counts(mesh, 1.0, autan::io::CellOrder::rcm,
                                            autan::io::RanksRequest{.ranks = ranks, .groups = {}});
  REQUIRE(counts.partition);
  CHECK(partitioned["owned_cells"] == Json(counts.partition->cells));
  CHECK(partitioned["ghost_cells"] == Json(counts.partition->ghosts));

  // Every cell from 7 on diverges, on several ranks: the least is reported,
  // as a single rank names it.
  const Outcome diverged = run(doc, {.factor = 0.9, .diverge_at = 5, .diverge_cell = 7});
  const Json d = Json::parse(read(run_on_ranks(doc, ranks,
                                               {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_DIVERGE_AT=5",
                                                "AUTAN_TEST_DIVERGE_CELL=7"}) /
                                  "run.json"));
  CHECK(d["status"] == "diverged");
  CHECK(d["diagnostics"] == Json::parse(read(diverged.run / "run.json"))["diagnostics"]);

  // A physics registered for one rank refuses several before any directory,
  // and a scope refuses by the rank count it is given.
  CHECK(run_on_ranks(doc, ranks, {"AUTAN_TEST_ONE_RANK=1"}, 2).empty());
  // Every rank's scope must see the count for the run to start.
  CHECK(!run_on_ranks(doc, ranks, {"AUTAN_TEST_EXPECT_RANKS=" + std::to_string(ranks)}).empty());
  Json refusal;
  CHECK(run_on_ranks(doc, ranks, {"AUTAN_TEST_EXPECT_RANKS=" + std::to_string(ranks + 1)}, 2,
                     &refusal)
            .empty());
  CHECK(refusal["code"] == "E0706");
  CHECK(refusal["data"] == Json{{"ranks", ranks}, {"path", "physics"}});

  // A physics failure on the last rank alone, the others waiting for it in a
  // collective call, ends every rank with status 1 and a failed run.json.
  const Json f =
      Json::parse(read(run_on_ranks(doc, ranks,
                                    {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_THROW_AT=4",
                                     "AUTAN_TEST_THROW_RANK=" + std::to_string(ranks - 1)},
                                    1) /
                       "run.json"));
  CHECK(f["status"] == "failed");
  CHECK(f["iterations"] == 3);
  CHECK(f["diagnostics"][0]["code"] == "E0704");
  CHECK(f["diagnostics"][0]["data"]["rank"] == ranks - 1);

  // The same failure in the final monitor record (loads() called only by
  // finish, `every` beyond the run), the others inside the collective sum of
  // their loads(): every rank ends, none waits forever (ctest TIMEOUT).
  Json final_only = doc;
  final_only["monitors"][0]["every"] = 1000;
  const Json g =
      Json::parse(read(run_on_ranks(final_only, ranks,
                                    {"AUTAN_TEST_FACTOR=0.5", "AUTAN_TEST_THROW_IN_LOADS=1",
                                     "AUTAN_TEST_THROW_RANK=" + std::to_string(ranks - 1)},
                                    1) /
                       "run.json"));
  CHECK(g["status"] == "failed");
  CHECK(g["monitors"].is_null());
  CHECK(g["diagnostics"][0]["code"] == "E0704");
  CHECK(g["diagnostics"][0]["data"]["rank"] == ranks - 1);

  // A stop file the last rank creates during iteration 5: rank 0 sees it, and
  // all stop before iteration 6.
  const fs::path stop_file = fresh_dir() / "stop";
  const fs::path by_file = run_on_ranks(doc, ranks,
                                        {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_TOUCH_AT=5",
                                         "AUTAN_TEST_TOUCH_RANK=" + std::to_string(ranks - 1),
                                         "AUTAN_TEST_TOUCH_FILE=" + stop_file.string()},
                                        130, nullptr, {"--stop-file", stop_file.string()});
  CHECK(Json::parse(read(by_file / "run.json"))["status"] == "stopped");
  CHECK(Json::parse(read(by_file / "run.json"))["iterations"] == 5);

  // A signal reaching the last rank alone, after iteration 5, stops all of
  // them before iteration 6, with the solution written.
  const fs::path stopped = run_on_ranks(doc, ranks,
                                        {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_SIGNAL_AT=5",
                                         "AUTAN_TEST_SIGNAL_RANK=" + std::to_string(ranks - 1)},
                                        130);
  const Json st = Json::parse(read(stopped / "run.json"));
  CHECK(st["status"] == "stopped");
  CHECK(st["iterations"] == 5);
  CHECK(fs::exists(stopped / "solution.cgns"));
}

TEST_CASE("Several ranks checkpoint and continue from it as one", "[runtime][mpi]") {
  // results.md#checkpointcgns: the ranks gather the checkpoint as they gather the solution. A
  // run that fails at step 8 keeps its checkpoint of step 6 (every 3); continuing from it under
  // "ranks", on the same ranks or on one, gives the uninterrupted run of one rank bit for bit.
  const int ranks = 2;
  const int resume_on = GENERATE(1, 2);
  CAPTURE(resume_on);
  const auto steps = [](std::int64_t n) {
    return [n](Json& d) {
      d["reproducibility"] = "ranks";
      d["monitors"][0]["every"] = 1;
      d["time"]["steps"] = n;
      d["time"]["checkpoint"] = Json{{"iterations", 3}};
    };
  };
  // The physics' SolverState (a step count driving its growth) is rank 0's in the checkpoint and
  // every rank's on the continue.
  const std::vector<std::string> env{"AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1",
                                     "AUTAN_TEST_SCALARS=3"};
  const autan_test::RelaxationOptions one{.growth = 1e-3, .vary = true, .scalars = 3};
  const Outcome whole = run(input(steps(12), "hybrid-unsteady.solver-input.json"), one);
  std::vector<std::string> failing = env;
  failing.emplace_back("AUTAN_TEST_THROW_AT=8");
  const fs::path a =
      run_on_ranks(input(steps(12), "hybrid-unsteady.solver-input.json"), ranks, failing, 1);
  REQUIRE(Json::parse(read(a / "run.json"))["status"] == "failed");
  REQUIRE(checkpoint_iteration(a / "checkpoint.cgns") == 6);
  Json next = input(steps(12), "hybrid-unsteady.solver-input.json");
  next["start"] = start_from_checkpoint(a);
  const fs::path b = resume_on == 1 ? run(next, one).run : run_on_ranks(next, ranks, env);
  if (resume_on == 2) {
    // A SolverState that rank 1 alone returns of another length than registered fails the run.
    std::vector<std::string> wrong = env;
    wrong.emplace_back("AUTAN_TEST_SCALARS_WRONG_RANK=1");
    const fs::path bad =
        run_on_ranks(input(steps(12), "hybrid-unsteady.solver-input.json"), ranks, wrong, 1);
    const Json info = Json::parse(read(bad / "run.json"));
    CHECK(info["status"] == "failed");
    CHECK(std::ranges::count_if(info["diagnostics"],
                                [](const Json& d) { return d["code"] == "E0704"; }) == 1);
  }

  auto chain = lines_of(a / "history.jsonl");
  chain.resize(6);
  const auto more = lines_of(b / "history.jsonl");
  chain.insert(chain.end(), more.begin(), more.end());
  CHECK(chain == lines_of(whole.run / "history.jsonl"));
  for (const char* solution : {"FlowSolution", "FlowSolutionMean"}) {
    for (const char* field : {"Density", "Pressure"}) {
      CAPTURE(solution, field);
      CHECK(solution_field(b / "solution.cgns", solution, field) ==
            solution_field(whole.run / "solution.cgns", solution, field));
    }
  }
}

TEST_CASE("Several ranks average and continue as one", "[runtime][mpi]") {
  // Unsteady with averaging after step 3: two runs of 4 then 4 steps on
  // several ranks, the second continuing the first, give the 8 steps of one
  // rank: the start state is split among the ranks and the mean and restart
  // fields gathered back.
  const int ranks = GENERATE(2, 4);
  const std::string mode = GENERATE("run", "ranks");
  CAPTURE(ranks, mode);
  const auto steps = [mode](std::int64_t n) {
    return [n, mode](Json& d) {
      d["reproducibility"] = mode;
      d["reproducibility"] = "run";
      d["monitors"][0]["every"] = 1;
      d["time"]["steps"] = n;
    };
  };
  const std::vector<std::string> env{"AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1"};
  const Outcome whole =
      run(input(steps(8), "hybrid-unsteady.solver-input.json"), {.growth = 1e-3, .vary = true});
  const fs::path a = run_on_ranks(input(steps(4), "hybrid-unsteady.solver-input.json"), ranks, env);
  Json next = input(steps(8), "hybrid-unsteady.solver-input.json");
  next["start"] = start_from(a, "continue");
  const fs::path b = run_on_ranks(next, ranks, env);

  auto chain = lines_of(a / "history.jsonl");
  const auto more = lines_of(b / "history.jsonl");
  chain.insert(chain.end(), more.begin(), more.end());
  CHECK(chain == lines_of(whole.run / "history.jsonl"));
  for (const char* solution : {"FlowSolution", "FlowSolutionMean"}) {
    for (const char* field : {"Density", "Pressure"}) {
      CAPTURE(solution, field);
      const auto expected = solution_field(whole.run / "solution.cgns", solution, field);
      CHECK(!expected.empty());
      CHECK(solution_field(b / "solution.cgns", solution, field) == expected);
    }
  }
}
#endif

TEST_CASE("Cells renumbered in RCM order give the run of the mesh's order", "[runtime][renumber]") {
  // By default autan-run computes on the mesh renumbered in reverse
  // Cuthill-McKee order; every output keeps the mesh's numbering.
  // AUTAN_CELL_ORDER=input keeps the mesh's own order.
  struct InputOrder {
    InputOrder() { REQUIRE(setenv("AUTAN_CELL_ORDER", "input", 1) == 0); }
    ~InputOrder() { unsetenv("AUTAN_CELL_ORDER"); }
    InputOrder(const InputOrder&) = delete;
    InputOrder& operator=(const InputOrder&) = delete;
  };
  // The input planned in the mesh's order: mesh_info's lines are those of the run's order.
  const auto in_input_order = [](const Json& d) { return autan_test::planned(d, mesh_path); };
  SECTION("bit for bit when nothing sums over faces: the relaxation physics") {
    // Fields that vary per cell, forces at every iteration, a cell that
    // diverges, an initial state and the periodic box: each output mapped back.
    const bool periodic = GENERATE(false, true);
    CAPTURE(periodic);
    const auto edit = [](Json& d) {
      d["monitors"][0]["every"] = 1;
      d["time"]["max_iterations"] = 12;
    };
    const Json doc = periodic ? periodic_input(1.0, edit) : input(edit);
    const autan_test::RelaxationOptions options{
        .factor = 0.9, .pressure_offset = 1000.0, .growth = 1e-3, .vary = true};
    Outcome a;
    {
      const InputOrder input_order;
      a = run(in_input_order(doc), options);
    }
    REQUIRE(a.status == ExitStatus::finished);
    const Outcome b = run(doc, options);
    REQUIRE(b.status == ExitStatus::finished);
    // The order is recorded and part of the identity (results.md).
    const Json info_a = Json::parse(read(a.run / "run.json"));
    const Json info_b = Json::parse(read(b.run / "run.json"));
    CHECK(info_a["cell_order"] == "input");
    CHECK(info_b["cell_order"] == "rcm");
    CHECK(info_a["identity"] != info_b["identity"]);
    CHECK(lines_of(a.run / "history.jsonl") == lines_of(b.run / "history.jsonl"));
    for (const char* field : {"Density", "VelocityX", "Pressure"}) {
      CAPTURE(field);
      CHECK(solution_field(a.run / "solution.cgns", "FlowSolution", field) ==
            solution_field(b.run / "solution.cgns", "FlowSolution", field));
    }
    std::ifstream log_in(b.run / "log.jsonl");
    bool renumbered = false;
    for (std::string line; std::getline(log_in, line);)
      renumbered = renumbered || Json::parse(line)["data"].value("order", "") == "rcm";
    CHECK(renumbered);
    const auto diverged = [&](const Json& d) {
      const Outcome o = run(d, {.factor = 0.9, .diverge_at = 3, .diverge_cell = 7});
      return Json::parse(read(o.run / "run.json"))["diagnostics"];
    };
    const Json in_rcm = diverged(doc);
    {
      const InputOrder input_order;
      CHECK(diverged(in_input_order(doc)) == in_rcm);
    }
    if (!periodic) {
      Json plain = doc;
      inviscid(plain);
      const Json from_file = with_initial_state(plain, initial_state_document(mesh_path));
      const Outcome c = run(from_file, options);
      const InputOrder input_order;
      const Outcome d = run(in_input_order(from_file), options);
      CHECK(solution_field(c.run / "solution.cgns", "FlowSolution", "Pressure") ==
            solution_field(d.run / "solution.cgns", "FlowSolution", "Pressure"));
    }
  }
  SECTION("to round-off under the Euler physics, which sums faces in their order") {
    // Five explicit SSP-RK3 steps of an inviscid flow at incidence, so not
    // uniform near the symmetry plane zmax, on each mesh family and in each
    // precision: only the order of the face sums differs, so every field
    // agrees to a few units of the precision's epsilon of its magnitude.
    const char* family = GENERATE("tet", "hex", "poly", "hybrid");
    const char* precision = GENERATE("fp64", "fp32+acc64");
    CAPTURE(family, precision);
    const fs::path mesh_file = data_dir / "mesh" / family / "polyMesh";
    const autan::Mesh mesh = autan::io::read_mesh(mesh_file);
    const Json doc = input([&](Json& d) {
      d["mesh"]["path"] = fs::absolute(mesh_file).string();
      d["mesh"]["sha256"] = autan::io::mesh_hash(mesh_file);
      d["precision"] = precision;
      inviscid(d);
      for (auto& b : d["boundaries"]) {
        if (b["zone"] == "ymin") b["kind"] = "slip";
      }
      Json zones = Json::array();
      for (const auto z : autan::io::meshinfo_zones(mesh).cell) {
        Json zone = d["cell_zones"][0];
        zone["zone"] = mesh.cell_zones()[static_cast<std::size_t>(z)].name;
        zones.push_back(zone);
      }
      d["cell_zones"] = zones;
      d["time"] = {{"kind", "unsteady"},      {"steps", 5},
                   {"dt", nullptr},           {"cfl", 0.8},
                   {"average_from", nullptr}, {"checkpoint", {{"seconds", 1800.0}}}};
    });
    autan::runtime::PhysicsRegistry euler;
    autan::physics::register_euler(euler);
    Outcome a;
    {
      const InputOrder input_order;
      a = run(in_input_order(doc), euler);
    }
    REQUIRE(a.status == ExitStatus::finished);
    const Outcome b = run(doc, euler);
    REQUIRE(b.status == ExitStatus::finished);
    const double eps = std::string(precision) == "fp64" ? 0x1p-52 : 0x1p-23;
    for (const char* field : {"Density", "VelocityX", "VelocityZ", "Pressure"}) {
      CAPTURE(field);
      const auto x = solution_field(a.run / "solution.cgns", "FlowSolution", field);
      const auto y = solution_field(b.run / "solution.cgns", "FlowSolution", field);
      REQUIRE(x.size() == y.size());
      double scale = 0.0, worst = 0.0;
      for (std::size_t c = 0; c < x.size(); ++c) {
        scale = std::max(scale, std::abs(x[c]));
        worst = std::max(worst, std::abs(x[c] - y[c]));
      }
      CAPTURE(worst / (eps * scale));
      CHECK(worst <= 64 * eps * scale);
    }
  }
}

TEST_CASE("A run logs the memory it used", "[runtime]") {
  // The physics holds 256 MiB of host memory from its construction on: the
  // peak and the after-setup figures count it; a CPU run uses no device memory.
  autan_test::RelaxationOptions holding;
  holding.hold_bytes = std::int64_t{256} << 20;
  const Outcome o = run(input(), holding);
  REQUIRE(o.status == ExitStatus::finished);
  const Json memory = memory_event(o.run);
  REQUIRE(memory.is_object());
  const auto host = memory["host_peak_bytes"].get<std::int64_t>();
  CHECK(host >= holding.hold_bytes);
  CHECK(memory["after_setup"]["host"].get<std::int64_t>() >= holding.hold_bytes);
  CHECK(memory["after_setup"]["host"].get<std::int64_t>() <= host);
  CHECK(memory["host_peak_bytes_by_rank"] == Json::array({host}));
  CHECK(memory["device_peak_bytes"] == 0);
  CHECK(memory["device_peak_bytes_by_rank"] == Json::array({0}));
}

TEST_CASE("A mesh in inches gives the run of the same mesh in metres", "[runtime][unit]") {
  // The hybrid mesh, and a copy whose points are divided by 0.0254 and
  // declared in inches (mesh.scale = 0.0254, exactly one inch in metres):
  // five explicit Euler steps of an inviscid flow at incidence give the same
  // fields and forces to round-off of the precision, and the solution's
  // coordinates in metres. (The SDK's test_solver_backend runs one file in m
  // and in in through the SDK: its coordinates are then bit for bit.)
  const char* precision = GENERATE("fp64", "fp32+acc64");
  CAPTURE(precision);
  const fs::path inches = fresh_dir() / "polyMesh";
  fs::copy(mesh_path, inches);
  {
    std::ifstream in(mesh_path / "points");
    std::ostringstream out;
    const std::regex point(R"(^\(([^ ]+) ([^ ]+) ([^ ]+)\)$)");
    char buffer[96];
    for (std::string line; std::getline(in, line);) {
      std::smatch m;
      if (std::regex_match(line, m, point)) {
        std::snprintf(buffer, sizeof buffer, "(%.17g %.17g %.17g)", std::stod(m[1]) / 0.0254,
                      std::stod(m[2]) / 0.0254, std::stod(m[3]) / 0.0254);
        line = buffer;
      }
      out << line << '\n';
    }
    std::ofstream(inches / "points") << out.str();
  }
  const auto euler = [&](Json& d) {
    d["precision"] = precision;
    inviscid(d);
    for (auto& b : d["boundaries"]) {
      if (b["zone"] == "ymin") b["kind"] = "slip";
    }
    d["monitors"][0]["every"] = 1;
    d["time"] = {{"kind", "unsteady"},      {"steps", 5},
                 {"dt", nullptr},           {"cfl", 0.8},
                 {"average_from", nullptr}, {"checkpoint", {{"seconds", 1800.0}}}};
  };
  const Json metres = input(euler);
  Json in_inches = metres;
  {
    std::string toml = in_inches["case"]["toml"];
    const std::string unit = "mesh_unit = \"m\"";
    REQUIRE(toml.find(unit) != std::string::npos);
    in_inches["case"]["toml"] = toml.replace(toml.find(unit), unit.size(), "mesh_unit = \"in\"");
  }
  in_inches["mesh"]["path"] = inches.string();
  in_inches["mesh"]["sha256"] = autan::io::mesh_hash(inches);
  in_inches["mesh"]["scale"] = 0.0254;
  in_inches = autan_test::planned(in_inches, mesh_path);
  autan::runtime::PhysicsRegistry registry;
  autan::physics::register_euler(registry);
  const Outcome a = run(metres, registry);
  const Outcome b = run(in_inches, registry);
  REQUIRE(a.status == ExitStatus::finished);
  REQUIRE(b.status == ExitStatus::finished);
  // The archived MeshInfo is the planned one, of the file in its units.
  CHECK(read(b.run / "mesh-info.json") == in_inches["mesh_info"].dump(2));

  // The fields' precision; the coordinates are FP64 in either.
  double eps = std::string(precision) == "fp64" ? 0x1p-52 : 0x1p-23;
  const auto close = [&](const std::vector<double>& x, const std::vector<double>& y) {
    REQUIRE(x.size() == y.size());
    double scale = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      scale = std::max(scale, std::abs(x[i]));
      worst = std::max(worst, std::abs(x[i] - y[i]));
    }
    CAPTURE(worst / (eps * scale));
    CHECK(worst <= 64 * eps * scale);
  };
  for (const char* field : {"Density", "VelocityX", "VelocityZ", "Pressure"}) {
    CAPTURE(field);
    close(solution_field(a.run / "solution.cgns", "FlowSolution", field),
          solution_field(b.run / "solution.cgns", "FlowSolution", field));
  }
  const Json fa = history(a.run).back()["monitors"]["forces"];
  const Json fb = history(b.run).back()["monitors"]["forces"];
  const auto values = [](const Json& f, std::initializer_list<const char*> keys) {
    std::vector<double> v;
    for (const char* k : keys) v.push_back(f[k].get<double>());
    return v;
  };
  close(values(fa, {"fx", "fy", "fz"}), values(fb, {"fx", "fy", "fz"}));
  close(values(fa, {"mx", "my", "mz"}), values(fb, {"mx", "my", "mz"}));
  close(values(fa, {"cd", "cs", "cl"}), values(fb, {"cd", "cs", "cl"}));
  eps = 0x1p-52;
  for (const char* axis : {"CoordinateX", "CoordinateY", "CoordinateZ"}) {
    CAPTURE(axis);
    close(coordinates(a.run / "solution.cgns", axis), coordinates(b.run / "solution.cgns", axis));
  }
}

TEST_CASE("A run archives the zone groups its case was resolved with", "[runtime][groups]") {
  // The hybrid mesh copied with a zone-groups sidecar: the input names the
  // sidecar's hash (mesh.groups_sha256); run.json records it, the identity
  // covers it, the archived MeshInfo carries the groups. A sidecar that is not
  // the one named, present or absent, is refused before any directory.
  const fs::path mesh = fresh_dir() / "polyMesh";
  fs::copy(mesh_path, mesh, fs::copy_options::recursive);
  const std::string hash = autan::io::mesh_hash(mesh);
  const fs::path sidecar = mesh.string() + ".zones.json";
  std::ofstream(sidecar) << Json{{"schema", "autan.zone-groups/1"},
                                 {"mesh_sha256", hash},
                                 {"provenance", Json::object()},
                                 {"groups", {{"walls", {"ymin", "zmax"}}}}}
                                .dump();
  const std::string groups_hash = autan::io::mesh_hash(sidecar);
  // An input as the SDK writes it for a case planned with the sidecar `groups_sha256` names, or
  // with none: its MeshInfo's groups agree with it.
  const auto on = [&](const Json& groups_sha256) {
    Json d = input([&](Json& e) {
      e["mesh"]["path"] = mesh.string();
      e["mesh"]["sha256"] = hash;
    });
    d["mesh"]["groups_sha256"] = groups_sha256;
    if (groups_sha256.is_null()) {
      d["mesh_info"]["groups"] = Json::array();
      d["mesh_info"]["groups_source"] = nullptr;
    } else {
      d["mesh_info"]["groups"] = Json::parse(R"([{"name": "walls", "zones": ["ymin", "zmax"]}])");
      d["mesh_info"]["groups_source"] = {
          {"path", sidecar.string()}, {"sha256", groups_sha256}, {"provenance", Json::object()}};
    }
    return d;
  };
  const Outcome o = run(on(groups_hash), {.factor = 0.5});
  REQUIRE(o.status == ExitStatus::finished);
  const Json info = Json::parse(read(o.run / "run.json"));
  CHECK(info["mesh"]["groups_sha256"] == groups_hash);
  const Json archived = Json::parse(read(o.run / "mesh-info.json"));
  CHECK(archived["groups"] == Json::parse(R"([{"name": "walls", "zones": ["ymin", "zmax"]}])"));
  CHECK(archived["groups_source"]["sha256"] == groups_hash);
  // The identity covers the groups' hash (results.md), the cell order RCM by default.
  std::string expected;
  picosha2::hash256_hex_string(info["case"]["sha256"].get<std::string>() + "\n" + hash + "\n" +
                                   groups_hash + "\n" + build.commit +
                                   "\nfp32+acc64\ncpu\ncpu\n1\nrcm",
                               expected);
  CHECK(info["identity"] == expected);

  require_error(run(on(nullptr)), "E0137");               // a sidecar the input does not name
  require_error(run(on(std::string(64, 'a'))), "E0137");  // another sidecar
  // One the input does not name is refused unread, whatever it holds: its plan counted no
  // memory for reading it (plan.md, zone groups), and reading this one would be E0136.
  std::ofstream(sidecar) << "{not json";
  require_error(run(on(nullptr)), "E0137");
  // A MeshInfo whose groups the runtime cannot take, or from another sidecar than the input
  // names, is refused as the input's (solver-input.md, mesh_info).
  Json bad = on(groups_hash);
  bad["mesh_info"]["groups"] = 42;
  require_error(run(bad), "E0705");
  bad = on(groups_hash);
  bad["mesh_info"]["groups_source"]["sha256"] = std::string(64, 'b');
  require_error(run(bad), "E0705");
  bad = on(groups_hash);
  bad["mesh_info"]["groups_source"]["provenance"] = {{"url", Json::array()}};
  require_error(run(bad), "E0705");
  bad = on(groups_hash);
  bad["mesh_info"]["groups"][0]["zones"] = {"ymin", "nowhere"};
  require_error(run(bad), "E0705");
  fs::remove(sidecar);
  require_error(run(on(groups_hash)), "E0137");  // the named sidecar is gone
}
