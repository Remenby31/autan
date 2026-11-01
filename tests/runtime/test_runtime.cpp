#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cgnslib.h>
#include <nlohmann/json.hpp>
#include <picosha2.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include "autan/io/cgns.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/runtime/input.hpp"
#include "autan/runtime/run.hpp"
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
const autan::runtime::SolverBuild build{.version = "0.0.0", .commit = std::string(40, 'a')};

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
  return doc;
}

struct Outcome {
  ExitStatus status;
  fs::path runs;
  fs::path run;  // empty when no directory was created
  std::string err;
};

Outcome run(const Json& doc, const autan::runtime::PhysicsRegistry& registry) {
  const fs::path dir = fresh_dir();
  const fs::path input_file = dir / "solver-input.json";
  std::ofstream(input_file) << doc.dump();
  std::FILE* out = std::tmpfile();
  std::FILE* err = std::tmpfile();
  const ExitStatus status =
      autan::runtime::run_case(input_file, dir / "runs", registry, build, out, err);
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

Outcome run(const Json& doc, autan_test::RelaxationOptions options = {}) {
  return run(doc, autan_test::relaxation_registry(options));
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

// The `start` the SDK writes for a run that starts from `previous`
// (solver-input.md): its run.json, the hash of its solution and, for a
// steady `continue`, the residuals at iteration 1 of the chain's first run.
Json start_from(const fs::path& previous, const std::string& mode) {
  const Json info = Json::parse(read(previous / "run.json"));
  Json reference = nullptr;
  if (mode == "continue" && info["time"].is_null()) {
    const bool chained = !info["start"].is_null() && info["start"]["mode"] == "continue";
    reference = chained ? info["start"]["residual_reference"] : history(previous)[0]["residuals"];
  }
  return Json{{"run", previous.string()},
              {"run_id", info["run_id"]},
              {"identity", info["identity"]},
              {"mode", mode},
              {"iterations", info["iterations"]},
              {"time", info["time"]},
              {"solution_sha256", file_sha256(previous / "solution.cgns")},
              {"residual_reference", reference}};
}

std::vector<std::string> lines_of(const fs::path& file) {
  std::vector<std::string> lines;
  std::ifstream in(file);
  for (std::string line; std::getline(in, line);) lines.push_back(line);
  return lines;
}

}  // namespace

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
  CHECK(std::regex_match(info["run_id"].get<std::string>(),
                         std::regex("run-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}")));
  CHECK(o.run.filename() == info["run_id"].get<std::string>());
  CHECK(std::regex_match(info["finished"].get<std::string>(),
                         std::regex("[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z")));
  CHECK(info["mesh"]["path"] == fs::absolute(mesh_path).lexically_normal().string());
  CHECK(info["solver"] == Json{{"version", "0.0.0"}, {"commit", build.commit}, {"backend", "cpu"}});
  // identity: SHA-256 of the seven fields joined by "\n" (results.md).
  std::string expected;
  picosha2::hash256_hex_string(doc["case"]["sha256"].get<std::string>() + "\n" +
                                   doc["mesh"]["sha256"].get<std::string>() + "\n" + build.commit +
                                   "\nfp32+acc64\ncpu\ncpu\n1",
                               expected);
  CHECK(info["identity"] == expected);
  CHECK(read(o.run / "case.toml") == doc["case"]["toml"].get<std::string>());
  CHECK(read(o.run / "mesh-info.json") ==
        autan::io::mesh_info_json(autan::io::read_mesh(mesh_path),
                                  doc["mesh"]["sha256"].get<std::string>()));

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

  const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
  const autan::Mesh back = autan::io::read_mesh(o.run / "solution.cgns");
  CHECK(back.n_cells() == mesh.n_cells());
  const auto density = solution_field(o.run / "solution.cgns", "FlowSolution", "Density");
  REQUIRE(density.size() == static_cast<std::size_t>(mesh.n_cells()));
  for (const double rho : density)
    CHECK_THAT(rho, WithinRel(doc["freestream"]["density"].get<double>(), 1e-15));
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
    // The diverged state holds a NaN: no solution can satisfy the readers.
    CHECK_FALSE(fs::exists(o.run / "solution.cgns"));
  }
  SECTION("failed: E0704, history kept, exit 1") {
    const Outcome o = run(input(), {.factor = 0.9, .throw_at = 4});
    CHECK(o.status == ExitStatus::failed);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "failed");
    CHECK(info["diagnostics"][0]["code"] == "E0704");
    CHECK(history(o.run).size() == 3);
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
  }
  SECTION("E0705 value a kind forbids") {
    require_error(run(input([](Json& d) { d["boundaries"][0]["static_pressure"] = 1e5; })),
                  "E0705");
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
  SECTION("E0705 zone absent from the mesh") {
    require_error(run(input([](Json& d) { d["boundaries"][0]["zone"] = "nowhere"; })), "E0705");
  }
  SECTION("E0203 mesh hash") {
    require_error(run(input([](Json& d) { d["mesh"]["sha256"] = std::string(64, '0'); })), "E0203");
  }
  SECTION("E0706 no physics in the build") {
    require_error(run(input(), autan::runtime::PhysicsRegistry{}), "E0706");
  }
}

TEST_CASE("A physics' scope refuses an input before any directory", "[runtime]") {
  const bool refuse = GENERATE(true, false);
  autan::runtime::PhysicsRegistry registry;
  registry.add(
      {.kind = "*", .turbulence = "*", .precision = "*", .device = "*"},
      [](const autan::runtime::PhysicsContext& context) {
        return std::make_unique<autan_test::Relaxation>(context, autan_test::RelaxationOptions{});
      },
      [refuse](
          const autan::runtime::SolverInput& in) -> std::optional<autan::runtime::PhysicsRefusal> {
        if (!refuse) return std::nullopt;
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
  const Json doc = input([](Json& d) { d["deterministic"] = true; });
  const Outcome a = run(doc);
  const Outcome b = run(doc);
  CHECK(read(a.run / "history.jsonl") == read(b.run / "history.jsonl"));
  for (const char* field : {"Density", "Pressure", "TurbulentSANuTilde"}) {
    CHECK(solution_field(a.run / "solution.cgns", "FlowSolution", field) ==
          solution_field(b.run / "solution.cgns", "FlowSolution", field));
  }
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
  const Json f = history(o.run)[4]["monitors"]["forces"];
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
  for (std::size_t k = 0; k < 3; ++k) {
    CHECK_THAT(f["f" + axes[k]].get<double>(), Catch::Matchers::WithinAbs(force[k], 1e-9));
    CHECK_THAT(f["m" + axes[k]].get<double>(), Catch::Matchers::WithinAbs(moment[k], 1e-9));
    CHECK_THAT(f["cm" + axes[k]].get<double>(), Catch::Matchers::WithinAbs(moment[k] / q, 1e-9));
  }
  CHECK_THAT(f["cd"].get<double>(), Catch::Matchers::WithinAbs(dot(force, drag) / q, 1e-12));
  CHECK_THAT(f["cl"].get<double>(), Catch::Matchers::WithinAbs(dot(force, lift) / q, 1e-12));
  CHECK_THAT(f["cs"].get<double>(), Catch::Matchers::WithinAbs(dot(force, side) / q, 1e-12));
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
  SECTION("mesh path is absolute") {
    require_error(run(input([](Json& d) { d["mesh"]["path"] = "relative/polyMesh"; })), "E0705");
  }
  SECTION("hashes are 64 lowercase hexadecimal characters") {
    require_error(run(input([](Json& d) { d["case"]["sha256"] = "ABC"; })), "E0705");
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
  CHECK(lines[1]["message"] == "relaxation physics ready");  // the physics' own line
  // The run.json diagnostic appears with its code and data.
  const Json info = Json::parse(read(o.run / "run.json"));
  const auto d0107 =
      std::ranges::find_if(lines, [](const Json& l) { return l["code"] == "D0107"; });
  REQUIRE(d0107 != lines.end());
  CHECK((*d0107)["data"] == info["diagnostics"][0]["data"]);
  CHECK((*d0107)["iteration"] == 4);
  CHECK(lines.back()["data"]["status"] == "diverged");
}

TEST_CASE("A continued run goes on bit for bit from the run it continues", "[runtime]") {
  // The state grows at each step, so that the history (forces) and the
  // solution depend on the state handed over; the uninterrupted run is the
  // reference (results.md: same identity parts, same iterations).
  const bool steady = GENERATE(true, false);
  CAPTURE(steady);
  const autan_test::RelaxationOptions options{.growth = 1e-3};
  const auto budget = [&](std::int64_t n) {
    return [steady, n](Json& d) {
      d["deterministic"] = true;
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
  // identity: the seven parts, then the start's identity, mode, iterations
  // and solution hash, joined by "\n" (results.md).
  const auto& s = second["start"];
  std::string identity;
  picosha2::hash256_hex_string(second["case"]["sha256"].get<std::string>() + "\n" +
                                   second["mesh"]["sha256"].get<std::string>() + "\n" +
                                   build.commit + "\nfp32+acc64\ncpu\ncpu\n1\n" +
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
  CHECK(info["start"]["residual_reference"].is_null());
  const auto previous = solution_field(a.run / "solution.cgns", "FlowSolution", "Density");
  CHECK(solution_field(b.run / "solution.cgns", "FlowSolution", "Density") == previous);
  // 1.25^5 times the initial density: not the freestream state.
  CHECK_THAT(previous[0],
             WithinRel(doc["freestream"]["density"].get<double>() * std::pow(1.25, 5.0), 1e-14));
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
  SECTION("the previous run used another precision") {
    require_error(from([](Json& d) { d["precision"] = "fp64"; }), "E0708");
  }
  SECTION("the previous run solved other variables") {
    require_error(from([](Json& d) {
                    d["start"]["mode"] = "warm";
                    d["start"]["residual_reference"] = nullptr;
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
    // A solution.cgns whose RunMetadata has no `averaging`, with its hash.
    const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
    Json metadata = Json::parse(read(a.run / "run.json"));
    metadata.erase("averaging");
    autan::io::Solution solution;
    solution.run_metadata = metadata.dump();
    solution.boundary_bc_types.assign(autan::io::meshinfo_zones(mesh).boundary.size(),
                                      "BCFarfield");
    for (const auto& name :
         autan::runtime::field_names(autan::runtime::parse_solver_input(input().dump()))) {
      solution.fields.push_back(
          {.name = name,
           .values = solution_field(a.run / "solution.cgns", "FlowSolution", name.c_str())});
    }
    const fs::path previous = fresh_dir();
    autan::io::write_solution(previous / "solution.cgns", mesh, solution);
    require_error(from([&](Json& d) {
                    d["start"]["run"] = previous.string();
                    d["start"]["solution_sha256"] = file_sha256(previous / "solution.cgns");
                  }),
                  "E0707");
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
    require_error(from([](Json& d) { d["start"]["mode"] = "warm"; }), "E0705");
    require_error(from([](Json& d) { d["time"]["max_iterations"] = 4; }), "E0705");
    require_error(from([](Json& d) { d["start"]["run"] = "relative/run"; }), "E0705");
    require_error(from([](Json& d) { d["start"]["residual_reference"].erase("energy"); }), "E0705");
  }
}

TEST_CASE("A mesh with periodic pairs is E0706 until the runtime solves them", "[runtime]") {
  const fs::path box = data_dir / "mesh" / "periodic-box" / "periodic-box.ascii.msh";
  const auto with_pairs = [&](double tx) {
    return input([&](Json& d) {
      d["mesh"]["path"] = fs::absolute(box).string();
      d["mesh"]["sha256"] = autan::io::mesh_hash(box);
      d["periodic"] = Json::array(
          {Json{{"zones", {"xmin", "xmax"}},
                {"transform", {{"kind", "translation"}, {"vector", {tx, 0.0, 0.0}}}}},
           Json{{"zones", {"ymin", "ymax"}},
                {"transform", {{"kind", "translation"}, {"vector", {0.0, 1.0, 0.0}}}}}});
    });
  };
  require_error(run(with_pairs(1.0)), "E0706");
  require_error(run(with_pairs(2.0)), "E0705");
}

// face-states.md: tests/data/mesh/ringleb, its states written by autan-vv, on the four in-plane
// sides; `edit` changes the document before it is written beside the input.
Json ringleb_input(const std::function<void(Json&)>& edit = {}) {
  const fs::path mesh = data_dir / "mesh" / "ringleb" / "ringleb-n004.msh";
  Json states = Json::parse(read(data_dir / "mesh" / "ringleb" / "ringleb-n004.states.json"));
  std::string bytes = read(data_dir / "mesh" / "ringleb" / "ringleb-n004.states.json");
  if (edit) {
    edit(states);
    bytes = states.dump();
  }
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
}

TEST_CASE("Face states that do not fit the case exit 2 before any directory", "[runtime]") {
  SECTION("E0420 another file than states.sha256") {
    Json doc = ringleb_input();
    doc["boundaries"][0]["states"]["sha256"] = std::string(64, '0');
    require_error(run(doc), "E0420");
  }
  SECTION("E0420 not a face-state document") {
    require_error(run(ringleb_input([](Json& s) { s["variables"][4] = "temperature"; })), "E0420");
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
