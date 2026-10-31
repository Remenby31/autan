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
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cgnslib.h>
#include <nlohmann/json.hpp>
#include <picosha2.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
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
    cgsize_t size[3];
    REQUIRE(cg_zone_read(fn, 1, 1, zone_name, size) == CG_OK);
    values.resize(static_cast<std::size_t>(size[1]));
    const cgsize_t first = 1;
    REQUIRE(cg_field_read(fn, 1, 1, s, name, CGNS_ENUMV(RealDouble), &first, &size[1],
                          values.data()) == CG_OK);
  }
  cg_close(fn);
  return values;
}

void require_error(const Outcome& o, const std::string& code) {
  CHECK(o.status == ExitStatus::invalid_input);
  CHECK(o.run.empty());
  CHECK((!fs::exists(o.runs) || fs::is_empty(o.runs)));
  REQUIRE(!o.err.empty());
  CHECK(Json::parse(o.err)["code"] == code);
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
  CHECK(files == std::vector<std::string>{"case.toml", "history.jsonl", "mesh-info.json",
                                          "run.json", "solution.cgns"});

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
    for (const auto& e : equations)
      CHECK(lines[k]["residuals"][e] == std::pow(0.5, static_cast<double>(k)));
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
    // Written by `autan prepare` with time = Unsteady(steps=6, dt=0.01 s, average_from=3).
    const Outcome o = run(input({}, "hybrid-unsteady.solver-input.json"));
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
    CHECK(solution_field(o.run / "solution.cgns", "FlowSolutionMean", "Density") ==
          solution_field(o.run / "solution.cgns", "FlowSolution", "Density"));
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
