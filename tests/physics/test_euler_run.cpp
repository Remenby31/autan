// The Euler physics through autan-run's public entry point, run_case, on tests/data/mesh/hybrid:
// solver inputs written by `autan prepare` (tests/data/runtime), edited per test.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cgnslib.h>
#include <nlohmann/json.hpp>
#include <picosha2.h>

#include "autan/physics/euler_physics.hpp"
#include "autan/runtime/run.hpp"

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;
using autan::runtime::ExitStatus;

const fs::path data_dir{AUTAN_TEST_DATA_DIR};
const fs::path mesh_path = data_dir / "mesh" / "hybrid" / "polyMesh";
const fs::path mms_path = data_dir / ".." / ".." / "py" / "tests" / "fixtures" / "euler.mms.json";
const autan::runtime::SolverBuild build{.version = "0.0.0", .commit = std::string(40, 'a')};

std::string read(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

Json input(const char* file, const std::function<void(Json&)>& edit) {
  Json doc = Json::parse(read(data_dir / "runtime" / file));
  doc["mesh"]["path"] = fs::absolute(mesh_path).string();
  if (doc["verification"].is_object())
    doc["verification"]["path"] = fs::absolute(mms_path).string();
  edit(doc);
  return doc;
}

struct Outcome {
  ExitStatus status;
  fs::path run;
  fs::path runs;  // the runs directory passed to run_case
  std::string err;
};

Outcome run(const Json& doc) {
  std::random_device random;
  const fs::path dir = fs::temp_directory_path() / ("autan_euler-" + std::to_string(random()));
  fs::create_directories(dir);
  std::ofstream(dir / "solver-input.json") << doc.dump();
  autan::runtime::PhysicsRegistry registry;
  autan::physics::register_euler(registry);
  std::FILE* out = std::tmpfile();
  std::FILE* err = std::tmpfile();
  const ExitStatus status =
      autan::runtime::run_case(dir / "solver-input.json", dir / "runs", registry, build, out, err);
  auto text = [](std::FILE* f) {
    std::rewind(f);
    std::string s;
    for (int c = std::fgetc(f); c != EOF; c = std::fgetc(f)) s.push_back(static_cast<char>(c));
    std::fclose(f);
    return s;
  };
  std::string printed = text(out);
  if (!printed.empty() && printed.back() == '\n') printed.pop_back();
  return Outcome{.status = status, .run = printed, .runs = dir / "runs", .err = text(err)};
}

std::vector<Json> history(const fs::path& run) {
  std::vector<Json> lines;
  std::ifstream in(run / "history.jsonl");
  for (std::string line; std::getline(in, line);) lines.push_back(Json::parse(line));
  return lines;
}

// The `start` the SDK writes to continue `previous` (solver-input.md).
Json continue_from(const fs::path& previous) {
  const Json info = Json::parse(read(previous / "run.json"));
  std::string hash;
  picosha2::hash256_hex_string(read(previous / "solution.cgns"), hash);
  Json reference = nullptr;
  if (info["time"].is_null()) reference = history(previous)[0]["residuals"];
  return Json{{"run", previous.string()},         {"run_id", info["run_id"]},
              {"identity", info["identity"]},     {"mode", "continue"},
              {"iterations", info["iterations"]}, {"time", info["time"]},
              {"solution_sha256", hash},          {"residual_reference", reference}};
}

// Field `name` of the cell-centred FlowSolution_t `solution` of a run's solution.cgns.
std::vector<double> field(const fs::path& run, const char* name,
                          const char* solution = "FlowSolution") {
  int fn = 0;
  REQUIRE(cg_open((run / "solution.cgns").string().c_str(), CG_MODE_READ, &fn) == CG_OK);
  char zone_name[33];
  cgsize_t size[3];
  REQUIRE(cg_zone_read(fn, 1, 1, zone_name, size) == CG_OK);
  int count = 0;
  REQUIRE(cg_nsols(fn, 1, 1, &count) == CG_OK);
  std::vector<double> values;
  for (int s = 1; s <= count; ++s) {
    char sol_name[33];
    CGNS_ENUMT(GridLocation_t) location{};
    REQUIRE(cg_sol_info(fn, 1, 1, s, sol_name, &location) == CG_OK);
    if (std::string(sol_name) != solution) continue;
    values.resize(static_cast<std::size_t>(size[1]));
    const cgsize_t first = 1;
    REQUIRE(cg_field_read(fn, 1, 1, s, name, CGNS_ENUMV(RealDouble), &first, &size[1],
                          values.data()) == CG_OK);
  }
  cg_close(fn);
  REQUIRE(!values.empty());
  return values;
}

// The case of hybrid.solver-input.json made inviscid, with the freestream along x so that it is
// tangent to the slip wall ymin and the symmetry plane zmax; farfield elsewhere.
void inviscid_along_x(Json& doc, const char* precision) {
  doc["precision"] = precision;
  auto& t = doc["physics"]["turbulence"];
  t["model"] = "inviscid";
  t["nu_tilde_ratio"] = nullptr;
  auto& f = doc["freestream"];
  f["velocity"] = {f["speed"], 0.0, 0.0};
  f["aoa"] = 0.0;
  f["nu_tilde"] = nullptr;
  for (auto& z : doc["cell_zones"]) {
    z["initial"]["velocity"] = f["velocity"];
    z["initial"]["nu_tilde"] = nullptr;
  }
  for (auto& b : doc["boundaries"]) {
    if (b["kind"] == "wall") b["kind"] = "slip";
  }
}

}  // namespace

TEST_CASE("a uniform flow along slip walls and farfield faces stays uniform", "[euler]") {
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const bool steady = GENERATE(true, false);
  INFO(precision << (steady ? " steady" : " unsteady"));
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    if (steady) {
      d["time"]["max_iterations"] = 20;
    } else {
      d["time"] = {
          {"kind", "unsteady"}, {"steps", 20}, {"dt", nullptr}, {"cfl", 0.8}, {"average_from", 10}};
    }
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const Json info = Json::parse(read(out.run / "run.json"));
  CHECK(info["diagnostics"].empty());
  const auto& f = doc["freestream"];
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  const double speed = f["speed"].get<double>();
  const double p = f["pressure"].get<double>();
  const double rho = f["density"].get<double>();
  // Each step changes the state by round-off of the face fluxes: 20 steps stay within 64 eps.
  for (const double v : field(out.run, "Density")) CHECK(std::abs(v - rho) <= 64 * eps * rho);
  for (const double v : field(out.run, "Pressure")) CHECK(std::abs(v - p) <= 64 * eps * p);
  for (const double v : field(out.run, "VelocityX")) CHECK(std::abs(v - speed) <= 64 * eps * speed);
  for (const double v : field(out.run, "VelocityZ")) CHECK(std::abs(v) <= 64 * eps * speed);
  // The pressure on the slip wall is the freestream's: no force.
  const auto lines = history(out.run);
  const auto& forces = lines.back()["monitors"]["forces"];
  for (const char* k : {"fx", "fy", "fz"}) CHECK(std::abs(forces[k].get<double>()) <= 64 * eps * p);
  if (!steady) CHECK(info["time"].get<double>() > 0.0);
}

TEST_CASE("a verification case converges to its discrete solution", "[euler]") {
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const auto doc = input("hybrid-verification.solver-input.json", [&](Json& d) {
    d["precision"] = precision;
    d["time"]["max_iterations"] = 20000;
    d["time"]["residual_orders"] = std::string(precision) == "fp64" ? 10.0 : 5.0;
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const Json info = Json::parse(read(out.run / "run.json"));
  CHECK(info["status"] == "converged");
  CHECK(fs::exists(out.run / "manufactured.json"));
}

TEST_CASE("a continued run reproduces the uninterrupted one bit for bit", "[euler]") {
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const bool steady = GENERATE(true, false);
  INFO(precision << (steady ? " steady" : " unsteady"));
  auto doc = [&](int budget) {
    return input("hybrid-verification.solver-input.json", [&](Json& d) {
      d["precision"] = precision;
      if (steady) {
        d["time"]["max_iterations"] = budget;
        d["time"]["residual_orders"] = 30.0;
      } else {
        d["verification"] = nullptr;
        for (auto& b : d["boundaries"]) b["kind"] = "farfield";
        // Two cell zones at different states: a nonuniform, evolving flow.
        auto& init = d["cell_zones"].back()["initial"];
        init["density"] = 1.2 * init["density"].get<double>();
        init["pressure"] = 1.3 * init["pressure"].get<double>();
        d["time"] = {{"kind", "unsteady"},
                     {"steps", budget},
                     {"dt", nullptr},
                     {"cfl", 0.8},
                     {"average_from", 2}};
      }
    });
  };
  const auto whole = run(doc(12));
  REQUIRE(whole.status == ExitStatus::finished);
  const auto first = run(doc(6));
  REQUIRE(first.status == ExitStatus::finished);
  auto next = doc(12);
  next["start"] = continue_from(first.run);
  const auto second = run(next);
  INFO(second.err);
  REQUIRE(second.status == ExitStatus::finished);
  auto lines = [](const fs::path& f) {
    std::vector<std::string> out;
    std::ifstream in(f);
    for (std::string l; std::getline(in, l);) out.push_back(l);
    return out;
  };
  auto chain = lines(first.run / "history.jsonl");
  const auto more = lines(second.run / "history.jsonl");
  chain.insert(chain.end(), more.begin(), more.end());
  CHECK(chain == lines(whole.run / "history.jsonl"));
  for (const char* name : {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}) {
    CHECK(field(second.run, name) == field(whole.run, name));
    if (!steady) {
      CHECK(field(second.run, name, "FlowSolutionMean") ==
            field(whole.run, name, "FlowSolutionMean"));
    }
  }
  for (const char* name : {"Density", "MomentumX", "EnergyStagnationDensity"}) {
    CHECK(field(second.run, name, "RestartState") == field(whole.run, name, "RestartState"));
  }
  if (!steady) {
    CHECK(field(second.run, "AverageWeight", "RestartState") ==
          field(whole.run, "AverageWeight", "RestartState"));
  }
}

TEST_CASE("cases outside the Euler physics are refused before any directory exists", "[euler]") {
  // Two gases, or a solid zone: E0706 on stderr, exit 2, no run directory.
  const bool solid = GENERATE(false, true);
  INFO((solid ? "solid zone" : "two gases"));
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, "fp64");
    auto& z = d["cell_zones"].back();
    if (solid) {
      z["medium"] = {{"kind", "solid"},
                     {"name", "steel"},
                     {"density", 7800.0},
                     {"specific_heat", 500.0},
                     {"conductivity", 40.0}};
      z["initial"] = {{"kind", "solid"}, {"temperature", 300.0}};
    } else {
      z["medium"]["name"] = "helium";
      z["medium"]["gas_constant"] = 2077.1;
      z["medium"]["gamma"] = 5.0 / 3.0;
    }
  });
  const auto out = run(doc);
  INFO(out.err);
  CHECK(out.status == ExitStatus::invalid_input);
  CHECK(out.run.empty());
  CHECK(!fs::exists(out.runs));
  const auto diagnostic = Json::parse(out.err.substr(0, out.err.find('\n')));
  CHECK(diagnostic["code"] == "E0706");
  CHECK(diagnostic["data"]["zone"] == doc["cell_zones"].back()["zone"]);
  CHECK(diagnostic["data"]["path"] ==
        "cell_zones[" + std::to_string(doc["cell_zones"].size() - 1) + "].medium");
}
