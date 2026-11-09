// The Euler and Navier-Stokes physics through autan-run's public entry point, run_case, on
// tests/data/mesh/hybrid: solver inputs written by `autan prepare` (tests/data/runtime), edited
// per test.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cgnslib.h>
#include <nlohmann/json.hpp>
#include <picosha2.h>
#ifdef AUTAN_MPIEXEC
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

#include "../runtime/planned.hpp"
#include "autan/core/cellwise.hpp"
#include "autan/core/geometry.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/io/write_polymesh.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/partition/partition.hpp"
#include "autan/physics/euler_physics.hpp"
#include "autan/physics/first_order_start.hpp"
#include "autan/physics/manufactured.hpp"
#include "autan/runtime/physics.hpp"
#include "autan/runtime/run.hpp"
#include "fixtures.hpp"

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
  return autan_test::planned(doc, mesh_path);
}

// How many times a run logged that it takes the planar loops (EulerPhysics, #412).
int planar_events(const fs::path& run) {
  int n = 0;
  std::istringstream log(read(run / "log.jsonl"));
  for (std::string l; std::getline(log, l);) {
    const auto e = Json::parse(l);
    if (e["source"] == "solver" &&
        e["message"].get<std::string>().starts_with("every face is planar")) {
      ++n;
    }
  }
  return n;
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
  autan::physics::register_navier_stokes(registry);
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
Json continue_from(const fs::path& previous, const std::string& mode = "continue") {
  const Json info = Json::parse(read(previous / "run.json"));
  std::string hash;
  picosha2::hash256_hex_string(read(previous / "solution.cgns"), hash);
  Json reference = nullptr;
  Json woken = nullptr;
  if (info["time"].is_null()) {
    // case-schema.md#time: a steady continue or warm start takes the reference;
    // continue also the equations below 1e-12 of its largest that went above that floor.
    const auto lines = history(previous);
    reference = lines[0]["residuals"];
    double largest = 0.0;
    for (const auto& [name, f] : reference.items()) largest = std::max(largest, f.get<double>());
    std::vector<std::string> names;
    for (const auto& [name, f] : reference.items()) {
      const bool left = std::ranges::any_of(lines, [&](const Json& line) {
        return line["residuals"][name].get<double>() > 1e-12 * largest;
      });
      if (f.get<double>() < 1e-12 * largest && left) names.push_back(name);
    }
    std::ranges::sort(names);
    woken = mode == "warm" ? Json::array() : Json(names);
  }
  return Json{{"run", previous.string()},         {"run_id", info["run_id"]},
              {"identity", info["identity"]},     {"mode", mode},
              {"iterations", info["iterations"]}, {"time", info["time"]},
              {"solution_sha256", hash},          {"file", "solution"},
              {"residual_reference", reference},  {"woken", woken}};
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

// Every data array a run computes in its solution.cgns, keyed "node/name": the fields of each
// FlowSolution_t (FlowSolution, RestartState) and the arrays of the WallSolution subregion
// (results.md, solution.cgns).
std::map<std::string, std::vector<double>> data_arrays(const fs::path& run) {
  std::map<std::string, std::vector<double>> out;
  int fn = 0;
  REQUIRE(cg_open((run / "solution.cgns").string().c_str(), CG_MODE_READ, &fn) == CG_OK);
  char zone_name[33];
  cgsize_t size[3];
  REQUIRE(cg_zone_read(fn, 1, 1, zone_name, size) == CG_OK);
  int solutions = 0;
  REQUIRE(cg_nsols(fn, 1, 1, &solutions) == CG_OK);
  for (int s = 1; s <= solutions; ++s) {
    char sol_name[33];
    CGNS_ENUMT(GridLocation_t) location{};
    REQUIRE(cg_sol_info(fn, 1, 1, s, sol_name, &location) == CG_OK);
    int fields = 0;
    REQUIRE(cg_nfields(fn, 1, 1, s, &fields) == CG_OK);
    for (int f = 1; f <= fields; ++f) {
      CGNS_ENUMT(DataType_t) type{};
      char name[33];
      REQUIRE(cg_field_info(fn, 1, 1, s, f, &type, name) == CG_OK);
      std::vector<double> values(static_cast<std::size_t>(size[1]));
      const cgsize_t first = 1;
      REQUIRE(cg_field_read(fn, 1, 1, s, name, CGNS_ENUMV(RealDouble), &first, &size[1],
                            values.data()) == CG_OK);
      out[std::string(sol_name) + "/" + name] = std::move(values);
    }
  }
  int regions = 0;
  REQUIRE(cg_nsubregs(fn, 1, 1, &regions) == CG_OK);
  for (int r = 1; r <= regions; ++r) {
    char name[33];
    int dimension = 0, bc_name_length = 0, gc_name_length = 0;
    CGNS_ENUMT(GridLocation_t) location{};
    CGNS_ENUMT(PointSetType_t) set{};
    cgsize_t points = 0;
    REQUIRE(cg_subreg_info(fn, 1, 1, r, name, &dimension, &location, &set, &points, &bc_name_length,
                           &gc_name_length) == CG_OK);
    if (std::string(name) != "WallSolution") continue;
    REQUIRE(cg_goto(fn, 1, "Zone_t", 1, "ZoneSubRegion_t", r, "end") == CG_OK);
    int arrays = 0;
    REQUIRE(cg_narrays(&arrays) == CG_OK);
    for (int a = 1; a <= arrays; ++a) {
      char array_name[33];
      CGNS_ENUMT(DataType_t) type{};
      int dims = 0;
      cgsize_t extent[12];
      REQUIRE(cg_array_info(a, array_name, &type, &dims, extent) == CG_OK);
      std::vector<double> values(static_cast<std::size_t>(extent[0]));
      REQUIRE(cg_array_read_as(a, CGNS_ENUMV(RealDouble), values.data()) == CG_OK);
      out[std::string("WallSolution/") + array_name] = std::move(values);
    }
  }
  cg_close(fn);
  return out;
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

// The same case with laminar viscous flow.
void laminar_along_x(Json& doc, const char* precision) {
  inviscid_along_x(doc, precision);
  doc["physics"]["turbulence"]["model"] = "laminar";
}

const fs::path ringleb_n004 = data_dir / "mesh" / "ringleb" / "ringleb-n004.msh";

// tests/data/mesh/ringleb, one cell thick in z, inviscid along x from a uniform state at 1.3 times
// the freestream pressure and an initial z-velocity of w times the speed: farfield on its sides,
// symmetry on its faces of constant z, `iterations` steps at most.
Json ringleb_slab(double w, std::int64_t iterations) {
  return input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, "fp64");
    d["mesh"]["path"] = fs::absolute(ringleb_n004).string();
    d["mesh"]["sha256"] = autan::io::mesh_hash(ringleb_n004);
    const double speed = d["freestream"]["speed"].get<double>();
    Json zone = d["cell_zones"][0];
    zone["zone"] = "fluid";
    zone["initial"]["pressure"] = 1.3 * zone["initial"]["pressure"].get<double>();
    zone["initial"]["velocity"] = {speed, 0.0, w * speed};
    d["cell_zones"] = Json::array({zone});
    Json boundaries = Json::array();
    for (const char* name : {"k-min", "k-max", "inflow", "outflow", "back", "front"}) {
      Json b = d["boundaries"][0];
      for (auto& [key, value] : b.items()) {
        if (key != "zone" && key != "kind") value = nullptr;
      }
      b["zone"] = name;
      const bool side = std::string(name) != "back" && std::string(name) != "front";
      b["kind"] = side ? "farfield" : "symmetry";
      boundaries.push_back(b);
    }
    d["boundaries"] = boundaries;
    d["monitors"] = Json::array();
    d["time"]["max_iterations"] = iterations;
    d["time"]["residual_orders"] = 30.0;
  });
}

}  // namespace

TEST_CASE("a uniform flow along slip walls and farfield faces stays uniform", "[euler]") {
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const bool steady = GENERATE(true, false);
  INFO(precision << (steady ? " steady" : " unsteady"));
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    // fp32+acc64 starts at round-off: it converges once its residuals are on a plateau
    // (case-schema.md#time), after the first-order start's provisional steps and 200 more.
    d["monitors"][0]["every"] = 1;
    if (steady) {
      d["time"]["max_iterations"] = 400;
    } else {
      d["time"] = {{"kind", "unsteady"}, {"steps", 20},
                   {"dt", nullptr},      {"cfl", 0.8},
                   {"average_from", 10}, {"checkpoint", {{"seconds", 1800.0}}}};
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
  // Each step changes the state by round-off of the face fluxes: the run stays within 64 eps.
  for (const double v : field(out.run, "Density")) CHECK(std::abs(v - rho) <= 64 * eps * rho);
  for (const double v : field(out.run, "Pressure")) CHECK(std::abs(v - p) <= 64 * eps * p);
  for (const double v : field(out.run, "VelocityX")) CHECK(std::abs(v - speed) <= 64 * eps * speed);
  for (const double v : field(out.run, "VelocityZ")) CHECK(std::abs(v) <= 64 * eps * speed);
  // The pressure on the slip wall is the freestream's: no force.
  const auto lines = history(out.run);
  const auto& forces = lines.back()["monitors"]["forces"];
  for (const char* k : {"fx", "fy", "fz"}) CHECK(std::abs(forces[k].get<double>()) <= 64 * eps * p);
  if (!steady) CHECK(info["time"].get<double>() > 0.0);
  // case-schema.md#time: a steady fp32+acc64 run converges on a round-off plateau near its floor
  // 0.5 * 2^-23 * Phi_k, Phi_k = f_k sqrt(sum_i A_i^2 / N) from the freestream flux f_k and the
  // cell surfaces.
  if (!steady || std::string(precision) == "fp64") {
    CHECK(info["convergence"].is_null());
    return;
  }
  CHECK(info["status"] == "converged");
  const auto mesh = autan::io::read_mesh(mesh_path);
  const auto g = autan::compute_geometry(mesh);
  std::vector<double> surface(static_cast<std::size_t>(mesh.n_cells()), 0.0);
  for (std::size_t face = 0; face < static_cast<std::size_t>(mesh.n_faces()); ++face) {
    surface[static_cast<std::size_t>(mesh.owner()[face])] += g.face_area[face];
    if (face < static_cast<std::size_t>(mesh.n_interior_faces()))
      surface[static_cast<std::size_t>(mesh.neighbour()[face])] += g.face_area[face];
  }
  double sum = 0.0;
  for (const double a : surface) sum += a * a;
  const double area = std::sqrt(sum / static_cast<double>(surface.size()));
  const double gamma = doc["cell_zones"][0]["medium"]["gamma"].get<double>();
  const double c = std::sqrt(gamma * p / rho);
  const double q = speed + c;
  const double h = c * c / (gamma - 1.0) + 0.5 * speed * speed;
  const double unit = 0.5 * 0x1p-23 * area;
  const Json expected = {{"density", unit * rho * q},
                         {"momentum-x", unit * rho * q * q},
                         {"momentum-y", unit * rho * q * q},
                         {"momentum-z", unit * rho * q * q},
                         {"energy", unit * rho * h * q}};
  const auto& floor = info["convergence"]["floor"];
  REQUIRE(floor.size() == expected.size());
  for (const auto& [name, value] : expected.items()) {
    INFO(name);
    CHECK(std::abs(floor[name].get<double>() - value.get<double>()) <= 1e-12 * value.get<double>());
  }
}

TEST_CASE("a run takes the planar loops exactly when every face is planar and the flow inviscid",
          "[euler]") {
  // EulerPhysics logs its choice (#412): the loops without the warped-face terms on the
  // axis-aligned hexahedra, never on the polyhedra's warped faces, and never for Navier-Stokes,
  // whose no-slip walls the planar boundary loops would send to the Riemann solver (#540). Closed
  // by no-slip walls and the symmetry plane zmax, a laminar flow keeps its mass: after explicit
  // steps sum(rho V) is the initial rho sum(V), within each step's round-off.
  const bool hex = GENERATE(true, false);
  const bool viscous = GENERATE(false, true);
  const fs::path path = data_dir / "mesh" / (hex ? "hex/hex.ascii.msh" : "poly/polyMesh");
  INFO(path << (viscous ? " laminar" : " inviscid"));
  const auto mesh = autan::io::read_mesh(path);
  const auto geometry = autan::compute_geometry(mesh);
  const bool planar = autan::kernels::pack_geometry(mesh, geometry).all_planar;
  REQUIRE(planar == hex);
  const int steps = 5;
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    Json wall;
    for (const auto& b : d["boundaries"]) {
      if (b["kind"] == "wall") wall = b;
    }
    if (viscous) {
      laminar_along_x(d, "fp64");
      for (auto& b : d["boundaries"]) {
        if (b["zone"] == "zmax") continue;
        Json w = wall;
        w["zone"] = b["zone"];
        b = w;
      }
      d["time"] = {{"kind", "unsteady"}, {"steps", steps},
                   {"dt", nullptr},      {"cfl", 0.8},
                   {"average_from", 1},  {"checkpoint", {{"seconds", 1800.0}}}};
    } else {
      inviscid_along_x(d, "fp64");
      d["time"]["max_iterations"] = 2;
    }
    d["mesh"]["path"] = fs::absolute(path).string();
    d["mesh"]["sha256"] = autan::io::mesh_hash(path);
    Json zone = d["cell_zones"][0];
    zone["zone"] = "fluid";
    d["cell_zones"] = Json::array({zone});
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  CHECK(planar_events(out.run) == (planar && !viscous ? 1 : 0));
  if (!viscous) return;
  const double rho = doc["cell_zones"][0]["initial"]["density"].is_number()
                         ? doc["cell_zones"][0]["initial"]["density"].get<double>()
                         : doc["freestream"]["density"].get<double>();
  const auto density = field(out.run, "Density");
  REQUIRE(density.size() == geometry.cell_volume.size());
  double mass = 0.0, volume = 0.0;
  for (std::size_t c = 0; c < density.size(); ++c) {
    mass += density[c] * geometry.cell_volume[c];
    volume += geometry.cell_volume[c];
  }
  CHECK(std::abs(mass - rho * volume) <=
        64.0 * steps * std::numeric_limits<double>::epsilon() * rho * volume);
}

// Two runs of `doc`, one with the default cadence and one with a checkpoint after every
// iteration: the same history and solution, bit for bit (results.md#checkpointcgns), and no
// checkpoint left once the run ends.
void check_checkpoints_change_nothing(Json doc) {
  doc["reproducibility"] = "run";
  const auto plain = run(doc);
  INFO(plain.err);
  REQUIRE(plain.status == ExitStatus::finished);
  doc["time"]["checkpoint"] = Json{{"iterations", 1}};
  const auto every = run(doc);
  INFO(every.err);
  REQUIRE(every.status == ExitStatus::finished);
  CHECK(read(every.run / "history.jsonl") == read(plain.run / "history.jsonl"));
  for (const char* name : {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}) {
    CAPTURE(name);
    const auto a = field(plain.run, name), b = field(every.run, name);
    REQUIRE(a.size() == b.size());
    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
  }
  const Json info = Json::parse(read(every.run / "run.json"));
  CHECK(info["checkpoint"].is_object());
  CHECK(!fs::exists(every.run / "checkpoint.cgns"));
}

TEST_CASE("a continued steady run ends its first-order start where the uninterrupted one does",
          "[euler]") {
  // results.md: a `continue` run gives the uninterrupted run's history, bit for bit. A uniform
  // flow's first-order drop is round-off, so its first-order start ends on its stall
  // (euler.md#time-stepping), about 100 steps in: a continuation from iteration 60 must carry
  // that count over.
  auto doc = input("hybrid.solver-input.json", [](Json& d) {
    inviscid_along_x(d, "fp64");
    d["time"]["max_iterations"] = 200;
  });
  const auto whole = run(doc);
  INFO(whole.err);
  REQUIRE(whole.status == ExitStatus::finished);
  auto part = doc;
  part["time"]["max_iterations"] = 60;
  const auto first = run(part);
  REQUIRE(first.status == ExitStatus::finished);
  auto rest = doc;
  rest["start"] = continue_from(first.run);
  const auto second = run(rest);
  INFO(second.err);
  REQUIRE(second.status == ExitStatus::finished);
  const auto expected = history(whole.run);
  auto chain = history(first.run);
  for (const auto& line : history(second.run)) chain.push_back(line);
  REQUIRE(chain.size() == expected.size());
  for (std::size_t i = 0; i < chain.size(); ++i) {
    INFO("iteration " << i + 1);
    CHECK(chain[i]["residuals"] == expected[i]["residuals"]);
  }
}

TEST_CASE("the history marks provisional exactly the steps of the first-order start", "[euler]") {
  // results.md#history: a steady line carries `provisional`, true for the first-order start's
  // steps (euler.md#time-stepping). A uniform flow's start ends on its stall, before the cap, and
  // the stalling step is not counted by FirstOrderStart: n < 201 marked lines, a prefix, and
  // n - 1 counted steps in the restart state, second order on.
  const auto out = run(input("hybrid.solver-input.json", [](Json& d) {
    inviscid_along_x(d, "fp64");
    d["time"]["max_iterations"] = 250;
  }));
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const auto lines = history(out.run);
  std::size_t marked = 0;
  while (marked < lines.size() && lines[marked]["provisional"] == true) ++marked;
  for (std::size_t i = marked; i < lines.size(); ++i) CHECK(lines[i]["provisional"] == false);
  REQUIRE(marked >= 1);
  REQUIRE(marked < lines.size());
  CHECK(field(out.run, "SolverSecondOrder", "RestartState").front() == 1.0);
  const double counted = field(out.run, "SolverFirstOrderSteps", "RestartState").front();
  CHECK(marked < 201);
  CHECK(counted == static_cast<double>(marked - 1));
  // Unsteady lines leave it out.
  const auto steps = run(input("hybrid.solver-input.json", [](Json& d) {
    inviscid_along_x(d, "fp64");
    d["time"] = {{"kind", "unsteady"},      {"steps", 3},
                 {"dt", nullptr},           {"cfl", 0.8},
                 {"average_from", nullptr}, {"checkpoint", {{"seconds", 1800.0}}}};
  }));
  REQUIRE(steps.status == ExitStatus::finished);
  for (const auto& line : history(steps.run)) CHECK_FALSE(line.contains("provisional"));
}

TEST_CASE("checkpoints change no Euler result", "[euler]") {
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const bool steady = GENERATE(true, false);
  CAPTURE(precision, steady);
  check_checkpoints_change_nothing(input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    for (auto& b : d["boundaries"]) b["kind"] = "farfield";
    auto& init = d["cell_zones"].back()["initial"];
    init["pressure"] = 1.3 * init["pressure"].get<double>();
    if (steady) {
      d["time"]["max_iterations"] = 8;
    } else {
      d["time"] = {{"kind", "unsteady"}, {"steps", 8},
                   {"dt", nullptr},      {"cfl", 0.8},
                   {"average_from", 4},  {"checkpoint", {{"seconds", 1800.0}}}};
    }
  }));
}

TEST_CASE("a steady run whose transverse momenta wake at iteration 2 converges", "[euler]") {
  // #432: from a state along x the transverse momenta are round-off at iteration 1, below the
  // exemption threshold, and wake at iteration 2 (W0710); they then fall with the other
  // equations, so a reachable reference lets the run converge within its budget.
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, "fp64");
    for (auto& b : d["boundaries"]) b["kind"] = "farfield";
    // Pressure and density 1.3 times the freestream's, at its temperature.
    for (auto& z : d["cell_zones"]) {
      for (const char* key : {"pressure", "density"})
        z["initial"][key] = 1.3 * z["initial"][key].get<double>();
    }
    d["time"]["max_iterations"] = 1000;
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const Json info = Json::parse(read(out.run / "run.json"));
  CHECK(info["status"] == "converged");
  CHECK(std::ranges::count_if(info["diagnostics"],
                              [](const Json& d) { return d["code"] == "W0710"; }) == 2);
}

TEST_CASE("a verification case converges to its discrete solution", "[euler]") {
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const char* linear = GENERATE("multigrid", "gauss-seidel", "lines");
  INFO(precision << " " << linear);
  const auto doc = input("hybrid-verification.solver-input.json", [&](Json& d) {
    d["precision"] = precision;
    d["time"]["linear_solver"] = linear;
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

TEST_CASE("the multigrid builds the levels autan-mesh-info publishes", "[euler]") {
  // mesh-info.md, quality.multigrid: the hierarchy a steady multigrid run builds on one rank, in
  // the cell order it computes in, against the run's own log of it.
  const char* order = GENERATE("rcm", "input");
  CAPTURE(order);
  REQUIRE(setenv("AUTAN_CELL_ORDER", order, 1) == 0);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, "fp64");
    d["time"]["linear_solver"] = "multigrid";
    d["time"]["max_iterations"] = 1;
  });
  const auto out = run(doc);
  unsetenv("AUTAN_CELL_ORDER");
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  Json built;
  std::ifstream log_in(out.run / "log.jsonl");
  for (std::string line; std::getline(log_in, line);) {
    const Json event = Json::parse(line);
    // The linear solver's event of the hierarchy: its nodes and edges per level.
    if (event["source"] == "linear" && event["data"].contains("edges")) built = event["data"];
  }
  REQUIRE(built.is_object());
  const Json& published = doc["mesh_info"]["quality"]["multigrid"];
  CHECK(doc["mesh_info"]["quality"]["lines"]["order"] == order);
  CHECK(built["nodes"] == published["cells_by_level"]);
  CHECK(built["edges"] == published["edges_by_level"]);
}

TEST_CASE("the line smoother sweeps the same lines whatever the flow model", "[navier-stokes]") {
  // linear_solver = "lines" keeps each of the mesh's lines whole and colours the lines
  // (euler_groups): an SA case sweeps the colours of the inviscid one, not those of single cells.
  auto colours = [](bool inviscid) {
    const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
      if (inviscid) inviscid_along_x(d, "fp64");
      d["time"]["linear_solver"] = "lines";
      d["time"]["max_iterations"] = 1;
    });
    const auto out = run(doc);
    INFO(out.err);
    REQUIRE(out.status == ExitStatus::finished);
    Json sweeps;
    std::ifstream log_in(out.run / "log.jsonl");
    for (std::string line; std::getline(log_in, line);) {
      const Json event = Json::parse(line);
      if (event["source"] == "linear" && event["data"].contains("colours")) sweeps = event["data"];
    }
    REQUIRE(sweeps.is_object());
    return sweeps["colours"];
  };
  CHECK(colours(false) == colours(true));
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
        d["time"] = {{"kind", "unsteady"}, {"steps", budget},
                     {"dt", nullptr},      {"cfl", 0.8},
                     {"average_from", 2},  {"checkpoint", {{"seconds", 1800.0}}}};
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

TEST_CASE("a run continued right after a shortened step reproduces the uninterrupted one",
          "[euler]") {
  // A shortened step caps the next CFL number; the cap is solver state and must cross a restart.
  // Two zones at very different states make the first steps shorten.
  auto doc = [&](int budget) {
    return input("hybrid.solver-input.json", [&](Json& d) {
      inviscid_along_x(d, "fp64");
      auto& init = d["cell_zones"].back()["initial"];
      init["density"] = 3.0 * init["density"].get<double>();
      init["pressure"] = 5.0 * init["pressure"].get<double>();
      d["time"]["max_iterations"] = budget;
      d["time"]["residual_orders"] = 30.0;
    });
  };
  auto lines = [](const fs::path& f) {
    std::vector<std::string> out;
    std::ifstream in(f);
    for (std::string l; std::getline(in, l);) out.push_back(l);
    return out;
  };
  constexpr int budget = 40;
  const auto whole = run(doc(budget));
  REQUIRE(whole.status == ExitStatus::finished);
  int shortened = 0;
  for (const auto& l : lines(whole.run / "log.jsonl")) {
    const auto e = Json::parse(l);
    if (e["level"] == "warning" && e["source"] == "solver" && e["iteration"].is_number()) {
      shortened = e["iteration"].get<int>();
      break;
    }
  }
  INFO("first shortened step " << shortened);
  REQUIRE(shortened > 0);
  REQUIRE(shortened < budget);
  const auto first = run(doc(shortened));
  REQUIRE(first.status == ExitStatus::finished);
  auto next = doc(budget);
  next["start"] = continue_from(first.run);
  const auto second = run(next);
  REQUIRE(second.status == ExitStatus::finished);
  auto chain = lines(first.run / "history.jsonl");
  const auto more = lines(second.run / "history.jsonl");
  chain.insert(chain.end(), more.begin(), more.end());
  CHECK(chain == lines(whole.run / "history.jsonl"));
  for (const char* name : {"Density", "VelocityX", "Pressure"}) {
    CHECK(field(second.run, name) == field(whole.run, name));
  }
}

TEST_CASE("a steady run reports the residual of its second-order scheme from the start",
          "[euler]") {
  // The steady solver starts at first order, but convergence is judged on the history, which
  // must hold the second-order residual: at iteration 1 it equals the unsteady run's, which is
  // second order from its first step. Two zones at different states make the flow nonuniform.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  auto doc = [&](bool steady) {
    return input("hybrid.solver-input.json", [&](Json& d) {
      inviscid_along_x(d, precision);
      auto& init = d["cell_zones"].back()["initial"];
      init["density"] = 1.2 * init["density"].get<double>();
      init["pressure"] = 1.3 * init["pressure"].get<double>();
      if (steady) {
        d["time"]["max_iterations"] = 1;
      } else {
        d["time"] = {{"kind", "unsteady"},      {"steps", 1},
                     {"dt", nullptr},           {"cfl", 0.8},
                     {"average_from", nullptr}, {"checkpoint", {{"seconds", 1800.0}}}};
      }
    });
  };
  const auto steady = run(doc(true));
  const auto unsteady = run(doc(false));
  REQUIRE(steady.status == ExitStatus::finished);
  REQUIRE(unsteady.status == ExitStatus::finished);
  CHECK(history(steady.run).front()["residuals"] == history(unsteady.run).front()["residuals"]);
}

namespace {

// tests/data/mesh/periodic-box, periodic in x and y: zmin a slip wall, zmax a symmetry plane,
// one cell zone; the case of hybrid.solver-input.json otherwise, inviscid.
void periodic_box(Json& d, const char* precision) {
  const fs::path box = data_dir / "mesh" / "periodic-box" / "periodic-box.ascii.msh";
  inviscid_along_x(d, precision);
  d["mesh"]["path"] = fs::absolute(box).string();
  d["mesh"]["sha256"] = autan::io::mesh_hash(box);
  d["periodic"] =
      Json::array({Json{{"zones", {"xmin", "xmax"}},
                        {"transform", {{"kind", "translation"}, {"vector", {1.0, 0.0, 0.0}}}}},
                   Json{{"zones", {"ymin", "ymax"}},
                        {"transform", {{"kind", "translation"}, {"vector", {0.0, 1.0, 0.0}}}}}});
  // Periodic zones take no condition.
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
  d["monitors"][0]["zones"] = {"zmin"};
  Json zone = d["cell_zones"][0];
  zone["zone"] = "fluid";
  d["cell_zones"] = Json::array({zone});
}

// A flow along x and y through both periodic pairs, `steps` SSP-RK3 steps.
void unsteady_through_pairs(Json& d, int steps) {
  auto& f = d["freestream"];
  const double speed = f["speed"].get<double>();
  f["velocity"] = {0.6 * speed, 0.8 * speed, 0.0};
  for (auto& z : d["cell_zones"]) z["initial"]["velocity"] = f["velocity"];
  d["time"] = {{"kind", "unsteady"},      {"steps", steps},
               {"dt", nullptr},           {"cfl", 0.8},
               {"average_from", nullptr}, {"checkpoint", {{"seconds", 1800.0}}}};
}

}  // namespace

TEST_CASE("a case runs the convective flux its physics names", "[euler]") {
  // case-schema.md#physics: `flux` picks Roe, HLLC or AUSM+-up. The fluxes differ wherever the
  // flow is not uniform, so each gives its own residuals from the second step on.
  const auto residuals = [](const std::string& flux) {
    const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
      inviscid_along_x(d, "fp64");
      d["physics"]["flux"] = flux;
      d["time"]["max_iterations"] = 3;
      // One zone at 1.3 times the freestream's pressure and density: not a uniform flow.
      for (const char* key : {"pressure", "density"}) {
        auto& v = d["cell_zones"][1]["initial"][key];
        v = 1.3 * v.get<double>();
      }
    });
    const auto out = run(doc);
    INFO(flux << " " << out.err);
    REQUIRE(out.status == ExitStatus::finished);
    std::vector<Json> steps;
    for (const auto& line : history(out.run)) steps.push_back(line["residuals"]);
    return steps;
  };
  const auto roe = residuals("roe");
  const auto hllc = residuals("hllc");
  const auto ausm = residuals("ausm+up");
  REQUIRE(roe.size() == 3);
  CHECK(roe.back() != hllc.back());
  CHECK(roe.back() != ausm.back());
  CHECK(hllc.back() != ausm.back());
  CHECK(residuals("hllc") == hllc);
  // Any other value is an invalid document at `physics.flux`.
  const auto out = run(input("hybrid.solver-input.json", [](Json& d) {
    inviscid_along_x(d, "fp64");
    d["physics"]["flux"] = "jst";
  }));
  CHECK(out.status == ExitStatus::invalid_input);
  const auto diagnostic = Json::parse(out.err.substr(0, out.err.find('\n')));
  CHECK(diagnostic["code"] == "E0705");
  CHECK(diagnostic["location"]["path"] == "physics.flux");
}

TEST_CASE("steady cases the steady solver cannot run are refused before any directory exists",
          "[euler]") {
  // The multigrid agglomerates one mesh's cells and takes no image ghosts: E0706 at
  // `time.linear_solver`. The multigrid sums in local cell order: reproducibility "ranks" is
  // E1109. The coupled turbulence step is SA's and the matrix-free step laminar flow's
  // (case-schema.md#time): E0426 and E0429 on inviscid flow. Neither is written for the
  // Navier-Stokes physics yet: E0426 and E0429 on SA and laminar flow as well.
  const std::string reason =
      GENERATE("periodic", "ranks", "coupled", "matrix-free", "coupled sa", "matrix-free laminar");
  CAPTURE(reason);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    d["time"]["max_iterations"] = 2;
    if (reason == "periodic") {
      periodic_box(d, "fp64");
    } else if (reason == "coupled sa") {
      d["time"]["coupled_turbulence"] = true;
    } else if (reason == "matrix-free laminar") {
      laminar_along_x(d, "fp64");
      d["time"]["matrix_free"] = true;
    } else {
      inviscid_along_x(d, "fp64");
      if (reason == "ranks") d["reproducibility"] = "ranks";
      if (reason == "coupled") d["time"]["coupled_turbulence"] = true;
      if (reason == "matrix-free") d["time"]["matrix_free"] = true;
    }
  });
  const auto out = run(doc);
  INFO(out.err);
  CHECK(out.status == ExitStatus::invalid_input);
  CHECK(!fs::exists(out.runs));
  const auto diagnostic = Json::parse(out.err.substr(0, out.err.find('\n')));
  const std::map<std::string, std::pair<std::string, std::string>> expected = {
      {"periodic", {"E0706", "time.linear_solver"}},
      {"ranks", {"E1109", "reproducibility"}},
      {"coupled", {"E0426", "time.coupled_turbulence"}},
      {"matrix-free", {"E0429", "time.matrix_free"}},
      {"coupled sa", {"E0426", "time.coupled_turbulence"}},
      {"matrix-free laminar", {"E0429", "time.matrix_free"}}};
  CHECK(diagnostic["code"] == expected.at(reason).first);
  CHECK(diagnostic["location"]["path"] == expected.at(reason).second);
}

TEST_CASE("a Navier-Stokes case outside the physics' scope is refused before any directory exists",
          "[navier-stokes]") {
  // Across a periodic pair the boundary fits would need the image ghosts' translated stencils:
  // E0706 at `periodic`, steady or not. Under
  // reproducibility "ranks" the steady multigrid still sums in local cell order: E1109, laminar
  // and SA alike.
  const auto [model, ranks] =
      GENERATE(std::pair{"laminar", false}, std::pair{"laminar", true}, std::pair{"sa", true});
  INFO(model << (ranks ? " under reproducibility ranks" : " through periodic pairs"));
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    if (!ranks) {
      periodic_box(d, "fp64");
      unsteady_through_pairs(d, 2);
      d["physics"]["turbulence"]["model"] = "laminar";
      return;
    }
    // The hybrid case is SA; laminar_along_x makes it laminar.
    if (std::string(model) == "laminar") laminar_along_x(d, "fp64");
    d["reproducibility"] = "ranks";
    d["time"]["linear_solver"] = "multigrid";
  });
  const auto out = run(doc);
  INFO(out.err);
  CHECK(out.status == ExitStatus::invalid_input);
  CHECK(!fs::exists(out.runs));
  const auto diagnostic = Json::parse(out.err.substr(0, out.err.find('\n')));
  CHECK(diagnostic["code"] == (ranks ? "E1109" : "E0706"));
  CHECK(diagnostic["location"]["path"] == (ranks ? "reproducibility" : "periodic"));
}

TEST_CASE("an SA case with the flow on multigrid takes SA steps by colour", "[navier-stokes]") {
  // ADR 0005 (B), #163 step 7a: SA sweeps the runtime's cell colours whatever the flow's linear
  // solver. A steady SA run with the flow on multigrid (reproducibility "run": "ranks" refuses
  // it, E1109) completes past the second-order switch, where SA steps start, with rho nt changed
  // from its uniform start.
  const auto doc = input("hybrid.solver-input.json", [](Json& d) {
    d["time"]["linear_solver"] = "multigrid";
    d["time"]["max_iterations"] = 400;
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  REQUIRE(field(out.run, "SolverSecondOrder", "RestartState").front() == 1.0);
  const auto rnu = field(out.run, "DensityNuTilde", "RestartState");
  REQUIRE(!rnu.empty());
  const auto [lo, hi] = std::ranges::minmax_element(rnu);
  CHECK(*lo < *hi);
}

TEST_CASE(
    "a Navier-Stokes case under reproducibility ranks gives the same bits in every cell order",
    "[navier-stokes][long]") {
  // case-schema.md, reproducibility: "ranks" repeats the bits in any cell order. The hybrid case,
  // SA and laminar, keeps its no-slip wall so that the viscous boundary fit is built. Steady runs
  // go past the second-order switch, so that the Jacobian, SA's assembly and its sweeps are
  // exercised; laminar flow also runs unsteady. The history and every data array of solution.cgns
  // (fields, restart state, wall values), computed in RCM order and in input order, are the same
  // bits (#587).
  const auto [model, linear, steady] =
      GENERATE(std::tuple{"laminar", "gauss-seidel", true}, std::tuple{"laminar", "lines", true},
               std::tuple{"sa", "gauss-seidel", true}, std::tuple{"sa", "lines", true},
               std::tuple{"laminar", "", false});
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(model << " " << linear << (steady ? " steady " : " unsteady ") << precision);
  // input() writes the mesh information of the cell order in force: set it first.
  auto in_order = [&](const char* order) {
    REQUIRE(setenv("AUTAN_CELL_ORDER", order, 1) == 0);
    const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
      d["precision"] = precision;
      if (std::string(model) == "laminar") {
        d["physics"]["turbulence"]["model"] = "laminar";
        d["physics"]["turbulence"]["nu_tilde_ratio"] = nullptr;
        d["freestream"]["nu_tilde"] = nullptr;
        for (auto& z : d["cell_zones"]) z["initial"]["nu_tilde"] = nullptr;
      }
      d["reproducibility"] = "ranks";
      if (steady) {
        d["time"]["linear_solver"] = linear;
        d["time"]["max_iterations"] = 400;
      } else {
        d["time"] = {{"kind", "unsteady"}, {"steps", 20},
                     {"dt", nullptr},      {"cfl", 0.8},
                     {"average_from", 10}, {"checkpoint", {{"seconds", 1800.0}}}};
      }
    });
    const auto out = run(doc);
    unsetenv("AUTAN_CELL_ORDER");
    INFO(out.err);
    REQUIRE(out.status == ExitStatus::finished);
    const Json info = Json::parse(read(out.run / "run.json"));
    REQUIRE((info["status"] == "completed" || info["status"] == "converged"));
    if (steady) REQUIRE(field(out.run, "SolverSecondOrder", "RestartState").front() == 1.0);
    return out.run;
  };
  const auto rcm = in_order("rcm"), plain = in_order("input");
  CHECK(read(rcm / "history.jsonl") == read(plain / "history.jsonl"));
  auto bits = [](const std::vector<double>& v) {
    std::vector<std::uint64_t> out(v.size());
    std::ranges::transform(v, out.begin(),
                           [](double x) { return std::bit_cast<std::uint64_t>(x); });
    return out;
  };
  const auto a = data_arrays(rcm), b = data_arrays(plain);
  REQUIRE(a.contains("FlowSolution/Density"));
  REQUIRE(a.contains("WallSolution/WallShearStressX"));
  REQUIRE(a.contains("RestartState/SolverPenalty") == steady);
  REQUIRE(a.contains("FlowSolution/TurbulentSANuTilde") == (std::string(model) == "sa"));
  CHECK(a.size() == b.size());
  for (const auto& [name, values] : a) {
    INFO(name);
    REQUIRE(b.contains(name));
    CHECK(bits(values) == bits(b.at(name)));
  }
}

TEST_CASE("a case with an initial-state file starts from the file's states", "[euler]") {
  // initial-state.md: the file gives every cell the freestream while both cell zones start at
  // twice its density. Started from the file, the flow is uniform with farfield all around and
  // stays so; started from the zones, it would not.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
  const autan::Geometry g = autan::compute_geometry(mesh);
  Json doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    for (auto& b : d["boundaries"]) b["kind"] = "farfield";
    for (auto& z : d["cell_zones"]) {
      z["initial"]["density"] = 2.0 * d["freestream"]["density"].get<double>();
    }
    d["time"] = {{"kind", "unsteady"},      {"steps", 3},
                 {"dt", nullptr},           {"cfl", 0.8},
                 {"average_from", nullptr}, {"checkpoint", {{"seconds", 1800.0}}}};
  });
  const auto& f = doc["freestream"];
  const double rho = f["density"].get<double>(), p = f["pressure"].get<double>();
  const double speed = f["speed"].get<double>();
  Json centroids = Json::array(), values = Json::array();
  for (std::size_t c = 0; c < static_cast<std::size_t>(mesh.n_cells()); ++c) {
    centroids.push_back({g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]});
    values.push_back({rho, speed, 0.0, 0.0, p});
  }
  const Json state{{"schema", "autan.initial-state/1"},
                   {"oracle", "freestream"},
                   {"mesh_sha256", autan::io::mesh_hash(mesh_path)},
                   {"variables", {"density", "velocity-x", "velocity-y", "velocity-z", "pressure"}},
                   {"cells", mesh.n_cells()},
                   {"centroids", centroids},
                   {"values", values}};
  std::random_device random;
  const fs::path file = fs::temp_directory_path() / ("autan_initial-" + std::to_string(random()));
  std::ofstream(file) << state.dump();
  std::string sha;
  picosha2::hash256_hex_string(read(file), sha);
  doc["initial_state"] = {{"path", file.string()}, {"sha256", sha}};
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  for (const double v : field(out.run, "Density")) CHECK(std::abs(v - rho) <= 64 * eps * rho);
  for (const double v : field(out.run, "Pressure")) CHECK(std::abs(v - p) <= 64 * eps * p);
  for (const double v : field(out.run, "VelocityX")) {
    CHECK(std::abs(v - speed) <= 64 * eps * speed);
  }
}

TEST_CASE("a force monitor records the forces of its own iteration, whatever its cadence",
          "[euler]") {
  // Two zones at different states evolve over four unsteady steps. Recorded every iteration by
  // two monitors, or at the fourth only, the forces of iteration 4 are the same bits: the boundary
  // pressure every monitor of an iteration reads belongs to that iteration's state.
  auto doc = [](int every, int count) {
    return input("hybrid.solver-input.json", [&](Json& d) {
      inviscid_along_x(d, "fp64");
      for (auto& b : d["boundaries"]) b["kind"] = "farfield";
      auto& init = d["cell_zones"].back()["initial"];
      init["density"] = 1.2 * init["density"].get<double>();
      init["pressure"] = 1.3 * init["pressure"].get<double>();
      d["time"] = {{"kind", "unsteady"},      {"steps", 4},
                   {"dt", nullptr},           {"cfl", 0.8},
                   {"average_from", nullptr}, {"checkpoint", {{"seconds", 1800.0}}}};
      Json monitor = d["monitors"][0];
      monitor["every"] = every;
      d["monitors"] = Json::array();
      for (int k = 0; k < count; ++k) {
        monitor["name"] = "forces" + std::to_string(k);
        d["monitors"].push_back(monitor);
      }
    });
  };
  const auto each = run(doc(1, 2));
  const auto last = run(doc(4, 1));
  INFO(each.err << last.err);
  REQUIRE(each.status == ExitStatus::finished);
  REQUIRE(last.status == ExitStatus::finished);
  const auto a = history(each.run), b = history(last.run);
  REQUIRE(a.size() == 4);
  REQUIRE(b.size() == 4);
  const Json& at4 = a[3]["monitors"];
  CHECK(at4["forces0"] == at4["forces1"]);
  CHECK(at4["forces0"] == b[3]["monitors"]["forces0"]);
  CHECK(a[0]["monitors"]["forces0"] != at4["forces0"]);  // the forces do evolve
}

TEST_CASE("a uniform flow through periodic pairs stays uniform", "[euler]") {
  // Image ghosts hold their owners' values: a uniform flow crossing both pairs is steady.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    periodic_box(d, precision);
    unsteady_through_pairs(d, 20);
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  const auto& f = doc["freestream"];
  const double rho = f["density"].get<double>(), p = f["pressure"].get<double>();
  const double speed = f["speed"].get<double>();
  for (const double v : field(out.run, "Density")) CHECK(std::abs(v - rho) <= 64 * eps * rho);
  for (const double v : field(out.run, "Pressure")) CHECK(std::abs(v - p) <= 64 * eps * p);
  for (const double v : field(out.run, "VelocityX"))
    CHECK(std::abs(v - 0.6 * speed) <= 64 * eps * speed);
  for (const double v : field(out.run, "VelocityY"))
    CHECK(std::abs(v - 0.8 * speed) <= 64 * eps * speed);
}

TEST_CASE("cases outside the Euler physics are refused before any directory exists", "[euler]") {
  // E0706 on stderr, exit 2, no run directory; data names the zone and what it holds.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const std::string refused =
      GENERATE("solid", "second gas", "transport", "wall", "sideslip", "sonic", "unsteady vortex");
  // The point vortex's limits hold for laminar flow too.
  const bool laminar = GENERATE(false, true);
  const bool vortex = refused == "sideslip" || refused == "sonic" || refused == "unsteady vortex";
  if (laminar && !vortex) return;
  INFO(precision << " " << refused << (laminar ? " laminar" : ""));
  std::string path = "cell_zones[1].medium";
  std::string field = "medium";
  Json value = "helium";
  std::size_t boundary = 2;
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    if (laminar) {
      laminar_along_x(d, precision);
    } else {
      inviscid_along_x(d, precision);
    }
    d["time"]["max_iterations"] = 2;
    auto& z = d["cell_zones"][1];
    if (vortex) {
      // A point vortex on the farfield xmin, from the lift of the slip wall ymin.
      boundary = 0;
      d["boundaries"][0]["point_vortex"] = {{"zones", {"ymin"}}, {"center", {0.5, 0.0, -1.0}}};
    }
    if (refused == "solid") {
      z["medium"] = {{"kind", "solid"},
                     {"name", "steel"},
                     {"density", 7800.0},
                     {"specific_heat", 500.0},
                     {"conductivity", 40.0}};
      z["initial"] = {{"kind", "solid"}, {"temperature", 300.0}};
      value = "solid";
    } else if (refused == "second gas") {
      z["medium"]["name"] = "helium";
      z["medium"]["gas_constant"] = 2077.1;
      z["medium"]["gamma"] = 5.0 / 3.0;
    } else if (refused == "transport") {
      z["medium"]["name"] = "helium";
      z["medium"]["prandtl"] = 0.67;
    } else if (refused == "wall") {
      d["boundaries"][2]["kind"] = "wall";
      path = "boundaries[2].kind";
      field = "condition";
      value = "wall";
    } else if (refused == "sideslip") {
      d["freestream"]["sideslip"] = 0.1;
      path = "freestream.sideslip";
      field = "sideslip";
      value = 0.1;
    } else if (refused == "sonic") {
      d["freestream"]["mach"] = 1.0;
      path = "freestream.mach";
      field = "mach";
      value = 1.0;
    } else {
      d["time"] = {{"kind", "unsteady"},      {"steps", 2},
                   {"dt", nullptr},           {"cfl", 0.8},
                   {"average_from", nullptr}, {"checkpoint", {{"seconds", 1800.0}}}};
      path = "time.kind";
      field = "time";
      value = "unsteady";
    }
  });
  const auto out = run(doc);
  INFO(out.err);
  CHECK(out.status == ExitStatus::invalid_input);
  CHECK(out.run.empty());
  CHECK(!fs::exists(out.runs));
  const auto diagnostic = Json::parse(out.err.substr(0, out.err.find('\n')));
  CHECK(diagnostic["code"] == "E0706");
  CHECK(diagnostic["data"]["path"] == path);
  CHECK(diagnostic["data"][field] == value);
  CHECK(diagnostic["data"]["zone"] ==
        (field == "medium" ? doc["cell_zones"][1]["zone"] : doc["boundaries"][boundary]["zone"]));
}

#ifdef AUTAN_MPIEXEC
namespace {

// autan-run under the launcher on `ranks` ranks: its run directory, none for an invalid input;
// its standard error in `err` when given.
fs::path run_on_ranks(const Json& doc, int ranks, int expected_status = 0,
                      std::string* err = nullptr) {
  std::random_device random;
  const fs::path dir = fs::temp_directory_path() / ("autan_euler_mpi-" + std::to_string(random()));
  fs::create_directories(dir);
  const std::string err_file = (dir / "stderr").string();
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, 2, err_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
                                   0644);
  // solver-input.md, `devices`: one rank per device, so the input counts the ranks.
  Json with_count = doc;
  if (with_count["devices"]["count"] == 1) with_count["devices"]["count"] = ranks;
  std::ofstream(dir / "solver-input.json") << with_count.dump();
  std::vector<std::string> args{AUTAN_MPIEXEC,
                                AUTAN_MPIEXEC_NUMPROC_FLAG,
                                std::to_string(ranks),
                                AUTAN_RUN,
                                "--input",
                                (dir / "solver-input.json").string(),
                                "--runs",
                                (dir / "runs").string()};
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = 0;
  REQUIRE(posix_spawn(&pid, AUTAN_MPIEXEC, &actions, nullptr, argv.data(), environ) == 0);
  posix_spawn_file_actions_destroy(&actions);
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  if (err != nullptr) *err = read(err_file);
  REQUIRE(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == expected_status);
  fs::path run;
  if (fs::exists(dir / "runs")) {
    for (const auto& entry : fs::directory_iterator(dir / "runs")) run = entry.path();
  }
  return run;
}

// Two runs of reproducibility "ranks" on different rank counts: the same history text and the
// same bit patterns in every field (a signed zero or a NaN cannot pass for an equal value).
void check_bit_for_bit(const fs::path& one, const fs::path& many) {
  CHECK(read(many / "history.jsonl") == read(one / "history.jsonl"));
  for (const char* name : {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}) {
    INFO(name);
    const auto a = field(one, name), b = field(many, name);
    REQUIRE(std::ranges::all_of(a, [](double x) { return std::isfinite(x); }));
    std::vector<std::uint64_t> bits_one(a.size()), bits_many(b.size());
    std::ranges::transform(a, bits_one.begin(),
                           [](double x) { return std::bit_cast<std::uint64_t>(x); });
    std::ranges::transform(b, bits_many.begin(),
                           [](double x) { return std::bit_cast<std::uint64_t>(x); });
    CHECK(bits_one == bits_many);
  }
}

}  // namespace

TEST_CASE("several ranks give the one-rank solution to round-off", "[euler][mpi]") {
  // runtime.md, Ranks: Euler on N ranks equals one rank to round-off under reproducibility
  // "run" (faces summed in local order), and bit for bit under "ranks"; at a fixed rank count,
  // runs repeat bit for bit. Two zones at different states on the hybrid mesh (farfield
  // around), or a flow through the periodic box's pairs.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const bool periodic = GENERATE(false, true);
  const int ranks = GENERATE(2, 4);
  const char* reproducibility = GENERATE("run", "ranks");
  CAPTURE(precision, periodic, ranks, reproducibility);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    d["reproducibility"] = reproducibility;
    if (periodic) {
      // Through both pairs and into the slip wall zmin, so that the flow evolves.
      periodic_box(d, precision);
      unsteady_through_pairs(d, 12);
      const double speed = d["freestream"]["speed"].get<double>();
      d["cell_zones"][0]["initial"]["velocity"] = {0.6 * speed, 0.7 * speed, -0.3 * speed};
    } else {
      inviscid_along_x(d, precision);
      for (auto& b : d["boundaries"]) b["kind"] = "farfield";
      auto& init = d["cell_zones"].back()["initial"];
      init["density"] = 1.2 * init["density"].get<double>();
      init["pressure"] = 1.3 * init["pressure"].get<double>();
      d["time"] = {{"kind", "unsteady"},      {"steps", 12},
                   {"dt", nullptr},           {"cfl", 0.8},
                   {"average_from", nullptr}, {"checkpoint", {{"seconds", 1800.0}}}};
    }
  });
  const auto one = run(doc);
  INFO(one.err);
  REQUIRE(one.status == ExitStatus::finished);
  const fs::path many = run_on_ranks(doc, ranks);
  REQUIRE(!many.empty());
  // The hybrid mesh's faces are all planar: these runs take the planar loops (#412).
  if (!periodic) CHECK(planar_events(many) == 1);
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  // History: every residual norm.
  const auto h1 = history(one.run), hn = history(many);
  REQUIRE(h1.size() == hn.size());
  double worst_norm = 0.0;
  for (std::size_t i = 0; i < h1.size(); ++i) {
    for (const auto& [name, v] : h1[i]["residuals"].items()) {
      const double a = v.get<double>(), b = hn[i]["residuals"][name].get<double>();
      worst_norm = std::max(worst_norm, std::abs(a - b) / (std::max(std::abs(a), 1e-300) * eps));
    }
  }
  // Fields: the largest difference over the field's largest magnitude (the largest speed for
  // the velocity components).
  const auto magnitude = [](const std::vector<double>& v) {
    double m = 0.0;
    for (const double x : v) m = std::max(m, std::abs(x));
    return m;
  };
  double speed = 0.0;
  {
    const auto vx = field(one.run, "VelocityX"), vy = field(one.run, "VelocityY"),
               vz = field(one.run, "VelocityZ");
    for (std::size_t c = 0; c < vx.size(); ++c) {
      speed = std::max(speed, std::hypot(vx[c], vy[c], vz[c]));
    }
  }
  double worst_field = 0.0;
  for (const char* name : {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}) {
    const auto a = field(one.run, name), b = field(many, name);
    REQUIRE(a.size() == b.size());
    double diff = 0.0;
    for (std::size_t c = 0; c < a.size(); ++c) diff = std::max(diff, std::abs(a[c] - b[c]));
    const double scale = std::string(name).starts_with("Velocity") ? speed : magnitude(a);
    worst_field = std::max(worst_field, diff / (scale * eps));
  }
  // Under "run": measured at most 7.5 eps on norms and 11.3 eps on fields over these 12 steps
  // (fp64, 4 ranks).
  INFO("norms " << worst_norm << " eps, fields " << worst_field << " eps");
  if (std::string(reproducibility) == "ranks") {
    check_bit_for_bit(one.run, many);
  } else {
    CHECK(worst_norm <= 256.0);
    CHECK(worst_field <= 256.0);
  }
  // A fixed rank count repeats bit for bit.
  const fs::path again = run_on_ranks(doc, ranks);
  CHECK(read(again / "history.jsonl") == read(many / "history.jsonl"));
  for (const char* name : {"Density", "VelocityX", "Pressure"}) {
    CHECK(field(again, name) == field(many, name));
  }
}

// The steps each steady configuration of hybrid.solver-input.json takes: its last first-order
// (provisional) step plus 10, so that every run goes past the second-order switch (Jacobian, SA's
// assembly and sweeps) and stays within CI's per-test timeout unoptimised. Measured at 83b40ec8 in
// gpu-tests' unoptimised build on 2026-11-09 (t648-o0b): the fp32+acc64 first-order start ends on
// its round-off plateau, later than fp64's drop.
static int past_switch(const std::string& model, const std::string& linear,
                       const std::string& precision) {
  const bool fp64 = precision == "fp64";
  int last_first_order = 0;
  if (model == "laminar") {
    last_first_order = linear == "gauss-seidel" ? (fp64 ? 53 : 162) : (fp64 ? 44 : 201);
  } else {
    last_first_order = linear == "gauss-seidel" ? (fp64 ? 44 : 146) : (fp64 ? 37 : 184);
  }
  return last_first_order + 10;
}

// runtime.md, Ranks: under reproducibility "ranks" the Navier-Stokes physics writes on N ranks the
// history and every array of solution.cgns of a run on one (#587). The hybrid case keeps its
// no-slip wall, so that the boundary fits are built across parts; steady runs go past the
// second-order switch, laminar flow also runs unsteady.
static void ns_on_ranks(const char* model, const char* linear, bool steady, const char* precision,
                        int ranks) {
  INFO(model << " " << linear << (steady ? " steady " : " unsteady ") << precision << " on "
             << ranks << " ranks");
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    d["precision"] = precision;
    if (std::string(model) == "laminar") {
      d["physics"]["turbulence"]["model"] = "laminar";
      d["physics"]["turbulence"]["nu_tilde_ratio"] = nullptr;
      d["freestream"]["nu_tilde"] = nullptr;
      for (auto& z : d["cell_zones"]) z["initial"]["nu_tilde"] = nullptr;
    }
    d["reproducibility"] = "ranks";
    if (steady) {
      d["time"]["linear_solver"] = linear;
      d["time"]["max_iterations"] = past_switch(model, linear, precision);
    } else {
      d["time"] = {{"kind", "unsteady"}, {"steps", 20},
                   {"dt", nullptr},      {"cfl", 0.8},
                   {"average_from", 10}, {"checkpoint", {{"seconds", 1800.0}}}};
    }
  });
  const auto one = run(doc);
  INFO(one.err);
  REQUIRE(one.status == ExitStatus::finished);
  if (steady) {
    REQUIRE(field(one.run, "SolverSecondOrder", "RestartState").front() == 1.0);
    // The 10 steps past the switch ran: no stop by convergence before them.
    const auto steps = history(one.run);
    REQUIRE(std::ranges::count_if(steps, [](const Json& e) { return e["provisional"] == false; }) ==
            10);
  }
  std::string err;
  const fs::path many = run_on_ranks(doc, ranks, 0, &err);
  INFO(err);
  REQUIRE(!many.empty());
  CHECK(read(many / "history.jsonl") == read(one.run / "history.jsonl"));
  auto bits = [](const std::vector<double>& v) {
    std::vector<std::uint64_t> out(v.size());
    std::ranges::transform(v, out.begin(),
                           [](double x) { return std::bit_cast<std::uint64_t>(x); });
    return out;
  };
  const auto same_arrays = [&](const fs::path& reference, const fs::path& other) {
    const auto a = data_arrays(reference), b = data_arrays(other);
    REQUIRE(a.contains("WallSolution/WallShearStressX"));
    CHECK(a.size() == b.size());
    for (const auto& [name, values] : a) {
      INFO(name);
      REQUIRE(b.contains(name));
      CHECK(bits(values) == bits(b.at(name)));
    }
  };
  same_arrays(one.run, many);
  if (steady && ranks == 4 && std::string(model) == "sa" && std::string(linear) == "gauss-seidel") {
    // Written on 2 ranks 5 steps after the switch to second order, continued on 4 for the last 5:
    // the restart state carries SA's variable and the switch, so the two runs are the
    // uninterrupted one. The orders are out of reach, so neither run stops early.
    const int steps = past_switch(model, linear, precision);
    Json whole_doc = doc;
    whole_doc["time"]["residual_orders"] = 30.0;
    whole_doc["time"]["max_iterations"] = steps;
    const auto whole = run(whole_doc);
    INFO(whole.err);
    REQUIRE(whole.status == ExitStatus::finished);
    REQUIRE(history(whole.run).size() == static_cast<std::size_t>(steps));
    Json first_doc = whole_doc;
    first_doc["time"]["max_iterations"] = steps - 5;
    const fs::path first = run_on_ranks(first_doc, 2);
    REQUIRE(!first.empty());
    REQUIRE(field(first, "SolverSecondOrder", "RestartState").front() == 1.0);
    Json next = whole_doc;
    next["start"] = continue_from(first);
    const fs::path rest = run_on_ranks(next, 4, 0, &err);
    INFO(err);
    REQUIRE(!rest.empty());
    CHECK(read(first / "history.jsonl") + read(rest / "history.jsonl") ==
          read(whole.run / "history.jsonl"));
    same_arrays(whole.run, rest);
  }
}

TEST_CASE(
    "laminar Navier-Stokes on several ranks gives the one-rank solution bit for bit in fp64 and "
    "unsteady",
    "[navier-stokes][mpi]") {
  const auto [linear, steady, precision] =
      GENERATE(std::tuple{"gauss-seidel", true, "fp64"}, std::tuple{"lines", true, "fp64"},
               std::tuple{"", false, "fp64"}, std::tuple{"", false, "fp32+acc64"});
  ns_on_ranks("laminar", linear, steady, precision, GENERATE(2, 4));
}

TEST_CASE(
    "laminar Navier-Stokes on several ranks gives the one-rank solution bit for bit in fp32+acc64",
    "[navier-stokes][mpi]") {
  ns_on_ranks("laminar", GENERATE("gauss-seidel", "lines"), true, "fp32+acc64", GENERATE(2, 4));
}

TEST_CASE("Spalart-Allmaras on several ranks gives the one-rank solution bit for bit in fp64",
          "[navier-stokes][mpi]") {
  ns_on_ranks("sa", GENERATE("gauss-seidel", "lines"), true, "fp64", GENERATE(2, 4));
}

TEST_CASE("Spalart-Allmaras on several ranks gives the one-rank solution bit for bit in fp32+acc64",
          "[navier-stokes][mpi]") {
  ns_on_ranks("sa", GENERATE("gauss-seidel", "lines"), true, "fp32+acc64", GENERATE(2, 4));
}

TEST_CASE("Spalart-Allmaras on ranks of which one holds no ghost runs as on one rank",
          "[navier-stokes][mpi]") {
  // Two boxes of hexahedra apart, 64 and 125 cells, on 3 ranks: the partition gives the small one
  // a rank of its own, without ghosts, and splits the other. Every ghost exchange is collective,
  // the SA variable's after each colour of its sweeps included: the rank without ghosts must take
  // part in each, or the ranks wait on each other. Under "ranks" the run is one rank's, bit for
  // bit; under "run" it ends.
  const char* reproducibility = GENERATE("ranks", "run");
  CAPTURE(reproducibility);
  autan::CellwiseMesh cells = autan::test::box(4, autan::test::BoxFill::hexa);
  {
    const autan::CellwiseMesh b = autan::test::box(5, autan::test::BoxFill::hexa);
    const auto nodes = static_cast<autan::index_t>(cells.x.size());
    const auto faces = static_cast<autan::offset_t>(cells.face_node_offsets.size() - 1);
    const auto face_nodes = static_cast<autan::offset_t>(cells.face_nodes.size());
    const auto labelled = static_cast<autan::offset_t>(cells.labelled_face_nodes.size());
    for (std::size_t i = 0; i < b.x.size(); ++i) {
      cells.x.push_back(b.x[i] + 2.0);
      cells.y.push_back(b.y[i]);
      cells.z.push_back(b.z[i]);
    }
    for (std::size_t c = 1; c < b.cell_face_offsets.size(); ++c) {
      cells.cell_face_offsets.push_back(b.cell_face_offsets[c] + faces);
    }
    for (std::size_t f = 1; f < b.face_node_offsets.size(); ++f) {
      cells.face_node_offsets.push_back(b.face_node_offsets[f] + face_nodes);
    }
    for (const auto v : b.face_nodes) cells.face_nodes.push_back(v + nodes);
    cells.cell_zone.insert(cells.cell_zone.end(), b.cell_zone.begin(), b.cell_zone.end());
    for (std::size_t f = 1; f < b.labelled_face_node_offsets.size(); ++f) {
      cells.labelled_face_node_offsets.push_back(b.labelled_face_node_offsets[f] + labelled);
    }
    for (const auto v : b.labelled_face_nodes) cells.labelled_face_nodes.push_back(v + nodes);
    cells.labelled_face_zone.insert(cells.labelled_face_zone.end(), b.labelled_face_zone.begin(),
                                    b.labelled_face_zone.end());
  }
  const auto mesh = autan::assemble(autan::to_face_based(std::move(cells)));
  REQUIRE(mesh.n_cells() == 64 + 125);
  std::random_device random;
  const fs::path dir =
      fs::temp_directory_path() / ("autan_sa_ghostless-" + std::to_string(random()));
  autan::io::write_polymesh(dir / "polyMesh", mesh);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    d["reproducibility"] = reproducibility;
    d["mesh"]["path"] = (dir / "polyMesh").string();
    d["mesh"]["sha256"] = autan::io::mesh_hash(dir / "polyMesh");
    Json zone = d["cell_zones"][0];
    zone["zone"] = "fluid";
    d["cell_zones"] = Json::array({zone});
    d["time"]["linear_solver"] = "gauss-seidel";
    d["time"]["max_iterations"] = 200;
  });
  const auto one = run(doc);
  INFO(one.err);
  REQUIRE(one.status == ExitStatus::finished);
  REQUIRE(field(one.run, "SolverSecondOrder", "RestartState").front() == 1.0);
  std::string err;
  const fs::path many = run_on_ranks(doc, 3, 0, &err);
  INFO(err);
  REQUIRE(!many.empty());
  // The partition this test needs: a rank without ghosts beside ranks with some.
  std::vector<std::int64_t> ghosts;
  std::istringstream log(read(many / "log.jsonl"));
  for (std::string l; std::getline(log, l);) {
    const auto e = Json::parse(l);
    if (e["source"] == "mpi") ghosts = e["data"]["ghost_cells"].get<std::vector<std::int64_t>>();
  }
  REQUIRE(ghosts.size() == 3);
  CHECK(std::ranges::count(ghosts, 0) == 1);
  if (std::string(reproducibility) == "ranks") {
    CHECK(read(many / "history.jsonl") == read(one.run / "history.jsonl"));
    const auto a = data_arrays(one.run), b = data_arrays(many);
    REQUIRE(a.size() == b.size());
    for (const auto& [name, values] : a) {
      INFO(name);
      REQUIRE(b.contains(name));
      REQUIRE(values.size() == b.at(name).size());
      CHECK(std::memcmp(values.data(), b.at(name).data(), values.size() * sizeof(double)) == 0);
    }
  } else {
    CHECK(Json::parse(read(many / "run.json"))["status"] ==
          Json::parse(read(one.run / "run.json"))["status"]);
  }
}

TEST_CASE("ranks that disagree on planar faces all run the full loops", "[euler][mpi]") {
  // A box of hexahedra with one interior node moved along z, in the part that autan-run's
  // partition gives rank 1 and away from rank 0's ghost layers: rank 0's faces are all planar,
  // rank 1's are not. EulerPhysics takes the minimum over the ranks: no rank logs the planar
  // loops, and under reproducibility "ranks" two ranks give the one-rank run bit for bit.
  constexpr int n = 10;
  auto box = autan::test::box(n, autan::test::BoxFill::hexa);
  const auto plain = autan::assemble(autan::to_face_based(autan::CellwiseMesh(box)));
  const auto parts =
      autan::partition_cells(plain, 2, {.cell_weights = autan::cell_cost_weights(plain)});
  const auto g = autan::compute_geometry(plain);
  const double h = 1.0 / n;
  // The interior node whose nearest rank-0 cell centroid is farthest.
  int best = -1;
  double far = 0.0;
  for (int k = 1; k < n; ++k) {
    for (int j = 1; j < n; ++j) {
      for (int i = 1; i < n; ++i) {
        double nearest = 1e300;
        for (std::size_t c = 0; c < parts.size(); ++c) {
          if (parts[c] != 0) continue;
          nearest = std::min(nearest,
                             std::hypot(g.cell_centroid_x[c] - i * h, g.cell_centroid_y[c] - j * h,
                                        g.cell_centroid_z[c] - k * h));
        }
        if (nearest > far) {
          far = nearest;
          best = i + (n + 1) * (j + (n + 1) * k);
        }
      }
    }
  }
  // Two ghost layers of rank 0 reach centroids within 2.5 h of its part: the moved node's faces
  // lie beyond them.
  REQUIRE(far > 3.6 * h);
  box.z[static_cast<std::size_t>(best)] += 0.25 * h;
  const auto mesh = autan::assemble(autan::to_face_based(std::move(box)));
  REQUIRE_FALSE(autan::kernels::pack_geometry(mesh, autan::compute_geometry(mesh)).all_planar);
  std::random_device random;
  const fs::path dir =
      fs::temp_directory_path() / ("autan_euler_disagree-" + std::to_string(random()));
  autan::io::write_polymesh(dir / "polyMesh", mesh);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    d["reproducibility"] = "ranks";
    inviscid_along_x(d, "fp64");
    for (auto& b : d["boundaries"]) b["kind"] = "farfield";
    d["mesh"]["path"] = (dir / "polyMesh").string();
    d["mesh"]["sha256"] = autan::io::mesh_hash(dir / "polyMesh");
    Json zone = d["cell_zones"][0];
    zone["zone"] = "fluid";
    zone["initial"]["density"] = 1.2 * zone["initial"]["density"].get<double>();
    d["cell_zones"] = Json::array({zone});
    d["time"] = {{"kind", "unsteady"},      {"steps", 6},
                 {"dt", nullptr},           {"cfl", 0.8},
                 {"average_from", nullptr}, {"checkpoint", {{"seconds", 1800.0}}}};
  });
  const auto one = run(doc);
  INFO(one.err);
  REQUIRE(one.status == ExitStatus::finished);
  const fs::path many = run_on_ranks(doc, 2);
  REQUIRE(!many.empty());
  CHECK(planar_events(one.run) == 0);
  CHECK(planar_events(many) == 0);
  check_bit_for_bit(one.run, many);
}

TEST_CASE("a steady multigrid case on several ranks is refused before any directory exists",
          "[euler][mpi]") {
  // The multigrid agglomerates one mesh's cells: E0706 at `time.linear_solver`, exit 2.
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) { inviscid_along_x(d, "fp64"); });
  std::string err;
  CHECK(run_on_ranks(doc, 2, 2, &err).empty());
  INFO(err);
  // The launcher may write its own lines; the diagnostic is the first JSON object.
  Json diagnostic;
  std::istringstream lines(err);
  for (std::string line; std::getline(lines, line);) {
    if (line.starts_with('{')) {
      diagnostic = Json::parse(line);
      break;
    }
  }
  CHECK(diagnostic["code"] == "E0706");
  CHECK(diagnostic["location"]["path"] == "time.linear_solver");
}

TEST_CASE("a steady run whose state is not physical on some ranks stops every rank",
          "[euler][mpi]") {
  // Zone `lower` at 1e-9 times the freestream density rounds to zero density in fp32+acc64. On 4
  // ranks some parts hold none of it: they must take the divergence branch with the others, or the
  // ranks enter different collectives and the run hangs instead of reporting D0107.
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, "fp32+acc64");
    for (auto& z : d["cell_zones"]) {
      if (z["zone"] == "lower")
        z["initial"]["density"] = 1e-9 * d["freestream"]["density"].get<double>();
    }
    d["time"]["linear_solver"] = "gauss-seidel";
    d["time"]["max_iterations"] = 5;
  });
  const fs::path many = run_on_ranks(doc, 4);
  REQUIRE(!many.empty());
  const Json info = Json::parse(read(many / "run.json"));
  INFO(info["diagnostics"].dump());
  CHECK(info["status"] == "diverged");
  const auto& diagnostics = info["diagnostics"];
  const auto d107 = std::find_if(diagnostics.begin(), diagnostics.end(),
                                 [](const Json& g) { return g["code"] == "D0107"; });
  REQUIRE(d107 != diagnostics.end());
  CHECK((*d107)["data"]["zone"] == "lower");
}

TEST_CASE("a steady Gauss-Seidel or lines case on several ranks converges to the one-rank solution",
          "[euler][mpi]") {
  // Gauss-Seidel sweeps the runtime's colours by global id, and the step length, the worst cell
  // and the norms reduce over the ranks, so N ranks run the one-rank algorithm. "lines" sweeps
  // the multigrid's fine lines, kept whole by the partition, over the runtime's colouring of the
  // lines. Under reproducibility "run" the face sums follow local order, so the bits differ and a
  // fixed rank count repeats bit for bit. Converged in fp64, the fields agree within 1e-9 of their
  // magnitude (the largest speed for velocity components). In fp32+acc64 the runs stop near the
  // round-off floor, where the state drifts with the summation order (partition, cell order,
  // processor), so the N-rank state is held to the stop rule instead (case-schema.md#time): one
  // rank, started from it, reports for each equation a residual within the one-rank run's
  // threshold, the larger of its iteration-1 residual times 10^-orders and the floor, plus the
  // floor, which the contract places above the round-off of evaluating a residual, here of one
  // state in another summation order. Under
  // "ranks" every sum follows the global order: bit for bit. Two zones at different states on the
  // hybrid mesh, farfield around.
  const int ranks = GENERATE(2, 4);
  const char* linear = GENERATE("gauss-seidel", "lines");
  const char* reproducibility = GENERATE("run", "ranks");
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const bool fp64 = std::string(precision) == "fp64";
  CAPTURE(ranks, linear, reproducibility, precision);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    d["reproducibility"] = reproducibility;
    inviscid_along_x(d, precision);
    for (auto& b : d["boundaries"]) b["kind"] = "farfield";
    auto& init = d["cell_zones"].back()["initial"];
    init["density"] = 1.2 * init["density"].get<double>();
    init["pressure"] = 1.3 * init["pressure"].get<double>();
    d["time"]["linear_solver"] = linear;
    d["time"]["max_iterations"] = 2000;
    d["time"]["residual_orders"] = fp64 ? 10.0 : 5.0;
  });
  const auto one = run(doc);
  INFO(one.err);
  REQUIRE(one.status == ExitStatus::finished);
  CHECK(Json::parse(read(one.run / "run.json"))["status"] == "converged");
  const fs::path many = run_on_ranks(doc, ranks);
  REQUIRE(!many.empty());
  CHECK(Json::parse(read(many / "run.json"))["status"] == "converged");
  {
    // Each rank owns the cells, and holds the ghosts, of autan-mesh-info --ranks N, with
    // --groups lines when the case forms the lines (mesh-info.md, partition).
    Json partitioned;
    std::ifstream log_in(many / "log.jsonl");
    for (std::string line; std::getline(log_in, line);) {
      const Json event = Json::parse(line);
      if (event["source"] == "mpi") partitioned = event["data"];
    }
    const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
    const auto groups =
        std::string(linear) == "lines" ? autan::io::RankGroups::lines : autan::io::RankGroups::none;
    const auto counts =
        autan::io::run_counts(mesh, 1.0, autan::io::CellOrder::rcm,
                              autan::io::RanksRequest{.ranks = ranks, .groups = groups});
    REQUIRE(counts.partition);
    CHECK(partitioned["owned_cells"] == Json(counts.partition->cells));
    CHECK(partitioned["ghost_cells"] == Json(counts.partition->ghosts));
  }
  if (fp64) {
    const auto vx = field(one.run, "VelocityX"), vy = field(one.run, "VelocityY"),
               vz = field(one.run, "VelocityZ");
    double speed = 0.0;
    for (std::size_t c = 0; c < vx.size(); ++c) {
      speed = std::max(speed, std::hypot(vx[c], vy[c], vz[c]));
    }
    for (const char* name : {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}) {
      const auto a = field(one.run, name), b = field(many, name);
      REQUIRE(a.size() == b.size());
      double diff = 0.0, scale = 0.0;
      for (std::size_t c = 0; c < a.size(); ++c) {
        diff = std::max(diff, std::abs(a[c] - b[c]));
        scale = std::max(scale, std::abs(a[c]));
      }
      if (std::string(name).starts_with("Velocity")) scale = speed;
      INFO(name << " differs by " << diff / scale << " of its magnitude");
      CHECK(diff <= 1e-9 * scale);
    }
  } else {
    // History line 1 of a steady run is the residual of the state it starts from (results.md).
    auto check = doc;
    check["start"] = continue_from(many, "warm");
    check["time"]["max_iterations"] = 1;
    const auto from_many = run(check);
    INFO(from_many.err);
    REQUIRE(from_many.status == ExitStatus::finished);
    const Json first = history(one.run).front()["residuals"];
    const Json floor = Json::parse(read(one.run / "run.json"))["convergence"]["floor"];
    const double drop = std::pow(10.0, -doc["time"]["residual_orders"].get<double>());
    const Json residuals = history(from_many.run).front()["residuals"];
    REQUIRE(residuals.size() == first.size());
    for (const auto& [name, r] : residuals.items()) {
      const double f = floor[name].get<double>();
      const double bound =
          std::max(first[name].get<double>() * drop, autan::runtime::round_off_band * f) + f;
      INFO(name << ": residual " << r.get<double>() << ", bound " << bound);
      CHECK(r.get<double>() <= bound);
    }
  }
  if (std::string(reproducibility) == "ranks") check_bit_for_bit(one.run, many);
  const fs::path again = run_on_ranks(doc, ranks);
  CHECK(read(again / "history.jsonl") == read(many / "history.jsonl"));
}

TEST_CASE("steady Gauss-Seidel steps through periodic pairs follow the one-rank steps",
          "[euler][mpi]") {
  // The periodic box: image ghosts take du after each colour. Not converged: 40 steps, the
  // fields agree within 1e-9 of their magnitude, and bit for bit under reproducibility "ranks".
  const int ranks = GENERATE(2, 4);
  const char* reproducibility = GENERATE("run", "ranks");
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const bool fp64 = std::string(precision) == "fp64";
  CAPTURE(ranks, reproducibility, precision);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    periodic_box(d, precision);
    d["reproducibility"] = reproducibility;
    const double speed = d["freestream"]["speed"].get<double>();
    d["cell_zones"][0]["initial"]["velocity"] = {0.6 * speed, 0.7 * speed, -0.3 * speed};
    d["time"]["linear_solver"] = "gauss-seidel";
    d["time"]["max_iterations"] = 40;
    d["time"]["residual_orders"] = 30.0;
  });
  const auto one = run(doc);
  INFO(one.err);
  REQUIRE(one.status == ExitStatus::finished);
  const fs::path many = run_on_ranks(doc, ranks);
  REQUIRE(!many.empty());
  // Both ran their 40 steps (not an early divergence with finite fields) and the flow moved.
  for (const fs::path& r : {one.run, many}) {
    const Json info = Json::parse(read(r / "run.json"));
    CHECK(info["status"] == "completed");
    CHECK(info["iterations"] == 40);
  }
  const double rho0 = doc["cell_zones"][0]["initial"]["density"].get<double>();
  const auto density = field(one.run, "Density");
  CHECK(std::ranges::any_of(density, [&](double r) { return std::abs(r - rho0) > 1e-3 * rho0; }));
  const auto vx = field(one.run, "VelocityX"), vy = field(one.run, "VelocityY"),
             vz = field(one.run, "VelocityZ");
  double speed = 0.0;
  for (std::size_t c = 0; c < vx.size(); ++c) {
    speed = std::max(speed, std::hypot(vx[c], vy[c], vz[c]));
  }
  for (const char* name : {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}) {
    const auto a = field(one.run, name), b = field(many, name);
    REQUIRE(a.size() == b.size());
    double diff = 0.0, scale = 0.0;
    for (std::size_t c = 0; c < a.size(); ++c) {
      diff = std::max(diff, std::abs(a[c] - b[c]));
      scale = std::max(scale, std::abs(a[c]));
    }
    if (std::string(name).starts_with("Velocity")) scale = speed;
    INFO(name << " differs by " << diff / scale << " of its magnitude");
    CHECK(diff <= (fp64 ? 1e-9 : 1e-5) * scale);
  }
  if (std::string(reproducibility) == "ranks") check_bit_for_bit(one.run, many);
}
#endif

TEST_CASE("an equation at the round-off floor does not set the steady CFL number", "[euler]") {
  // tests/data/mesh/ringleb is one cell thick in z: the z-momentum is zero and its residual
  // round-off. Two runs differ only by a uniform initial w of 1e-20 times the speed, which
  // changes that round-off and the other equations by an ulp at most; the CFL law must ignore
  // it (case-schema.md#time, the runtime's exemption), so over 40 steps the other equations'
  // residuals agree within 1e-6 of each other plus the round-off they converge to. A cell's
  // residual sums its n_f = 6 face fluxes, whose order (the cell order) changes it by at most
  // n_f ε Σ|F| (Higham 2002, §4.2), Σ|F| the cell's flux magnitude, which the physics' flux scale
  // Φ_k bounds (case-schema.md#time, computed here from its definition); the history's norm is
  // an RMS over cells, as Φ_k, so it moves by as much: 2 n_f ε Φ_k with a margin of 2. Measured
  // at 0.006 of that bound; with the exemption disabled, 3e4 times it.
  const fs::path mesh = ringleb_n004;
  auto doc = [](double w) { return ringleb_slab(w, 40); };
  const auto a = run(doc(0.0));
  INFO(a.err);
  REQUIRE(a.status == ExitStatus::finished);
  const auto b = run(doc(1e-20));
  REQUIRE(b.status == ExitStatus::finished);
  const auto ha = history(a.run), hb = history(b.run);
  REQUIRE(ha.size() == hb.size());
  // Φ_k as case-schema.md#time defines it: f_k √(Σ_i A_i² / N), A_i the surface of cell i.
  const Json spec = doc(0.0);
  const auto m = autan::io::read_mesh(mesh);
  const auto g = autan::compute_geometry(m);
  std::vector<double> surface(static_cast<std::size_t>(m.n_cells()), 0.0);
  for (std::size_t f = 0; f < static_cast<std::size_t>(m.n_faces()); ++f) {
    surface[static_cast<std::size_t>(m.owner()[f])] += g.face_area[f];
    if (f < static_cast<std::size_t>(m.n_interior_faces())) {
      surface[static_cast<std::size_t>(m.neighbour()[f])] += g.face_area[f];
    }
  }
  double area = 0.0;
  for (const double s : surface) area += s * s;
  area = std::sqrt(area / static_cast<double>(surface.size()));
  const auto& fs = spec["freestream"];
  const double gamma = spec["cell_zones"][0]["medium"]["gamma"].get<double>();
  const double rho = fs["density"].get<double>(), speed = fs["speed"].get<double>();
  const double sound = std::sqrt(gamma * fs["pressure"].get<double>() / rho);
  const double q = speed + sound;
  const double enthalpy = sound * sound / (gamma - 1.0) + 0.5 * speed * speed;
  const std::map<std::string, double> scale = {{"density", rho * q * area},
                                               {"momentum-x", rho * q * q * area},
                                               {"momentum-y", rho * q * q * area},
                                               {"energy", rho * enthalpy * q * area}};
  bool noisy = false;
  double worst = 0.0;  // the largest difference, in units of the bound
  for (std::size_t i = 0; i < ha.size(); ++i) {
    for (const char* k : {"density", "momentum-x", "momentum-y", "energy"}) {
      const double x = ha[i]["residuals"][k].get<double>();
      const double y = hb[i]["residuals"][k].get<double>();
      const double flux = scale.at(k);
      const double bound =
          1e-6 * std::abs(x) + 12.0 * std::numeric_limits<double>::epsilon() * flux;
      worst = std::max(worst, std::abs(x - y) / bound);
    }
    noisy = noisy || hb[i]["residuals"]["momentum-z"] != ha[i]["residuals"]["momentum-z"];
  }
  INFO("histories differ by " << worst);
  CHECK(worst <= 1.0);
  // A continuation carries the exemption: from 20 steps, without `start.woken` it repeats the
  // uninterrupted run. With the z-momentum woken, that equation drops from the largest momentum
  // residual (#432), against which its round-off is nothing: the CFL law still ignores it.
  auto half = doc(1e-20);
  half["time"]["max_iterations"] = 20;
  const auto first = run(half);
  REQUIRE(first.status == ExitStatus::finished);
  auto rest = doc(1e-20);
  rest["start"] = continue_from(first.run);
  REQUIRE(rest["start"]["woken"] == Json::array());
  const auto plain = run(rest);
  INFO(plain.err);
  REQUIRE(plain.status == ExitStatus::finished);
  CHECK(read(first.run / "history.jsonl") + read(plain.run / "history.jsonl") ==
        read(b.run / "history.jsonl"));
  rest["start"]["woken"] = {"momentum-z"};
  const auto woken = run(rest);
  REQUIRE(woken.status == ExitStatus::finished);
  CHECK(read(woken.run / "history.jsonl") == read(plain.run / "history.jsonl"));
  CHECK(noisy);  // the injected noise did reach the z-momentum residual
}

TEST_CASE("the steady CFL penalty halves after a rise of the relative residual above 10 %",
          "[euler]") {
  // euler.md, time stepping: the penalty halves after a step whose largest residual relative to
  // its value at the start of the current order rose by more than 10 % over the step before's,
  // and otherwise recovers by half, up to 1 (a shortened step, or a failed linear solve, lowers it
  // too; steps next to one, which log a warning, are not judged). Past the first-order start
  // those residuals are the history's, relative to the first second-order line, whose residuals
  // the restart state holds as the reference; the z-momentum, below 1e-12 of the largest residual
  // at iteration 1, is left out (case-schema.md#time). The restart state also holds the penalty
  // after a run's last step, so runs of k - 1 and k steps give step k's change. Steps away from
  // the margin are judged: a rise above 15 % and one below 5 %.
  const std::int64_t steps = 120;
  const auto whole = run(ringleb_slab(0.0, steps));
  INFO(whole.err);
  REQUIRE(whole.status == ExitStatus::finished);
  REQUIRE(field(whole.run, "SolverSecondOrder", "RestartState").front() == 1.0);
  const std::vector<std::string> names{"density", "momentum-x", "momentum-y", "momentum-z",
                                       "energy"};
  std::vector<double> reference;
  for (std::size_t k = 0; k < names.size(); ++k) {
    const std::string name = "SolverReferenceResidual" + std::to_string(k);
    reference.push_back(field(whole.run, name.c_str(), "RestartState").front());
  }
  const auto h = history(whole.run);
  const auto at = [&](std::size_t i, std::size_t k) {
    return h[i]["residuals"][names[k]].get<double>();
  };
  std::size_t first = 0;
  while (first < h.size() && !std::ranges::all_of(
                                 std::views::iota(std::size_t{0}, names.size()),
                                 [&](std::size_t k) { return at(first, k) == reference[k]; })) {
    ++first;
  }
  INFO("first second-order line " << first + 1 << " of " << h.size());
  REQUIRE(first + 2 < h.size());
  double largest = 0.0;
  for (std::size_t k = 0; k < names.size(); ++k) largest = std::max(largest, at(0, k));
  auto relative = [&](std::size_t i) {
    double drop = 0.0;
    for (std::size_t k = 0; k < names.size(); ++k) {
      if (at(0, k) >= 1e-12 * largest) drop = std::max(drop, at(i, k) / reference[k]);
    }
    return drop;
  };
  std::set<std::int64_t> warned;
  {
    std::ifstream log_in(whole.run / "log.jsonl");
    for (std::string line; std::getline(log_in, line);) {
      const Json event = Json::parse(line);
      if (event["level"] == "warning" && event["iteration"].is_number_integer()) {
        warned.insert(event["iteration"].get<std::int64_t>());
      }
    }
  }
  std::optional<std::int64_t> rise, small;  // iterations
  for (std::size_t i = first + 1; i < h.size(); ++i) {
    const std::int64_t k = h[i]["iteration"].get<std::int64_t>();
    if (warned.contains(k) || warned.contains(k - 1)) continue;
    const double ratio = relative(i) / relative(i - 1);
    if (!rise && ratio > 1.15) rise = k;
    if (!small && ratio > 1.0 && ratio < 1.05) small = k;
  }
  REQUIRE(rise);
  REQUIRE(small);
  auto penalty = [](std::int64_t k) {
    const auto r = run(ringleb_slab(0.0, k));
    REQUIRE(r.status == ExitStatus::finished);
    return field(r.run, "SolverPenalty", "RestartState").front();
  };
  INFO("rise at " << *rise << ", small rise at " << *small);
  CHECK(penalty(*rise) == 0.5 * penalty(*rise - 1));
  CHECK(penalty(*small) == std::min(1.0, 1.5 * penalty(*small - 1)));
}

TEST_CASE("a diverged run names the cell whose stored state is not physical", "[euler]") {
  // Zone `lower` starts at 1e-9 times the freestream density: FP32 stores it as a difference from
  // the freestream that rounds to zero density, so the residual is not finite and the run
  // diverges; D0107 must name a cell of that zone. In FP64 the state is representable; whatever
  // happens, a divergence names its cell.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    for (auto& z : d["cell_zones"]) {
      if (z["zone"] == "lower")
        z["initial"]["density"] = 1e-9 * d["freestream"]["density"].get<double>();
    }
    d["time"]["max_iterations"] = 5;
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(!out.run.empty());  // a run directory, whatever the exit status
  const Json info = Json::parse(read(out.run / "run.json"));
  INFO(info["diagnostics"].dump());
  if (std::string(precision) == "fp32+acc64") {
    CHECK(info["status"] == "diverged");
  } else {
    CHECK(info["status"] == "completed");  // FP64 represents the state and runs on
  }
  if (info["status"] == "diverged") {
    const auto& diagnostics = info["diagnostics"];
    const auto d107 = std::find_if(diagnostics.begin(), diagnostics.end(),
                                   [](const Json& g) { return g["code"] == "D0107"; });
    REQUIRE(d107 != diagnostics.end());
    CHECK((*d107)["data"]["zone"] == "lower");
  }
}

TEST_CASE("a fast flow at low pressure that the kernels resolve is not taken for divergence",
          "[euler]") {
  // rho = 1, |v| = 1e4, p = 0.7: the absolute energy rounds the pressure away in FP32, but the
  // stored differences and the gauge pressure keep it; the check must judge the state as the
  // kernels read it.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    auto& f = d["freestream"];
    const double r = d["cell_zones"][0]["medium"]["gas_constant"].get<double>();
    f["density"] = 1.0;
    f["pressure"] = 0.7;
    f["temperature"] = 0.7 / r;
    f["speed"] = 1e4;
    f["velocity"] = {1e4, 0.0, 0.0};
    f["mach"] = 1e4 / std::sqrt(1.4 * 0.7);
    for (auto& z : d["cell_zones"]) {
      z["initial"]["density"] = 1.0;
      z["initial"]["pressure"] = 0.7;
      z["initial"]["temperature"] = 0.7 / r;
      z["initial"]["velocity"] = f["velocity"];
    }
    for (auto& b : d["boundaries"]) b["kind"] = "farfield";
    d["time"] = {{"kind", "unsteady"},      {"steps", 3},
                 {"dt", nullptr},           {"cfl", 0.5},
                 {"average_from", nullptr}, {"checkpoint", {{"seconds", 1800.0}}}};
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  CHECK(Json::parse(read(out.run / "run.json"))["status"] == "completed");
}

TEST_CASE("oracle-state faces take their paired states, whatever the row order", "[euler]") {
  // tests/data/mesh/ringleb with the exact Ringleb states (autan-vv) on its four in-plane sides,
  // in the oracle's units (R = 1/gamma); the same rows in reverse order within each zone give
  // the same solution bit for bit, because faces pair with rows by centroid (face-states.md).
  const fs::path dir = data_dir / "mesh" / "ringleb";
  auto doc_for = [&](bool reversed) {
    Json states = Json::parse(read(dir / "ringleb-n004.states.json"));
    if (reversed) {
      for (auto& [name, z] : states["zones"].items()) {
        std::reverse(z["centroids"].begin(), z["centroids"].end());
        std::reverse(z["values"].begin(), z["values"].end());
      }
    }
    const std::string bytes = states.dump();
    std::random_device random;
    const fs::path file =
        fs::temp_directory_path() / ("autan_ringleb-" + std::to_string(random()) + ".json");
    std::ofstream(file, std::ios::binary) << bytes;
    std::string sha;
    picosha2::hash256_hex_string(bytes, sha);
    return input("hybrid.solver-input.json", [&](Json& d) {
      inviscid_along_x(d, "fp64");
      const fs::path mesh = dir / "ringleb-n004.msh";
      d["mesh"]["path"] = fs::absolute(mesh).string();
      d["mesh"]["sha256"] = autan::io::mesh_hash(mesh);
      const double gamma = 1.4, rho = 0.8, p = 0.6, r = 1.0 / gamma;
      auto& f = d["freestream"];
      f["density"] = rho;
      f["pressure"] = p;
      f["temperature"] = p / (rho * r);
      f["speed"] = 0.5;
      f["velocity"] = {0.0, -0.5, 0.0};
      f["mach"] = 0.5 / std::sqrt(gamma * p / rho);
      Json zone = d["cell_zones"][0];
      zone["zone"] = "fluid";
      zone["medium"]["gas_constant"] = r;
      zone["initial"]["density"] = rho;
      zone["initial"]["pressure"] = p;
      zone["initial"]["temperature"] = p / (rho * r);
      zone["initial"]["velocity"] = f["velocity"];
      d["cell_zones"] = Json::array({zone});
      Json boundaries = Json::array();
      for (const char* name : {"k-min", "k-max", "inflow", "outflow", "back", "front"}) {
        Json b = d["boundaries"][0];
        for (auto& [key, value] : b.items()) {
          if (key != "zone" && key != "kind") value = nullptr;
        }
        b["zone"] = name;
        const bool side = std::string(name) != "back" && std::string(name) != "front";
        b["kind"] = side ? "oracle-state" : "symmetry";
        if (side) b["states"] = Json{{"path", file.string()}, {"sha256", sha}};
        boundaries.push_back(b);
      }
      d["boundaries"] = boundaries;
      d["monitors"][0]["zones"] = {"k-min"};
      d["time"]["max_iterations"] = 200;
    });
  };
  const auto a = run(doc_for(false));
  INFO(a.err);
  REQUIRE(a.status == ExitStatus::finished);
  const auto b = run(doc_for(true));
  REQUIRE(b.status == ExitStatus::finished);
  for (const char* name : {"Density", "VelocityX", "VelocityY", "Pressure"}) {
    CHECK(field(a.run, name) == field(b.run, name));
  }
  // The states reach the cells: the flow is no longer the uniform initial state.
  const auto rho = field(a.run, "Density");
  CHECK(*std::ranges::max_element(rho) - *std::ranges::min_element(rho) > 0.1);
}

TEST_CASE("oracle-state faces impose their states component by component", "[euler]") {
  // Every side of the hybrid mesh takes one uniform state with all five components distinct and
  // non-zero; started there, the flow stays there in both precisions, inviscid or laminar.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const bool laminar = GENERATE(false, true);
  INFO(precision << (laminar ? " laminar" : ""));
  const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
  const autan::Geometry g = autan::compute_geometry(mesh);
  const std::array<double, 5> w{1.2, 80.0, -30.0, 20.0, 1.1e5};
  Json zones = Json::object();
  const auto off = mesh.boundary_zone_offsets();
  for (std::size_t z = 0; z < mesh.face_zones().size(); ++z) {
    Json centroids = Json::array(), values = Json::array();
    for (auto f = off[z]; f < off[z + 1]; ++f) {
      const auto i = static_cast<std::size_t>(f);
      centroids.push_back({g.face_centroid_x[i], g.face_centroid_y[i], g.face_centroid_z[i]});
      values.push_back(w);
    }
    if (!centroids.empty()) {
      zones[mesh.face_zones()[z].name] = {
          {"faces", centroids.size()}, {"centroids", centroids}, {"values", values}};
    }
  }
  const Json states = {
      {"schema", "autan.face-states/1"},
      {"oracle", "uniform"},
      {"mesh_sha256", autan::io::mesh_hash(mesh_path)},
      {"variables", {"density", "velocity-x", "velocity-y", "velocity-z", "pressure"}},
      {"zones", zones}};
  const std::string bytes = states.dump();
  std::random_device random;
  const fs::path file =
      fs::temp_directory_path() / ("autan_uniform-" + std::to_string(random()) + ".json");
  std::ofstream(file, std::ios::binary) << bytes;
  std::string sha;
  picosha2::hash256_hex_string(bytes, sha);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    if (laminar) {
      laminar_along_x(d, precision);
    } else {
      inviscid_along_x(d, precision);
    }
    for (auto& z : d["cell_zones"]) {
      z["initial"]["density"] = w[0];
      z["initial"]["velocity"] = {w[1], w[2], w[3]};
      z["initial"]["pressure"] = w[4];
    }
    for (auto& b : d["boundaries"]) {
      b["kind"] = "oracle-state";
      b["states"] = Json{{"path", file.string()}, {"sha256", sha}};
    }
    d["time"]["max_iterations"] = 20;
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  const double speed = std::hypot(w[1], w[2], w[3]);
  const std::array<std::pair<const char*, double>, 5> expected{{{"Density", w[0]},
                                                                {"VelocityX", w[1]},
                                                                {"VelocityY", w[2]},
                                                                {"VelocityZ", w[3]},
                                                                {"Pressure", w[4]}}};
  for (const auto& [name, value] : expected) {
    INFO(name);
    const double scale =
        std::string(name) == "Density" || std::string(name) == "Pressure" ? std::abs(value) : speed;
    for (const double v : field(out.run, name)) CHECK(std::abs(v - value) <= 64 * eps * scale);
  }
}

TEST_CASE("a Spalart-Allmaras run steps its six equations", "[navier-stokes]") {
  // The contract's hybrid case as it is: SA-neg, a no-slip wall, farfield faces.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    d["precision"] = precision;
    d["time"]["max_iterations"] = 3;
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const Json info = Json::parse(read(out.run / "run.json"));
  CHECK(info["status"] == "completed");
  const auto lines = history(out.run);
  REQUIRE(lines.size() == 3);
  for (const auto& line : lines) {
    CHECK(line["residuals"].size() == 6);
    for (const auto& [name, r] : line["residuals"].items()) {
      INFO(name);
      CHECK(std::isfinite(r.get<double>()));
    }
  }
}

TEST_CASE("a continued Spalart-Allmaras run repeats the uninterrupted one bit for bit",
          "[navier-stokes]") {
  // The residual is a function of the state: the eddy viscosity the flow reads is the current
  // state's, which a restart rebuilds. Continued one step before and one after the switch to
  // second order (SolverSecondOrder in the restart state), where the SA update starts.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  CAPTURE(precision);
  auto doc = [&](int budget) {
    return input("hybrid.solver-input.json", [&](Json& d) {
      d["precision"] = precision;
      d["time"]["max_iterations"] = budget;
      d["time"]["residual_orders"] = 30.0;
    });
  };
  auto finished = [&](const Json& d) {
    const auto r = run(d);
    INFO(r.err);
    REQUIRE(r.status == ExitStatus::finished);
    return r.run;
  };
  // A few steps past the switch to second order (step 35 in fp64, 192 in fp32+acc64); the
  // checks below fail if it moves past the budget.
  const int budget = std::string(precision) == "fp64" ? 40 : 200;
  const auto whole = finished(doc(budget));
  const auto steps = static_cast<int>(history(whole).size());
  REQUIRE(field(whole, "SolverSecondOrder", "RestartState").front() == 1.0);
  auto second_order = [&](int n) {
    return field(finished(doc(n)), "SolverSecondOrder", "RestartState").front() == 1.0;
  };
  // The start counts its first-order steps, the one that switches excluded: the switch comes
  // after s steps, which the runs of s - 1 and s steps confirm.
  const int s = static_cast<int>(field(whole, "SolverFirstOrderSteps", "RestartState").front()) + 1;
  INFO("switch at step " << s);
  REQUIRE(!second_order(s - 1));
  REQUIRE(second_order(s));
  REQUIRE(s >= 2);
  REQUIRE(steps >= s + 2);
  const auto all = read(whole / "history.jsonl");
  for (const int at : {s - 1, s + 1}) {
    const auto first = finished(doc(at));
    auto next = doc(budget);
    next["start"] = continue_from(first);
    const auto rest = finished(next);
    CHECK(read(first / "history.jsonl") + read(rest / "history.jsonl") == all);
  }
}

TEST_CASE("a point vortex takes the Kutta-Joukowski circulation of its surfaces' lift", "[euler]") {
  // tests/data/mesh/hybrid with y halved (span 0.5); the slip walls xmin and zmin carry the
  // vortex of every other side. Uniform at p1 = 1.3 p_inf, the walls carry (p1 - p_inf) A on
  // their outward normals -x and -z (A = 0.5 each), so with e_L = (-sin a, 0, cos a) the lift per
  // unit span is L' = (p1 - p_inf)(sin a - cos a), and the first step's residual gives
  // Gamma = L' / (rho_inf V_inf).
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  std::random_device random;
  const fs::path mesh = fs::temp_directory_path() / ("autan_halfspan-" + std::to_string(random()));
  fs::create_directories(mesh);
  for (const auto& entry : fs::directory_iterator(mesh_path)) {
    fs::copy_file(entry.path(), mesh / entry.path().filename());
  }
  {
    std::istringstream in(read(mesh_path / "points"));
    std::ofstream out(mesh / "points", std::ios::trunc);
    for (std::string line; std::getline(in, line);) {
      double x = 0, y = 0, z = 0;
      if (std::sscanf(line.c_str(), "(%lf %lf %lf)", &x, &y, &z) == 3) {
        char text[128];
        std::snprintf(text, sizeof text, "(%.17g %.17g %.17g)", x, 0.5 * y, z);
        line = text;
      }
      out << line << '\n';
    }
  }
  const double a = 0.2;
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    d["mesh"]["path"] = mesh.string();
    d["mesh"]["sha256"] = autan::io::mesh_hash(mesh);
    auto& f = d["freestream"];
    const double speed = f["speed"].get<double>();
    f["aoa"] = a;
    f["velocity"] = {speed * std::cos(a), 0.0, speed * std::sin(a)};
    const double r = d["cell_zones"][0]["medium"]["gas_constant"].get<double>();
    const double p1 = 1.3 * f["pressure"].get<double>();
    for (auto& z : d["cell_zones"]) {
      z["initial"]["velocity"] = f["velocity"];
      z["initial"]["pressure"] = p1;
      z["initial"]["temperature"] = p1 / (z["initial"]["density"].get<double>() * r);
    }
    for (auto& b : d["boundaries"]) {
      if (b["zone"] == "xmin" || b["zone"] == "zmin") {
        b["kind"] = "slip";
      } else {
        b["kind"] = "farfield";
        b["point_vortex"] = {{"zones", {"xmin", "zmin"}}, {"center", {0.5, 0.0, -0.5}}};
      }
    }
    d["monitors"] = Json::array();
    d["time"]["max_iterations"] = 1;
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const auto& f = doc["freestream"];
  const double p = f["pressure"].get<double>(), rho = f["density"].get<double>();
  const double speed = f["speed"].get<double>();
  const double expected = 0.3 * p * (std::sin(a) - std::cos(a)) / (rho * speed);
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  const double circulation = field(out.run, "VortexCirculation0", "RestartState").front();
  CHECK(std::abs(circulation - expected) <= 64 * eps * 1.3 * p / (rho * speed));
}

TEST_CASE("a point vortex over periodic image ghosts writes its circulation for the owned cells",
          "[euler]") {
  // The periodic box on one rank holds image ghosts after its cells; restart fields, the
  // circulation included, cover the cells of the mesh only (results.md, RestartState).
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    periodic_box(d, "fp64");
    for (auto& b : d["boundaries"]) {
      if (b["zone"] == "zmax") {
        b["kind"] = "farfield";
        b["point_vortex"] = {{"zones", {"zmin"}}, {"center", {0.5, 0.5, -0.5}}};
      }
    }
    d["time"]["linear_solver"] = "gauss-seidel";
    d["time"]["max_iterations"] = 2;
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  CHECK(Json::parse(read(out.run / "run.json"))["status"] == "completed");
  CHECK(field(out.run, "VortexCirculation0", "RestartState").size() ==
        field(out.run, "Density").size());
}

TEST_CASE("a point vortex reaches the flow and continues bit for bit", "[euler]") {
  // Steady flow at zero incidence over the slip wall zmin, two cell zones at different states;
  // every other side is farfield with the vortex of zmin's lift below the wall.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  auto doc = [&](int iterations) {
    return input("hybrid.solver-input.json", [&](Json& d) {
      inviscid_along_x(d, precision);
      for (auto& b : d["boundaries"]) {
        if (b["zone"] == "zmin") {
          b["kind"] = "slip";
        } else {
          b["kind"] = "farfield";
          b["point_vortex"] = {{"zones", {"zmin"}}, {"center", {0.5, 0.0, -0.3}}};
        }
      }
      auto& init = d["cell_zones"].back()["initial"];
      init["density"] = 1.2 * init["density"].get<double>();
      init["pressure"] = 1.3 * init["pressure"].get<double>();
      d["monitors"][0]["zones"] = {"zmin"};
      d["time"]["max_iterations"] = iterations;
      d["time"]["residual_orders"] = 30.0;
    });
  };
  const auto whole = run(doc(8));
  INFO(whole.err);
  REQUIRE(whole.status == ExitStatus::finished);
  CHECK(field(whole.run, "VortexCirculation0", "RestartState").front() != 0.0);
  auto plain = doc(8);
  for (auto& b : plain["boundaries"]) b["point_vortex"] = nullptr;
  const auto without = run(plain);
  REQUIRE(without.status == ExitStatus::finished);
  CHECK(field(without.run, "VelocityZ") != field(whole.run, "VelocityZ"));

  const auto first = run(doc(4));
  REQUIRE(first.status == ExitStatus::finished);
  auto next = doc(8);
  next["start"] = continue_from(first.run);
  const auto second = run(next);
  INFO(second.err);
  REQUIRE(second.status == ExitStatus::finished);
  CHECK(read(first.run / "history.jsonl") + read(second.run / "history.jsonl") ==
        read(whole.run / "history.jsonl"));
  for (const char* name : {"Density", "VelocityX", "VelocityZ", "Pressure"}) {
    CHECK(field(second.run, name) == field(whole.run, name));
  }
  CHECK(field(second.run, "VortexCirculation0", "RestartState") ==
        field(whole.run, "VortexCirculation0", "RestartState"));
}

TEST_CASE("a point vortex takes the viscous lift of its walls", "[navier-stokes]") {
  // Laminar flow over the adiabatic wall zmin of the unit cube, every other side farfield with the
  // vortex of zmin's lift. The initial state (initial-state.md) is the shear u = (k z, 0, 0) at the
  // freestream's density and pressure: the wall carries no pressure force and the traction
  // (mu k, 0, 0), so with e_L = (-sin a, 0, cos a) the lift per unit span is L' = -sin a mu k, and
  // the first step's residual gives Gamma = L' / (rho_inf V_inf), as the force monitors count it.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
  const autan::Geometry g = autan::compute_geometry(mesh);
  constexpr double a = 0.2;
  constexpr double mu = 1.0;
  constexpr double k = 100.0;
  Json doc = input("hybrid.solver-input.json", [&](Json& d) {
    laminar_along_x(d, precision);
    auto& f = d["freestream"];
    const double speed = f["speed"].get<double>();
    f["aoa"] = a;
    f["velocity"] = {speed * std::cos(a), 0.0, speed * std::sin(a)};
    f["viscosity"] = mu;
    for (auto& z : d["cell_zones"]) {
      z["medium"]["viscosity"] = {{"law", "constant"}, {"mu", mu}};
      z["initial"]["velocity"] = f["velocity"];
    }
    for (auto& b : d["boundaries"]) {
      if (b["zone"] == "zmin") {
        b["kind"] = "wall";
      } else {
        b["kind"] = "farfield";
        b["point_vortex"] = {{"zones", {"zmin"}}, {"center", {0.5, 0.0, -0.5}}};
      }
    }
    d["monitors"] = Json::array();
    d["time"]["max_iterations"] = 1;
  });
  const auto& f = doc["freestream"];
  const double rho = f["density"].get<double>(), p = f["pressure"].get<double>();
  const double speed = f["speed"].get<double>();
  Json centroids = Json::array(), values = Json::array();
  for (std::size_t c = 0; c < static_cast<std::size_t>(mesh.n_cells()); ++c) {
    centroids.push_back({g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]});
    values.push_back({rho, k * g.cell_centroid_z[c], 0.0, 0.0, p});
  }
  const Json state{{"schema", "autan.initial-state/1"},
                   {"oracle", "freestream"},
                   {"mesh_sha256", autan::io::mesh_hash(mesh_path)},
                   {"variables", {"density", "velocity-x", "velocity-y", "velocity-z", "pressure"}},
                   {"cells", mesh.n_cells()},
                   {"centroids", centroids},
                   {"values", values}};
  std::random_device random;
  const fs::path file = fs::temp_directory_path() / ("autan_initial-" + std::to_string(random()));
  std::ofstream(file) << state.dump();
  std::string sha;
  picosha2::hash256_hex_string(read(file), sha);
  doc["initial_state"] = {{"path", file.string()}, {"sha256", sha}};
  const auto out = run(doc);
  fs::remove(file);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const double expected = -std::sin(a) * mu * k / (rho * speed);
  // The wall pressure is the cell's, recovered from the stored energy: its round-off times the
  // wall's area bounds the pressure force.
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  const double circulation = field(out.run, "VortexCirculation0", "RestartState").front();
  CHECK(std::abs(circulation - expected) <= 64 * eps * (mu * k + p / 0.4) / (rho * speed));
}

#ifdef AUTAN_PHYSICS_CUDA
namespace {

void on_device(Json& d) { d["devices"] = {{"kind", "cuda"}, {"count", 1}}; }

}  // namespace

TEST_CASE("on a device, a uniform flow along slip walls and farfield faces stays uniform",
          "[euler][gpu]") {
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const bool steady = GENERATE(true, false);
  INFO(precision << (steady ? " steady" : " unsteady"));
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    on_device(d);
    if (steady) {
      d["time"]["max_iterations"] = 20;
    } else {
      d["time"] = {{"kind", "unsteady"}, {"steps", 20},
                   {"dt", nullptr},      {"cfl", 0.8},
                   {"average_from", 10}, {"checkpoint", {{"seconds", 1800.0}}}};
    }
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const auto& f = doc["freestream"];
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  const double rho = f["density"].get<double>(), p = f["pressure"].get<double>();
  const double speed = f["speed"].get<double>();
  for (const double v : field(out.run, "Density")) CHECK(std::abs(v - rho) <= 64 * eps * rho);
  for (const double v : field(out.run, "Pressure")) CHECK(std::abs(v - p) <= 64 * eps * p);
  for (const double v : field(out.run, "VelocityX")) CHECK(std::abs(v - speed) <= 64 * eps * speed);
  for (const double v : field(out.run, "VelocityZ")) CHECK(std::abs(v) <= 64 * eps * speed);
}

TEST_CASE("on a device, checkpoints change no Euler result", "[euler][gpu]") {
  const char* precision = GENERATE("fp64", "fp32+acc64");
  CAPTURE(precision);
  check_checkpoints_change_nothing(input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    on_device(d);
    for (auto& b : d["boundaries"]) b["kind"] = "farfield";
    auto& init = d["cell_zones"].back()["initial"];
    init["pressure"] = 1.3 * init["pressure"].get<double>();
    d["time"]["max_iterations"] = 8;
  }));
}

TEST_CASE("on a device, a verification case converges to the host's discrete solution",
          "[euler][gpu]") {
  // The device steps explicitly (local time steps), the host implicitly: both converge to the same
  // discrete solution. fp64: they differ by at most 1e-3 times its discretisation error, per field
  // in the volume-weighted L2 norm. fp32+acc64: they agree as fp32+acc64 agrees with fp64
  // (docs/validation/acceptance.md, FP32 conventions: L2 difference over the field's scale, the
  // RMS speed for velocity components, at most 500 x 2^-23).
  const std::string precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  auto doc = [&](bool device) {
    return input("hybrid-verification.solver-input.json", [&](Json& d) {
      d["precision"] = precision;
      d["time"]["max_iterations"] = device ? 200000 : 20000;
      d["time"]["residual_orders"] = precision == "fp64" ? 10.0 : 5.0;
      if (device) on_device(d);
    });
  };
  const auto host = run(doc(false));
  const auto dev = run(doc(true));
  INFO(dev.err);
  REQUIRE(host.status == ExitStatus::finished);
  REQUIRE(dev.status == ExitStatus::finished);
  CHECK(Json::parse(read(dev.run / "run.json"))["status"] == "converged");
  // The device run sets up no host linear solver (no BlockMatrix, no multigrid); the host run does.
  auto linear_events = [](const fs::path& run) {
    int count = 0;
    std::istringstream lines(read(run / "log.jsonl"));
    for (std::string line; std::getline(lines, line);) {
      if (!line.empty() && Json::parse(line)["source"] == "linear") ++count;
    }
    return count;
  };
  CHECK(linear_events(host.run) > 0);
  CHECK(linear_events(dev.run) == 0);
  const auto mesh = autan::io::read_mesh(mesh_path);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  std::vector<std::array<double, 3>> points(n);
  double volume = 0.0;
  for (std::size_t c = 0; c < n; ++c) {
    points[c] = {g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]};
    volume += g.cell_volume[c];
  }
  const autan::physics::Manufactured mms(read(mms_path));
  const std::array<std::string, 5> variables = {"rho", "u", "v", "w", "p"};
  const auto exact = mms.evaluate("fields", variables, points);
  auto l2 = [&](const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t c = 0; c < n; ++c) s += g.cell_volume[c] * std::pow(a[c] - b[c], 2);
    return std::sqrt(s / volume);
  };
  const std::array<const char*, 5> names = {"Density", "VelocityX", "VelocityY", "VelocityZ",
                                            "Pressure"};
  double speed = 0.0;
  for (std::size_t k = 1; k < 4; ++k) {
    const auto v = field(host.run, names[k]);
    speed += std::pow(l2(v, std::vector<double>(n, 0.0)), 2);
  }
  speed = std::sqrt(speed);
  for (std::size_t k = 0; k < 5; ++k) {
    INFO(names[k]);
    const auto a = field(host.run, names[k]), b = field(dev.run, names[k]);
    const double diff = l2(a, b);
    if (precision == "fp64") {
      const double error = l2(a, exact[k]);
      INFO("difference " << diff / error << " of the discretisation error");
      CHECK(diff <= 1e-3 * error);
    } else {
      const double scale = k >= 1 && k <= 3 ? speed : l2(a, std::vector<double>(n, 0.0));
      INFO("difference " << diff / scale / std::numeric_limits<float>::epsilon() << " eps32");
      CHECK(diff <= 500.0 * std::numeric_limits<float>::epsilon() * scale);
    }
  }
}

TEST_CASE("on a device, a run repeats bit for bit and continues bit for bit", "[euler][gpu]") {
  // Deterministic face loops (store-gather) and exact norms: the same input gives the same bits,
  // and a continued run the uninterrupted one's.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  auto doc = [&](int steps) {
    return input("hybrid.solver-input.json", [&](Json& d) {
      inviscid_along_x(d, precision);
      on_device(d);
      for (auto& b : d["boundaries"]) b["kind"] = "farfield";
      auto& init = d["cell_zones"].back()["initial"];
      init["density"] = 1.2 * init["density"].get<double>();
      init["pressure"] = 1.3 * init["pressure"].get<double>();
      d["time"] = {{"kind", "unsteady"}, {"steps", steps},
                   {"dt", nullptr},      {"cfl", 0.8},
                   {"average_from", 2},  {"checkpoint", {{"seconds", 1800.0}}}};
    });
  };
  const auto whole = run(doc(12));
  INFO(whole.err);
  REQUIRE(whole.status == ExitStatus::finished);
  const auto again = run(doc(12));
  REQUIRE(again.status == ExitStatus::finished);
  CHECK(read(again.run / "history.jsonl") == read(whole.run / "history.jsonl"));
  const auto first = run(doc(6));
  REQUIRE(first.status == ExitStatus::finished);
  auto next = doc(12);
  next["start"] = continue_from(first.run);
  const auto second = run(next);
  INFO(second.err);
  REQUIRE(second.status == ExitStatus::finished);
  CHECK(read(first.run / "history.jsonl") + read(second.run / "history.jsonl") ==
        read(whole.run / "history.jsonl"));
  for (const char* name : {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}) {
    CHECK(field(again.run, name) == field(whole.run, name));
    CHECK(field(second.run, name) == field(whole.run, name));
    CHECK(field(second.run, name, "FlowSolutionMean") ==
          field(whole.run, name, "FlowSolutionMean"));
  }
}

TEST_CASE("on a device, a flow through periodic pairs follows the host's", "[euler][gpu]") {
  // Unsteady, through both pairs and into the slip wall zmin: the device fills the periodic
  // images through the runtime's device ghost exchange, as the host through its own. After 12
  // steps the fields agree to round-off: the largest difference over the field's largest
  // magnitude (the largest speed for velocity components) within 256 eps of the precision, the
  // bound of the multi-rank test, as both sum the same fluxes in different orders.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  auto doc = [&](bool device) {
    return input("hybrid.solver-input.json", [&](Json& d) {
      periodic_box(d, precision);
      unsteady_through_pairs(d, 12);
      const double speed = d["freestream"]["speed"].get<double>();
      d["cell_zones"][0]["initial"]["velocity"] = {0.6 * speed, 0.7 * speed, -0.3 * speed};
      if (device) on_device(d);
    });
  };
  const auto host = run(doc(false));
  const auto dev = run(doc(true));
  INFO(dev.err);
  REQUIRE(host.status == ExitStatus::finished);
  REQUIRE(dev.status == ExitStatus::finished);
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  const auto vx = field(host.run, "VelocityX"), vy = field(host.run, "VelocityY"),
             vz = field(host.run, "VelocityZ");
  double speed = 0.0;
  for (std::size_t c = 0; c < vx.size(); ++c)
    speed = std::max(speed, std::hypot(vx[c], vy[c], vz[c]));
  double worst = 0.0;
  for (const char* name : {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}) {
    const auto a = field(host.run, name), b = field(dev.run, name);
    REQUIRE(a.size() == b.size());
    double diff = 0.0, scale = 0.0;
    for (std::size_t c = 0; c < a.size(); ++c) {
      diff = std::max(diff, std::abs(a[c] - b[c]));
      scale = std::max(scale, std::abs(a[c]));
    }
    if (std::string(name).starts_with("Velocity")) scale = speed;
    worst = std::max(worst, diff / (scale * eps));
  }
  INFO("fields differ by " << worst << " eps");
  CHECK(worst <= 256.0);
}

TEST_CASE("on a device, what the device steps cannot honour is refused before any directory",
          "[euler][gpu]") {
  // The device steps take no circulation from the lift (the host's steady step measures it), so a
  // point vortex on a device would keep its initial circulation: E0706, not a silent mismatch.
  // They sum in local order, so reproducibility "ranks" is E1109.
  const bool vortex = GENERATE(true, false);
  CAPTURE(vortex);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, "fp64");
    on_device(d);
    if (!vortex) {
      d["reproducibility"] = "ranks";
      return;
    }
    for (auto& b : d["boundaries"]) {
      if (b["zone"] == "zmin") {
        b["kind"] = "slip";
      } else {
        b["kind"] = "farfield";
        b["point_vortex"] = {{"zones", {"zmin"}}, {"center", {0.5, 0.0, -0.3}}};
      }
    }
    d["monitors"][0]["zones"] = {"zmin"};
  });
  const auto out = run(doc);
  INFO(out.err);
  CHECK(out.status == ExitStatus::invalid_input);
  CHECK(!fs::exists(out.runs));
  const auto diagnostic = Json::parse(out.err.substr(0, out.err.find('\n')));
  CHECK(diagnostic["code"] == (vortex ? "E0706" : "E1109"));
  CHECK(diagnostic["location"]["path"].get<std::string>().ends_with(vortex ? ".point_vortex"
                                                                           : "reproducibility"));
}

TEST_CASE("on a device, a diverged run names the cell whose stored state is not physical",
          "[euler][gpu]") {
  // As on the host: zone `lower` at 1e-9 times the freestream density rounds to zero density in
  // fp32+acc64; D0107 names a cell of that zone.
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, "fp32+acc64");
    on_device(d);
    for (auto& z : d["cell_zones"]) {
      if (z["zone"] == "lower")
        z["initial"]["density"] = 1e-9 * d["freestream"]["density"].get<double>();
    }
    d["time"]["max_iterations"] = 5;
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(!out.run.empty());
  const Json info = Json::parse(read(out.run / "run.json"));
  CHECK(info["status"] == "diverged");
  const auto& diagnostics = info["diagnostics"];
  const auto d107 = std::find_if(diagnostics.begin(), diagnostics.end(),
                                 [](const Json& g) { return g["code"] == "D0107"; });
  REQUIRE(d107 != diagnostics.end());
  CHECK((*d107)["data"]["zone"] == "lower");
}
#endif

TEST_CASE("the first-order start ends with its 201st step or earlier on a drop or a stall",
          "[euler]") {
  // euler.md#time-stepping. Each sequence gives the step the start ends with.
  const auto ends_with = [](const auto& drop) {
    autan::physics::FirstOrderStart start;
    for (int step = 1; step <= 1000; ++step) {
      if (start.ends_after(drop(step))) return step;
    }
    return 0;
  };
  // Never three orders down, a new plateau reference every 50 steps (0.85 < 0.9): the 201st.
  CHECK(ends_with([](int step) { return 0.5 * std::pow(0.85, (step - 1) / 50); }) == 201);
  // The same drop throughout: 100 steps without a new reference after the first.
  CHECK(ends_with([](int) { return 0.5; }) == 101);
  // Three orders down at once.
  CHECK(ends_with([](int) { return 1e-4; }) == 1);
}
