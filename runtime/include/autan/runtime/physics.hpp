#pragma once

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/cgns.hpp"
#include "autan/runtime/input.hpp"

namespace autan::runtime {

/// What one iteration (steady) or one time step (unsteady) reports.
struct StepReport {
  /// One norm per equation of Physics::equations(), defined as in
  /// docs/contracts/results.md#history (FP64, owned fluid cells).
  std::vector<double> residuals;
  /// Physical time advanced by an unsteady step, in seconds; 0 when steady.
  double time_step = 0.0;
  /// First cell whose state or residual is non-finite or non-physical; set
  /// when the solution diverged.
  std::optional<index_t> diverged_cell;
};

/// Force and moment the fluid exerts on a monitor's surfaces (results.md,
/// forces monitor): F (N) and M about the monitor's moment centre (N·m), in
/// mesh axes. The runtime derives the coefficients.
struct Loads {
  std::array<double, 3> force{};
  std::array<double, 3> moment{};
};

/// The run's log (log.jsonl of results.md): one event per call, stamped
/// with the wall-clock time and the current iteration by the runtime.
class Log {
 public:
  virtual ~Log() = default;
  /// `level`: info, warning or error; `source`: [a-z][a-z0-9-]* (solver,
  /// linear, io, mpi…); `code`: a registry code or empty; `data_json`: a
  /// JSON object.
  virtual void write(std::string_view level, std::string_view source, std::string_view message,
                     std::string_view code = {}, std::string_view data_json = "{}") = 0;
};

/// The state of the previous run a run starts from (input.start), read from
/// its solution.cgns.
struct StartState {
  std::vector<io::Field> fields;  ///< FlowSolution: field_names(input), in order
  /// `continue` only (empty for `warm`): FlowSolutionMean when that run
  /// averaged, and its RestartState.
  std::vector<io::Field> mean_fields;
  std::vector<io::Field> restart_fields;
};

/// Valid until the run ends; a physics may keep a copy.
struct PhysicsContext {
  const Mesh& mesh;
  const Geometry& geometry;
  const SolverInput& input;
  Log& log;
  const StartState* start;  ///< null unless input.start
};

/// A discretisation of the case's equations. The runtime owns the iteration
/// loop, convergence, signals and outputs; an implementation owns its state
/// arrays (host or device), its kernels and their schedule. Constructed from
/// a PhysicsContext that outlives it; the state starts at the cell zones'
/// `initial` values, or at `context.start` when the run starts from another
/// (docs/contracts/runtime.md).
class Physics {
 public:
  virtual ~Physics() = default;
  /// Advances one pseudo-time iteration (steady) or one time step (unsteady).
  virtual StepReport step(std::int64_t iteration) = 0;
  /// Adds the current state to the time average with weight `weight` (the
  /// step's time step), for unsteady runs with average_from.
  virtual void accumulate_average(double weight) = 0;
  /// Cell-centred fields with the SIDS names of results.md (field_names()),
  /// current state or time average.
  [[nodiscard]] virtual std::vector<io::Field> fields(bool mean) const = 0;
  /// A constant flux scale per equation of residual_names(input), in that
  /// order, finite and >= 0; empty for none. Called once, after
  /// construction: in fp32+acc64, an equation converges at a residual of
  /// 0.25 * 2^-23 times its scale at the latest (docs/contracts/runtime.md).
  [[nodiscard]] virtual std::vector<double> residual_scales() const = 0;
  /// Cell-centred state beyond fields(false) that a `continue` needs to go
  /// on bit for bit (earlier time levels, averaging sums); empty when none.
  [[nodiscard]] virtual std::vector<io::Field> restart_fields() const = 0;
  /// Pressure and WallShearStressX/Y/Z on `wall_faces` (ascending).
  [[nodiscard]] virtual std::vector<io::Field> wall_fields(
      std::span<const index_t> wall_faces) const = 0;
  /// Loads on the boundary faces of `zones` (indices into Mesh::face_zones()).
  [[nodiscard]] virtual Loads loads(std::span<const index_t> zones,
                                    const std::array<double, 3>& moment_center) const = 0;
};

/// What a physics implementation can run: each member matches the input's
/// value, or "*" any value.
struct PhysicsKey {
  std::string kind;        ///< physics.kind
  std::string turbulence;  ///< physics.turbulence.model
  std::string precision;   ///< precision
  std::string device;      ///< devices.kind
};

using PhysicsFactory = std::function<std::unique_ptr<Physics>(const PhysicsContext&)>;

/// Why a physics whose key matches an input cannot run it: the runtime reports it as E0706,
/// before any directory exists.
struct PhysicsRefusal {
  std::string message;
  std::string path;              ///< document path of the offending key; also `data.path`
  std::string data_json = "{}";  ///< machine-readable details: the text of a JSON object
};

/// What a physics supports beyond its key (media, zones, conditions): nullopt when it can run
/// the input. A scope must not throw: it reports what it refuses as a PhysicsRefusal.
using PhysicsScope = std::function<std::optional<PhysicsRefusal>(const SolverInput&)>;

/// The physics a build of autan-run provides; the first matching key wins.
class PhysicsRegistry {
 public:
  struct Entry {
    PhysicsFactory factory;
    PhysicsScope scope;  ///< empty: every input the key matches
  };
  void add(PhysicsKey key, PhysicsFactory factory, PhysicsScope scope = {});
  /// The entry for `input`, or null when this build has none (E0706).
  [[nodiscard]] const Entry* find(const SolverInput& input) const;

 private:
  std::vector<std::pair<PhysicsKey, Entry>> entries_;
};

}  // namespace autan::runtime
