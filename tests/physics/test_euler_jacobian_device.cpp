// The implicit step's matrix assembled on a CUDA device (DeviceStepper::assemble) against the
// host's (Euler<P>::assemble into a BlockMatrix), block by block.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "autan/core/cellwise.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/physics/block_matrix.hpp"
#include "autan/physics/euler.hpp"
#include "autan/physics/euler_device.hpp"
#include "autan/runtime/ranks.hpp"
#include "fixtures.hpp"
#include "stretched_slab.hpp"

namespace {

namespace fs = std::filesystem;
using autan::physics::BlockMatrix;
using autan::physics::BoundaryKind;
using autan::physics::Euler;
using autan::physics::Flux;
using autan::physics::Gas;
using autan::physics::Scheme;
using autan::test::BoxFill;

const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
constexpr Gas air{1.4, 287.0};

class NoGhosts final : public autan::runtime::GhostExchange {
 public:
  void forward(std::span<float* const> /*c*/, autan::MemorySpace /*s*/, void* /*stream*/) override {
  }
  void forward(std::span<double* const> /*c*/, autan::MemorySpace /*s*/,
               void* /*stream*/) override {}
};

// A smooth subsonic flow over the unit cube: rho, u, v, w, p at (x, y, z).
std::array<double, 5> flow(double x, double y, double z) {
  return {1.2 + 0.1 * std::sin(2.0 * x + y), 80.0 + 15.0 * std::cos(x + 2.0 * z),
          -20.0 + 10.0 * std::sin(y + z), 5.0 * std::cos(3.0 * x),
          1.0e5 + 4.0e3 * std::sin(x + y + z)};
}

std::array<double, 5> conserved(const std::array<double, 5>& w) {
  const double q2 = w[1] * w[1] + w[2] * w[2] + w[3] * w[3];
  return {w[0], w[0] * w[1], w[0] * w[2], w[0] * w[3], w[4] / (air.gamma - 1.0) + 0.5 * w[0] * q2};
}

// The test meshes as read, and boxes whose interior nodes move by 20 % of the spacing.
using Source = std::variant<const char*, BoxFill>;
autan::Mesh mesh_of(const Source& s) {
  if (const auto* file = std::get_if<const char*>(&s))
    return autan::io::read_mesh(mesh_dir / *file);
  return autan::assemble(autan::to_face_based(autan::test::box(4, std::get<BoxFill>(s), 0.2)));
}

std::string name_of(const Source& s) {
  if (const auto* file = std::get_if<const char*>(&s)) return *file;
  return "perturbed " + autan::test::to_string(std::get<BoxFill>(s));
}

std::int64_t bits(double v) { return std::bit_cast<std::int64_t>(v); }

}  // namespace

TEMPLATE_TEST_CASE("the device assembles the host's implicit matrix bit for bit",
                   "[euler][jacobian][gpu]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // #218: under the mesh's own numbering as global keys both sides compute the face blocks with
  // the same operations in the same order (#586, criterion 1), and the device assembles them in
  // Euler<P>::assemble's order. On state, slip, outlet and wall faces every diagonal block has the
  // host's bits, and every off-diagonal block the host's value rounded once to P::jac. Farfield
  // and inlet faces call exp and log, whose last bits differ between host and device: their
  // blocks are assembled but not verified here (#218, PR 2b).
  using P = TestType;
  using J = typename P::jac;
  const Source source =
      GENERATE(Source{"tet-gmsh/tet-gmsh.ascii.msh"}, Source{"hex/hex.ascii.msh"},
               Source{"poly/polyMesh"}, Source{"hybrid/hybrid.ascii.msh"}, Source{BoxFill::tetra},
               Source{BoxFill::hexa}, Source{BoxFill::prism}, Source{BoxFill::hybrid});
  const auto flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  const bool limiter = GENERATE(false, true);
  const bool second_order = GENERATE(false, true);
  CAPTURE(name_of(source), static_cast<int>(flux), limiter, second_order);
  const auto mesh = mesh_of(source);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  std::vector<std::int64_t> face_id(ni + nb);
  std::iota(face_id.begin(), face_id.end(), std::int64_t{0});
  const std::vector<std::uint8_t> not_reversed(ni + nb, 0);
  const autan::kernels::GlobalKeys keys{
      .cell = mesh.cell_global_id(), .face = face_id, .reversed = not_reversed, .shift = {}};

  const auto ref = flow(0.5, 0.5, 0.5);
  Scheme s;
  s.flux = flux;
  s.limiter = limiter;
  const double c_ref = std::sqrt(air.gamma * ref[4] / ref[0]);
  s.mach_ref = std::hypot(ref[1], ref[2], ref[3]) / c_ref;
  s.scale = {ref[0], 100.0, 100.0, 100.0, ref[4]};
  s.reference = conserved(ref);
  s.venkat_k = 0.1;
  Euler<P> host(mesh, g, air, s, &keys);
  // Outlet on x, slip walls on y, the flow's state on zmin, an adiabatic wall on zmax.
  std::vector<BoundaryKind> kind(nb);
  std::vector<double> data(5 * nb, 0.0);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto f = ni + b;
    const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
    std::array<double, 5> w = ref;
    if (zone[0] == 'x') {
      kind[b] = BoundaryKind::outlet;
    } else if (zone[0] == 'y') {
      kind[b] = BoundaryKind::slip;
    } else if (zone == "zmax") {
      kind[b] = BoundaryKind::wall;
      w = {0.0, 0.0, 0.0, 0.0, 0.0};
    } else {
      kind[b] = BoundaryKind::state;
      w = flow(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
    }
    for (std::size_t k = 0; k < 5; ++k) data[k * nb + b] = w[k];
  }
  host.set_boundary(kind, data);
  host.set_second_order(second_order);
  std::vector<double> u(5 * n), shift(n);
  for (std::size_t c = 0; c < n; ++c) {
    const auto w =
        conserved(flow(g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]));
    for (std::size_t k = 0; k < 5; ++k) u[k * n + c] = w[k];
    shift[c] = g.cell_volume[c] * (1.0e3 + static_cast<double>(c % 7));
  }
  std::vector<typename P::state> stored(5 * n);
  host.state(u, stored);
  std::vector<typename P::residual> r_host(5 * n);
  host.residual_and_jacobian(stored, r_host);
  BlockMatrix a(mesh);
  host.assemble(shift, a);

  autan::runtime::OneRank rank;
  NoGhosts ghosts;
  auto device = autan::physics::make_device_stepper<P>(mesh, g, host, air.gamma, second_order, rank,
                                                       ghosts, nullptr, &keys);
  device->upload(u);
  device->assemble(shift);
  const auto b = device->jacobian_blocks();

  // Each row holds its cell's faces, interior and boundary, by increasing global face id (here
  // the face index): taken from the mesh, not from the device.
  std::vector<std::vector<autan::index_t>> faces_of(n);
  for (std::size_t f = 0; f < ni + nb; ++f) {
    faces_of[static_cast<std::size_t>(mesh.owner()[f])].push_back(static_cast<autan::index_t>(f));
    if (f < ni) {
      faces_of[static_cast<std::size_t>(mesh.neighbour()[f])].push_back(
          static_cast<autan::index_t>(f));
    }
  }
  REQUIRE(b.diagonal.size() == 25 * n);
  REQUIRE(b.offsets.size() == n + 1);
  REQUIRE(b.faces.size() == 2 * ni + nb);
  REQUIRE(b.off_diagonal.size() == 25 * b.faces.size());
  for (std::size_t c = 0; c < n; ++c) {
    const auto first = static_cast<std::ptrdiff_t>(b.offsets[c]);
    const auto last = static_cast<std::ptrdiff_t>(b.offsets[c + 1]);
    REQUIRE(std::vector<autan::index_t>(b.faces.begin() + first, b.faces.begin() + last) ==
            faces_of[c]);
  }
  std::size_t diag_differ = 0, off_differ = 0;
  for (std::size_t c = 0; c < n; ++c) {
    const auto host_diag = std::span(a.diagonals()).subspan(25 * c, 25);
    for (std::size_t q = 0; q < 25; ++q) {
      REQUIRE(std::isfinite(host_diag[q]));
      if (bits(b.diagonal[25 * c + q]) != bits(host_diag[q])) ++diag_differ;
    }
    for (auto k = b.offsets[c]; k < b.offsets[c + 1]; ++k) {
      const auto i = static_cast<std::size_t>(k);
      const auto f = static_cast<std::size_t>(b.faces[i]);
      for (std::size_t q = 0; q < 25; ++q) {
        double expected = 0.0;
        if (f < ni) {
          const bool row_owner = static_cast<std::size_t>(mesh.owner()[f]) == c;
          const auto blocks = row_owner ? a.uppers() : a.lowers();
          expected = static_cast<double>(static_cast<J>(blocks[25 * f + q]));
        }
        if (bits(b.off_diagonal[25 * i + q]) != bits(expected)) ++off_differ;
      }
    }
  }
  CHECK(diag_differ == 0);
  CHECK(off_differ == 0);
}

TEST_CASE("the device Jacobian needs the face moments a planar stepper omits",
          "[euler][jacobian][gpu]") {
  using P = autan::kernels::fp64;
  const auto mesh = autan::io::read_mesh(mesh_dir / "hex/hex.ascii.msh");
  const auto g = autan::compute_geometry(mesh);
  const auto ref = flow(0.5, 0.5, 0.5);
  Scheme s;
  s.reference = conserved(ref);
  s.scale = {ref[0], 100.0, 100.0, 100.0, ref[4]};
  Euler<P> host(mesh, g, air, s);
  host.set_planar(true);
  autan::runtime::OneRank rank;
  NoGhosts ghosts;
  auto device = autan::physics::make_device_stepper<P>(mesh, g, host, air.gamma, false, rank,
                                                       ghosts, nullptr);
  const std::vector<double> shift(static_cast<std::size_t>(mesh.n_cells()), 1.0);
  CHECK_THROWS_AS(device->assemble(shift), std::logic_error);
}

TEMPLATE_TEST_CASE("the device boundary gradient tangent has the host's bits",
                   "[euler][jacobian][gpu]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // Euler<P>::gradient_tangent over the boundary faces (the term the steady
  // operator adds to the assembled matrix), the tangent of the kept cells' boundary values
  // (#581) included, against the host under the mesh's own numbering as global keys. The
  // one-layer slab has kept cells; the test meshes have none.
  using P = TestType;
  using autan::physics::GradientFit;
  const std::array<Source, 6> sources = {
      Source{"tet-gmsh/tet-gmsh.ascii.msh"}, Source{"hex/hex.ascii.msh"}, Source{"poly/polyMesh"},
      Source{"hybrid/hybrid.ascii.msh"},     Source{BoxFill::prism},      Source{BoxFill::hexa}};
  const auto which = static_cast<std::size_t>(GENERATE(0, 1, 2, 3, 4, 5));
  const bool slab = which == 5;  // the stretched one-layer slab in place of the hexahedra box
  const Source source = sources[which];
  const auto flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  const bool limiter = GENERATE(false, true);
  const auto fit = GENERATE(GradientFit::linear, GradientFit::quadratic);
  CAPTURE(slab, name_of(source), static_cast<int>(flux), limiter, static_cast<int>(fit));
  const auto mesh = slab ? autan::test::stretched_slab({.first = 0.02}) : mesh_of(source);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  std::vector<std::int64_t> face_id(ni + nb);
  std::iota(face_id.begin(), face_id.end(), std::int64_t{0});
  const std::vector<std::uint8_t> not_reversed(ni + nb, 0);
  const autan::kernels::GlobalKeys keys{
      .cell = mesh.cell_global_id(), .face = face_id, .reversed = not_reversed, .shift = {}};

  const auto ref = flow(0.5, 0.5, 0.5);
  Scheme s;
  s.flux = flux;
  s.limiter = limiter;
  s.gradient_fit = fit;
  const double c_ref = std::sqrt(air.gamma * ref[4] / ref[0]);
  s.mach_ref = std::hypot(ref[1], ref[2], ref[3]) / c_ref;
  s.scale = {ref[0], 100.0, 100.0, 100.0, ref[4]};
  s.reference = conserved(ref);
  s.venkat_k = 0.1;
  Euler<P> host(mesh, g, air, s, &keys);
  // Outlet on x, an adiabatic wall on ymin (the slab) or zmax, slip on y and on the slab's sides,
  // the flow's state elsewhere.
  std::vector<BoundaryKind> kind(nb);
  std::vector<double> data(5 * nb, 0.0);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto f = ni + b;
    const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
    std::array<double, 5> w = ref;
    if (zone[0] == 'x') {
      kind[b] = BoundaryKind::outlet;
    } else if ((slab && zone == "ymin") || (!slab && zone == "zmax")) {
      kind[b] = BoundaryKind::wall;
      w = {0.0, 0.0, 0.0, 0.0, 0.0};
    } else if (zone[0] == 'y' || zone == "sides") {
      kind[b] = BoundaryKind::slip;
    } else {
      kind[b] = BoundaryKind::state;
      w = flow(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
    }
    for (std::size_t k = 0; k < 5; ++k) data[k * nb + b] = w[k];
  }
  host.set_boundary(kind, data);
  host.set_second_order(true);
  if (slab) REQUIRE(!host.kept_cells().empty());
  std::vector<double> u(5 * n), shift(n, 1.0), du(5 * n);
  for (std::size_t c = 0; c < n; ++c) {
    const auto w =
        conserved(flow(g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]));
    for (std::size_t k = 0; k < 5; ++k) {
      u[k * n + c] = w[k];
      du[k * n + c] = 1e-3 * w[k] * std::sin(static_cast<double>(7 * c + k));
    }
  }
  std::vector<typename P::state> stored(5 * n), dv(5 * n);
  host.state(u, stored);
  for (std::size_t i = 0; i < 5 * n; ++i) dv[i] = static_cast<typename P::state>(du[i]);
  std::vector<typename P::residual> r(5 * n), dres_host(5 * n, typename P::residual(0));
  host.residual_and_jacobian(stored, r);
  host.gradient_tangent(stored, dv, dres_host, Euler<P>::Faces::boundary);

  autan::runtime::OneRank rank;
  NoGhosts ghosts;
  auto device = autan::physics::make_device_stepper<P>(mesh, g, host, air.gamma, true, rank, ghosts,
                                                       nullptr, &keys);
  device->upload(u);
  device->assemble(shift);
  std::vector<double> dres_device(5 * n);
  device->boundary_gradient_tangent(du, dres_device);
  // Another state evaluated after the assembly: the tangent is refused, not taken there.
  std::vector<double> r_other(5 * n);
  device->residual(u, r_other);
  CHECK_THROWS_AS(device->boundary_gradient_tangent(du, dres_device), std::logic_error);

  REQUIRE(std::ranges::any_of(dres_host, [](auto v) { return v != 0; }));
  std::size_t differ = 0;
  for (std::size_t i = 0; i < 5 * n; ++i) {
    REQUIRE(std::isfinite(static_cast<double>(dres_host[i])));
    if (bits(dres_device[i]) != bits(static_cast<double>(dres_host[i]))) ++differ;
  }
  CHECK(differ == 0);
}
