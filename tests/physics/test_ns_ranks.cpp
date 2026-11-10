// The Navier-Stokes physics on the ranks of this process's MPI world, under the global keys of
// reproducibility "ranks": at ghost depth 3, its ghosts' fit rows received from their owners,
// against depth 4, every rank fitting them itself (#651, G-fit v2 on design (4): P0, B1, B5).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <mpi.h>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/ghost_stencil.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/partition/cell_rows.hpp"
#include "autan/partition/halo_exchange.hpp"
#include "autan/partition/partition.hpp"
#include "autan/physics/block_matrix.hpp"
#include "autan/physics/euler.hpp"
#include "autan/physics/manufactured.hpp"
#include "autan/physics/navier_stokes.hpp"
#include "fixtures.hpp"
#include "navier_stokes_cpu.hpp"
#include "stretched_slab.hpp"

namespace {

namespace fs = std::filesystem;
using autan::index_t;
using autan::physics::BoundaryKind;
using autan::physics::Flux;
using autan::physics::Gas;
using autan::physics::Scheme;

const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
const std::array<const char*, 4> meshes = {"tet-gmsh/tet-gmsh.ascii.msh", "hex/hex.ascii.msh",
                                           "poly/polyMesh", "hybrid/hybrid.ascii.msh"};
constexpr Gas air{1.4, 287.0};

int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

int world_size() {
  int n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

// Whether `local` holds on every rank.
bool everywhere(bool local) {
  int in = local ? 1 : 0, out = 0;
  MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return out == 1;
}

long long total(long long local) {
  long long out = 0;
  MPI_Allreduce(&local, &out, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  return out;
}

std::array<double, 5> conserved(double rho, std::array<double, 3> v, double p) {
  const double q2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  return {rho, rho * v[0], rho * v[1], rho * v[2], p / (air.gamma - 1.0) + 0.5 * rho * q2};
}

Scheme scheme() {
  Scheme s;
  s.flux = Flux::roe;
  s.mach_ref = 0.5;
  s.scale = {1.0, 100.0, 100.0, 100.0, 1e5};
  s.reference = conserved(1.1, {10.0, 5.0, 0.0}, 0.95e5);
  return s;
}

// One call of forward_rows as a rank saw it (B5).
struct RowsCall {
  int stride;
  long long selected;  // ghost slots this rank asked for
  long long filled;    // rows it received non-empty
  bool operator==(const RowsCall&) const = default;
};

// The partition's halo as the physics' ghost fill: states forward, fit rows through
// partition's forward_rows. Every call of forward_rows is recorded.
struct Halo final : autan::physics::GhostFill {
  autan::HaloExchange exchange;
  const autan::HaloPattern& pattern;
  std::vector<RowsCall> calls;
  std::vector<autan::CellRows> received;

  explicit Halo(const autan::LocalPartition& part)
      : exchange(part.halo, part.mesh.n_owned_cells(), MPI_COMM_WORLD, autan::MemorySpace::host),
        pattern(part.halo) {}
  void forward(std::span<float* const> components) override { exchange.forward(components); }
  void forward(std::span<double* const> components) override { exchange.forward(components); }
  autan::CellRows forward_rows(const autan::CellRows& owned,
                               std::span<const std::uint8_t> wanted) override {
    autan::CellRows got = autan::forward_rows(pattern, owned, wanted, MPI_COMM_WORLD);
    RowsCall call{owned.stride, 0, 0};
    for (const auto w : wanted) call.selected += w;
    for (std::size_t g = 0; g + 1 < got.offsets.size(); ++g) {
      call.filled += got.offsets[g + 1] > got.offsets[g] ? 1 : 0;
    }
    calls.push_back(call);
    received.push_back(got);
    return got;
  }
};

// A part of `global` at ghost depth `ghosts`, its ghost geometry from the owners.
struct Part {
  autan::LocalPartition p;
  std::unique_ptr<Halo> halo;
  autan::kernels::GlobalKeys keys;

  Part(const autan::Mesh& global, const autan::Geometry& gg, const std::vector<std::int32_t>& parts,
       autan::GhostStencil ghosts)
      : p(autan::extract_partition(global, gg, parts, world_size(), world_rank(), ghosts)) {
    halo = std::make_unique<Halo>(p);
    autan::exchange_ghost_geometry(halo->exchange, p.mesh, p.geometry);
    keys = {.cell = p.mesh.cell_global_id(),
            .face = p.face_global_id,
            .reversed = p.face_reversed,
            .shift = p.mesh.ghost_translation()};
  }
};

// The boundary of a case: every face of `wall_zone` (all zones when empty) a no-slip wall at
// t_wall (adiabatic at 0), the others slip.
struct Walls {
  std::string zone;
  double t_wall;
};

// What a run of one part gives, keyed by global ids.
struct Result {
  std::map<std::int64_t, std::array<std::uint64_t, 5>> residual;  // owned cells
  // Placed blocks of owned rows: (row, column) global cell ids.
  std::map<std::pair<std::int64_t, std::int64_t>, std::array<std::uint64_t, 25>> blocks;
  // Fitted coefficients (7 x 15 per cell) of owned cells and fitted ghost face neighbours.
  std::map<std::int64_t, std::vector<std::uint64_t>> fits;
  // Owned cells' boundary-fit rows: (entry key, 19 weights) in stencil order. A cell entry's
  // key is its global id, a boundary face's -2 - its global face id.
  std::map<std::int64_t, std::vector<std::pair<std::int64_t, std::vector<std::uint64_t>>>> rows;
  std::map<std::int64_t, std::array<std::uint64_t, 4>> geometry;  // owned: volume, centroid
  std::set<std::int64_t> fitted_near_ghosts;
};

template <class T>
std::uint64_t bits(T v) {
  if constexpr (sizeof(T) == 8) {
    std::uint64_t b = 0;
    std::memcpy(&b, &v, 8);
    return b;
  } else {
    std::uint32_t b = 0;
    std::memcpy(&b, &v, 4);
    return b;
  }
}

template <class P>
using NS = autan::physics::NavierStokes<P>;

template <class P>
std::vector<std::pair<std::int64_t, std::vector<std::uint64_t>>> row_of(const NS<P>& ns,
                                                                        const Part& part,
                                                                        std::size_t c) {
  const auto& d = ns.data();
  const auto& m = part.p.mesh;
  const auto nst = static_cast<std::size_t>(d.topo.n_stencil);
  const std::array q{d.q0, d.q1, d.q2, d.q3, d.q4, d.q5, d.q6};
  std::vector<std::pair<std::int64_t, std::vector<std::uint64_t>>> row;
  for (auto k = d.topo.stencil_offsets[c]; k < d.topo.stencil_offsets[c + 1]; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    const index_t e = d.topo.stencil[uk];
    const std::int64_t key = e >= 0 ? m.cell_global_id()[static_cast<std::size_t>(e)]
                                    : -2 - part.p.face_global_id[static_cast<std::size_t>(-e - 1)];
    std::vector<std::uint64_t> w(19);
    for (std::size_t i = 0; i < 19; ++i) w[i] = bits(q[i / 3][(i % 3) * nst + uk]);
    row.emplace_back(key, std::move(w));
  }
  return row;
}

template <class P>
Result run(Part& part, const Walls& walls, const autan::physics::Manufactured& mms,
           std::unique_ptr<NS<P>>& keep) {
  const auto& m = part.p.mesh;
  auto ns = std::make_unique<NS<P>>(m, part.p.geometry, air, scheme(), autan::physics::Transport{},
                                    &part.keys, part.p.ghosts);
  ns->set_ghosts(part.halo.get());
  const auto nb = static_cast<std::size_t>(m.n_boundary_faces());
  const auto ni = static_cast<std::size_t>(m.n_interior_faces());
  std::vector<BoundaryKind> kind(nb, BoundaryKind::slip);
  std::vector<double> data(5 * nb, 0.0);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto zone = static_cast<std::size_t>(m.face_zone()[ni + b]);
    if (walls.zone.empty() || m.face_zones()[zone].name == walls.zone) {
      kind[b] = BoundaryKind::wall;
      data[b] = walls.t_wall;
    }
  }
  ns->set_boundary(kind, data);
  const auto nc = static_cast<std::size_t>(m.n_cells());
  const auto& g = part.p.geometry;
  std::vector<std::array<double, 3>> points(nc);
  for (std::size_t c = 0; c < nc; ++c) {
    points[c] = {g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]};
  }
  const std::vector<std::string> names{"rho", "rhou", "rhov", "rhow", "rhoE"};
  const auto values = mms.evaluate("conserved", names, points);
  std::vector<double> u(5 * nc);
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t c = 0; c < nc; ++c) u[k * nc + c] = values[k][c];
  }
  std::vector<typename P::state> s(5 * nc);
  ns->state(u, s);
  std::vector<typename P::residual> res(5 * nc);
  ns->residual_and_jacobian(s, res);
  autan::physics::BlockMatrix a(m);
  const std::vector<double> shift(nc, 0.0);
  ns->assemble(shift, a);

  Result out;
  const auto owned = static_cast<std::size_t>(m.n_owned_cells());
  const auto gid = [&](std::size_t c) { return m.cell_global_id()[c]; };
  for (std::size_t c = 0; c < owned; ++c) {
    auto& r = out.residual[gid(c)];
    for (std::size_t k = 0; k < 5; ++k) r[k] = bits(res[k * nc + c]);
    auto& b = out.blocks[{gid(c), gid(c)}];
    for (std::size_t e = 0; e < 25; ++e) b[e] = bits(a.diagonal(c)[e]);
    out.geometry[gid(c)] = {bits(g.cell_volume[c]), bits(g.cell_centroid_x[c]),
                            bits(g.cell_centroid_y[c]), bits(g.cell_centroid_z[c])};
    out.rows[gid(c)] = row_of(*ns, part, c);
  }
  for (std::size_t f = 0; f < ni; ++f) {
    const auto o = static_cast<std::size_t>(m.owner()[f]);
    const auto n = static_cast<std::size_t>(m.neighbour()[f]);
    // upper(f): the owner's row, the neighbour's column; lower(f): the reverse.
    if (o < owned) {
      auto& b = out.blocks[{gid(o), gid(n)}];
      for (std::size_t e = 0; e < 25; ++e) b[e] = bits(a.upper(f)[e]);
    }
    if (n < owned) {
      auto& b = out.blocks[{gid(n), gid(o)}];
      for (std::size_t e = 0; e < 25; ++e) b[e] = bits(a.lower(f)[e]);
    }
  }
  const auto& d = ns->data();
  const std::array fit{d.fit0, d.fit1, d.fit2, d.fit3, d.fit4, d.fit5, d.fit6};
  std::vector<char> near(nc, 0);
  std::fill(near.begin(), near.begin() + static_cast<std::ptrdiff_t>(owned), 1);
  for (std::size_t f = 0; f < ni; ++f) {
    const auto o = static_cast<std::size_t>(m.owner()[f]);
    const auto n = static_cast<std::size_t>(m.neighbour()[f]);
    if (o < owned) near[n] = 1;
    if (n < owned) near[o] = 1;
  }
  for (std::size_t c = 0; c < nc; ++c) {
    if (near[c] == 0 || !(static_cast<double>(d.quad[c]) > 0.5)) continue;
    if (c >= owned) out.fitted_near_ghosts.insert(gid(c));
    auto& v = out.fits[gid(c)];
    for (const auto* b : fit) {
      for (std::size_t j = 0; j < 15; ++j) v.push_back(bits(b[j * nc + c]));
    }
  }
  keep = std::move(ns);
  return out;
}

const autan::physics::Manufactured& manufactured() {
  static const autan::physics::Manufactured mms = [] {
    std::ifstream in(AUTAN_TEST_MMS);
    std::stringstream text;
    text << in.rdbuf();
    return autan::physics::Manufactured(text.str());
  }();
  return mms;
}

}  // namespace

TEMPLATE_TEST_CASE("Navier-Stokes at ghost depth 3 gives the bits of depth 4 under global keys",
                   "[navier-stokes][g-fit]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // G-fit v2, design (4) (651): B1 with its precondition P0. On the same parts and the same
  // prescribed state (the manufactured solution's conserved fields at the cell centroids), the
  // physics with vertex3 ghosts, whose ghost face neighbours' fit rows arrive from their owners,
  // and with vertex4 ghosts, fitting them itself, give the same owned residuals, placed blocks of
  // owned rows and fitted coefficients, bit for bit.
  using P = TestType;
  if (world_size() < 2) SKIP("a partition needs two ranks");
  const int which = GENERATE(range(0, static_cast<int>(meshes.size()) + 1));
  const bool slab = which == static_cast<int>(meshes.size());
  const double t_wall = GENERATE(300.0, 0.0);  // isothermal, adiabatic
  if (slab && t_wall == 0.0) return;           // the slab's wall is isothermal only
  const Walls walls{slab ? "ymin" : "", t_wall};
  INFO((slab ? "stretched slab, wall ymin" : meshes[static_cast<std::size_t>(which)])
       << ", wall at " << t_wall << " K on " << world_size() << " ranks");
  const autan::Mesh global =
      slab ? autan::test::stretched_slab()
           : autan::io::read_mesh(mesh_dir / meshes[static_cast<std::size_t>(which)]);
  const autan::Geometry gg = autan::compute_geometry(global);
  // The slab's first column of cells, whose fits take three rings, is a part of its own: the next
  // part receives those rows. The rest of the slab is split across x; the meshes by METIS.
  std::vector<std::int32_t> parts;
  if (slab) {
    std::vector<double> columns(gg.cell_centroid_x.begin(), gg.cell_centroid_x.end());
    std::ranges::sort(columns);
    columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
    const double first = columns.front(), second = columns.at(1), hi = columns.back();
    const int rest = world_size() - 1;
    for (const double x : gg.cell_centroid_x) {
      if (x == first) {
        parts.push_back(0);
        continue;
      }
      const auto k = static_cast<std::int32_t>((x - second) / (hi - second) * rest);
      parts.push_back(1 + std::min(k, rest - 1));
    }
  } else {
    parts = autan::partition_cells(global, world_size());
  }
  Part d3(global, gg, parts, autan::GhostStencil::vertex3);
  Part d4(global, gg, parts, autan::GhostStencil::vertex4);
  std::unique_ptr<NS<P>> ns3, ns4;
  const Result a = run<P>(d3, walls, manufactured(), ns3);
  const Result b = run<P>(d4, walls, manufactured(), ns4);

  // P0, part 1: on the depth-4 path, the row each rank computes for a fitted ghost face
  // neighbour is its owner's row, bit for bit: the owners' rows forwarded against them.
  {
    const auto& m = d4.p.mesh;
    const auto owned = static_cast<std::size_t>(m.n_owned_cells());
    const auto nc = static_cast<std::size_t>(m.n_cells());
    autan::CellRows rows;
    rows.stride = 19;
    for (std::size_t c = 0; c < owned; ++c) {
      for (const auto& [key, w] : row_of(*ns4, d4, c)) {
        rows.keys.push_back(key);
        for (const auto x : w) {
          double v = 0.0;
          if constexpr (sizeof(typename P::grad) == 8) {
            std::memcpy(&v, &x, 8);
          } else {
            float f = 0.0F;
            const auto lo = static_cast<std::uint32_t>(x);
            std::memcpy(&f, &lo, 4);
            v = f;
          }
          rows.values.push_back(v);
        }
      }
      rows.offsets.push_back(static_cast<autan::offset_t>(rows.keys.size()));
    }
    std::vector<std::uint8_t> wanted(nc - owned, 0);
    for (std::size_t g = owned; g < nc; ++g) {
      wanted[g - owned] = b.fitted_near_ghosts.contains(m.cell_global_id()[g]) ? 1 : 0;
    }
    const auto got = autan::forward_rows(d4.p.halo, rows, wanted, MPI_COMM_WORLD);
    long long compared = 0;
    for (std::size_t g = owned; g < nc; ++g) {
      if (wanted[g - owned] == 0) continue;
      const auto local = row_of(*ns4, d4, g);
      const auto begin = static_cast<std::size_t>(got.offsets[g - owned]);
      const auto end = static_cast<std::size_t>(got.offsets[g - owned + 1]);
      INFO("ghost " << m.cell_global_id()[g]);
      REQUIRE(end - begin == local.size());
      for (std::size_t e = begin; e < end; ++e) {
        CHECK(got.keys[e] == local[e - begin].first);
        for (std::size_t i = 0; i < 19; ++i) {
          const double v = got.values[19 * e + i];
          const auto w = sizeof(typename P::grad) == 8 ? bits(v) : bits(static_cast<float>(v));
          CHECK(w == local[e - begin].second[i]);
        }
      }
      ++compared;
    }
    CHECK(total(compared) > 0);
  }
  // P0, part 2: owned rows, geometry and membership the same on both paths.
  CHECK(a.geometry == b.geometry);
  CHECK(a.rows == b.rows);
  CHECK(a.fitted_near_ghosts == b.fitted_near_ghosts);
  // A fit over a third vertex ring at a part boundary: a fitted ghost face neighbour whose row
  // reaches beyond two rings of it (the stretched slab's ends), which depth 3 receives.
  if (slab) {
    const auto vertex = autan::kernels::cell_stencil(
        global, autan::kernels::pack_geometry(global, gg), autan::kernels::StencilKind::vertex);
    long long three_rings = 0;
    const auto& m = d4.p.mesh;
    for (auto g = static_cast<std::size_t>(m.n_owned_cells());
         g < static_cast<std::size_t>(m.n_cells()); ++g) {
      const auto id = m.cell_global_id()[g];
      if (!b.fitted_near_ghosts.contains(id)) continue;
      std::set<std::int64_t> near{id};
      for (int ring = 0; ring < 2; ++ring) {
        const std::vector<std::int64_t> reached(near.begin(), near.end());
        for (const auto c : reached) {
          const auto uc = static_cast<std::size_t>(c);
          for (auto k = vertex.offsets[uc]; k < vertex.offsets[uc + 1]; ++k) {
            const auto e = vertex.entries[static_cast<std::size_t>(k)];
            if (e >= 0) near.insert(e);
          }
        }
      }
      bool beyond = false;
      for (const auto& [key, w] : row_of(*ns4, d4, g))
        beyond = beyond || (key >= 0 && !near.contains(key));
      three_rings += beyond ? 1 : 0;
    }
    CHECK(total(three_rings) > 0);
  }
  // A ghost face neighbour owning an adiabatic wall face: its row, and that face's part in it,
  // come from another rank.
  if (!slab && t_wall == 0.0) {
    std::set<std::int64_t> walled;
    for (index_t f = global.n_interior_faces(); f < global.n_faces(); ++f) {
      walled.insert(global.owner()[static_cast<std::size_t>(f)]);
    }
    long long remote = 0;
    for (const auto id : a.fitted_near_ghosts) remote += walled.contains(id) ? 1 : 0;
    CHECK(total(remote) > 0);
  }
  // B1.
  CHECK(a.residual == b.residual);
  CHECK(a.blocks == b.blocks);
  CHECK(a.fits == b.fits);
  CHECK(everywhere(a.residual == b.residual && a.blocks == b.blocks && a.fits == b.fits));
}

TEST_CASE("every rank takes part in each fit-row exchange, one without ghosts too",
          "[navier-stokes][g-fit]") {
  // G-fit v2, B5. Two boxes apart, 64 and 125 hexahedra, on 3 ranks: the partition gives the
  // small one a rank of its own, without ghosts. At ghost depth 3 every rank calls forward_rows
  // the same number of times with the same stride, with keys and without; the ghostless rank
  // asks for nothing and gets nothing. A second set_boundary with other conditions exchanges again
  // and changes the rows received.
  using P = autan::kernels::fp64;
  if (world_size() != 3) SKIP("this case runs on 3 ranks");
  autan::CellwiseMesh cells = autan::test::box(4, autan::test::BoxFill::hexa);
  {
    const autan::CellwiseMesh b = autan::test::box(5, autan::test::BoxFill::hexa);
    const auto nodes = static_cast<index_t>(cells.x.size());
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
  const autan::Mesh global = autan::assemble(autan::to_face_based(std::move(cells)));
  const autan::Geometry gg = autan::compute_geometry(global);
  const auto parts = autan::partition_cells(global, 3);
  Part part(global, gg, parts, autan::GhostStencil::vertex3);
  const bool ghostless = part.p.mesh.n_cells() == part.p.mesh.n_owned_cells();
  REQUIRE(total(ghostless ? 1 : 0) == 1);
  const bool keyed = GENERATE(true, false);
  CAPTURE(keyed);
  auto ns = std::make_unique<NS<P>>(part.p.mesh, part.p.geometry, air, scheme(),
                                    autan::physics::Transport{}, keyed ? &part.keys : nullptr,
                                    part.p.ghosts);
  ns->set_ghosts(part.halo.get());
  const auto& m = part.p.mesh;
  const auto nb = static_cast<std::size_t>(m.n_boundary_faces());
  std::vector<BoundaryKind> kind(nb, BoundaryKind::wall);
  std::vector<double> data(5 * nb, 0.0);
  for (std::size_t b = 0; b < nb; ++b) data[b] = 300.0;
  ns->set_boundary(kind, data);
  // Half the walls slip: other cells are fitted, other rows travel.
  const auto ni = static_cast<std::size_t>(m.n_interior_faces());
  for (std::size_t b = 0; b < nb; ++b) {
    const auto zone = static_cast<std::size_t>(m.face_zone()[ni + b]);
    const auto& name = m.face_zones()[zone].name;
    if (name == "xmin" || name == "ymin" || name == "zmin") kind[b] = BoundaryKind::slip;
  }
  ns->set_boundary(kind, data);
  const auto& calls = part.halo->calls;
  REQUIRE(calls.size() == 2);
  // The same call sequence and stride on every rank.
  const auto n_calls = static_cast<long long>(calls.size());
  CHECK(total(n_calls) == 3 * n_calls);
  for (const auto& call : calls) {
    CHECK(call.stride == 19);
    CHECK(total(call.stride) == 3 * 19);
    if (ghostless) {
      CHECK(call.selected == 0);
      CHECK(call.filled == 0);
    }
    CHECK(total(call.filled) > 0);
  }
  // The second exchange consumed other rows.
  const auto& r = part.halo->received;
  const bool changed =
      r[0].keys != r[1].keys || r[0].values != r[1].values || r[0].offsets != r[1].offsets;
  CHECK(total(changed ? 1 : 0) > 0);

  // One evaluation: residual, assembled blocks and the fit Jacobian's norms, as bits.
  struct Evaluation {
    std::vector<std::uint64_t> residual, blocks;
    std::pair<double, double> norms;
  };
  const auto& mms = manufactured();
  const std::vector<std::string> names{"rho", "rhou", "rhov", "rhow", "rhoE"};
  const auto evaluate = [&](NS<P>& flow, const autan::Mesh& mesh, const autan::Geometry& g) {
    const auto nc = static_cast<std::size_t>(mesh.n_cells());
    std::vector<std::array<double, 3>> points(nc);
    for (std::size_t c = 0; c < nc; ++c) {
      points[c] = {g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]};
    }
    const auto values = mms.evaluate("conserved", names, points);
    std::vector<double> u(5 * nc);
    for (std::size_t k = 0; k < 5; ++k) {
      for (std::size_t c = 0; c < nc; ++c) u[k * nc + c] = values[k][c];
    }
    std::vector<typename P::state> s(5 * nc);
    flow.state(u, s);
    std::vector<typename P::residual> res(5 * nc);
    flow.residual_and_jacobian(s, res);
    autan::physics::BlockMatrix a(mesh);
    flow.assemble(std::vector<double>(nc, 0.0), a);
    Evaluation out;
    for (const auto v : res) out.residual.push_back(bits(v));
    for (const auto span : {a.diagonals(), a.uppers(), a.lowers()}) {
      for (const double v : span) out.blocks.push_back(bits(v));
    }
    out.norms = flow.fit_jacobian_norms();
    return out;
  };
  // The second set_boundary's rows are the ones the physics uses: its evaluation has the bits of a
  // physics given those conditions first.
  const Evaluation refreshed = evaluate(*ns, m, part.p.geometry);
  {
    NS<P> fresh(part.p.mesh, part.p.geometry, air, scheme(), autan::physics::Transport{},
                keyed ? &part.keys : nullptr, part.p.ghosts);
    fresh.set_ghosts(part.halo.get());
    fresh.set_boundary(kind, data);
    const Evaluation first = evaluate(fresh, m, part.p.geometry);
    CHECK(everywhere(refreshed.residual == first.residual && refreshed.blocks == first.blocks));
  }

  // The fit Jacobian's norms count the owned rows only: under the global keys, where every block
  // has the bits of one rank's (B1, B3), the ranks' squares sum to the whole mesh's, up to the
  // order of the sums. Every entry of a received row is held here, so none is left out of the
  // outside norm.
  if (!keyed) return;
  {
    std::set<std::int64_t> held(m.cell_global_id().begin(), m.cell_global_id().end());
    const auto& last = r[1];
    bool all_held = true;
    for (const auto key : last.keys) all_held = all_held && (key < 0 || held.contains(key));
    REQUIRE(everywhere(all_held));
  }
  const auto [placed, outside] = refreshed.norms;
  const auto one = autan::extract_partition(
      global, gg, std::vector<std::int32_t>(gg.cell_volume.size(), 0), 1, 0);
  const autan::kernels::GlobalKeys one_keys{.cell = one.mesh.cell_global_id(),
                                            .face = one.face_global_id,
                                            .reversed = one.face_reversed,
                                            .shift = one.mesh.ghost_translation()};
  NS<P> whole(one.mesh, one.geometry, air, scheme(), autan::physics::Transport{}, &one_keys);
  const auto gnb = static_cast<std::size_t>(one.mesh.n_boundary_faces());
  const auto gni = static_cast<std::size_t>(one.mesh.n_interior_faces());
  std::vector<BoundaryKind> gkind(gnb, BoundaryKind::wall);
  std::vector<double> gdata(5 * gnb, 0.0);
  for (std::size_t b = 0; b < gnb; ++b) {
    gdata[b] = 300.0;
    const auto& name =
        one.mesh.face_zones()[static_cast<std::size_t>(one.mesh.face_zone()[gni + b])].name;
    if (name == "xmin" || name == "ymin" || name == "zmin") gkind[b] = BoundaryKind::slip;
  }
  whole.set_boundary(gkind, gdata);
  const auto [g_placed, g_outside] = evaluate(whole, one.mesh, one.geometry).norms;
  // Two sums of the same N nonnegative terms in other orders, each within gamma(N) = N u /
  // (1 - N u) of the exact sum (u = 2^-53, every addition rounded to nearest), the ranks' three
  // more: N bounds the terms by two rows per side of every face, each side's stencil entries.
  const auto& t = whole.data().topo;
  std::size_t widest = 0;
  for (std::size_t c = 0; c + 1 < static_cast<std::size_t>(global.n_cells()) + 1; ++c) {
    widest =
        std::max(widest, static_cast<std::size_t>(t.stencil_offsets[c + 1] - t.stencil_offsets[c]));
  }
  const double n_terms =
      4.0 * static_cast<double>(global.n_faces()) * static_cast<double>(widest + 1) + 3.0;
  const double nu = n_terms * 0x1p-53;
  const double gamma = nu / (1.0 - nu);
  double sums[2] = {placed * placed, outside * outside};
  double ranks_sum[2] = {0.0, 0.0};
  MPI_Allreduce(sums, ranks_sum, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double whole_sum[2] = {g_placed * g_placed, g_outside * g_outside};
  for (int k = 0; k < 2; ++k) {
    INFO((k == 0 ? "placed" : "outside")
         << ": ranks " << ranks_sum[k] << ", one rank " << whole_sum[k] << ", gamma " << gamma);
    CHECK(std::abs(ranks_sum[k] - whole_sum[k]) <= 2.0 * gamma * (ranks_sum[k] + whole_sum[k]));
  }
  // Neither norm is zero: the check compares two definite sums.
  CHECK(whole_sum[0] > 0.0);
  CHECK(whole_sum[1] > 0.0);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Catch::Session session;
  int result = session.applyCommandLine(argc, argv);
  // Every rank must run the same tests in the same order: their exchanges pair up.
  session.configData().runOrder = Catch::TestRunOrder::Declared;
  if (result == 0) result = session.run();
  int worst = 0;
  MPI_Allreduce(&result, &worst, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return worst;
}
