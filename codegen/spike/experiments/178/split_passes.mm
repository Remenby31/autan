// Throwaway: GPU time of the store and gather passes of advection store-gather, separately.
#include <algorithm>
#include <cstdio>
#include <vector>

#include "advection_metal.hpp"
#include "autan/core/renumber.hpp"
#include "autan/io/read_mesh.hpp"
#include "metal_device.hpp"
using namespace autan;
namespace mtl = autan::kernels::metal;
namespace av = autan::kernels::advection;
template <class P>
void run(const char* path) {
  auto mesh = io::read_mesh(path);
  mesh = renumber_cells(mesh, rcm_order(mesh));
  spike::SpikeMesh m{std::move(mesh), {}, {}, {}, {}, {}};
  m.geometry = compute_geometry(m.mesh);
  m.packed = kernels::pack_geometry(m.mesh, m.geometry);
  m.topo = kernels::topology(m.mesh.view());
  m.twins = kernels::incidence_twins(m.mesh);
  m.topo.incidence_twin = m.twins.data();
  mtl::Context ctx;
  spike::metal::DeviceMesh<P> dm(ctx, m, 256);
  spike::GeomCopy<P> hg(m.packed);
  spike::metal::DeviceGeom<P> g(ctx, hg);
  const auto nc = size_t(m.topo.n_cells);
  spike::metal::Array<typename P::state> q(ctx, std::vector<typename P::state>(nc, 1.0f));
  spike::metal::Array<typename P::grad> gr(ctx, std::vector<typename P::grad>(3 * nc, 0.1f));
  spike::metal::Array<typename P::residual> r(ctx, nc);
  av::Data<P> d{};
  d.topo = dm.topo;
  d.q = q.get();
  d.grad = gr.get();
  d.res = r.get();
  spike::bind(d.face_r_left, g.face_r_left);
  spike::bind(d.face_r_right, g.face_r_right);
  spike::bind(d.face_normal, g.face_normal);
  spike::bind(d.face_area_magnitude, g.face_area_magnitude);
  d.velocity[0] = 1;
  d.velocity[1] = 0.7;
  d.velocity[2] = 0.4;
  auto* scratch = static_cast<typename P::compute*>(dm.schedule.face_scratch);
  const index_t ni = m.topo.n_owned_interior_faces;
  auto time = [&](auto f) {
    std::vector<double> t;
    for (int i = 0; i < 25; ++i) {
      for (int b = 0; b < 5; ++b) f();
      double s = ctx.synchronise();
      if (i >= 5) t.push_back(s / 5);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2] / double(nc) * 1e9;
  };
  double ts = time([&] {
    ctx.dispatch(av::metal_loops::library, "residual_store", P::tag, d,
                 mtl::StoreArgs<typename P::compute>{0, ni, scratch}, ni, 256);
  });
  double tg = time([&] {
    ctx.dispatch(av::metal_loops::library, "residual_gather", P::tag, d,
                 mtl::GatherArgs<typename P::compute>{scratch}, m.topo.n_cells, 256);
  });
  double tb = time(
      [&] { av::metal_loops::residual(ctx, d, kernels::Strategy::store_gather, dm.schedule); });
  std::printf("%s store %.3f gather %.3f both %.3f ns/cell\n", P::name, ts, tg, tb);
}
int main(int, char** argv) {
  run<mtl::fp32>(argv[1]);
  run<mtl::fp32_acc64>(argv[1]);
}
