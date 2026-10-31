// C ABI of the GPU spike driver: the generated CUDA loops on device copies of a spike mesh, for
// the tests that compare them with the host loops and check determinism.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "advection_diffusion_gpu.cuh"
#include "advection_gpu.cuh"
#include "autan/kernels/colouring.hpp"
#include "autan/kernels/gpu.hpp"
#include "gradients_gpu.cuh"
#include "spike.hpp"

namespace {

using autan::kernels::index_t;
using autan::kernels::offset_t;
using autan::spike::GeomCopy;
using autan::spike::guarded;
using autan::spike::SpikeMesh;
namespace gpu = autan::kernels::gpu;

template <class T>
class DeviceArray {
 public:
  explicit DeviceArray(std::size_t n) : n_(n) {
    gpu::check(cudaMalloc(&p_, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    gpu::check(cudaMemset(p_, 0, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMemset");
  }
  explicit DeviceArray(const std::vector<T>& v) : DeviceArray(v.size()) { upload(v.data()); }
  DeviceArray(const T* host, std::size_t n) : DeviceArray(n) { upload(host); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray() { cudaFree(p_); }
  void upload(const T* host) {
    gpu::check(cudaMemcpy(p_, host, n_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
  }
  std::vector<T> download() const {
    std::vector<T> v(n_);
    gpu::check(cudaMemcpy(v.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
  }
  T* get() const { return p_; }

 private:
  T* p_ = nullptr;
  std::size_t n_;
};

/// Topology, colourings and scratch of a spike mesh on the device.
struct DeviceMesh {
  DeviceArray<index_t> owner, neighbour, cell_faces;
  DeviceArray<offset_t> offsets;
  autan::kernels::HostColouring int_colours, bnd_colours;
  DeviceArray<index_t> int_faces, bnd_faces;
  DeviceArray<double> scratch;
  autan::kernels::Topology topo;
  gpu::Schedule schedule;

  DeviceMesh(const SpikeMesh& m, int block)
      : owner(m.topo.owner, static_cast<std::size_t>(m.topo.n_faces)),
        neighbour(m.topo.neighbour, static_cast<std::size_t>(m.topo.n_interior_faces)),
        cell_faces(m.topo.cell_faces, static_cast<std::size_t>(m.topo.n_incidences)),
        offsets(m.topo.cell_face_offsets, static_cast<std::size_t>(m.topo.n_cells) + 1),
        int_colours(autan::kernels::colour_faces(m.topo, 0, m.topo.n_interior_faces)),
        bnd_colours(
            autan::kernels::colour_faces(m.topo, m.topo.n_interior_faces, m.topo.n_faces)),
        int_faces(int_colours.faces),
        bnd_faces(bnd_colours.faces),
        scratch(3 * static_cast<std::size_t>(m.topo.n_faces)),
        topo(m.topo) {
    topo.owner = owner.get();
    topo.neighbour = neighbour.get();
    topo.cell_faces = cell_faces.get();
    topo.cell_face_offsets = offsets.get();
    schedule.block = block;
    schedule.interior = {int_faces.get(), int_colours.offsets};
    schedule.boundary = {bnd_faces.get(), bnd_colours.offsets};
    schedule.face_scratch = scratch.get();
  }
};

template <class P>
struct DeviceGeom {
  using G = typename P::geom;
  DeviceArray<G> face_area, face_centroid, face_moment, face_normal, face_area_magnitude,
      face_r_left, face_r_right, cell_centroid, cell_volume, lsq_weight;
  explicit DeviceGeom(const GeomCopy<P>& g)
      : face_area(g.face_area),
        face_centroid(g.face_centroid),
        face_moment(g.face_moment),
        face_normal(g.face_normal),
        face_area_magnitude(g.face_area_magnitude),
        face_r_left(g.face_r_left),
        face_r_right(g.face_r_right),
        cell_centroid(g.cell_centroid),
        cell_volume(g.cell_volume),
        lsq_weight(g.lsq_weight) {}
};

gpu::Strategy strategy_of(std::string_view s) {
  for (auto k : {gpu::Strategy::atomic, gpu::Strategy::warp_aggregated, gpu::Strategy::coloured,
                 gpu::Strategy::store_gather, gpu::Strategy::gather_recompute}) {
    if (s == gpu::name(k)) {
      return k;
    }
  }
  throw std::invalid_argument("E1103: unknown race strategy");
}

template <class T>
std::vector<T> converted_in(const double* p, std::size_t n) {
  return std::vector<T>(p, p + n);
}

template <class T>
void copy_out(const std::vector<T>& v, double* p) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    p[i] = static_cast<double>(v[i]);
  }
}

void synchronise() { gpu::check(cudaDeviceSynchronize(), "kernel execution"); }

template <class P>
void advdiff_residual(const SpikeMesh& m, gpu::Strategy strategy, int block, const double* q,
                      const double* q_b, const double* source, const double* velocity, double nu,
                      double* res, double* grad) {
  namespace ad = autan::kernels::advection_diffusion;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  DeviceMesh dm(m, block);
  GeomCopy<P> hg(m.packed);
  DeviceGeom<P> g(hg);
  DeviceArray<typename P::state> qs(converted_in<typename P::state>(q, nc));
  DeviceArray<typename P::state> qbs(converted_in<typename P::state>(q_b, nf));
  DeviceArray<typename P::state> ss(converted_in<typename P::state>(source, nc));
  DeviceArray<typename P::grad> gs(3 * nc);
  DeviceArray<typename P::residual> rs(nc);
  ad::Data<P> d{};
  d.topo = dm.topo;
  d.q = qs.get();
  d.q_b = qbs.get();
  d.lsq_weight = g.lsq_weight.get();
  d.grad = gs.get();
  d.source = ss.get();
  d.cell_volume = g.cell_volume.get();
  d.res = rs.get();
  d.cell_centroid = g.cell_centroid.get();
  d.face_centroid = g.face_centroid.get();
  d.face_area = g.face_area.get();
  d.face_moment = g.face_moment.get();
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = typename P::param(velocity[i]);
  }
  d.nu[0] = typename P::param(nu);
  ad::gpu_loops::lsq_gradient(d, dm.schedule);
  ad::gpu_loops::init_residual(d, dm.schedule);
  ad::gpu_loops::flux(d, strategy, dm.schedule);
  ad::gpu_loops::boundary_flux(d, strategy, dm.schedule);
  synchronise();
  copy_out(rs.download(), res);
  copy_out(gs.download(), grad);
}

template <class P>
void gradients(const SpikeMesh& m, std::string_view method, gpu::Strategy strategy, int block,
               const double* q, const double* q_b, int iterations, double* grad, double* qmin,
               double* qmax) {
  namespace gr = autan::kernels::gradients;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  DeviceMesh dm(m, block);
  GeomCopy<P> hg(m.packed);
  DeviceGeom<P> g(hg);
  DeviceArray<typename P::state> qs(converted_in<typename P::state>(q, nc));
  DeviceArray<typename P::state> qbs(converted_in<typename P::state>(q_b, nf));
  DeviceArray<typename P::grad> gs(converted_in<typename P::grad>(grad, 3 * nc));
  DeviceArray<typename P::state> mins(nc), maxs(nc);
  DeviceArray<typename P::residual> acc(3 * nc);
  gr::Data<P> d{};
  d.topo = dm.topo;
  d.q = qs.get();
  d.q_b = qbs.get();
  d.grad = gs.get();
  d.qmin = mins.get();
  d.qmax = maxs.get();
  d.gg_sum = acc.get();
  d.lsq_weight = g.lsq_weight.get();
  d.cell_volume = g.cell_volume.get();
  d.cell_centroid = g.cell_centroid.get();
  d.face_centroid = g.face_centroid.get();
  d.face_area = g.face_area.get();
  d.face_moment = g.face_moment.get();
  if (method == "lsq") {
    gr::gpu_loops::lsq_gradient_bounds(d, dm.schedule);
  } else {
    for (int it = 0; it < iterations; ++it) {
      gpu::check(cudaMemset(acc.get(), 0, 3 * nc * sizeof(typename P::residual)), "memset");
      gr::gpu_loops::green_gauss_interior(d, strategy, dm.schedule);
      gr::gpu_loops::green_gauss_boundary(d, strategy, dm.schedule);
      gr::gpu_loops::green_gauss_finalise(d, dm.schedule);
    }
  }
  synchronise();
  copy_out(gs.download(), grad);
  copy_out(mins.download(), qmin);
  copy_out(maxs.download(), qmax);
}

template <class P>
void advection(const SpikeMesh& m, gpu::Strategy strategy, int block, const double* q,
               const double* grad, const double* velocity, double* res) {
  namespace av = autan::kernels::advection;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  DeviceMesh dm(m, block);
  GeomCopy<P> hg(m.packed);
  DeviceGeom<P> g(hg);
  DeviceArray<typename P::state> qs(converted_in<typename P::state>(q, nc));
  DeviceArray<typename P::grad> gs(converted_in<typename P::grad>(grad, 3 * nc));
  DeviceArray<typename P::residual> rs(nc);
  av::Data<P> d{};
  d.topo = dm.topo;
  d.q = qs.get();
  d.grad = gs.get();
  d.res = rs.get();
  d.face_r_left = g.face_r_left.get();
  d.face_r_right = g.face_r_right.get();
  d.face_normal = g.face_normal.get();
  d.face_area_magnitude = g.face_area_magnitude.get();
  for (int i = 0; i < 3; ++i) {
    d.velocity[i] = typename P::param(velocity[i]);
  }
  av::gpu_loops::residual(d, strategy, dm.schedule);
  synchronise();
  copy_out(rs.download(), res);
}

}  // namespace

extern "C" {

int autan_spike_gpu_advdiff_residual(const void* mesh, const char* policy, const char* strategy,
                                     int block, const double* q, const double* q_b,
                                     const double* source, const double* velocity, double nu,
                                     double* res, double* grad, char* err, std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::with_policy(policy, [&](auto p) {
      advdiff_residual<decltype(p)>(*m, s, block, q, q_b, source, velocity, nu, res, grad);
    });
  });
}

int autan_spike_gpu_gradients(const void* mesh, const char* policy, const char* method,
                              const char* strategy, int block, const double* q, const double* q_b,
                              int iterations, double* grad, double* qmin, double* qmax, char* err,
                              std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::with_policy(policy, [&](auto p) {
      gradients<decltype(p)>(*m, method, s, block, q, q_b, iterations, grad, qmin, qmax);
    });
  });
}

int autan_spike_gpu_advection(const void* mesh, const char* policy, const char* strategy,
                              int block, const double* q, const double* grad,
                              const double* velocity, double* res, char* err,
                              std::size_t err_len) {
  const auto* m = static_cast<const SpikeMesh*>(mesh);
  return guarded(err, err_len, [&] {
    const auto s = strategy_of(strategy);
    autan::spike::with_policy(policy, [&](auto p) {
      advection<decltype(p)>(*m, s, block, q, grad, velocity, res);
    });
  });
}

}  // extern "C"
