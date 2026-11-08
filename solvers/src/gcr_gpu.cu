#include <cstdint>
#include <stdexcept>
#include <vector>

#include "autan/kernels/device.cuh"
#include "autan/runtime/exact_sum.cuh"
#include "autan/solvers/gcr_gpu.hpp"

namespace autan::solvers::gpu {

namespace {

namespace kg = autan::kernels::gpu;
namespace rg = autan::runtime::gpu;

constexpr int block = 256;

unsigned int grid(std::int64_t n) { return static_cast<unsigned int>((n + block - 1) / block); }

kg::stream_t as_stream(void* s) { return static_cast<kg::stream_t>(s); }

// The iteration's scalars, written by one-thread kernels, read by the vector kernels.
struct State {
  int active;    // the iteration goes on
  int cycles;    // KrylovReport::cycles
  double scale;  // largest |r0|, NaN counted as +inf
  double r0;     // |r0 / scale|
  double rr;     // |r / scale|^2 after the last update
  double beta;   // the current projection's coefficient
  double alpha;  // the current step
  double drop;   // KrylovReport::residual_drop
  int finished;  // drop is set (scale not finite or zero)
};

// Products u_i v_i, as the host's `sum.add(u * v)`.
struct Dot {
  const double* u;
  const double* v;
  __device__ double operator()(std::int64_t i, int /*k*/) const { return __dmul_rn(u[i], v[i]); }
};

struct Dot2 {
  const double* u;
  const double* v;
  const double* p;
  const double* q;
  __device__ double operator()(std::int64_t i, int k) const {
    return k == 0 ? __dmul_rn(u[i], v[i]) : __dmul_rn(p[i], q[i]);
  }
};

// Largest |r| over the vector, NaN as +inf (owned_max_abs): one block, a fixed tree. A maximum
// is exact, so the order does not change it.
__global__ void __launch_bounds__(1024) max_abs_kernel(std::int64_t m, const double* r, State* s) {
  __shared__ double best[1024];
  double b = 0.0;
  for (std::int64_t i = threadIdx.x; i < m; i += blockDim.x) {
    const double a = fabs(r[i]);
    b = isnan(a) ? __longlong_as_double(0x7ff0000000000000LL) : fmax(b, a);
  }
  best[threadIdx.x] = b;
  __syncthreads();
  for (unsigned int h = blockDim.x / 2; h > 0; h /= 2) {
    if (threadIdx.x < h) best[threadIdx.x] = fmax(best[threadIdx.x], best[threadIdx.x + h]);
    __syncthreads();
  }
  if (threadIdx.x == 0) s->scale = best[0];
}

// The start (Gcr::solve before its loop): a scale that is not finite or zero ends the solve.
__global__ void start_kernel(State* s) {
  s->cycles = 0;
  s->finished = 0;
  s->active = 1;
  if (!isfinite(s->scale)) {
    s->drop = __longlong_as_double(0x7ff8000000000000LL);
    s->finished = 1;
    s->active = 0;
  } else if (s->scale == 0.0) {
    s->drop = 0.0;
    s->finished = 1;
    s->active = 0;
  }
}

__global__ void scale_kernel(std::int64_t m, double* v, const State* s, bool divide) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= m || s->finished != 0) return;
  v[i] = divide ? __ddiv_rn(v[i], s->scale) : __dmul_rn(v[i], s->scale);
}

__global__ void norm0_kernel(const rg::ExactWords* sums, State* s) {
  if (s->finished != 0) return;
  s->r0 = __dsqrt_rn(rg::exact_value(sums[0]));
  s->rr = __dmul_rn(s->r0, s->r0);
}

__global__ void zero_kernel(std::int64_t m, double* v, const int* active) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= m || *active == 0) return;
  v[i] = 0.0;
}

__global__ void beta_kernel(const rg::ExactWords* sums, State* s) {
  if (s->active == 0) return;
  s->beta = __ddiv_rn(rg::exact_value(sums[0]), rg::exact_value(sums[1]));
}

__global__ void project_kernel(std::int64_t m, double* z, double* w, const double* d,
                               const double* wj, const State* s) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= m || s->active == 0) return;
  z[i] = __dadd_rn(z[i], -__dmul_rn(s->beta, d[i]));
  w[i] = __dadd_rn(w[i], -__dmul_rn(s->beta, wj[i]));
}

// After the projections: ww = w . w, rw = r . w; the cycle counts; a direction whose image is
// zero or not finite ends the solve (a non-finite one fails it).
__global__ void step_kernel(const rg::ExactWords* sums, State* s, int k) {
  if (s->active == 0) return;
  const double ww = rg::exact_value(sums[0]);
  s->cycles = k + 1;
  if (!(ww > 0.0)) {
    if (!isfinite(ww)) s->rr = ww;
    s->active = 0;
    return;
  }
  s->alpha = __ddiv_rn(rg::exact_value(sums[1]), ww);
}

__global__ void update_kernel(std::int64_t m, double* x, double* r, const double* z,
                              const double* w, const State* s) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= m || s->active == 0) return;
  x[i] = __dadd_rn(x[i], __dmul_rn(__dmul_rn(s->scale, s->alpha), z[i]));
  r[i] = __dadd_rn(r[i], -__dmul_rn(s->alpha, w[i]));
}

__global__ void converged_kernel(const rg::ExactWords* sums, State* s, double tolerance) {
  if (s->active == 0) return;
  s->rr = rg::exact_value(sums[0]);
  if (__dsqrt_rn(s->rr) <= __dmul_rn(tolerance, s->r0)) s->active = 0;
}

__global__ void report_kernel(State* s) {
  if (s->finished != 0) return;
  s->drop = isfinite(s->rr) ? __ddiv_rn(__dsqrt_rn(s->rr), s->r0)
                            : __longlong_as_double(0x7ff8000000000000LL);
}

}  // namespace

struct Gcr::Impl {
  std::int64_t m = 0;
  int kmax = 0;
  void* stream = nullptr;
  std::vector<kg::DeviceArray<double>> directions, images;
  kg::DeviceArray<State> state;
  kg::DeviceArray<rg::ExactWords> sums;
};

Gcr::Gcr(std::int64_t m, int max_iterations, void* stream) : impl_(std::make_unique<Impl>()) {
  if (m < 0 || max_iterations < 0) throw std::invalid_argument("Gcr: negative size");
  impl_->m = m;
  impl_->kmax = max_iterations;
  impl_->stream = stream;
  for (int k = 0; k < max_iterations; ++k) {
    impl_->directions.emplace_back(static_cast<std::size_t>(m), as_stream(stream));
    impl_->images.emplace_back(static_cast<std::size_t>(m), as_stream(stream));
  }
  impl_->state = kg::DeviceArray<State>(1, as_stream(stream));
  impl_->sums = kg::DeviceArray<rg::ExactWords>(2, as_stream(stream));
}

Gcr::~Gcr() = default;
Gcr::Gcr(Gcr&&) noexcept = default;
Gcr& Gcr::operator=(Gcr&&) noexcept = default;

// Eisenstat, Elman and Schultz (1983), SIAM J. Numer. Anal. 20(2):345-357, as solvers::Gcr.
KrylovReport Gcr::solve(const Operator& apply, const Operator& precondition, double* r, double* x,
                        int max_iterations, double tolerance) {
  auto& d = *impl_;
  if (max_iterations > d.kmax) throw std::invalid_argument("Gcr: more iterations than kept");
  if (d.m == 0) return {.cycles = 0, .residual_drop = 0.0};  // the host's: a zero scale
  const auto st = as_stream(d.stream);
  const std::int64_t m = d.m;
  State* s = d.state.get();
  const int* active = &s->active;
  max_abs_kernel<<<1, 1024, 0, st>>>(m, r, s);
  start_kernel<<<1, 1, 0, st>>>(s);
  scale_kernel<<<grid(m), block, 0, st>>>(m, r, s, true);
  rg::exact_sums<1>(m, Dot{r, r}, d.sums.get(), st, 0, active);
  norm0_kernel<<<1, 1, 0, st>>>(d.sums.get(), s);
  for (int k = 0; k < max_iterations; ++k) {
    double* z = d.directions[static_cast<std::size_t>(k)].get();
    double* w = d.images[static_cast<std::size_t>(k)].get();
    zero_kernel<<<grid(m), block, 0, st>>>(m, z, active);
    precondition(r, z, active, d.stream);
    apply(z, w, active, d.stream);
    for (int j = 0; j < k; ++j) {
      const double* dj = d.directions[static_cast<std::size_t>(j)].get();
      const double* wj = d.images[static_cast<std::size_t>(j)].get();
      rg::exact_sums<2>(m, Dot2{w, wj, wj, wj}, d.sums.get(), st, 0, active);
      beta_kernel<<<1, 1, 0, st>>>(d.sums.get(), s);
      project_kernel<<<grid(m), block, 0, st>>>(m, z, w, dj, wj, s);
    }
    rg::exact_sums<2>(m, Dot2{w, w, r, w}, d.sums.get(), st, 0, active);
    step_kernel<<<1, 1, 0, st>>>(d.sums.get(), s, k);
    update_kernel<<<grid(m), block, 0, st>>>(m, x, r, z, w, s);
    rg::exact_sums<1>(m, Dot{r, r}, d.sums.get(), st, 0, active);
    converged_kernel<<<1, 1, 0, st>>>(d.sums.get(), s, tolerance);
  }
  scale_kernel<<<grid(m), block, 0, st>>>(m, r, s, false);
  report_kernel<<<1, 1, 0, st>>>(s);
  kg::check(kg::last_error(), "gcr");
  State host{};
  kg::check(kg::detail::to_host(&host, s, sizeof(State), st), "gcr report");
  kg::check(kg::detail::synchronize(st), "gcr report");
  return {.cycles = host.cycles, .residual_drop = host.drop};
}

}  // namespace autan::solvers::gpu
