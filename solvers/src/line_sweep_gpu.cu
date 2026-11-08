#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "autan/kernels/device.cuh"
#include "autan/solvers/line_sweep_gpu.hpp"
#include "blocks_gpu.cuh"

namespace autan::solvers::gpu {

namespace {

namespace kg = autan::kernels::gpu;
using detail::b2;
using detail::block;
using detail::factor_kernel;
using detail::grid;
using detail::Lines;

kg::stream_t as_stream(void* s) { return static_cast<kg::stream_t>(s); }

// sum_j a(q, j) v_j from +0.0 in increasing j, as the host's `double v = 0.0; v += ...`.
template <class A>
__device__ double row(const A* a, std::int64_t stride, std::int64_t at, int q, const double* v) {
  double s = 0.0;
  for (int j = 0; j < 5; ++j) {
    s = __dadd_rn(s, __dmul_rn(static_cast<double>(a[(5 * q + j) * stride + at]), v[j]));
  }
  return s;
}

// The units of one colour (LineSmoother::solve_unit): forward, each cell's right-hand side from
// the current x of every neighbour but the previous and next cells of its unit, minus the
// coupling to the previous cell's w, times D'^-1; then backward, w_k minus D'^-1 U_k x_{k+1}, x
// written as it is known. `work` holds w, 5 per entry.
template <class Off>
__global__ void __launch_bounds__(block) solve_units(Lines l, std::int64_t count,
                                                     const index_t* units, const Off* off,
                                                     const double* dinv, const double* lower,
                                                     const double* gain, const double* b, double* x,
                                                     double* work) {
  const std::int64_t t = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= count) return;
  const std::int64_t unit = units[t];
  const offset_t first = l.line_offsets[unit], last = l.line_offsets[unit + 1];
  const std::int64_t n = l.n_nodes, ne = l.n_line_nodes;
  for (offset_t k = first; k < last; ++k) {
    const std::int64_t i = l.line_nodes[k];
    double r[5], w[5], xm[5];
    for (int q = 0; q < 5; ++q) r[q] = b[q * n + i];
    for (offset_t e = l.edge_offsets[k]; e < l.edge_offsets[k + 1]; ++e) {
      const std::int64_t m = l.edge_nodes[e];
      for (int j = 0; j < 5; ++j) xm[j] = x[j * n + m];
      for (int q = 0; q < 5; ++q) {
        r[q] = __dadd_rn(r[q], -row(off, l.n_slots, l.edge_slots[e], q, xm));
      }
    }
    if (k > first) {
      for (int j = 0; j < 5; ++j) w[j] = work[j * ne + k - 1];
      for (int q = 0; q < 5; ++q) r[q] = __dadd_rn(r[q], -row(lower, ne, k, q, w));
    }
    for (int q = 0; q < 5; ++q) work[q * ne + k] = row(dinv, ne, k, q, r);
  }
  double s[5];
  for (offset_t k = last - 1; k >= first; --k) {
    double w[5];
    for (int q = 0; q < 5; ++q) w[q] = work[q * ne + k];
    if (k + 1 < last) {
      for (int q = 0; q < 5; ++q) w[q] = __dadd_rn(w[q], -row(gain, ne, k, q, s));
    }
    for (int q = 0; q < 5; ++q) {
      s[q] = w[q];
      work[q * ne + k] = w[q];
      x[q * n + l.line_nodes[k]] = w[q];
    }
  }
}

// y = A x (LineSmoother::multiply): per cell, its diagonal block's terms, then each interior
// face's block terms in the host's face order, every term added to the running sum.
template <class Off>
__global__ void __launch_bounds__(block) multiply_kernel(
    std::int64_t n, std::int64_t n_slots, const offset_t* row_offsets, const offset_t* row_slots,
    const index_t* row_cells, const double* diag, const Off* off, const double* x, double* y) {
  const std::int64_t c = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= n) return;
  double s[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (int q = 0; q < 5; ++q) {
    for (int j = 0; j < 5; ++j)
      s[q] = __dadd_rn(s[q], __dmul_rn(diag[(5 * q + j) * n + c], x[j * n + c]));
  }
  for (offset_t p = row_offsets[c]; p < row_offsets[c + 1]; ++p) {
    const offset_t slot = row_slots[p];
    const std::int64_t m = row_cells[p];
    for (int q = 0; q < 5; ++q) {
      for (int j = 0; j < 5; ++j) {
        s[q] = __dadd_rn(
            s[q], __dmul_rn(static_cast<double>(off[(5 * q + j) * n_slots + slot]), x[j * n + m]));
      }
    }
  }
  for (int q = 0; q < 5; ++q) y[q * n + c] = s[q];
}

}  // namespace

struct LineSweep::Impl {
  std::int64_t n_units = 0, n_entries = 0, n_cells = 0, n_slots = 0;
  kg::DeviceArray<offset_t> unit_offsets, prev_offsets, prev_slots, next_offsets, next_slots,
      edge_offsets, edge_slots, row_offsets, row_slots;
  kg::DeviceArray<index_t> entry_cells, edge_cells, row_cells, colour_units;
  std::vector<std::int64_t> colour_offsets;
  kg::DeviceArray<double> dinv, lower, gain, work;
  kg::DeviceArray<int> singular;
  const double* diag = nullptr;
  const void* off = nullptr;
  bool single = false;

  [[nodiscard]] Lines view() const {
    return {n_units,
            n_entries,
            n_cells,
            n_slots,
            unit_offsets.get(),
            entry_cells.get(),
            prev_offsets.get(),
            prev_slots.get(),
            next_offsets.get(),
            next_slots.get(),
            edge_offsets.get(),
            edge_slots.get(),
            edge_cells.get()};
  }

  template <class Off>
  void set_operator(const double* d, const Off* o, void* stream) {
    diag = nullptr;
    off = nullptr;
    if (n_units > 0) {
      kg::check(kg::detail::zero(singular.get(), sizeof(int), as_stream(stream)), "reset");
      factor_kernel<Off><<<grid(n_units), block, 0, as_stream(stream)>>>(
          view(), d, o, dinv.get(), lower.get(), gain.get(), singular.get());
      kg::check(kg::last_error(), "line factorisation");
      int flag = 0;
      kg::check(kg::detail::to_host(&flag, singular.get(), sizeof(int), as_stream(stream)),
                "line factorisation");
      kg::check(kg::detail::synchronize(as_stream(stream)), "line factorisation");
      if (flag != 0) throw std::domain_error("singular diagonal block");
    }
    diag = d;
    off = o;
    single = std::is_same_v<Off, float>;
  }

  template <class Off>
  void multiply(const double* x, double* y, void* stream) const {
    multiply_kernel<Off><<<grid(n_cells), block, 0, as_stream(stream)>>>(
        n_cells, n_slots, row_offsets.get(), row_slots.get(), row_cells.get(), diag,
        static_cast<const Off*>(off), x, y);
    kg::check(kg::last_error(), "line sweep product");
  }

  template <class Off>
  void relax(std::size_t colour, const double* b, double* x, void* stream) const {
    const std::int64_t first = colour_offsets[colour], count = colour_offsets[colour + 1] - first;
    if (count == 0) return;
    solve_units<Off><<<grid(count), block, 0, as_stream(stream)>>>(
        view(), count, colour_units.get() + first, static_cast<const Off*>(off), dinv.get(),
        lower.get(), gain.get(), b, x, work.get());
    kg::check(kg::last_error(), "line sweep");
  }

  template <class Off>
  void sweep(const double* b, double* x, int sweeps, void* stream) const {
    const std::size_t colours = colour_offsets.size() - 1;
    for (int s = 0; s < sweeps; ++s) {
      for (std::size_t c = 0; c < colours; ++c) relax<Off>(c, b, x, stream);
      for (std::size_t c = colours; c-- > 0;) relax<Off>(c, b, x, stream);
    }
  }
};

LineSweep::LineSweep(const solvers::LineSmoother& host, std::span<const offset_t> incidence_offsets,
                     std::span<const index_t> incidence_faces)
    : impl_(std::make_unique<Impl>()) {
  const std::size_t n = host.n_cells();
  if (host.n_owned() != n) throw std::invalid_argument("LineSweep: one rank, no ghost cells");
  if (incidence_offsets.size() != n + 1) {
    throw std::invalid_argument("LineSweep: an incidence row per cell");
  }
  const auto offsets = host.cell_face_offsets();
  const auto faces = host.cell_faces();
  const auto owner = host.owner();
  const auto neighbour = host.neighbour();
  auto slot_of = [&](std::size_t c, index_t f) {
    for (auto k = incidence_offsets[c]; k < incidence_offsets[c + 1]; ++k) {
      if (incidence_faces[static_cast<std::size_t>(k)] == f) return k;
    }
    throw std::invalid_argument("LineSweep: an interior face of a cell has no slot");
  };
  auto other = [&](std::size_t c, index_t f) {
    const auto fs = static_cast<std::size_t>(f);
    return static_cast<std::size_t>(static_cast<std::size_t>(owner[fs]) == c ? neighbour[fs]
                                                                             : owner[fs]);
  };
  std::vector<offset_t> row_offsets{0}, row_slots;
  std::vector<index_t> row_cells;
  for (std::size_t c = 0; c < n; ++c) {
    for (auto p = offsets[c]; p < offsets[c + 1]; ++p) {
      const index_t f = faces[static_cast<std::size_t>(p)];
      row_slots.push_back(slot_of(c, f));
      row_cells.push_back(static_cast<index_t>(other(c, f)));
    }
    row_offsets.push_back(static_cast<offset_t>(row_slots.size()));
  }
  const auto cells = host.cells();
  std::vector<offset_t> unit_offsets{0}, prev_offsets{0}, next_offsets{0}, edge_offsets{0};
  std::vector<offset_t> prev_slots, next_slots, edge_slots;
  std::vector<index_t> entry_cells, edge_cells;
  for (const auto& u : host.units()) {
    for (std::size_t k = 0; k < u.size; ++k) {
      const auto c = static_cast<std::size_t>(cells[u.first + k]);
      const std::size_t prev = k > 0 ? static_cast<std::size_t>(cells[u.first + k - 1]) : SIZE_MAX;
      const std::size_t next =
          k + 1 < u.size ? static_cast<std::size_t>(cells[u.first + k + 1]) : SIZE_MAX;
      entry_cells.push_back(static_cast<index_t>(c));
      for (auto p = row_offsets[c]; p < row_offsets[c + 1]; ++p) {
        const auto ps = static_cast<std::size_t>(p);
        const auto m = static_cast<std::size_t>(row_cells[ps]);
        if (m == prev) prev_slots.push_back(row_slots[ps]);
        if (m == next) next_slots.push_back(row_slots[ps]);
        if (m != prev && m != next) {
          edge_slots.push_back(row_slots[ps]);
          edge_cells.push_back(row_cells[ps]);
        }
      }
      prev_offsets.push_back(static_cast<offset_t>(prev_slots.size()));
      next_offsets.push_back(static_cast<offset_t>(next_slots.size()));
      edge_offsets.push_back(static_cast<offset_t>(edge_slots.size()));
    }
    unit_offsets.push_back(static_cast<offset_t>(entry_cells.size()));
  }
  std::vector<index_t> colour_units;
  impl_->colour_offsets.push_back(0);
  for (const auto& colour : host.colour_units()) {
    for (const std::size_t u : colour) colour_units.push_back(static_cast<index_t>(u));
    impl_->colour_offsets.push_back(static_cast<std::int64_t>(colour_units.size()));
  }
  impl_->n_units = static_cast<std::int64_t>(unit_offsets.size() - 1);
  impl_->n_entries = static_cast<std::int64_t>(entry_cells.size());
  impl_->n_cells = static_cast<std::int64_t>(n);
  impl_->n_slots = static_cast<std::int64_t>(incidence_faces.size());
  auto up = [](const auto& v) {
    using T = typename std::decay_t<decltype(v)>::value_type;
    return kg::DeviceArray<T>(std::span<const T>(v));
  };
  impl_->unit_offsets = up(unit_offsets);
  impl_->entry_cells = up(entry_cells);
  impl_->prev_offsets = up(prev_offsets);
  impl_->prev_slots = up(prev_slots);
  impl_->next_offsets = up(next_offsets);
  impl_->next_slots = up(next_slots);
  impl_->edge_offsets = up(edge_offsets);
  impl_->edge_slots = up(edge_slots);
  impl_->edge_cells = up(edge_cells);
  impl_->row_offsets = up(row_offsets);
  impl_->row_slots = up(row_slots);
  impl_->row_cells = up(row_cells);
  impl_->colour_units = up(colour_units);
  const auto entries = static_cast<std::size_t>(b2) * entry_cells.size();
  impl_->dinv = kg::DeviceArray<double>(entries);
  impl_->lower = kg::DeviceArray<double>(entries);
  impl_->gain = kg::DeviceArray<double>(entries);
  impl_->work = kg::DeviceArray<double>(5 * entry_cells.size());
  impl_->singular = kg::DeviceArray<int>(1);
}

LineSweep::~LineSweep() = default;
LineSweep::LineSweep(LineSweep&&) noexcept = default;
LineSweep& LineSweep::operator=(LineSweep&&) noexcept = default;

void LineSweep::set_operator(const double* diag, const double* off, void* stream) {
  impl_->set_operator(diag, off, stream);
}
void LineSweep::set_operator(const double* diag, const float* off, void* stream) {
  impl_->set_operator(diag, off, stream);
}

void LineSweep::multiply(const double* x, double* y, void* stream) const {
  if (impl_->diag == nullptr) throw std::logic_error("LineSweep: set_operator first");
  if (impl_->single) {
    impl_->multiply<float>(x, y, stream);
  } else {
    impl_->multiply<double>(x, y, stream);
  }
}

void LineSweep::sweep(const double* b, double* x, int sweeps, void* stream) const {
  if (impl_->diag == nullptr) throw std::logic_error("LineSweep: set_operator first");
  if (impl_->single) {
    impl_->sweep<float>(b, x, sweeps, stream);
  } else {
    impl_->sweep<double>(b, x, sweeps, stream);
  }
}

}  // namespace autan::solvers::gpu
