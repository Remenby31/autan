#include "autan/partition/halo_exchange.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "autan/core/error.hpp"
#include "halo_kernels.hpp"

#ifdef AUTAN_HAVE_CUDA
#include <cuda_runtime.h>
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>
#endif
#endif

namespace autan {
namespace {

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

constexpr int forward_tag = 7101;
constexpr int reverse_tag = 7102;

#ifdef AUTAN_HAVE_CUDA
void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw Error(ErrorCode::halo_device_unsupported,
                std::string(what) + ": " + cudaGetErrorString(status));
  }
}

bool mpi_is_cuda_aware() {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  return MPIX_Query_cuda_support() == 1;
#else
  return false;
#endif
}
#endif

/// Throws `code` on every rank if any rank found a problem, so that no rank
/// is left waiting in a collective the others skipped.
void agree(MPI_Comm comm, bool ok, ErrorCode code, const std::string& message) {
  int local = ok ? 1 : 0;
  int all = 0;
  MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, comm);
  if (all == 0) {
    throw Error(code, ok ? "another rank: " + message : message);
  }
}

}  // namespace

struct HaloExchange::State {
  MPI_Comm comm = MPI_COMM_NULL;
  MemorySpace space = MemorySpace::host;
  std::vector<std::int32_t> neighbours;
  std::vector<index_t> send_offsets, recv_offsets;
  // Host copies of the index arrays; `layout` points at them or at their
  // device copies in `device_arrays`.
  std::vector<index_t> send_cells, send_message, ghost_message;
  std::vector<index_t> reverse_cells, reverse_offsets, reverse_slots;
  halo::Layout layout{};
  std::vector<void*> device_arrays;
  // Buffer `slots` holds messages laid out by send slot, `ghosts` by ghost.
  std::vector<std::byte> host_slots, host_ghosts;
  void* slots = nullptr;
  void* ghosts = nullptr;
  std::vector<MPI_Request> receives, sends;

  bool pending = false;
  bool pending_reverse = false;
  bool pending_double = false;
  std::array<std::byte, sizeof(halo::Components<double>)> pending_components{};
  void* pending_stream = nullptr;

  ~State() {
#ifdef AUTAN_HAVE_CUDA
    for (void* p : device_arrays) cudaFree(p);
    if (space == MemorySpace::cuda) {
      cudaFree(slots);
      cudaFree(ghosts);
    }
#endif
    if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
  }

  const index_t* place(const std::vector<index_t>& host) {
    if (space == MemorySpace::host || host.empty()) return host.data();
#ifdef AUTAN_HAVE_CUDA
    void* device = nullptr;
    cuda_check(cudaMalloc(&device, host.size() * sizeof(index_t)), "cudaMalloc");
    device_arrays.push_back(device);
    cuda_check(
        cudaMemcpy(device, host.data(), host.size() * sizeof(index_t), cudaMemcpyHostToDevice),
        "cudaMemcpy");
    return static_cast<const index_t*>(device);
#else
    return nullptr;
#endif
  }

  void* allocate(std::vector<std::byte>& host, std::size_t bytes) {
    if (space == MemorySpace::host) {
      host.resize(bytes);
      return host.data();
    }
#ifdef AUTAN_HAVE_CUDA
    void* device = nullptr;
    if (bytes > 0) cuda_check(cudaMalloc(&device, bytes), "cudaMalloc");
    return device;
#else
    return nullptr;
#endif
  }

  template <class T>
  void run(halo::Kernel kernel, halo::Components<T> c, void* buffer, std::size_t count,
           void* stream) {
    if (count == 0) return;
    T* b = static_cast<T*>(buffer);
    if (space == MemorySpace::cuda) {
#ifdef AUTAN_HAVE_CUDA
      halo::launch<T>(kernel, layout, c, b, count, stream);
#endif
      return;
    }
    for (std::size_t i = 0; i < count; ++i) {
      switch (kernel) {
        case halo::Kernel::pack_forward:
          halo::pack_forward(layout, c, b, i);
          break;
        case halo::Kernel::unpack_forward:
          halo::unpack_forward(layout, c, b, i);
          break;
        case halo::Kernel::pack_reverse:
          halo::pack_reverse(layout, c, b, i);
          break;
        case halo::Kernel::unpack_reverse:
          halo::unpack_reverse(layout, c, b, i);
          break;
      }
    }
  }

  void synchronize([[maybe_unused]] void* stream) const {
#ifdef AUTAN_HAVE_CUDA
    if (space == MemorySpace::cuda) {
      cuda_check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
    }
#endif
  }

  /// Posts one message per neighbour: receives into `in` laid out by
  /// `in_offsets`, sends from `out` laid out by `out_offsets`.
  template <class T>
  void post_receives(void* in, const std::vector<index_t>& in_offsets, int n, int tag) {
    const MPI_Datatype type = std::is_same_v<T, double> ? MPI_DOUBLE : MPI_FLOAT;
    receives.assign(neighbours.size(), MPI_REQUEST_NULL);
    for (std::size_t i = 0; i < neighbours.size(); ++i) {
      const auto count = n * (in_offsets[i + 1] - in_offsets[i]);
      MPI_Irecv(static_cast<T*>(in) + at(n) * at(in_offsets[i]), count, type, neighbours[i], tag,
                comm, &receives[i]);
    }
  }
  template <class T>
  void post_sends(void* out, const std::vector<index_t>& out_offsets, int n, int tag) {
    const MPI_Datatype type = std::is_same_v<T, double> ? MPI_DOUBLE : MPI_FLOAT;
    sends.assign(neighbours.size(), MPI_REQUEST_NULL);
    for (std::size_t i = 0; i < neighbours.size(); ++i) {
      const auto count = n * (out_offsets[i + 1] - out_offsets[i]);
      MPI_Isend(static_cast<T*>(out) + at(n) * at(out_offsets[i]), count, type, neighbours[i], tag,
                comm, &sends[i]);
    }
  }

  template <class T>
  halo::Components<T> components(std::span<T* const> fields) {
    if (pending) {
      throw Error(ErrorCode::partition_invalid_request,
                  "a halo exchange is already in flight on this HaloExchange");
    }
    if (fields.empty() || fields.size() > static_cast<std::size_t>(halo::max_components)) {
      throw Error(ErrorCode::partition_invalid_request,
                  "a halo exchange moves 1 to " + std::to_string(halo::max_components) +
                      " components, not " + std::to_string(fields.size()));
    }
    halo::Components<T> c{};
    c.n = static_cast<int>(fields.size());
    for (std::size_t k = 0; k < fields.size(); ++k) c.data[k] = fields[k];
    return c;
  }

  template <class T>
  void begin(std::span<T* const> fields, void* stream, bool reverse) {
    const halo::Components<T> c = components(fields);
    const auto n = static_cast<std::size_t>(c.n);
    if (reverse) {
      post_receives<T>(slots, send_offsets, c.n, reverse_tag);
      run(halo::Kernel::pack_reverse, c, ghosts, n * at(layout.n_ghost), stream);
      synchronize(stream);
      post_sends<T>(ghosts, recv_offsets, c.n, reverse_tag);
    } else {
      post_receives<T>(ghosts, recv_offsets, c.n, forward_tag);
      run(halo::Kernel::pack_forward, c, slots, n * at(layout.n_send), stream);
      synchronize(stream);
      post_sends<T>(slots, send_offsets, c.n, forward_tag);
    }
    pending = true;
    pending_reverse = reverse;
    pending_double = std::is_same_v<T, double>;
    std::memcpy(pending_components.data(), &c, sizeof c);
    pending_stream = stream;
  }

  template <class T>
  void finish() {
    halo::Components<T> c;
    std::memcpy(&c, pending_components.data(), sizeof c);
    const auto n = static_cast<std::size_t>(c.n);
    MPI_Waitall(static_cast<int>(receives.size()), receives.data(), MPI_STATUSES_IGNORE);
    if (pending_reverse) {
      run(halo::Kernel::unpack_reverse, c, slots, n * at(layout.n_reverse), pending_stream);
    } else {
      run(halo::Kernel::unpack_forward, c, ghosts, n * at(layout.n_ghost), pending_stream);
    }
    synchronize(pending_stream);
    MPI_Waitall(static_cast<int>(sends.size()), sends.data(), MPI_STATUSES_IGNORE);
    pending = false;
  }

  void end(bool reverse) {
    if (!pending || pending_reverse != reverse) {
      throw Error(ErrorCode::partition_invalid_request, reverse
                                                            ? "end_reverse without begin_reverse"
                                                            : "end_forward without begin_forward");
    }
    if (pending_double) {
      finish<double>();
    } else {
      finish<float>();
    }
  }
};

HaloExchange::HaloExchange(const HaloPattern& pattern, index_t n_owned_cells, MPI_Comm comm,
                           MemorySpace space)
    : state_(std::make_unique<State>()) {
  State& s = *state_;
  MPI_Comm_dup(comm, &s.comm);
  s.space = space;
  int size = 0;
  int rank = 0;
  MPI_Comm_size(s.comm, &size);
  MPI_Comm_rank(s.comm, &rank);
  agree(s.comm, size == pattern.n_ranks && rank == pattern.rank, ErrorCode::halo_inconsistent,
        "rank " + std::to_string(rank) + " of " + std::to_string(size) +
            " holds the halo pattern of part " + std::to_string(pattern.rank) + " of " +
            std::to_string(pattern.n_ranks));

  bool device_ok = space == MemorySpace::host;
  std::string device_problem = "this build has no CUDA (configure with AUTAN_ENABLE_CUDA=ON)";
#ifdef AUTAN_HAVE_CUDA
  if (space == MemorySpace::cuda) {
    device_ok = mpi_is_cuda_aware();
    device_problem = "the MPI library is not CUDA-aware";
  }
#endif
  agree(s.comm, device_ok, ErrorCode::halo_device_unsupported, device_problem);

  // Each rank's send count to every other must equal that rank's ghost count
  // from it; checked with one all-to-all so that a one-sided neighbour list
  // fails here instead of hanging in an exchange.
  const auto n_neighbours = pattern.neighbours.size();
  bool shape_ok = pattern.send_offsets.size() == n_neighbours + 1 &&
                  pattern.recv_offsets.size() == n_neighbours + 1;
  std::vector<int> send_count(at(size), 0);
  std::vector<int> expected(at(size), 0);
  for (std::size_t i = 0; shape_ok && i < n_neighbours; ++i) {
    const std::int32_t q = pattern.neighbours[i];
    shape_ok = q >= 0 && q < size && q != rank && (i == 0 || q > pattern.neighbours[i - 1]);
    if (!shape_ok) break;
    send_count[at(q)] = pattern.send_offsets[i + 1] - pattern.send_offsets[i];
    expected[at(q)] = pattern.recv_offsets[i + 1] - pattern.recv_offsets[i];
  }
  std::vector<int> received(at(size), 0);
  MPI_Alltoall(send_count.data(), 1, MPI_INT, received.data(), 1, MPI_INT, s.comm);
  agree(s.comm, shape_ok && received == expected, ErrorCode::halo_inconsistent,
        "rank " + std::to_string(rank) + " expects ghost counts its neighbours do not send");

  s.neighbours = pattern.neighbours;
  s.send_offsets = pattern.send_offsets;
  s.recv_offsets = pattern.recv_offsets;
  s.send_cells = pattern.send_cells;
  s.reverse_cells = pattern.reverse_cells;
  s.reverse_offsets = pattern.reverse_offsets;
  s.reverse_slots = pattern.reverse_slots;
  for (std::size_t i = 0; i < n_neighbours; ++i) {
    for (index_t j = s.send_offsets[i]; j < s.send_offsets[i + 1]; ++j) {
      s.send_message.push_back(static_cast<index_t>(i));
    }
    for (index_t j = s.recv_offsets[i]; j < s.recv_offsets[i + 1]; ++j) {
      s.ghost_message.push_back(static_cast<index_t>(i));
    }
  }

  halo::Layout& l = s.layout;
  l.n_owned = n_owned_cells;
  l.n_send = static_cast<index_t>(s.send_cells.size());
  l.n_ghost = static_cast<index_t>(s.ghost_message.size());
  l.n_reverse = static_cast<index_t>(s.reverse_cells.size());
  l.send_cells = s.place(s.send_cells);
  l.send_offsets = s.place(s.send_offsets);
  l.send_message = s.place(s.send_message);
  l.recv_offsets = s.place(s.recv_offsets);
  l.ghost_message = s.place(s.ghost_message);
  l.reverse_cells = s.place(s.reverse_cells);
  l.reverse_offsets = s.place(s.reverse_offsets);
  l.reverse_slots = s.place(s.reverse_slots);
  constexpr std::size_t widest = max_components * sizeof(double);
  s.slots = s.allocate(s.host_slots, widest * at(l.n_send));
  s.ghosts = s.allocate(s.host_ghosts, widest * at(l.n_ghost));
}

HaloExchange::~HaloExchange() = default;

MemorySpace HaloExchange::space() const noexcept { return state_->space; }

template <class T>
void HaloExchange::begin_forward(std::span<T* const> components, void* stream) {
  state_->begin(components, stream, false);
}
void HaloExchange::end_forward() { state_->end(false); }

template <class T>
void HaloExchange::begin_reverse(std::span<T* const> components, void* stream) {
  state_->begin(components, stream, true);
}
void HaloExchange::end_reverse() { state_->end(true); }

template void HaloExchange::begin_forward<float>(std::span<float* const>, void*);
template void HaloExchange::begin_forward<double>(std::span<double* const>, void*);
template void HaloExchange::begin_reverse<float>(std::span<float* const>, void*);
template void HaloExchange::begin_reverse<double>(std::span<double* const>, void*);

void exchange_ghost_geometry(HaloExchange& exchange, Geometry& geometry) {
  if (exchange.space() != MemorySpace::host) {
    throw Error(ErrorCode::partition_invalid_request,
                "exchange_ghost_geometry moves a host Geometry; this exchange works on the device");
  }
  const std::array<double*, 4> fields{geometry.cell_volume.data(), geometry.cell_centroid_x.data(),
                                      geometry.cell_centroid_y.data(),
                                      geometry.cell_centroid_z.data()};
  exchange.forward(std::span<double* const>(fields));
}

}  // namespace autan
