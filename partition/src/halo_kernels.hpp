#pragma once

#include <cstddef>

#include "autan/core/mesh.hpp"

#ifdef __CUDACC__
#define AUTAN_HD __host__ __device__
#else
#define AUTAN_HD
#endif

namespace autan::halo {

inline constexpr int max_components = 16;

template <class T>
struct Components {
  T* data[max_components];
  int n;
};

/// Device or host copies of the pattern arrays the kernels index. A message
/// to or from neighbour i holds n * count_i values, component-major, at
/// offset n * offsets[i] in its buffer.
struct Layout {
  index_t n_owned;
  index_t n_send;
  index_t n_ghost;
  const index_t* send_cells;     // n_send
  const index_t* send_offsets;   // n_neighbours + 1
  const index_t* send_message;   // n_send: neighbour of each send slot
  const index_t* recv_offsets;   // n_neighbours + 1
  const index_t* ghost_message;  // n_ghost: neighbour of each ghost
  index_t n_reverse;
  const index_t* reverse_cells;    // n_reverse
  const index_t* reverse_offsets;  // n_reverse + 1
  const index_t* reverse_slots;
};

AUTAN_HD inline std::size_t slot_position(const index_t* offsets, index_t message, index_t slot,
                                          int component, int n) {
  const auto first = static_cast<std::size_t>(offsets[message]);
  const auto count = static_cast<std::size_t>(offsets[message + 1]) - first;
  return static_cast<std::size_t>(n) * first + static_cast<std::size_t>(component) * count +
         (static_cast<std::size_t>(slot) - first);
}

/// Forward pack: send slot s, component k.
template <class T>
AUTAN_HD void pack_forward(const Layout& l, Components<T> c, T* buffer, std::size_t i) {
  const auto s = static_cast<index_t>(i % static_cast<std::size_t>(l.n_send));
  const int k = static_cast<int>(i / static_cast<std::size_t>(l.n_send));
  buffer[slot_position(l.send_offsets, l.send_message[s], s, k, c.n)] = c.data[k][l.send_cells[s]];
}

/// Forward unpack: ghost g, component k.
template <class T>
AUTAN_HD void unpack_forward(const Layout& l, Components<T> c, const T* buffer, std::size_t i) {
  const auto g = static_cast<index_t>(i % static_cast<std::size_t>(l.n_ghost));
  const int k = static_cast<int>(i / static_cast<std::size_t>(l.n_ghost));
  c.data[k][l.n_owned + g] = buffer[slot_position(l.recv_offsets, l.ghost_message[g], g, k, c.n)];
}

/// Reverse pack: ghost g, component k; the ghost is zeroed once sent.
template <class T>
AUTAN_HD void pack_reverse(const Layout& l, Components<T> c, T* buffer, std::size_t i) {
  const auto g = static_cast<index_t>(i % static_cast<std::size_t>(l.n_ghost));
  const int k = static_cast<int>(i / static_cast<std::size_t>(l.n_ghost));
  T& ghost = c.data[k][l.n_owned + g];
  buffer[slot_position(l.recv_offsets, l.ghost_message[g], g, k, c.n)] = ghost;
  ghost = T(0);
}

/// Reverse unpack: distinct send cell u, component k. Contributions are summed
/// in FP64 in ascending slot order, then added to the owned value once.
template <class T>
AUTAN_HD void unpack_reverse(const Layout& l, Components<T> c, const T* buffer, std::size_t i) {
  const auto u = static_cast<index_t>(i % static_cast<std::size_t>(l.n_reverse));
  const int k = static_cast<int>(i / static_cast<std::size_t>(l.n_reverse));
  T& value = c.data[k][l.reverse_cells[u]];
  double sum = static_cast<double>(value);
  for (index_t j = l.reverse_offsets[u]; j < l.reverse_offsets[u + 1]; ++j) {
    const index_t s = l.reverse_slots[j];
    sum += static_cast<double>(buffer[slot_position(l.send_offsets, l.send_message[s], s, k, c.n)]);
  }
  value = static_cast<T>(sum);
}

enum class Kernel { pack_forward, unpack_forward, pack_reverse, unpack_reverse };

#ifdef AUTAN_HAVE_CUDA
/// Launches `kernel` over `count` items on `stream` (cudaStream_t).
template <class T>
void launch(Kernel kernel, const Layout& layout, Components<T> c, T* buffer, std::size_t count,
            void* stream);
#endif

}  // namespace autan::halo
