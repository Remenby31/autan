#pragma once

// Host runtime of the generated Metal loops (`<module>_metal.hpp`). Plain C++: the Objective-C++
// implementation is in codegen/cpp/metal.mm. Buffers live in unified memory (shared storage); the
// pointers stored in a module's Data<P> are their GPU addresses, which the kernels dereference.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "autan/kernels/runtime.hpp"
#include "autan/kernels/strategy.hpp"

namespace autan::kernels::metal {

/// Double-float: x = hi + lo, the FP64 roles of the fp32+acc64 policy on Metal. Layout of
/// `df64` in the Metal prelude (codegen/src/autan_codegen/metal_prelude.metal).
struct df64 {
  float hi = 0.0f;
  float lo = 0.0f;
  df64() = default;
  df64(double x) : hi(static_cast<float>(x)), lo(static_cast<float>(x - static_cast<double>(hi))) {}
  explicit operator double() const { return static_cast<double>(hi) + static_cast<double>(lo); }
};

/// Metal precision policies: storage types on the host side of the kernels' `Data<P>`.
struct fp32 {
  using state = float;
  using grad = float;
  using residual = float;
  using geom = float;
  using param = float;
  using compute = float;
  using accum = float;
  static constexpr const char* name = "fp32";
  static constexpr const char* tag = "fp32";
  static constexpr bool float_atomics = true;
  using host = ::autan::kernels::fp32;  ///< the CPU policy this one equals to rounding
};

struct fp32_acc64 {
  using state = float;
  using grad = float;
  using residual = df64;
  using geom = df64;
  using param = df64;
  using compute = float;
  using accum = df64;
  static constexpr const char* name = "fp32+acc64";
  static constexpr const char* tag = "fp32_acc64";
  static constexpr bool float_atomics = false;
  using host = ::autan::kernels::fp32_acc64;  ///< the CPU policy this one equals to rounding
};

/// A generated module's Metal source, in pieces ending with nullptr; `name` keys the compiled
/// library in a Context.
struct Library {
  const char* name;
  const char* const* source;
};

// Argument blocks of the generated kernels (buffer 1): layouts of the Metal prelude's.
struct RangeArgs {
  index_t lo;
  index_t hi;
};
struct ColourArgs {
  const index_t* faces;
  index_t count;
};
template <class A>
struct StoreArgs {
  index_t lo;
  index_t hi;
  A* scratch;
};
template <class A>
struct GatherArgs {
  const A* scratch;
};

/// Everything a strategy needs besides the data. Pointers are GPU addresses (Buffer::device).
struct Schedule {
  int block = 256;
  Colouring interior;
  Colouring boundary;
  void* face_scratch = nullptr;  ///< store-gather: face_scratch_width x n_faces accumulation values
  int face_scratch_width = 0;
};

/// A shared-storage buffer: `host()` for the CPU, `device<T>()` for a Data<P> pointer field.
class Buffer {
 public:
  Buffer() = default;
  void* host() const { return host_; }
  std::size_t bytes() const { return bytes_; }
  template <class T>
  T* device() const {
    return reinterpret_cast<T*>(static_cast<std::uintptr_t>(gpu_));
  }
  template <class T>
  T* data() const {
    return static_cast<T*>(host_);
  }

 private:
  friend class Context;
  std::shared_ptr<void> handle_;  // the id<MTLBuffer>, released with the last copy
  void* host_ = nullptr;
  std::uint64_t gpu_ = 0;
  std::size_t bytes_ = 0;
};

/// The default Metal device, a command queue, the compiled libraries and pipelines, and the
/// command buffer being recorded. Dispatches are recorded in order with a buffer barrier between
/// them, and run on `synchronise()`. Errors throw E1107 (metal_runtime_error).
class Context {
 public:
  Context();
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  /// Identity of the GPU, as benchmark provenance records it.
  struct DeviceInfo {
    std::string name;
    std::uint64_t registry_id;
    int gpu_cores;  ///< 0 when the IO registry does not report it
    std::uint64_t recommended_max_working_set_bytes;
    std::uint64_t memory_bytes;
  };
  DeviceInfo info() const;
  /// A zeroed shared buffer of `bytes` (at least one byte is allocated).
  Buffer allocate(std::size_t bytes);
  template <class T>
  Buffer upload(const std::vector<T>& v) {
    Buffer b = allocate(v.size() * sizeof(T));
    if (!v.empty()) {
      std::memcpy(b.host(), v.data(), v.size() * sizeof(T));
    }
    return b;
  }

  template <class D, class Args>
  void dispatch(const Library& lib, std::string_view kernel, const char* tag, const D& d,
                const Args& args, index_t threads, int block) {
    static_assert(std::is_trivially_copyable_v<Args>);
    dispatch_raw(lib, kernel, tag, &d, sizeof(D), &args, sizeof(Args), threads, block);
  }
  template <class D>
  void dispatch(const Library& lib, std::string_view kernel, const char* tag, const D& d,
                index_t threads, int block) {
    dispatch_raw(lib, kernel, tag, &d, sizeof(D), nullptr, 0, threads, block);
  }

  /// Commits the recorded dispatches, waits for them, and returns the GPU time in seconds
  /// (0 when nothing was recorded).
  double synchronise();
  /// Seconds spent compiling libraries and pipelines so far (first use of each kernel).
  double compile_seconds() const;

 private:
  void dispatch_raw(const Library& lib, std::string_view kernel, const char* tag, const void* d,
                    std::size_t d_bytes, const void* args, std::size_t args_bytes, index_t threads,
                    int block);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace autan::kernels::metal
