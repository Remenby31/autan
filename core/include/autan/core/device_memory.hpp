#pragma once

#include <cstddef>
#include <cstdint>

// The device memory this process holds, as its allocators record it (#411): on a unified-memory
// device (GB10) the driver's free memory is the whole host's, so it cannot tell this process's
// share. Every device allocation of the code base records here (the allocators of
// codegen/include/autan/kernels/device.cuh, of partition/src and of
// core/src/wall_distance_device.cu; a test checks none bypasses them). Not counted: the CUDA
// context and what libraries allocate for themselves.
namespace autan::device_memory {

/// Records `bytes` of device memory allocated, or released. Thread-safe.
void allocated(std::size_t bytes) noexcept;
void released(std::size_t bytes) noexcept;

/// The bytes allocated and not released, and the most there were at once since the process
/// started or the last reset_peak().
[[nodiscard]] std::int64_t current() noexcept;
[[nodiscard]] std::int64_t peak() noexcept;
/// Starts a new peak from the bytes held now.
void reset_peak() noexcept;

}  // namespace autan::device_memory
