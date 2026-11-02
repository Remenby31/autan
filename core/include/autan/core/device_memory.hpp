#pragma once

#include <cstddef>
#include <cstdint>

// The device memory this process holds, as its allocators record it (#411): on a unified-memory
// device (GB10) the driver's free memory is the whole host's, so it cannot tell this process's
// share. Every device allocation of the code base records here (the allocators of
// codegen/include/autan/kernels/device.cuh and of partition/src; a test checks none bypasses
// them). Not counted: the CUDA context and what libraries allocate for themselves.
namespace autan::device_memory {

/// Records `bytes` of device memory allocated, or released. Thread-safe.
void allocated(std::size_t bytes) noexcept;
void released(std::size_t bytes) noexcept;

/// The bytes allocated and not released, and the most there ever were in this process.
[[nodiscard]] std::int64_t current() noexcept;
[[nodiscard]] std::int64_t peak() noexcept;

}  // namespace autan::device_memory
