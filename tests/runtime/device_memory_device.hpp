#pragma once

#include <cstddef>
#include <cstdint>

namespace autan_test {

// The device memory counter (autan::device_memory) while a DeviceArray of n doubles is alive.
std::int64_t counted_while_holding(std::size_t n);
// An allocation outside the code base's allocators (another library, another process's share).
void* raw_device_allocation(std::size_t bytes);
void raw_device_release(void* p);

}  // namespace autan_test
