#include <cstddef>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include "autan/core/device_memory.hpp"
#include "device.hpp"
#include "device_memory_device.hpp"

TEST_CASE("A run counts the device memory its allocators hold, and nothing else", "[gpu]") {
  // #411: on a unified-memory device the driver's free memory is the whole host's, so the run
  // counts its own allocations: n doubles are 8 n bytes, whatever else allocates meanwhile.
  const autan::runtime::Gpu gpu(0);
  const std::int64_t before = autan::device_memory::current();
  void* outside = autan_test::raw_device_allocation(std::size_t{256} << 20);
  REQUIRE(outside != nullptr);
  constexpr std::size_t n = 1'000'003;
  CHECK(autan_test::counted_while_holding(n) - before == static_cast<std::int64_t>(8 * n));
  CHECK(autan::device_memory::current() == before);
  CHECK(autan::device_memory::peak() >= before + static_cast<std::int64_t>(8 * n));
  CHECK(gpu.peak() == autan::device_memory::peak());
  autan_test::raw_device_release(outside);
  CHECK(autan::device_memory::current() == before);
}
