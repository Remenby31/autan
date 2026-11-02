#include "autan/core/device_memory.hpp"

#include <atomic>

namespace autan::device_memory {
namespace {

std::atomic<std::int64_t> held{0};
std::atomic<std::int64_t> most{0};

}  // namespace

void allocated(std::size_t bytes) noexcept {
  const std::int64_t now =
      held.fetch_add(static_cast<std::int64_t>(bytes)) + static_cast<std::int64_t>(bytes);
  std::int64_t seen = most.load();
  while (now > seen && !most.compare_exchange_weak(seen, now)) {
  }
}

void released(std::size_t bytes) noexcept { held.fetch_sub(static_cast<std::int64_t>(bytes)); }

std::int64_t current() noexcept { return held.load(); }

std::int64_t peak() noexcept { return most.load(); }

void reset_peak() noexcept { most.store(held.load()); }

}  // namespace autan::device_memory
