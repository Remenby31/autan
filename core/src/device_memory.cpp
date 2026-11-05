#include "autan/core/device_memory.hpp"

#include <algorithm>
#include <mutex>

namespace autan::device_memory {
namespace {

// One lock for both figures, so that a reset never loses an allocation it races with; every
// caller has just paid a device allocation or release, far costlier than the lock.
std::mutex lock;
std::int64_t held = 0;
std::int64_t most = 0;

}  // namespace

void allocated(std::size_t bytes) noexcept {
  const std::scoped_lock guard(lock);
  held += static_cast<std::int64_t>(bytes);
  most = std::max(most, held);
}

void released(std::size_t bytes) noexcept {
  const std::scoped_lock guard(lock);
  held -= static_cast<std::int64_t>(bytes);
}

std::int64_t current() noexcept {
  const std::scoped_lock guard(lock);
  return held;
}

std::int64_t peak() noexcept {
  const std::scoped_lock guard(lock);
  return most;
}

void reset_peak() noexcept {
  const std::scoped_lock guard(lock);
  most = held;
}

}  // namespace autan::device_memory
