#pragma once

#include <cstdint>
#include <string>

namespace autan::runtime {

/// The number of CUDA devices this process sees: 0 in a build without CUDA,
/// or when the driver reports none or fails.
int visible_gpus();

/// The free memory of the device a rank with `local_rank` runs on (as Gpu
/// chooses it), and whether that device is integrated, its memory the
/// host's (GB10). Throws std::runtime_error when CUDA fails; the caller has
/// checked that visible_gpus() > 0.
struct DeviceMemory {
  std::int64_t free_bytes = 0;
  bool integrated = false;
};
DeviceMemory device_memory(int local_rank);

/// The GPU a rank runs on for devices.kind = cuda: device `local_rank`
/// modulo the visible devices, made current, with a non-blocking stream of
/// its own. Throws std::runtime_error when CUDA fails; the caller has
/// checked that visible_gpus() > 0.
class Gpu {
 public:
  explicit Gpu(int local_rank);
  Gpu(const Gpu&) = delete;
  Gpu& operator=(const Gpu&) = delete;
  ~Gpu();

  [[nodiscard]] int index() const noexcept { return index_; }
  [[nodiscard]] void* stream() const noexcept { return stream_; }
  [[nodiscard]] const std::string& name() const noexcept { return name_; }

 private:
  int index_ = -1;
  void* stream_ = nullptr;
  std::string name_;
};

}  // namespace autan::runtime
