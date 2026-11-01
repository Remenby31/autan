#pragma once

// Device and runtime fields of the raw benchmark JSON of the Metal binaries, as autan-bench reads
// them (backend "metal"): runtime and driver versions are the macOS version, major * 1000 +
// minor * 10 + patch.
#include <cstdio>

#include "autan/kernels/metal.hpp"

namespace autan::spike::metal {

inline void print_device(const autan::kernels::metal::Context& ctx) {
  const auto d = ctx.info();
  const int version = d.os_major * 1000 + d.os_minor * 10 + d.os_patch;
  std::printf(
      "\"device\":{\"name\":\"%s\",\"registry_id\":\"%llu\",\"gpu_cores\":%d,"
      "\"recommended_max_working_set_bytes\":%llu,\"memory_bytes\":%llu},",
      d.name.c_str(), static_cast<unsigned long long>(d.registry_id), d.gpu_cores,
      static_cast<unsigned long long>(d.recommended_max_working_set_bytes),
      static_cast<unsigned long long>(d.memory_bytes));
  std::printf("\"backend\":\"metal\",\"runtime_version\":%d,\"driver_version\":%d,", version,
              version);
  std::printf("\"os_version\":\"%d.%d.%d\",", d.os_major, d.os_minor, d.os_patch);
}

}  // namespace autan::spike::metal
