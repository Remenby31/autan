#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "probe.hpp"

namespace nb = nanobind;

namespace {

nb::dict cuda_report() {
  nb::dict out;
#ifdef AUTAN_PROBE_CUDA
  const autan_probe::CudaReport report = autan_probe::probe_cuda();
  out["compiled"] = true;
  out["architectures"] = report.architectures;
  out["runtime_version"] = report.runtime_version;
  out["driver_version"] = report.driver_version;
  out["error_code"] = report.error_code;
  out["error"] = report.error;
  nb::list devices;
  for (const auto& d : report.devices) {
    nb::dict device;
    device["index"] = d.index;
    device["name"] = d.name;
    device["compute_capability"] = std::to_string(d.major) + "." + std::to_string(d.minor);
    device["memory_gb"] = d.memory_gb;
    device["kernel_ok"] = d.kernel_ok;
    device["error"] = d.error;
    devices.append(device);
  }
  out["devices"] = devices;
#else
  out["compiled"] = false;
#endif
  return out;
}

const char* cpu_target() {
#if defined(__aarch64__) && defined(__ARM_NEON)
  return "aarch64+neon";
#elif defined(__x86_64__) && defined(__AVX2__)
  return "x86_64+avx2";
#elif defined(__x86_64__)
  return "x86_64";
#else
  return "unknown";
#endif
}

}  // namespace

NB_MODULE(_native, m) {
  m.def("cuda_report", &cuda_report, "Compiled CUDA targets, driver and per-device kernel check.");
  m.def("cpu_target", &cpu_target, "Instruction set the CPU code was compiled for.");
}
