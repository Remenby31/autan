#pragma once

#include <string>
#include <vector>

namespace autan_probe {

struct Device {
  int index = 0;
  std::string name;
  int major = 0;
  int minor = 0;
  double memory_gb = 0.0;
  bool kernel_ok = false;
  std::string error;
};

struct CudaReport {
  bool compiled = false;
  std::string architectures;
  int runtime_version = 0;
  int driver_version = 0;  // highest CUDA version the installed driver supports, 0 without a driver
  std::string error;
  std::vector<Device> devices;
};

CudaReport probe_cuda();

}  // namespace autan_probe
