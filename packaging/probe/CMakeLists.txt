cmake_minimum_required(VERSION 3.28)
project(autan_probe LANGUAGES CXX)

option(AUTAN_PROBE_CUDA "Compile the CUDA probe kernel" OFF)
set(AUTAN_CUDA_ARCHITECTURES "" CACHE STRING "CUDA architectures embedded in the wheel")

if(AUTAN_PROBE_CUDA)
  if(NOT AUTAN_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "AUTAN_PROBE_CUDA needs AUTAN_CUDA_ARCHITECTURES")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES ${AUTAN_CUDA_ARCHITECTURES})
  enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_native STABLE_ABI NB_STATIC src/native.cpp)

if(AUTAN_PROBE_CUDA)
  target_sources(_native PRIVATE src/cuda_probe.cu)
  target_compile_definitions(_native PRIVATE AUTAN_PROBE_CUDA=1)
  string(REPLACE ";" "," _archs "${AUTAN_CUDA_ARCHITECTURES}")
  target_compile_definitions(_native PRIVATE AUTAN_CUDA_ARCHITECTURES="${_archs}")
  # cudart comes from the nvidia-cuda-runtime wheel, a dependency we do not redistribute; it
  # installs to site-packages/nvidia/cu13/lib. libcuda.so.1 comes from the driver.
  set_target_properties(
    _native
    PROPERTIES CUDA_RUNTIME_LIBRARY Shared INSTALL_RPATH "$ORIGIN/../nvidia/cu13/lib"
  )
  # Fatbin compression keeps one SASS per family within PyPI's per-file limit.
  target_compile_options(_native PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--compress-mode=size>)
endif()

install(TARGETS _native LIBRARY DESTINATION autan_probe)
