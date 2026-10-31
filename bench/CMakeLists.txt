cmake_minimum_required(VERSION 3.28)

# CFD_BENCH_BACKEND selects the GPU language: CUDA (default) or HIP. The architecture defaults
# to the GPU present at configure time; pass CMAKE_<LANG>_ARCHITECTURES to cross-build.
set(CFD_BENCH_BACKEND CUDA CACHE STRING "GPU backend: CUDA or HIP")
set_property(CACHE CFD_BENCH_BACKEND PROPERTY STRINGS CUDA HIP)
if(NOT CFD_BENCH_BACKEND MATCHES "^(CUDA|HIP)$")
  message(FATAL_ERROR "CFD_BENCH_BACKEND must be CUDA or HIP, not ${CFD_BENCH_BACKEND}")
endif()
if(NOT DEFINED CMAKE_${CFD_BENCH_BACKEND}_ARCHITECTURES AND CFD_BENCH_BACKEND STREQUAL "CUDA")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cfd_bench LANGUAGES CXX ${CFD_BENCH_BACKEND})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_${CFD_BENCH_BACKEND}_STANDARD 20)
set(CMAKE_${CFD_BENCH_BACKEND}_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(stream stream/stream.cu)
set_source_files_properties(stream/stream.cu PROPERTIES LANGUAGE ${CFD_BENCH_BACKEND})
if(CFD_BENCH_BACKEND STREQUAL "HIP")
  target_compile_definitions(stream PRIVATE CFD_BENCH_HIP)
else()
  target_compile_options(stream PRIVATE -lineinfo)
endif()
