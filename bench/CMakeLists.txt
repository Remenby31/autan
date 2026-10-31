cmake_minimum_required(VERSION 3.28)

# AUTAN_BENCH_BACKEND selects the GPU language: CUDA (default) or HIP. The architecture defaults
# to the GPU present at configure time; pass CMAKE_<LANG>_ARCHITECTURES to cross-build.
set(AUTAN_BENCH_BACKEND CUDA CACHE STRING "GPU backend: CUDA or HIP")
set_property(CACHE AUTAN_BENCH_BACKEND PROPERTY STRINGS CUDA HIP)
if(NOT AUTAN_BENCH_BACKEND MATCHES "^(CUDA|HIP)$")
  message(FATAL_ERROR "AUTAN_BENCH_BACKEND must be CUDA or HIP, not ${AUTAN_BENCH_BACKEND}")
endif()
if(NOT DEFINED CMAKE_${AUTAN_BENCH_BACKEND}_ARCHITECTURES AND AUTAN_BENCH_BACKEND STREQUAL "CUDA")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(autan_bench LANGUAGES CXX ${AUTAN_BENCH_BACKEND})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_${AUTAN_BENCH_BACKEND}_STANDARD 20)
set(CMAKE_${AUTAN_BENCH_BACKEND}_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(stream stream/stream.cu)
set_source_files_properties(stream/stream.cu PROPERTIES LANGUAGE ${AUTAN_BENCH_BACKEND})
if(AUTAN_BENCH_BACKEND STREQUAL "HIP")
  target_compile_definitions(stream PRIVATE AUTAN_BENCH_HIP)
else()
  target_compile_options(stream PRIVATE -lineinfo)
endif()

find_package(Git REQUIRED)
add_custom_target(
  autan_bench_revision
  COMMAND
    ${CMAKE_COMMAND} -DGIT=${GIT_EXECUTABLE} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/revision.h -P
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/revision.cmake
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/revision.h
)
add_dependencies(stream autan_bench_revision)
target_include_directories(stream PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
