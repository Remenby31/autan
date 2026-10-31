cmake_minimum_required(VERSION 3.28)
# Default to the GPU present at configure time; pass -DCMAKE_CUDA_ARCHITECTURES to cross-build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(cfd_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(stream stream/stream.cu)
target_compile_options(stream PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
