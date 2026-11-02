add_library(
  autan_core
  src/error.cpp
  src/mesh.cpp
  src/cellwise.cpp
  src/geometry.cpp
  src/quality.cpp
  src/renumber.cpp
  src/wall_distance.cpp
  src/periodic.cpp
)
add_library(autan::core ALIAS autan_core)

target_include_directories(autan_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(autan_core PUBLIC cxx_std_20)
target_link_libraries(autan_core PRIVATE autan_compile_options)

# The wall distance on a CUDA device: nvcc without the host warning flags,
# and without FMA contraction, so that it gives the host's bits.
if(AUTAN_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(autan_core_cuda OBJECT src/wall_distance_device.cu)
  target_include_directories(autan_core_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(autan_core_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(autan_core_cuda PRIVATE --fmad=false --expt-relaxed-constexpr)
  target_link_libraries(autan_core_cuda PRIVATE CUDA::cudart)
  target_sources(autan_core PRIVATE $<TARGET_OBJECTS:autan_core_cuda>)
  target_link_libraries(autan_core PUBLIC CUDA::cudart)
endif()
