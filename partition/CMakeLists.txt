add_library(autan_partition src/partition.cpp src/extract.cpp src/self_exchange.cpp src/graph.cpp)
add_library(autan::partition ALIAS autan_partition)
target_include_directories(
  autan_partition
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(autan_partition PUBLIC autan::core PRIVATE autan_compile_options)

# Periodic images in device memory: CUDA kernels, or a build without CUDA that refuses them.
if(AUTAN_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  # nvcc does not take the host warning flags of autan_compile_options.
  add_library(autan_partition_cuda OBJECT src/device_self.cu)
  target_link_libraries(autan_partition_cuda PRIVATE autan::core CUDA::cudart)
  target_include_directories(autan_partition_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(autan_partition_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_sources(autan_partition PRIVATE $<TARGET_OBJECTS:autan_partition_cuda>)
  target_link_libraries(autan_partition PUBLIC CUDA::cudart)
else()
  target_sources(autan_partition PRIVATE src/device_self_host.cpp)
endif()

if(AUTAN_ENABLE_METIS)
  find_path(METIS_INCLUDE_DIR metis.h REQUIRED)
  find_library(METIS_LIBRARY metis REQUIRED)
  target_include_directories(autan_partition SYSTEM PRIVATE ${METIS_INCLUDE_DIR})
  target_link_libraries(autan_partition PRIVATE ${METIS_LIBRARY})
  target_compile_definitions(autan_partition PRIVATE AUTAN_HAVE_METIS)
endif()

if(AUTAN_ENABLE_SCOTCH)
  find_path(SCOTCH_INCLUDE_DIR scotch.h PATH_SUFFIXES scotch REQUIRED)
  find_library(SCOTCH_LIBRARY scotch REQUIRED)
  find_library(SCOTCH_ERR_LIBRARY scotcherr REQUIRED)
  target_include_directories(autan_partition SYSTEM PRIVATE ${SCOTCH_INCLUDE_DIR})
  target_link_libraries(autan_partition PRIVATE ${SCOTCH_LIBRARY} ${SCOTCH_ERR_LIBRARY})
  target_compile_definitions(autan_partition PRIVATE AUTAN_HAVE_SCOTCH)
endif()

if(AUTAN_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  add_library(autan_halo src/halo_exchange.cpp)
  add_library(autan::halo ALIAS autan_halo)
  target_link_libraries(
    autan_halo
    PUBLIC autan::partition MPI::MPI_CXX
    PRIVATE autan_compile_options
  )
  if(AUTAN_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    # nvcc does not take the host warning flags of autan_compile_options.
    add_library(autan_halo_cuda OBJECT src/halo_kernels.cu)
    target_link_libraries(autan_halo_cuda PRIVATE autan::core)
    target_compile_definitions(autan_halo_cuda PRIVATE AUTAN_HAVE_CUDA)
    target_sources(autan_halo PRIVATE $<TARGET_OBJECTS:autan_halo_cuda>)
    target_compile_definitions(autan_halo PRIVATE AUTAN_HAVE_CUDA)
    target_link_libraries(autan_halo PUBLIC CUDA::cudart)
  endif()
endif()

add_executable(autan-partition-report tools/partition_report.cpp)
target_link_libraries(
  autan-partition-report
  PRIVATE autan::partition autan::io autan_compile_options
)
