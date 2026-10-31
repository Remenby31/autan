cmake_minimum_required(VERSION 3.28)
project(cfd_image_smoke LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(CGNS REQUIRED)
find_package(SCOTCH REQUIRED)
find_path(METIS_INCLUDE_DIR metis.h REQUIRED)
find_library(METIS_LIBRARY metis REQUIRED)

if(NOT HDF5_IS_PARALLEL)
  message(FATAL_ERROR "HDF5 is not built with MPI")
endif()

add_executable(smoke smoke.cpp)
target_include_directories(smoke PRIVATE ${METIS_INCLUDE_DIR})
target_link_libraries(smoke PRIVATE MPI::MPI_CXX HDF5::HDF5 CGNS::cgns-shared
                      SCOTCH::ptscotch SCOTCH::scotch ${METIS_LIBRARY})

add_executable(smoke_cuda smoke_cuda.cu)
target_link_libraries(smoke_cuda PRIVATE MPI::MPI_CXX CUDA::cudart)
