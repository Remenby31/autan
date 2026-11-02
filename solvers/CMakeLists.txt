# Linear solvers of the implicit step: agglomeration multigrid with line-implicit smoothing.
add_library(autan_solvers src/agglomeration.cpp src/dense.cpp src/multigrid.cpp)
add_library(autan::solvers ALIAS autan_solvers)
target_include_directories(
  autan_solvers
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(autan_solvers PUBLIC autan::core PRIVATE autan_compile_options)
set_target_properties(autan_solvers PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The multigrid's device kernels (docs/design/multigrid-gpu.md). nvcc does not take the host
# warning flags of autan_compile_options.
if(AUTAN_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(autan_solvers_cuda OBJECT src/multigrid_gpu.cu)
  target_link_libraries(autan_solvers_cuda PRIVATE autan::kernels CUDA::cudart)
  target_include_directories(autan_solvers_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(autan_solvers_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_sources(autan_solvers PRIVATE $<TARGET_OBJECTS:autan_solvers_cuda>)
  target_link_libraries(autan_solvers PUBLIC CUDA::cudart)
endif()

# The device transfers, timed by autan-bench (docs/infra/bench.md).
if(AUTAN_ENABLE_CUDA)
  add_executable(autan_mg_transfers_bench bench/transfers_bench.cpp)
  add_dependencies(autan_mg_transfers_bench autan_codegen_revision)
  target_include_directories(
    autan_mg_transfers_bench
    PRIVATE ${PROJECT_BINARY_DIR}/codegen/revision
  )
  target_link_libraries(
    autan_mg_transfers_bench
    PRIVATE autan::solvers autan::io CUDA::cudart autan_compile_options
  )
endif()
