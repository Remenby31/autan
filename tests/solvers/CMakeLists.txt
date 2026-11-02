add_executable(autan_solvers_tests test_multigrid.cpp)
target_link_libraries(
  autan_solvers_tests
  PRIVATE autan::solvers autan::io Catch2::Catch2WithMain autan_compile_options
)
target_compile_definitions(
  autan_solvers_tests
  PRIVATE AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
)
catch_discover_tests(autan_solvers_tests)

if(AUTAN_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(autan_solvers_gpu_tests test_multigrid_gpu.cpp)
  target_link_libraries(
    autan_solvers_gpu_tests
    PRIVATE autan::solvers autan::io CUDA::cudart Catch2::Catch2WithMain autan_compile_options
  )
  target_compile_definitions(
    autan_solvers_gpu_tests
    PRIVATE AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
  )
  catch_discover_tests(autan_solvers_gpu_tests PROPERTIES LABELS gpu)
endif()

add_executable(autan_galerkin_lists_tests test_galerkin_lists.cpp)
target_link_libraries(
  autan_galerkin_lists_tests
  PRIVATE autan::solvers autan::io Catch2::Catch2WithMain autan_compile_options
)
target_compile_definitions(
  autan_galerkin_lists_tests
  PRIVATE AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
)
catch_discover_tests(autan_galerkin_lists_tests)
