# Host loop benchmark (autan-bench run --suite host): the generated CPU loops of kernels/euler.py,
# timed one by one, built from the same generated code as autan_physics.
find_package(Git QUIET)
add_custom_target(
  autan_host_bench_revision
  COMMAND
    ${CMAKE_COMMAND} -DGIT=${GIT_EXECUTABLE} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
    "-DAUTAN_COMMIT=${AUTAN_COMMIT}" -DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/revision/revision.h
    -P ${PROJECT_SOURCE_DIR}/bench/cmake/revision.cmake
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/revision/revision.h
)
add_executable(autan_host_bench host_bench.cpp)
add_dependencies(autan_host_bench autan_host_bench_revision autan_physics)
# The generated headers are autan_physics's private includes: the same files, not a copy.
target_include_directories(
  autan_host_bench
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/revision ${PROJECT_BINARY_DIR}/physics/generated
)
# The loops are timed: optimized whatever the tree's build type, as CMake's Release (an empty
# CMAKE_BUILD_TYPE, gpu-bench's, compiles at -O0).
target_compile_options(autan_host_bench PRIVATE -O3)
target_compile_definitions(autan_host_bench PRIVATE NDEBUG)
target_link_libraries(
  autan_host_bench
  PRIVATE autan::physics autan::io autan::kernels autan_compile_options
)
