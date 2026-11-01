add_executable(autan_physics_tests test_euler.cpp)
target_link_libraries(
  autan_physics_tests
  PRIVATE autan::physics autan::io Catch2::Catch2WithMain autan_compile_options
)
target_compile_definitions(
  autan_physics_tests
  PRIVATE AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
)
catch_discover_tests(autan_physics_tests)
