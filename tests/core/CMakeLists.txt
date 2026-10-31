add_executable(autan_core_tests fixtures.cpp test_mesh.cpp test_geometry.cpp)
target_link_libraries(
  autan_core_tests
  PRIVATE autan::core Catch2::Catch2WithMain autan_compile_options
)
catch_discover_tests(autan_core_tests)
