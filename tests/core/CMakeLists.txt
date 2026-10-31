add_executable(cfd_core_tests
  fixtures.cpp
  test_mesh.cpp
  test_geometry.cpp)
target_link_libraries(cfd_core_tests PRIVATE cfd::core Catch2::Catch2WithMain cfd_compile_options)
catch_discover_tests(cfd_core_tests)
