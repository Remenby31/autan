add_executable(cfd_io_tests test_readers.cpp)
target_link_libraries(cfd_io_tests PRIVATE cfd::io Catch2::Catch2WithMain cfd_compile_options)
target_compile_definitions(
  cfd_io_tests
  PRIVATE CFD_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
)
catch_discover_tests(cfd_io_tests)
