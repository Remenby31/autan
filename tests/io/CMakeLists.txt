add_executable(autan_io_tests test_readers.cpp)
target_link_libraries(autan_io_tests PRIVATE autan::io Catch2::Catch2WithMain autan_compile_options)
target_compile_definitions(
  autan_io_tests
  PRIVATE AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
)
catch_discover_tests(autan_io_tests)
