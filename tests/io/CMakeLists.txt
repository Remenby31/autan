add_executable(autan_io_tests test_readers.cpp test_mesh_info.cpp)
target_link_libraries(
  autan_io_tests
  PRIVATE autan::io nlohmann_json::nlohmann_json Catch2::Catch2WithMain autan_compile_options
)
target_compile_definitions(
  autan_io_tests
  PRIVATE AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
)
catch_discover_tests(autan_io_tests)

# The command-line tool through its output and exit status.
set(mesh_dir ${PROJECT_SOURCE_DIR}/tests/data/mesh)
add_test(NAME autan-mesh-info.json COMMAND autan-mesh-info --json ${mesh_dir}/hanging/polyMesh)
set_tests_properties(
  autan-mesh-info.json
  PROPERTIES PASS_REGULAR_EXPRESSION "\"max_faces_per_cell\": 105"
)
add_test(NAME autan-mesh-info.text COMMAND autan-mesh-info ${mesh_dir}/hex/hex.binary.msh)
set_tests_properties(autan-mesh-info.text PROPERTIES PASS_REGULAR_EXPRESSION "cells +27\n")
add_test(NAME autan-mesh-info.coded-error COMMAND autan-mesh-info ${mesh_dir}/missing.msh)
set_tests_properties(autan-mesh-info.coded-error PROPERTIES PASS_REGULAR_EXPRESSION "E0130")
