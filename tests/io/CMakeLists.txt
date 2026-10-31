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
function(add_cli_test name code regex)
  add_test(
    NAME autan-mesh-info.${name}
    COMMAND
      ${CMAKE_COMMAND} -DTOOL=$<TARGET_FILE:autan-mesh-info> "-DARGS=${ARGN}" -DEXPECT_CODE=${code}
      "-DEXPECT=${regex}" -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli.cmake
  )
endfunction()
add_cli_test(json 0 "\"max_faces_per_cell\": 105" --json ${mesh_dir}/hanging/polyMesh)
add_cli_test(text 0 "cells +27\n" ${mesh_dir}/hex/hex.binary.msh)
add_cli_test(coded-error 1 "E0130" ${mesh_dir}/missing.msh)
add_cli_test(usage 2 "usage: autan-mesh-info" --unknown)
