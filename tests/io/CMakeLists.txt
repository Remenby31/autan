add_executable(
  autan_io_tests
  test_readers.cpp
  test_mesh_info.cpp
  test_mesh_counts.cpp
  ${PROJECT_SOURCE_DIR}/tests/core/fixtures.cpp
)
target_include_directories(autan_io_tests PRIVATE ${PROJECT_SOURCE_DIR}/tests/core)
target_link_libraries(
  autan_io_tests
  PRIVATE
    autan::io
    autan::mesh_counts
    nlohmann_json::nlohmann_json
    Catch2::Catch2WithMain
    autan_compile_options
)
target_compile_definitions(
  autan_io_tests
  PRIVATE
    AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
    $<$<BOOL:${AUTAN_ENABLE_METIS}>:AUTAN_HAVE_METIS>
)
if(AUTAN_ENABLE_CGNS)
  find_program(AUTAN_CGNSCHECK cgnscheck REQUIRED)
  target_sources(autan_io_tests PRIVATE test_cgns.cpp)
  target_link_libraries(autan_io_tests PRIVATE autan_cgns)
  target_compile_definitions(
    autan_io_tests
    PRIVATE AUTAN_HAVE_CGNS AUTAN_CGNSCHECK="${AUTAN_CGNSCHECK}"
  )
endif()
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
add_cli_test(scale 0 "\"scale\": 0.0254" --json --scale 0.0254 ${mesh_dir}/hex/polyMesh)
add_cli_test(no-scale 2 "usage: autan-mesh-info" --json --scale 0 ${mesh_dir}/hex/polyMesh)
if(AUTAN_ENABLE_METIS)
  add_cli_test(
    partition
    0
    "\"groups\": \"lines\""
    --json
    --cell-order
    input
    --ranks
    2
    --groups
    lines
    ${mesh_dir}/hybrid/polyMesh
  )
  add_cli_test(too-many-ranks 1 "E0140" --json --ranks 1000 ${mesh_dir}/hex/polyMesh)
else()
  add_cli_test(partition-without-metis 1 "E0142" --json --ranks 2 ${mesh_dir}/hex/polyMesh)
endif()
add_cli_test(groups-without-ranks 2 "usage: autan-mesh-info" --json --groups lines ${mesh_dir}/hex/polyMesh)
add_cli_test(no-ranks 2 "usage: autan-mesh-info" --json --ranks 0 ${mesh_dir}/hex/polyMesh)
add_cli_test(counts-without-json 2 "usage: autan-mesh-info" --ranks 2 ${mesh_dir}/hex/polyMesh)
add_test(
  NAME autan-mesh-info.surface
  COMMAND
    ${CMAKE_COMMAND} -DTOOL=$<TARGET_FILE:autan-mesh-info> -DMESH=${mesh_dir}/hex/polyMesh
    -DOUT=${CMAKE_CURRENT_BINARY_DIR}/hex.surface -P ${CMAKE_CURRENT_SOURCE_DIR}/surface_cli.cmake
)
add_cli_test(surface-and-json 2 "usage: autan-mesh-info" --json --surface out.bin ${mesh_dir}/hex/polyMesh)
add_cli_test(
  surface-unwritable
  1
  "E0130"
  --surface
  ${CMAKE_CURRENT_BINARY_DIR}/no-such-dir/out.bin
  ${mesh_dir}/hex/polyMesh
)
