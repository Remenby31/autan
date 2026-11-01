FetchContent_GetProperties(picosha2)

add_executable(autan-run-relaxation relaxation_main.cpp)
target_link_libraries(autan-run-relaxation PRIVATE autan::runtime autan_compile_options)

add_executable(autan_runtime_tests test_runtime.cpp)
target_include_directories(autan_runtime_tests SYSTEM PRIVATE ${picosha2_SOURCE_DIR})
target_link_libraries(
  autan_runtime_tests
  PRIVATE
    autan::runtime
    autan_cgns
    nlohmann_json::nlohmann_json
    Catch2::Catch2WithMain
    autan_compile_options
)
target_compile_definitions(
  autan_runtime_tests
  PRIVATE
    AUTAN_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
    AUTAN_RUN_RELAXATION="$<TARGET_FILE:autan-run-relaxation>"
)
add_dependencies(autan_runtime_tests autan-run-relaxation)
if(AUTAN_ENABLE_MPI AND AUTAN_ENABLE_METIS)
  target_compile_definitions(
    autan_runtime_tests
    PRIVATE
      AUTAN_MPIEXEC="${MPIEXEC_EXECUTABLE}"
      AUTAN_MPIEXEC_NUMPROC_FLAG="${MPIEXEC_NUMPROC_FLAG}"
  )
endif()
# The multi-rank tests carry the label `mpi` (ctest -L mpi).
catch_discover_tests(autan_runtime_tests TEST_SPEC "~[mpi]")
catch_discover_tests(autan_runtime_tests TEST_SPEC "[mpi]" PROPERTIES LABELS mpi)

add_executable(autan_exact_sum_tests test_exact_sum.cpp)
target_link_libraries(
  autan_exact_sum_tests
  PRIVATE autan::runtime Catch2::Catch2WithMain autan_compile_options
)
catch_discover_tests(autan_exact_sum_tests)

# autan-run as shipped, through its output and exit status: no physics yet.
set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/tests/data/runtime/hybrid.solver-input.json
)
file(READ ${PROJECT_SOURCE_DIR}/tests/data/runtime/hybrid.solver-input.json shipped_input)
string(
  REPLACE "@MESH@"
  "${PROJECT_SOURCE_DIR}/tests/data/mesh/hybrid/polyMesh"
  shipped_input
  "${shipped_input}"
)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/hybrid.solver-input.json "${shipped_input}")
add_test(
  NAME autan-run.no-physics
  COMMAND
    ${CMAKE_COMMAND} -DTOOL=$<TARGET_FILE:autan-run>
    "-DARGS=--input;${CMAKE_CURRENT_BINARY_DIR}/hybrid.solver-input.json;--runs;${CMAKE_CURRENT_BINARY_DIR}/runs"
    -DEXPECT_CODE=2 "-DEXPECT=E0706" -P ${PROJECT_SOURCE_DIR}/tests/io/run_cli.cmake
)
add_test(
  NAME autan-run.usage
  COMMAND
    ${CMAKE_COMMAND} -DTOOL=$<TARGET_FILE:autan-run> -DARGS=--bogus -DEXPECT_CODE=2
    "-DEXPECT=usage: autan-run" -P ${PROJECT_SOURCE_DIR}/tests/io/run_cli.cmake
)
