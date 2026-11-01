add_executable(autan_physics_tests test_euler.cpp test_implicit.cpp test_manufactured.cpp)
target_link_libraries(
  autan_physics_tests
  PRIVATE autan::physics autan::io Catch2::Catch2WithMain autan_compile_options
)
target_compile_definitions(
  autan_physics_tests
  PRIVATE
    AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
    AUTAN_EULER_MMS="${PROJECT_SOURCE_DIR}/py/tests/fixtures/euler.mms.json"
)
catch_discover_tests(autan_physics_tests)

# The Euler physics through autan-run's run_case; the runtime needs CGNS.
if(AUTAN_ENABLE_CGNS)
  add_executable(autan_euler_run_tests test_euler_run.cpp)
  target_link_libraries(
    autan_euler_run_tests
    PRIVATE
      autan::physics_runtime
      autan_cgns
      nlohmann_json::nlohmann_json
      Catch2::Catch2WithMain
      autan_compile_options
  )
  target_compile_definitions(
    autan_euler_run_tests
    PRIVATE AUTAN_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
  )
  catch_discover_tests(autan_euler_run_tests)
endif()
