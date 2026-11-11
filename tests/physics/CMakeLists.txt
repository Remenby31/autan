add_executable(
  autan_physics_tests
  test_euler.cpp
  test_implicit.cpp
  test_manufactured.cpp
  test_navier_stokes.cpp
  test_sa_sweeps.cpp
  ${PROJECT_SOURCE_DIR}/tests/core/fixtures.cpp
)
target_include_directories(autan_physics_tests PRIVATE ${PROJECT_SOURCE_DIR}/tests/core)
target_link_libraries(
  autan_physics_tests
  PRIVATE autan::physics autan::io autan::partition Catch2::Catch2WithMain autan_compile_options
)
target_compile_definitions(
  autan_physics_tests
  PRIVATE
    AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
    AUTAN_EULER_MMS="${PROJECT_SOURCE_DIR}/py/tests/fixtures/euler.mms.json"
)
# The generated loops' data, which Euler::data() returns.
target_include_directories(autan_physics_tests PRIVATE ${PROJECT_BINARY_DIR}/physics/generated)
catch_discover_tests(autan_physics_tests)
# The colour sweeps' certificate (T2c, #163), hidden from the run above: a ctest label of its own,
# run by gpu-tests nightly and on pull requests that touch the SA assembly or sweeps.
catch_discover_tests(
  autan_physics_tests
  TEST_SPEC "[certificate]"
  TEST_PREFIX "certificate: "
  PROPERTIES LABELS certificate
)

# The manufactured-solution study's C ABI, through the shared library euler_mms.py loads.
add_executable(autan_physics_mms_tests test_block_ilu.cpp test_mms_build_info.cpp)
target_link_libraries(
  autan_physics_mms_tests
  PRIVATE autan_physics_mms Catch2::Catch2WithMain autan_compile_options
)
# The build information the library was compiled with (runtime/cmake/build_info.cmake).
target_include_directories(
  autan_physics_mms_tests
  PRIVATE ${PROJECT_BINARY_DIR}/physics/mms_generated
)
add_dependencies(autan_physics_mms_tests autan_physics_mms_build_info)
catch_discover_tests(autan_physics_mms_tests)

# The Euler residual on the generated Metal loops against the CPU physics (macOS).
if(AUTAN_ENABLE_METAL)
  enable_language(OBJCXX)
  add_executable(autan_physics_metal_tests test_euler_metal.mm)
  target_compile_options(autan_physics_metal_tests PRIVATE -fobjc-arc)
  target_include_directories(
    autan_physics_metal_tests
    PRIVATE ${PROJECT_BINARY_DIR}/physics/generated
  )
  target_link_libraries(
    autan_physics_metal_tests
    PRIVATE autan::physics autan::io autan_metal Catch2::Catch2WithMain autan_compile_options
  )
  target_compile_definitions(
    autan_physics_metal_tests
    PRIVATE AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
  )
  catch_discover_tests(autan_physics_metal_tests)
endif()

# The Euler physics through autan-run's run_case; the runtime needs CGNS.
if(AUTAN_ENABLE_CGNS)
  add_executable(
    autan_euler_run_tests
    test_euler_run.cpp
    ${PROJECT_SOURCE_DIR}/tests/core/fixtures.cpp
  )
  target_include_directories(autan_euler_run_tests PRIVATE ${PROJECT_SOURCE_DIR}/tests/core)
  target_link_libraries(autan_euler_run_tests PRIVATE autan::picosha2)
  target_link_libraries(
    autan_euler_run_tests
    PRIVATE
      autan::physics_runtime
      autan::kernels
      autan::mesh_counts
      autan_cgns
      nlohmann_json::nlohmann_json
      Catch2::Catch2WithMain
      autan_compile_options
  )
  target_compile_definitions(
    autan_euler_run_tests
    PRIVATE AUTAN_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
  )
  # Several ranks: autan-run under the MPI launcher, as tests/runtime does.
  if(AUTAN_ENABLE_MPI AND AUTAN_ENABLE_METIS)
    target_compile_definitions(
      autan_euler_run_tests
      PRIVATE
        AUTAN_RUN="$<TARGET_FILE:autan-run>"
        AUTAN_MPIEXEC="${MPIEXEC_EXECUTABLE}"
        AUTAN_MPIEXEC_NUMPROC_FLAG="${MPIEXEC_NUMPROC_FLAG}"
    )
    add_dependencies(autan_euler_run_tests autan-run)
  endif()
  # [mpi] cases carry the CTest label mpi: CI's hosted MPI ranks job selects them by it.
  # Processes started by these tests share their executables' hash (run.json
  # solver.binary_sha256) through a cache in the build tree.
  set(hash_cache "AUTAN_BINARY_HASH_CACHE=${PROJECT_BINARY_DIR}/binary-hash-cache")
  catch_discover_tests(autan_euler_run_tests TEST_SPEC "~[gpu]~[mpi]~[long]" PROPERTIES ENVIRONMENT ${hash_cache})
  # [long]: cases that take longer than the default per-test timeout (600 s) on CI's hosted
  # runners; 1500 s is about twice the longest measured.
  catch_discover_tests(
    autan_euler_run_tests
    TEST_SPEC "[long]~[gpu]~[mpi]"
    PROPERTIES ENVIRONMENT ${hash_cache} TIMEOUT 1500
  )
  catch_discover_tests(autan_euler_run_tests TEST_SPEC "[mpi]~[gpu]" PROPERTIES ENVIRONMENT ${hash_cache} LABELS mpi)
  if(AUTAN_ENABLE_MPI AND AUTAN_ENABLE_METIS)
    # #651 G-fit (P0, B1, B5): the Navier-Stokes physics at ghost depths 3 and 4 on the ranks of
    # one MPI world, on 2, 3 and 4 ranks.
    add_executable(
      autan_ns_ranks_tests
      test_ns_ranks.cpp
      ${PROJECT_SOURCE_DIR}/tests/core/fixtures.cpp
    )
    target_include_directories(
      autan_ns_ranks_tests
      PRIVATE ${PROJECT_SOURCE_DIR}/tests/core ${PROJECT_BINARY_DIR}/physics/generated
    )
    target_link_libraries(
      autan_ns_ranks_tests
      PRIVATE
        autan::physics
        autan::io
        autan::partition
        autan::halo
        Catch2::Catch2
        autan_compile_options
    )
    target_compile_definitions(
      autan_ns_ranks_tests
      PRIVATE
        AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
        AUTAN_TEST_MMS="${PROJECT_SOURCE_DIR}/py/tests/fixtures/rans_sa.mms.json"
    )
    foreach(ranks 2 3 4)
      add_test(
        NAME autan_ns_ranks_tests_${ranks}_ranks
        COMMAND
          ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:autan_ns_ranks_tests> ${MPIEXEC_POSTFLAGS}
      )
      set_tests_properties(
        autan_ns_ranks_tests_${ranks}_ranks
        PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS mpi
      )
    endforeach()
  endif()
  if(AUTAN_ENABLE_CUDA)
    target_link_libraries(autan_euler_run_tests PRIVATE autan::physics)
    catch_discover_tests(autan_euler_run_tests TEST_SPEC "[gpu]" PROPERTIES ENVIRONMENT ${hash_cache} LABELS gpu)
  endif()
endif()

# The Euler and Navier-Stokes residuals on a CUDA device against the host's.
if(AUTAN_ENABLE_CUDA AND AUTAN_ENABLE_CGNS)
  add_executable(
    autan_euler_device_tests
    test_euler_device.cpp
    test_navier_stokes_device.cpp
    ${PROJECT_SOURCE_DIR}/tests/core/fixtures.cpp
  )
  target_include_directories(
    autan_euler_device_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/tests/core ${PROJECT_BINARY_DIR}/physics/generated
  )
  target_link_libraries(
    autan_euler_device_tests
    PRIVATE
      autan::physics_runtime
      autan::physics
      autan::io
      Catch2::Catch2WithMain
      autan_compile_options
  )
  target_compile_definitions(
    autan_euler_device_tests
    PRIVATE AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
  )
  catch_discover_tests(autan_euler_device_tests PROPERTIES LABELS gpu)
endif()
