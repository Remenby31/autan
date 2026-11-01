set(AUTAN_CORE_FIXTURES ${PROJECT_SOURCE_DIR}/tests/core/fixtures.cpp)

add_executable(autan_partition_tests ${AUTAN_CORE_FIXTURES} test_partition.cpp)
target_include_directories(autan_partition_tests PRIVATE ${PROJECT_SOURCE_DIR}/tests/core)
target_link_libraries(
  autan_partition_tests
  PRIVATE autan::partition Catch2::Catch2WithMain autan_compile_options
)
target_compile_definitions(
  autan_partition_tests
  PRIVATE
    $<$<BOOL:${AUTAN_ENABLE_METIS}>:AUTAN_HAVE_METIS>
    $<$<BOOL:${AUTAN_ENABLE_SCOTCH}>:AUTAN_HAVE_SCOTCH>
)
if(AUTAN_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(autan_partition_tests PRIVATE AUTAN_HAVE_CUDA)
  target_link_libraries(autan_partition_tests PRIVATE CUDA::cudart)
endif()
catch_discover_tests(autan_partition_tests TEST_SPEC "~[gpu]")
catch_discover_tests(autan_partition_tests TEST_SPEC "[gpu]" PROPERTIES LABELS gpu)

if(AUTAN_ENABLE_MPI AND AUTAN_ENABLE_METIS)
  add_executable(autan_halo_tests ${AUTAN_CORE_FIXTURES} test_halo.cpp)
  target_include_directories(autan_halo_tests PRIVATE ${PROJECT_SOURCE_DIR}/tests/core)
  target_link_libraries(autan_halo_tests PRIVATE autan::halo Catch2::Catch2 autan_compile_options)
  set(halo_labels mpi)
  if(AUTAN_ENABLE_CUDA)
    target_compile_definitions(autan_halo_tests PRIVATE AUTAN_HAVE_CUDA)
    set(halo_labels mpi gpu)
  endif()
  # The same tests on 1, 2, 4 and 8 ranks: results must not depend on the count.
  foreach(ranks 1 2 4 8)
    add_test(
      NAME autan_halo_tests_${ranks}_ranks
      COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:autan_halo_tests> ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(
      autan_halo_tests_${ranks}_ranks
      PROPERTIES PROCESSORS ${ranks} TIMEOUT 600 LABELS "${halo_labels}"
    )
  endforeach()
endif()
