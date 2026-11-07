FetchContent_GetProperties(picosha2)

add_executable(autan-run-relaxation relaxation_main.cpp)
target_link_libraries(autan-run-relaxation PRIVATE autan::runtime autan_compile_options)

add_executable(autan_runtime_tests test_runtime.cpp test_host_memory.cpp)
target_include_directories(autan_runtime_tests SYSTEM PRIVATE ${picosha2_SOURCE_DIR})
target_include_directories(autan_runtime_tests PRIVATE ${PROJECT_SOURCE_DIR}/runtime/src)
target_link_libraries(
  autan_runtime_tests
  PRIVATE
    autan::runtime
    autan::physics_runtime
    autan::mesh_counts
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
    AUTAN_TEST_CMAKE="${CMAKE_COMMAND}"
)
add_dependencies(autan_runtime_tests autan-run-relaxation)
if(AUTAN_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  foreach(target autan_runtime_tests autan-run-relaxation)
    target_compile_definitions(${target} PRIVATE AUTAN_HAVE_CUDA)
    target_link_libraries(${target} PRIVATE CUDA::cudart)
  endforeach()
endif()
if(AUTAN_ENABLE_MPI AND AUTAN_ENABLE_METIS)
  target_compile_definitions(
    autan_runtime_tests
    PRIVATE
      AUTAN_MPIEXEC="${MPIEXEC_EXECUTABLE}"
      AUTAN_MPIEXEC_NUMPROC_FLAG="${MPIEXEC_NUMPROC_FLAG}"
  )
endif()
# The multi-rank tests carry the label `mpi` (ctest -L mpi).
# Processes started by these tests share their executables' hash (run.json
# solver.binary_sha256) through a cache in the build tree.
set(hash_cache "AUTAN_BINARY_HASH_CACHE=${PROJECT_BINARY_DIR}/binary-hash-cache")
catch_discover_tests(autan_runtime_tests TEST_SPEC "~[mpi]~[gpu]" PROPERTIES ENVIRONMENT ${hash_cache})
catch_discover_tests(autan_runtime_tests TEST_SPEC "[gpu]~[mpi]" PROPERTIES ENVIRONMENT ${hash_cache} LABELS gpu)
catch_discover_tests(autan_runtime_tests TEST_SPEC "[mpi]~[gpu]" PROPERTIES ENVIRONMENT ${hash_cache} LABELS mpi)
# [mpi][gpu]: on several ranks, and on the GPU too in a CUDA build (its tags as labels).
if(AUTAN_ENABLE_CUDA)
  catch_discover_tests(autan_runtime_tests TEST_SPEC "[mpi][gpu]" ADD_TAGS_AS_LABELS PROPERTIES ENVIRONMENT ${hash_cache})
else()
  catch_discover_tests(autan_runtime_tests TEST_SPEC "[mpi][gpu]" PROPERTIES ENVIRONMENT ${hash_cache} LABELS mpi)
endif()

# Ranks through the runtime's communicator: one rank here, 1 to 4 under MPI.
add_executable(autan_ranks_tests test_ranks.cpp)
target_include_directories(autan_ranks_tests PRIVATE ${PROJECT_SOURCE_DIR}/runtime/src)
target_link_libraries(autan_ranks_tests PRIVATE autan::runtime Catch2::Catch2 autan_compile_options)
add_test(NAME autan_ranks_tests COMMAND autan_ranks_tests)
if(AUTAN_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_compile_definitions(autan_ranks_tests PRIVATE AUTAN_HAVE_MPI)
  target_link_libraries(autan_ranks_tests PRIVATE MPI::MPI_CXX)
  foreach(ranks 2 4)
    add_test(
      NAME autan_ranks_tests_${ranks}_ranks
      COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:autan_ranks_tests> ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(
      autan_ranks_tests_${ranks}_ranks
      PROPERTIES PROCESSORS ${ranks} TIMEOUT 300 LABELS mpi
    )
  endforeach()
endif()

add_executable(autan_exact_sum_tests test_exact_sum.cpp)
target_link_libraries(
  autan_exact_sum_tests
  PRIVATE autan::runtime Catch2::Catch2WithMain autan_compile_options
)
catch_discover_tests(autan_exact_sum_tests)

if(AUTAN_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  # The kernel launch, compiled by nvcc without the host warning flags.
  add_library(autan_exact_sum_device OBJECT exact_sum_device.cu)
  set_target_properties(
    autan_exact_sum_device
    PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(autan_exact_sum_device PRIVATE autan::runtime CUDA::cudart)
  add_executable(
    autan_exact_sum_device_tests
    test_exact_sum_device.cpp
    $<TARGET_OBJECTS:autan_exact_sum_device>
  )
  target_link_libraries(
    autan_exact_sum_device_tests
    PRIVATE autan::runtime CUDA::cudart Catch2::Catch2WithMain autan_compile_options
  )
  catch_discover_tests(autan_exact_sum_device_tests PROPERTIES LABELS gpu)
  # The device memory counter through the code base's DeviceArray (#411).
  add_library(autan_device_memory_device OBJECT device_memory_device.cu)
  set_target_properties(
    autan_device_memory_device
    PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(autan_device_memory_device PRIVATE autan::kernels CUDA::cudart)
  add_executable(
    autan_device_memory_tests
    test_device_memory.cpp
    $<TARGET_OBJECTS:autan_device_memory_device>
  )
  target_include_directories(autan_device_memory_tests PRIVATE ${PROJECT_SOURCE_DIR}/runtime/src)
  target_link_libraries(
    autan_device_memory_tests
    PRIVATE autan::runtime CUDA::cudart Catch2::Catch2WithMain autan_compile_options
  )
  catch_discover_tests(autan_device_memory_tests PROPERTIES LABELS gpu)
endif()

# Every device allocation of the shipped code is counted (#411), in any build.
add_test(
  NAME device-allocations-counted
  COMMAND
    ${CMAKE_COMMAND} -DROOT=${PROJECT_SOURCE_DIR} -P
    ${CMAKE_CURRENT_SOURCE_DIR}/device_allocations.cmake
)

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
# The same case with SST, which no physics of this autan-run registers, on any build or device.
set(sst_input "${shipped_input}")
foreach(
  pair
  "\"model\": \"sa\"|\"model\": \"sst\""
  "\"nu_tilde_ratio\": 3.0|\"nu_tilde_ratio\": null"
  "\"intensity\": null|\"intensity\": 0.001"
  "\"viscosity_ratio\": null|\"viscosity_ratio\": 0.009"
  "\"k\": null|\"k\": 1.0"
  "\"omega\": null|\"omega\": 100.0"
)
  string(REPLACE "|" ";" pair "${pair}")
  list(GET pair 0 from)
  list(GET pair 1 to)
  string(REPLACE "${from}" "${to}" sst_input "${sst_input}")
endforeach()
string(REGEX REPLACE "\"nu_tilde\": [0-9.e+-]+" "\"nu_tilde\": null" sst_input "${sst_input}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/hybrid-sst.solver-input.json "${sst_input}")
add_test(
  NAME autan-run.no-physics
  COMMAND
    ${CMAKE_COMMAND} -DTOOL=$<TARGET_FILE:autan-run>
    "-DARGS=--input;${CMAKE_CURRENT_BINARY_DIR}/hybrid-sst.solver-input.json;--runs;${CMAKE_CURRENT_BINARY_DIR}/runs"
    -DEXPECT_CODE=2 "-DEXPECT=E0706" -P ${PROJECT_SOURCE_DIR}/tests/io/run_cli.cmake
)
add_test(
  NAME autan-run.usage
  COMMAND
    ${CMAKE_COMMAND} -DTOOL=$<TARGET_FILE:autan-run> -DARGS=--bogus -DEXPECT_CODE=2
    "-DEXPECT=usage: autan-run" -P ${PROJECT_SOURCE_DIR}/tests/io/run_cli.cmake
)
