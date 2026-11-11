cmake_minimum_required(VERSION 3.28)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" AUTAN_VERSION LIMIT_COUNT 1)
project(autan VERSION ${AUTAN_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AUTAN_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(
  AUTAN_BUILD_FUZZERS
  "Build the libFuzzer harnesses of the mesh readers (clang; needs AUTAN_BUILD_TESTS)"
  OFF
)
option(AUTAN_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(AUTAN_ENABLE_CGNS "Read CGNS meshes and write solution.cgns (needs the CGNS library)" ON)
option(AUTAN_ENABLE_METIS "Partition with METIS" OFF)
option(AUTAN_ENABLE_SCOTCH "Partition with Scotch" OFF)
option(AUTAN_ENABLE_MPI "Build the MPI halo exchange" OFF)
option(AUTAN_ENABLE_CUDA "Build the CUDA paths" OFF)

# FetchContent's GitHub download (Catch2, for the tests) with a token when the
# environment holds one: GitHub slows unauthenticated downloads under load. CI sets it to the
# workflow's read-only GITHUB_TOKEN; the archive is checked against its SHA-256 either way.
# CMake before 3.31 does not verify TLS certificates by default: a token is sent only with it.
if(DEFINED ENV{AUTAN_GITHUB_TOKEN} AND NOT "$ENV{AUTAN_GITHUB_TOKEN}" STREQUAL "")
  set(
    AUTAN_GITHUB_DOWNLOAD
    TLS_VERIFY
    ON
    HTTP_HEADER
    "Authorization: Bearer $ENV{AUTAN_GITHUB_TOKEN}"
  )
else()
  set(AUTAN_GITHUB_DOWNLOAD "")
endif()

if(AUTAN_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  # No FMA contraction in device code, as -ffp-contract=off on the host: both sides run the same
  # arithmetic, so a device result can equal the host's bit for bit (#571). Explicit fma() calls
  # and the CUDA math library's own implementations are not affected. The host code of .cu files
  # goes to the host compiler, which contracts from -O2 on aarch64 unless told not to.
  string(APPEND CMAKE_CUDA_FLAGS " --fmad=false -Xcompiler=-ffp-contract=off")
endif()

# No FMA contraction in any C++ translation unit, whatever the build type and whether or not the
# target links autan_compile_options (the spike drivers' capi.cpp, the test framework): GCC
# contracts from -O2 on aarch64, which changes the last bits of host references.
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>")

add_library(autan_compile_options INTERFACE)
target_compile_options(
  autan_compile_options
  INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    # FMA contraction changes the last bits of geometry between compilers and
    # architectures; keep host results identical across the Mac and the Sparks.
    -ffp-contract=off
    $<$<BOOL:${AUTAN_WARNINGS_AS_ERRORS}>:-Werror>
)

add_subdirectory(third_party)
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(partition)
# autan-run writes solution.cgns: it needs the CGNS library.
if(AUTAN_ENABLE_CGNS)
  add_subdirectory(runtime)
else()
  message(STATUS "AUTAN_ENABLE_CGNS is OFF: autan-run is not built")
endif()
add_subdirectory(codegen)
add_subdirectory(solvers)
add_subdirectory(physics)

if(AUTAN_BUILD_TESTS)
  # Default per-test timeout of ctest, in seconds: a test that hangs fails after 10 minutes, with its
  # name, instead of holding its CI job until the job's own timeout. Tests that need longer set
  # their own TIMEOUT (the halo and ranks tests; the [long] tests of tests/physics). include(CTest)
  # reads it into ctest's configuration. A build directory configured before keeps its cached
  # value.
  set(DART_TESTING_TIMEOUT 600 CACHE STRING "Default per-test timeout of ctest, in seconds")
  include(CTest)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.16.1.tar.gz
    URL_HASH SHA256=d16bd1b8b2364918bd472cb216394acb93bde6784890df0d91bccfb3f28d777d
    ${AUTAN_GITHUB_DOWNLOAD}
  )
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  add_subdirectory(tests/core)
  add_subdirectory(tests/io)
  add_subdirectory(tests/fuzz)
  add_subdirectory(tests/partition)
  add_subdirectory(tests/physics)
  add_subdirectory(tests/solvers)
  if(AUTAN_ENABLE_CGNS)
    add_subdirectory(tests/runtime)
  endif()
  if(AUTAN_ENABLE_CUDA)
    add_subdirectory(tests/toolchain)
  endif()
endif()
