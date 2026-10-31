cmake_minimum_required(VERSION 3.28)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" AUTAN_VERSION LIMIT_COUNT 1)
project(autan VERSION ${AUTAN_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AUTAN_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(AUTAN_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

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

add_subdirectory(core)
add_subdirectory(codegen)

if(AUTAN_BUILD_TESTS)
  include(CTest)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.16.1.tar.gz
    URL_HASH SHA256=d16bd1b8b2364918bd472cb216394acb93bde6784890df0d91bccfb3f28d777d
  )
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  add_subdirectory(tests/core)
endif()
