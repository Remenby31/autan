FetchContent_GetProperties(picosha2)

add_library(
  autan_runtime
  src/comm.cpp
  src/exact_sum.cpp
  src/input.cpp
  src/run_directory.cpp
  src/run.cpp
  src/main.cpp
)
add_library(autan::runtime ALIAS autan_runtime)
target_include_directories(
  autan_runtime
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_include_directories(autan_runtime SYSTEM PRIVATE ${picosha2_SOURCE_DIR})
target_link_libraries(
  autan_runtime
  PUBLIC autan::io autan::core autan::partition
  PRIVATE autan_compile_options nlohmann_json::nlohmann_json
)
if(AUTAN_ENABLE_MPI)
  target_link_libraries(autan_runtime PRIVATE autan::halo)
  target_compile_definitions(autan_runtime PRIVATE AUTAN_HAVE_MPI)
endif()

# Version and commit of the build, refreshed at every build (run.json).
set(build_info ${CMAKE_CURRENT_BINARY_DIR}/generated/autan_build_info.hpp)
add_custom_target(
  autan_build_info
  COMMAND
    ${CMAKE_COMMAND} -DSOURCE=${PROJECT_SOURCE_DIR} -DOUT=${build_info} -DVERSION=${AUTAN_VERSION}
    "-DAUTAN_COMMIT=${AUTAN_COMMIT}" -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_info.cmake
  BYPRODUCTS ${build_info}
  VERBATIM
)
add_dependencies(autan_runtime autan_build_info)

# The SDK's diagnostics registry (docs/contracts/diagnostics.json) gives the
# severity, fix and doc of every code the runtime reports.
set(registry ${PROJECT_SOURCE_DIR}/docs/contracts/diagnostics.json)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${registry})
file(READ ${registry} registry_json)
file(
  WRITE ${CMAKE_CURRENT_BINARY_DIR}/generated/autan_diagnostics.hpp
  "#pragma once\n// Generated from docs/contracts/diagnostics.json.\ninline constexpr const char* autan_diagnostics_json = R\"autan_json(${registry_json})autan_json\";\n"
)

add_executable(autan-run tools/autan_run.cpp)
target_link_libraries(autan-run PRIVATE autan::runtime autan::physics_runtime autan_compile_options)
