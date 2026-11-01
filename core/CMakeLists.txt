add_library(
  autan_core
  src/error.cpp
  src/mesh.cpp
  src/cellwise.cpp
  src/geometry.cpp
  src/quality.cpp
  src/renumber.cpp
  src/wall_distance.cpp
  src/periodic.cpp
)
add_library(autan::core ALIAS autan_core)

target_include_directories(autan_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(autan_core PUBLIC cxx_std_20)
target_link_libraries(autan_core PRIVATE autan_compile_options)
