include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
)
FetchContent_Declare(
  picosha2
  URL https://github.com/okdshin/PicoSHA2/archive/161cb3fc4170fa7a3eca9e582cebd27cc4d1fe29.tar.gz
  URL_HASH SHA256=6cf473a00c98298d3ddee0aed853e3c799791f49dbc01c996ea46cb248e85802
  # Header only: a subdirectory without CMakeLists.txt skips its example build.
  SOURCE_SUBDIR no-build
)
FetchContent_MakeAvailable(nlohmann_json picosha2)

add_library(
  autan_io
  src/read_mesh.cpp
  src/gmsh.cpp
  src/fluent.cpp
  src/polymesh.cpp
  src/mesh_info.cpp
)
add_library(autan::io ALIAS autan_io)

if(AUTAN_ENABLE_CGNS)
  # The CGNS library brings HDF5 with it; a static build needs both on the
  # link line, so link what its installation provides.
  find_path(CGNS_INCLUDE_DIR cgnslib.h REQUIRED)
  find_library(CGNS_LIBRARY cgns REQUIRED)
  target_sources(autan_io PRIVATE src/cgns_read.cpp src/cgns_write.cpp)
  target_include_directories(autan_io SYSTEM PRIVATE ${CGNS_INCLUDE_DIR})
  target_link_libraries(autan_io PRIVATE ${CGNS_LIBRARY})
  target_compile_definitions(autan_io PRIVATE AUTAN_HAVE_CGNS)
endif()

target_include_directories(autan_io PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(autan_io SYSTEM PRIVATE ${picosha2_SOURCE_DIR})
target_link_libraries(
  autan_io
  PUBLIC autan::core
  PRIVATE autan_compile_options nlohmann_json::nlohmann_json
)

add_executable(autan-mesh-info tools/autan_mesh_info.cpp)
target_link_libraries(autan-mesh-info PRIVATE autan::io autan_compile_options)
