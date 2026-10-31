add_library(autan_io src/read_mesh.cpp src/gmsh.cpp src/fluent.cpp src/polymesh.cpp)
add_library(autan::io ALIAS autan_io)

target_include_directories(autan_io PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(autan_io PUBLIC autan::core PRIVATE autan_compile_options)
