add_library(cfd_io src/read_mesh.cpp src/gmsh.cpp src/fluent.cpp src/polymesh.cpp)
add_library(cfd::io ALIAS cfd_io)

target_include_directories(cfd_io PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(cfd_io PUBLIC cfd::core PRIVATE cfd_compile_options)
