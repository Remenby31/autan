add_library(cfd_core src/error.cpp src/mesh.cpp src/cellwise.cpp src/geometry.cpp)
add_library(cfd::core ALIAS cfd_core)

target_include_directories(cfd_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(cfd_core PUBLIC cxx_std_20)
target_link_libraries(cfd_core PRIVATE cfd_compile_options)
