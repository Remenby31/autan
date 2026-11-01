# Flow physics: the Euler residual from the generated loops of kernels/euler.py.
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
file(GLOB AUTAN_CODEGEN_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/codegen/src/autan_codegen/*.py)
set(AUTAN_PHYSICS_GENERATED ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(AUTAN_PHYSICS_KERNELS ${AUTAN_PHYSICS_GENERATED}/euler.hpp ${AUTAN_PHYSICS_GENERATED}/euler_cpu.hpp)
add_custom_command(
  OUTPUT ${AUTAN_PHYSICS_KERNELS}
  COMMAND
    ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}/codegen/src ${Python3_EXECUTABLE} -m
    autan_codegen ${PROJECT_SOURCE_DIR}/kernels/euler.py --out ${AUTAN_PHYSICS_GENERATED}
    --backends cpu
  DEPENDS ${PROJECT_SOURCE_DIR}/kernels/euler.py ${AUTAN_CODEGEN_SOURCES}
  COMMENT "Generating kernels from kernels/euler.py"
  VERBATIM
)

add_library(autan_physics src/euler.cpp ${AUTAN_PHYSICS_KERNELS})
add_library(autan::physics ALIAS autan_physics)
target_include_directories(
  autan_physics
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${AUTAN_PHYSICS_GENERATED}
)
target_link_libraries(autan_physics PUBLIC autan::core autan::kernels PRIVATE autan_compile_options)
set_target_properties(autan_physics PROPERTIES POSITION_INDEPENDENT_CODE ON)

# C ABI over the residual, loaded through ctypes by the manufactured-solution study (mms/).
add_library(autan_physics_mms SHARED mms/capi.cpp)
target_link_libraries(autan_physics_mms PRIVATE autan::physics autan::io autan_compile_options)
