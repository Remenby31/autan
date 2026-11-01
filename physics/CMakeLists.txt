# Flow physics: the Euler residual from the generated loops of kernels/euler.py.
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
file(
  GLOB AUTAN_CODEGEN_SOURCES
  CONFIGURE_DEPENDS
  ${PROJECT_SOURCE_DIR}/codegen/src/autan_codegen/*.py
)
set(AUTAN_PHYSICS_GENERATED ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(
  AUTAN_PHYSICS_KERNELS
  ${AUTAN_PHYSICS_GENERATED}/euler.hpp
  ${AUTAN_PHYSICS_GENERATED}/euler_cpu.hpp
)
set(backends cpu)
if(AUTAN_ENABLE_CUDA)
  set(backends cpu,gpu)
  list(
    APPEND AUTAN_PHYSICS_KERNELS
    ${AUTAN_PHYSICS_GENERATED}/euler_gpu.cuh
    ${AUTAN_PHYSICS_GENERATED}/euler_gpu.cu
  )
endif()
add_custom_command(
  OUTPUT ${AUTAN_PHYSICS_KERNELS}
  COMMAND
    ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}/codegen/src ${Python3_EXECUTABLE} -m
    autan_codegen ${PROJECT_SOURCE_DIR}/kernels/euler.py --out ${AUTAN_PHYSICS_GENERATED} --backends
    ${backends}
  DEPENDS ${PROJECT_SOURCE_DIR}/kernels/euler.py ${AUTAN_CODEGEN_SOURCES}
  COMMENT "Generating kernels from kernels/euler.py"
  VERBATIM
)

add_library(
  autan_physics
  src/euler.cpp
  src/manufactured.cpp
  ${AUTAN_PHYSICS_GENERATED}/euler.hpp
  ${AUTAN_PHYSICS_GENERATED}/euler_cpu.hpp
)
add_library(autan::physics ALIAS autan_physics)
target_include_directories(
  autan_physics
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${AUTAN_PHYSICS_GENERATED}
)
target_link_libraries(
  autan_physics
  PUBLIC autan::core autan::kernels
  PRIVATE autan_compile_options nlohmann_json::nlohmann_json
)
set_target_properties(autan_physics PROPERTIES POSITION_INDEPENDENT_CODE ON)

# C ABI over the residual, loaded through ctypes by the manufactured-solution study (mms/).
add_library(autan_physics_mms SHARED mms/capi.cpp)
target_link_libraries(autan_physics_mms PRIVATE autan::physics autan::io autan_compile_options)

# The Euler physics behind autan-run's Physics interface; the runtime needs CGNS.
if(AUTAN_ENABLE_CGNS)
  add_library(autan_physics_runtime src/euler_physics.cpp)
  add_library(autan::physics_runtime ALIAS autan_physics_runtime)
  target_include_directories(
    autan_physics_runtime
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(
    autan_physics_runtime
    PUBLIC autan::runtime
    PRIVATE autan::physics autan_compile_options
  )
endif()

# The full residual on the GPU, timed by autan-bench (docs/infra/bench.md).
if(AUTAN_ENABLE_CUDA)
  add_executable(autan_euler_bench bench/euler_bench.cu ${AUTAN_PHYSICS_GENERATED}/euler_gpu.cu)
  add_dependencies(autan_euler_bench autan_codegen_revision)
  target_include_directories(
    autan_euler_bench
    PRIVATE ${AUTAN_PHYSICS_GENERATED} ${PROJECT_BINARY_DIR}/codegen/revision
  )
  target_link_libraries(autan_euler_bench PRIVATE autan::physics autan::io)
  set_target_properties(autan_euler_bench PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(
    autan_euler_bench
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -Xptxas=-warn-spills>
  )
endif()
