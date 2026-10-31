# Kernel generator: runs autan_codegen on kernels/*.py at build time and builds the generated loops.
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)

set(AUTAN_KERNEL_MODULES advection advection_diffusion gradients roe)
# AUTAN_ENABLE_CUDA (root) adds the GPU loops and the GPU spike driver.
set(AUTAN_KERNEL_BACKENDS cpu)
if(AUTAN_ENABLE_CUDA)
  list(APPEND AUTAN_KERNEL_BACKENDS gpu)
endif()

file(
  GLOB AUTAN_CODEGEN_SOURCES
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/autan_codegen/*.py
)
set(AUTAN_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

set(AUTAN_GENERATED_FILES)
set(AUTAN_GENERATED_GPU_UNITS)
foreach(module IN LISTS AUTAN_KERNEL_MODULES)
  set(outputs ${AUTAN_GENERATED_DIR}/${module}.hpp)
  list(APPEND outputs ${AUTAN_GENERATED_DIR}/${module}_cpu.hpp)
  if(AUTAN_ENABLE_CUDA)
    list(
      APPEND outputs
      ${AUTAN_GENERATED_DIR}/${module}_gpu.cuh
      ${AUTAN_GENERATED_DIR}/${module}_gpu.cu
    )
    list(APPEND AUTAN_GENERATED_GPU_UNITS ${AUTAN_GENERATED_DIR}/${module}_gpu.cu)
  endif()
  list(JOIN AUTAN_KERNEL_BACKENDS "," backends)
  add_custom_command(
    OUTPUT ${outputs}
    COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src ${Python3_EXECUTABLE} -m
      autan_codegen ${PROJECT_SOURCE_DIR}/kernels/${module}.py --out ${AUTAN_GENERATED_DIR}
      --backends ${backends}
    DEPENDS ${PROJECT_SOURCE_DIR}/kernels/${module}.py ${AUTAN_CODEGEN_SOURCES}
    COMMENT "Generating kernels from kernels/${module}.py"
    VERBATIM
  )
  list(APPEND AUTAN_GENERATED_FILES ${outputs})
endforeach()
add_custom_target(autan_generated_kernels DEPENDS ${AUTAN_GENERATED_FILES})

add_library(autan_kernels cpp/geometry.cpp cpp/colouring.cpp)
add_library(autan::kernels ALIAS autan_kernels)
add_dependencies(autan_kernels autan_generated_kernels)
target_include_directories(
  autan_kernels
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${AUTAN_GENERATED_DIR}>
)
target_link_libraries(autan_kernels PUBLIC autan::core PRIVATE autan_compile_options)
set_target_properties(autan_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The spike drivers are shared libraries loaded by ctypes, so the static core they link must be
# position independent.
set_target_properties(autan_core autan_io PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Spike driver: C ABI over the generated host loops, loaded by codegen/tests through ctypes.
add_library(autan_spike SHARED spike/capi.cpp)
target_link_libraries(autan_spike PRIVATE autan::kernels autan::io autan_compile_options)

if(AUTAN_ENABLE_CUDA)
  # Every generated loop instantiated for every policy: the compile-time measure of the spike.
  add_library(autan_kernels_gpu OBJECT ${AUTAN_GENERATED_GPU_UNITS})
  target_link_libraries(autan_kernels_gpu PRIVATE autan::kernels)
  set_target_properties(autan_kernels_gpu PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

  # GPU spike driver: the same C ABI on device memory, and the loop benchmark.
  add_library(autan_spike_gpu SHARED spike/capi.cpp spike/capi_gpu.cu)
  target_link_libraries(autan_spike_gpu PRIVATE autan::kernels autan::io)
  set_target_properties(autan_spike_gpu PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(
    autan_spike_gpu
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xptxas=-warn-spills>
  )

  find_package(Git REQUIRED)
  add_custom_target(
    autan_codegen_revision
    COMMAND
      ${CMAKE_COMMAND} -DGIT=${GIT_EXECUTABLE} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
      -DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/revision/revision.h -P
      ${CMAKE_CURRENT_SOURCE_DIR}/bench/revision.cmake
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/revision/revision.h
  )
  add_executable(autan_loop_bench bench/loop_bench.cu spike/capi.cpp)
  add_dependencies(autan_loop_bench autan_codegen_revision)
  target_include_directories(autan_loop_bench PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/revision spike)
  target_link_libraries(autan_loop_bench PRIVATE autan::kernels autan::io)
  set_target_properties(autan_loop_bench PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(autan_loop_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xptxas=-v>)
endif()
