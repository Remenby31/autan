# Flow physics: the Euler and Navier-Stokes residuals from the generated loops of kernels/.
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
if(AUTAN_ENABLE_METAL)
  string(APPEND backends ",metal")
  list(APPEND AUTAN_PHYSICS_KERNELS ${AUTAN_PHYSICS_GENERATED}/euler_metal.hpp)
endif()
set(
  AUTAN_NS_KERNELS
  ${AUTAN_PHYSICS_GENERATED}/navier_stokes.hpp
  ${AUTAN_PHYSICS_GENERATED}/navier_stokes_cpu.hpp
  ${AUTAN_PHYSICS_GENERATED}/spalart_allmaras.hpp
  ${AUTAN_PHYSICS_GENERATED}/spalart_allmaras_cpu.hpp
)
set(ns_backends cpu)
if(AUTAN_ENABLE_CUDA)
  set(ns_backends cpu,gpu)
  list(
    APPEND AUTAN_NS_KERNELS
    ${AUTAN_PHYSICS_GENERATED}/navier_stokes_gpu.cuh
    ${AUTAN_PHYSICS_GENERATED}/navier_stokes_gpu.cu
    ${AUTAN_PHYSICS_GENERATED}/spalart_allmaras_gpu.cuh
    ${AUTAN_PHYSICS_GENERATED}/spalart_allmaras_gpu.cu
  )
endif()
add_custom_command(
  OUTPUT ${AUTAN_NS_KERNELS}
  COMMAND
    ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}/codegen/src ${Python3_EXECUTABLE} -m
    autan_codegen ${PROJECT_SOURCE_DIR}/kernels/navier_stokes.py --out ${AUTAN_PHYSICS_GENERATED}
    --backends ${ns_backends}
  COMMAND
    ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}/codegen/src ${Python3_EXECUTABLE} -m
    autan_codegen ${PROJECT_SOURCE_DIR}/kernels/spalart_allmaras.py --out ${AUTAN_PHYSICS_GENERATED}
    --backends ${ns_backends}
  DEPENDS
    ${PROJECT_SOURCE_DIR}/kernels/spalart_allmaras.py
    ${PROJECT_SOURCE_DIR}/kernels/navier_stokes.py
    ${PROJECT_SOURCE_DIR}/kernels/euler.py
    ${AUTAN_CODEGEN_SOURCES}
  COMMENT "Generating kernels from kernels/navier_stokes.py"
  VERBATIM
)
add_custom_command(
  OUTPUT ${AUTAN_PHYSICS_KERNELS}
  COMMAND
    ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}/codegen/src ${Python3_EXECUTABLE} -m
    autan_codegen ${PROJECT_SOURCE_DIR}/kernels/euler.py --out ${AUTAN_PHYSICS_GENERATED}
    --source-root ${PROJECT_SOURCE_DIR} --backends ${backends}
  DEPENDS
    ${PROJECT_SOURCE_DIR}/kernels/euler.py
    ${AUTAN_CODEGEN_SOURCES}
    ${PROJECT_SOURCE_DIR}/codegen/src/autan_codegen/metal_prelude.metal
  COMMENT "Generating kernels from kernels/euler.py"
  VERBATIM
)

add_library(
  autan_physics
  src/block_matrix.cpp
  src/euler.cpp
  src/manufactured.cpp
  src/navier_stokes.cpp
  src/spalart_allmaras.cpp
  ${AUTAN_PHYSICS_GENERATED}/euler.hpp
  ${AUTAN_PHYSICS_GENERATED}/euler_cpu.hpp
  ${AUTAN_PHYSICS_GENERATED}/navier_stokes.hpp
  ${AUTAN_PHYSICS_GENERATED}/navier_stokes_cpu.hpp
  ${AUTAN_PHYSICS_GENERATED}/spalart_allmaras.hpp
  ${AUTAN_PHYSICS_GENERATED}/spalart_allmaras_cpu.hpp
)
add_library(autan::physics ALIAS autan_physics)
target_include_directories(
  autan_physics
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${AUTAN_PHYSICS_GENERATED}
)
target_link_libraries(
  autan_physics
  PUBLIC autan::core autan::kernels autan::partition
  PRIVATE autan_compile_options nlohmann_json::nlohmann_json
)
set_target_properties(autan_physics PROPERTIES POSITION_INDEPENDENT_CODE ON)

# C ABI over the residual, loaded through ctypes by the manufactured-solution study (mms/).
add_library(autan_physics_mms SHARED mms/capi.cpp)
target_link_libraries(autan_physics_mms PRIVATE autan::physics autan::io autan_compile_options)
# The commit the library is built from, recorded by the study (as the runtime records its own).
set(mms_build_info ${CMAKE_CURRENT_BINARY_DIR}/mms_generated/autan_build_info.hpp)
add_custom_target(
  autan_physics_mms_build_info
  COMMAND
    ${CMAKE_COMMAND} -DSOURCE=${PROJECT_SOURCE_DIR} -DOUT=${mms_build_info}
    -DVERSION=${AUTAN_VERSION} "-DAUTAN_COMMIT=${AUTAN_COMMIT}" "-DAUTAN_DIRTY=${AUTAN_DIRTY}" -P
    ${PROJECT_SOURCE_DIR}/runtime/cmake/build_info.cmake
  BYPRODUCTS ${mms_build_info}
  VERBATIM
)
add_dependencies(autan_physics_mms autan_physics_mms_build_info)
target_include_directories(autan_physics_mms PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/mms_generated)

# The Euler physics behind autan-run's Physics interface; the runtime needs CGNS.
if(AUTAN_ENABLE_CGNS)
  add_library(autan_physics_runtime src/euler_physics.cpp src/partition_multigrid.cpp)
  add_library(autan::physics_runtime ALIAS autan_physics_runtime)
  target_include_directories(
    autan_physics_runtime
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(
    autan_physics_runtime
    PUBLIC autan::runtime
    PRIVATE autan::physics autan::solvers autan_compile_options nlohmann_json::nlohmann_json
  )
  if(AUTAN_ENABLE_CUDA)
    # The device step and the generated GPU loops it launches, compiled by nvcc without the host
    # warning flags.
    find_package(CUDAToolkit REQUIRED)
    add_library(
      autan_physics_device
      OBJECT
      src/euler_device.cu
      ${AUTAN_PHYSICS_GENERATED}/euler_gpu.cu
      src/navier_stokes_device.cu
      ${AUTAN_PHYSICS_GENERATED}/navier_stokes_gpu.cu
      ${AUTAN_PHYSICS_GENERATED}/spalart_allmaras_gpu.cu
    )
    target_include_directories(autan_physics_device PRIVATE ${AUTAN_PHYSICS_GENERATED})
    target_link_libraries(autan_physics_device PRIVATE autan::physics autan::runtime CUDA::cudart)
    set_target_properties(
      autan_physics_device
      PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(
      autan_physics_device
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    target_sources(autan_physics_runtime PRIVATE $<TARGET_OBJECTS:autan_physics_device>)
    target_link_libraries(autan_physics_runtime PUBLIC CUDA::cudart)
    target_compile_definitions(autan_physics_runtime PUBLIC AUTAN_PHYSICS_CUDA)
    # The device RK3 step against the host residual, per cell (docs/infra/bench.md).
    add_executable(autan_euler_step_bench bench/euler_step_bench.cpp)
    target_link_libraries(
      autan_euler_step_bench
      PRIVATE autan::physics_runtime autan::physics autan::io autan_compile_options
    )
    # The Navier-Stokes and Spalart-Allmaras loops on the GPU, through the device classes above,
    # timed by autan-bench (docs/infra/bench.md).
    add_executable(
      autan_navier_stokes_bench
      bench/navier_stokes_bench.cu
      $<TARGET_OBJECTS:autan_physics_device>
    )
    add_dependencies(autan_navier_stokes_bench autan_codegen_revision)
    target_include_directories(
      autan_navier_stokes_bench
      PRIVATE
        ${AUTAN_PHYSICS_GENERATED}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROJECT_BINARY_DIR}/codegen/revision
    )
    target_link_libraries(
      autan_navier_stokes_bench
      PRIVATE autan::physics autan::runtime autan::io CUDA::cudart
    )
    set_target_properties(
      autan_navier_stokes_bench
      PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
    )
    target_compile_options(
      autan_navier_stokes_bench
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -Xptxas=-warn-spills>
    )
  endif()
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
