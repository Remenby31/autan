# Build-configuration checks (#571): the flags every CUDA unit of the main build is compiled with.

# a * b + c, compiled to PTX like every other CUDA unit: nvcc contracts it into an fma unless
# --fmad=false, which keeps device arithmetic the host's (-ffp-contract=off).
add_library(autan_fmad_probe OBJECT fmad_probe.cu)
set_target_properties(autan_fmad_probe PROPERTIES CUDA_PTX_COMPILATION ON)
add_test(
  NAME "CUDA units are compiled without FMA contraction"
  COMMAND
    ${CMAKE_COMMAND} -DPTX=$<TARGET_OBJECTS:autan_fmad_probe> -P
    ${CMAKE_CURRENT_SOURCE_DIR}/check_no_fma.cmake
)
