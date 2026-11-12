add_executable(
  autan_core_tests
  fixtures.cpp
  test_mesh.cpp
  test_geometry.cpp
  test_quality.cpp
  test_renumber.cpp
  test_lines.cpp
  test_wall_distance.cpp
  test_periodic.cpp
  test_prisms.cpp
)
target_link_libraries(
  autan_core_tests
  PRIVATE autan::core Catch2::Catch2WithMain autan_compile_options
)
# The device tests ([gpu]) carry the label gpu: CI runs them under the GPU lock.
catch_discover_tests(autan_core_tests TEST_SPEC "~[gpu]")

if(AUTAN_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(autan_core_device_tests OBJECT wall_distance_device.cu)
  set_target_properties(
    autan_core_device_tests
    PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(autan_core_device_tests PRIVATE autan::core CUDA::cudart)
  target_sources(autan_core_tests PRIVATE $<TARGET_OBJECTS:autan_core_device_tests>)
  target_compile_definitions(autan_core_tests PRIVATE AUTAN_HAVE_CUDA)
  target_link_libraries(autan_core_tests PRIVATE CUDA::cudart)
  catch_discover_tests(autan_core_tests TEST_SPEC "[gpu]" PROPERTIES LABELS gpu)
endif()
