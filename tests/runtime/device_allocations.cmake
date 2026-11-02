# #411: every device allocation of the shipped code goes through an allocator that records it in
# autan::device_memory. Fails on a cudaMalloc/hipMalloc family call outside those allocators, or
# on an allocator that no longer records. Run as: cmake -DROOT=<source dir> -P device_allocations.cmake
set(
  allocators
  codegen/include/autan/kernels/device.cuh
  partition/src/device_self.cu
  partition/src/halo_exchange.cpp
)
set(
  pattern
  "(cuda|hip)(Malloc|MallocAsync|MallocManaged|MallocHost|MallocPitch|HostAlloc|Malloc3D)[ \t]*\\("
)
file(
  GLOB_RECURSE sources
  RELATIVE ${ROOT}
  ${ROOT}/core/*.cpp
  ${ROOT}/core/*.hpp
  ${ROOT}/core/*.cu
  ${ROOT}/core/*.cuh
  ${ROOT}/io/*.cpp
  ${ROOT}/io/*.hpp
  ${ROOT}/io/*.cu
  ${ROOT}/io/*.cuh
  ${ROOT}/partition/*.cpp
  ${ROOT}/partition/*.hpp
  ${ROOT}/partition/*.cu
  ${ROOT}/partition/*.cuh
  ${ROOT}/runtime/*.cpp
  ${ROOT}/runtime/*.hpp
  ${ROOT}/runtime/*.cu
  ${ROOT}/runtime/*.cuh
  ${ROOT}/codegen/include/*.cpp
  ${ROOT}/codegen/include/*.hpp
  ${ROOT}/codegen/include/*.cu
  ${ROOT}/codegen/include/*.cuh
  ${ROOT}/codegen/cpp/*.cpp
  ${ROOT}/codegen/cpp/*.hpp
  ${ROOT}/codegen/cpp/*.cu
  ${ROOT}/codegen/cpp/*.cuh
  ${ROOT}/solvers/src/*.cpp
  ${ROOT}/solvers/src/*.hpp
  ${ROOT}/solvers/src/*.cu
  ${ROOT}/solvers/src/*.cuh
  ${ROOT}/solvers/include/*.cpp
  ${ROOT}/solvers/include/*.hpp
  ${ROOT}/solvers/include/*.cu
  ${ROOT}/solvers/include/*.cuh
  ${ROOT}/physics/src/*.cpp
  ${ROOT}/physics/src/*.hpp
  ${ROOT}/physics/src/*.cu
  ${ROOT}/physics/src/*.cuh
  ${ROOT}/physics/include/*.cpp
  ${ROOT}/physics/include/*.hpp
  ${ROOT}/physics/include/*.cu
  ${ROOT}/physics/include/*.cuh
)
set(failures "")
foreach(source IN LISTS sources)
  file(READ ${ROOT}/${source} text)
  if(text MATCHES "${pattern}")
    list(FIND allocators ${source} known)
    if(known EQUAL -1)
      string(
        APPEND failures
        "\n  ${source}: allocates device memory outside the counted allocators"
      )
    elseif(NOT text MATCHES "device_memory::allocated")
      string(APPEND failures "\n  ${source}: allocates without recording in autan::device_memory")
    endif()
  endif()
endforeach()
if(failures)
  message(FATAL_ERROR "device allocations not counted (#411):${failures}")
endif()
