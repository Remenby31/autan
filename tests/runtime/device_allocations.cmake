# #411: every device allocation of the code base goes through an allocator that records it in
# autan::device_memory. Fails on a cudaMalloc/hipMalloc family call outside those allocators, or
# on an allocator file with fewer recordings than allocation calls.
# Run as: cmake -DROOT=<source dir> -P device_allocations.cmake
set(
  allocators
  codegen/include/autan/kernels/device.cuh
  partition/src/device_self.cu
  partition/src/halo_exchange.cpp
)
# Programs of their own that no run links: benchmarks, the codegen spike driver (loaded by the
# codegen tests through ctypes) and its generator tests, the image smoke test, the packaging probe.
set(
  standalone
  "^bench/"
  "^codegen/bench/"
  "^codegen/spike/"
  "^codegen/tests/"
  "^docker/"
  "^packaging/probe/"
  "^physics/bench/"
  "^solvers/bench/"
)
set(
  pattern
  "(cuda|hip)(Malloc|MallocAsync|MallocManaged|MallocHost|MallocPitch|HostAlloc|Malloc3D)[ \t]*\\("
)
# Every source directory but the tests, the docs and the web UI (no device code), and no build tree.
file(GLOB tops LIST_DIRECTORIES true RELATIVE ${ROOT} ${ROOT}/*)
set(sources "")
foreach(top IN LISTS tops)
  if(
    NOT IS_DIRECTORY ${ROOT}/${top}
    OR top MATCHES "^(\\.|tests$|docs$|ui$|website$)"
    OR EXISTS ${ROOT}/${top}/CMakeCache.txt
  )
    continue()
  endif()
  file(
    GLOB_RECURSE found
    RELATIVE ${ROOT}
    ${ROOT}/${top}/*.cpp
    ${ROOT}/${top}/*.cc
    ${ROOT}/${top}/*.h
    ${ROOT}/${top}/*.hpp
    ${ROOT}/${top}/*.cu
    ${ROOT}/${top}/*.cuh
    ${ROOT}/${top}/*.py
  )
  list(APPEND sources ${found})
endforeach()
list(FILTER sources EXCLUDE REGEX "/(\\.venv|node_modules)/")
set(failures "")
foreach(source IN LISTS sources)
  set(skip FALSE)
  foreach(prefix IN LISTS standalone)
    if(source MATCHES "${prefix}")
      set(skip TRUE)
    endif()
  endforeach()
  if(skip)
    continue()
  endif()
  file(READ ${ROOT}/${source} text)
  string(REGEX MATCHALL "${pattern}" calls "${text}")
  list(LENGTH calls n_calls)
  if(n_calls EQUAL 0)
    continue()
  endif()
  list(FIND allocators ${source} known)
  string(REGEX MATCHALL "device_memory::allocated\\(" records "${text}")
  list(LENGTH records n_records)
  if(known EQUAL -1)
    string(APPEND failures "\n  ${source}: allocates device memory outside the counted allocators")
  elseif(n_records LESS n_calls)
    string(
      APPEND failures
      "\n  ${source}: ${n_calls} allocation calls, ${n_records} recordings in autan::device_memory"
    )
  endif()
endforeach()
if(failures)
  message(FATAL_ERROR "device allocations not counted (#411):${failures}")
endif()
