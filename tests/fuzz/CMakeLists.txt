# libFuzzer harnesses (issue #75): opt-in, clang only.
#   cmake -S . -B build-fuzz -DCMAKE_CXX_COMPILER=clang++ -DAUTAN_BUILD_FUZZERS=ON
#   build-fuzz/tests/fuzz/fuzz-gmsh tests/fuzz/corpus/gmsh -max_total_time=3600
foreach(reader gmsh fluent polymesh cgns)
  if(reader STREQUAL "cgns" AND NOT AUTAN_ENABLE_CGNS)
    continue()
  endif()
  string(TOUPPER ${reader} upper)
  add_executable(fuzz-${reader} fuzz_reader.cpp)
  target_compile_definitions(
    fuzz-${reader}
    PRIVATE AUTAN_FUZZ_${upper} $<$<BOOL:${AUTAN_ENABLE_CGNS}>:AUTAN_HAVE_CGNS>
  )
  target_compile_options(fuzz-${reader} PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_options(fuzz-${reader} PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_libraries(fuzz-${reader} PRIVATE autan::io)
endforeach()
