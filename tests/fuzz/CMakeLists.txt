# Regression inputs from fuzzing (tests/data/fuzz), read without a fuzzer.
add_executable(autan_fuzz_regressions test_regressions.cpp)
target_link_libraries(autan_fuzz_regressions PRIVATE autan::io Catch2::Catch2WithMain autan_compile_options)
target_compile_definitions(
  autan_fuzz_regressions
  PRIVATE
    AUTAN_FUZZ_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data/fuzz"
    $<$<BOOL:${AUTAN_ENABLE_CGNS}>:AUTAN_HAVE_CGNS>
)
catch_discover_tests(autan_fuzz_regressions)

# libFuzzer harnesses (issue #75): opt-in, clang only; docs/testing/fuzzing.md.
if(AUTAN_BUILD_FUZZERS)
  foreach(reader gmsh fluent polymesh cgns)
    if(reader STREQUAL "cgns" AND NOT AUTAN_ENABLE_CGNS)
      continue()
    endif()
    add_executable(fuzz-${reader} fuzz_reader.cpp)
    target_compile_definitions(
      fuzz-${reader}
      PRIVATE AUTAN_FUZZ_READER=${reader} $<$<BOOL:${AUTAN_ENABLE_CGNS}>:AUTAN_HAVE_CGNS>
    )
    target_compile_options(fuzz-${reader} PRIVATE -fsanitize=fuzzer)
    target_link_options(fuzz-${reader} PRIVATE -fsanitize=fuzzer)
    target_link_libraries(fuzz-${reader} PRIVATE autan::io)
  endforeach()
endif()
