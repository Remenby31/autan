# Fuzzing the mesh readers

Each reader (Gmsh, Fluent, polyMesh, CGNS) has a libFuzzer harness, `fuzz_reader.cpp`. An input must read, or end with an `autan::Error` carrying an E01xx code. Anything else is a finding: a crash, a sanitizer report, an uncaught exception of another type, another code, or a memory spike beyond the run's limit.

## Run

The harnesses need clang with libFuzzer. On macOS that is Homebrew's `llvm`, linked against its own libc++; Apple clang has no libFuzzer.

```sh
LLVM=/opt/homebrew/opt/llvm
cmake -S . -B build-fuzz -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=$LLVM/bin/clang++ -DCMAKE_C_COMPILER=$LLVM/bin/clang \
  -DAUTAN_BUILD_FUZZERS=ON \
  "-DCMAKE_CXX_FLAGS=-fsanitize=fuzzer-no-link,address,undefined" \
  "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined -L$LLVM/lib/c++ -Wl,-rpath,$LLVM/lib/c++"
cmake --build build-fuzz --target fuzz-gmsh fuzz-fluent fuzz-polymesh fuzz-cgns
tests/fuzz/seed.sh build-fuzz/corpus
build-fuzz/tests/fuzz/fuzz-fluent build-fuzz/corpus/fluent -max_total_time=3600 -rss_limit_mb=4096
```

`seed.sh` builds the seed corpora from `tests/data/mesh`. A polyMesh input is its five files joined by NUL bytes (`read_input.hpp`).

## Findings

Minimise a finding (`-minimize_crash=1`), fix the reader, and add the input under `tests/data/fuzz/<reader>/`. The `autan_fuzz_regressions` test reads every file there on every build, without a fuzzer.

The CGNS harness also exercises the CGNS and HDF5 libraries. Report a finding inside them upstream; keep the input as a regression only once a fixed library version is in use.
