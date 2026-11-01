# Writes OUTPUT_FILE with the commit and dirty flag the benchmark binaries record, only when they
# change. In a git checkout whose top level is SOURCE_DIR: its HEAD and whether tracked files
# changed. Otherwise (an export: sdist, git archive) AUTAN_COMMIT when given, as for the runtime,
# recorded clean; without it "unknown", marked dirty, with a warning: autan-bench refuses to
# record such a binary, but the build goes on.
set(commit "")
set(dirty true)
if(GIT)
  execute_process(
    COMMAND ${GIT} -C ${SOURCE_DIR} rev-parse --show-toplevel
    OUTPUT_VARIABLE top
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE status
    ERROR_QUIET
  )
  if(status EQUAL 0)
    file(REAL_PATH "${top}" top)
    file(REAL_PATH "${SOURCE_DIR}" source)
  endif()
  # An export unpacked inside another repository must not take that repository's commit.
  if(status EQUAL 0 AND top STREQUAL source)
    execute_process(
      COMMAND ${GIT} -C ${SOURCE_DIR} rev-parse HEAD
      OUTPUT_VARIABLE head
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE status
      ERROR_QUIET
    )
    if(status EQUAL 0)
      set(commit "${head}")
      execute_process(
        COMMAND ${GIT} -C ${SOURCE_DIR} status --porcelain --untracked-files=no
        OUTPUT_VARIABLE changes
        RESULT_VARIABLE status
        ERROR_QUIET
      )
      if(status EQUAL 0 AND NOT changes)
        set(dirty false)
      endif()
    endif()
  endif()
endif()
if(commit STREQUAL "" AND DEFINED AUTAN_COMMIT AND NOT AUTAN_COMMIT STREQUAL "")
  if(NOT AUTAN_COMMIT MATCHES "^[0-9a-f]+$")
    message(
      FATAL_ERROR
      "AUTAN_COMMIT must be 40 lowercase hexadecimal characters, got '${AUTAN_COMMIT}'"
    )
  endif()
  string(LENGTH "${AUTAN_COMMIT}" length)
  if(NOT length EQUAL 40)
    message(
      FATAL_ERROR
      "AUTAN_COMMIT must be 40 lowercase hexadecimal characters, got '${AUTAN_COMMIT}'"
    )
  endif()
  set(commit "${AUTAN_COMMIT}")
  set(dirty false)
endif()
if(commit STREQUAL "")
  set(commit unknown)
  message(
    WARNING
    "no git checkout at ${SOURCE_DIR} and no AUTAN_COMMIT: benchmarks record commit 'unknown'"
  )
endif()
file(
  CONFIGURE
  OUTPUT ${OUTPUT_FILE}
  CONTENT "#define AUTAN_BENCH_COMMIT \"${commit}\"\n#define AUTAN_BENCH_DIRTY \"${dirty}\"\n"
)
