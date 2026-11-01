# Writes OUTPUT_FILE with the commit and dirty flag the benchmark binaries record, only when they
# change. The commit is AUTAN_COMMIT when given (builds from an export, as for the runtime), else
# the git commit of SOURCE_DIR; without either it is "unknown" and the build is marked dirty, with
# a warning: provenance is then incomplete, but the build goes on.
set(commit "")
set(dirty true)
if(DEFINED AUTAN_COMMIT AND NOT AUTAN_COMMIT STREQUAL "")
  set(commit "${AUTAN_COMMIT}")
  set(dirty false)
elseif(GIT)
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
if(commit STREQUAL "")
  set(commit unknown)
  message(
    WARNING
    "no git commit for ${SOURCE_DIR} and no AUTAN_COMMIT: benchmarks record commit 'unknown'"
  )
endif()
file(
  CONFIGURE
  OUTPUT ${OUTPUT_FILE}
  CONTENT "#define AUTAN_BENCH_COMMIT \"${commit}\"\n#define AUTAN_BENCH_DIRTY \"${dirty}\"\n"
)
