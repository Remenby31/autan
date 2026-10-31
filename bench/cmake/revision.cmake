# Writes OUTPUT_FILE with the git commit and dirty flag of SOURCE_DIR, only when they change, so the
# binary records the revision it was built from. Run at every build by autan_bench_revision.
execute_process(
  COMMAND ${GIT} -C ${SOURCE_DIR} rev-parse HEAD
  OUTPUT_VARIABLE commit
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE status
)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "cannot read the git commit of ${SOURCE_DIR}: build from a git checkout")
endif()
execute_process(
  COMMAND ${GIT} -C ${SOURCE_DIR} status --porcelain --untracked-files=no
  OUTPUT_VARIABLE changes
  RESULT_VARIABLE status
)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "cannot read the git status of ${SOURCE_DIR}")
endif()
if(changes)
  set(dirty true)
else()
  set(dirty false)
endif()
file(
  CONFIGURE
  OUTPUT ${OUTPUT_FILE}
  CONTENT "#define AUTAN_BENCH_COMMIT \"${commit}\"\n#define AUTAN_BENCH_DIRTY \"${dirty}\"\n"
)
