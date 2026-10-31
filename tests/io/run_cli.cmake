# Runs TOOL with ARGS (a ;-list) and checks its exit status and output:
#   cmake -DTOOL=... -DARGS=... -DEXPECT_CODE=0 -DEXPECT=<regex> -P run_cli.cmake
execute_process(
  COMMAND ${TOOL} ${ARGS}
  RESULT_VARIABLE code
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err
)
if(NOT code EQUAL EXPECT_CODE)
  message(FATAL_ERROR "exit status ${code}, expected ${EXPECT_CODE}\n${out}${err}")
endif()
if(NOT "${out}${err}" MATCHES "${EXPECT}")
  message(FATAL_ERROR "output does not match '${EXPECT}':\n${out}${err}")
endif()
