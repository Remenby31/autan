# Runs TOOL --surface OUT MESH and checks the file it writes: exit status 0,
# no output, and an autan.surface/1 header after its u32 length:
#   cmake -DTOOL=... -DMESH=... -DOUT=... -P surface_cli.cmake
file(REMOVE ${OUT})
execute_process(
  COMMAND ${TOOL} --surface ${OUT} ${MESH}
  RESULT_VARIABLE code
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err
)
if(NOT code EQUAL 0 OR NOT "${out}${err}" STREQUAL "")
  message(FATAL_ERROR "exit status ${code}, output:\n${out}${err}")
endif()
if(NOT EXISTS ${OUT})
  message(FATAL_ERROR "${OUT} was not written")
endif()
file(READ ${OUT} head LIMIT 64 HEX)
# "autan.surface/1" in hex, after the 4-byte length and '{"schema":"'.
if(NOT head MATCHES "^........7b22736368656d61223a22617574616e2e737572666163652f3122")
  message(FATAL_ERROR "${OUT} does not start with an autan.surface/1 header: ${head}")
endif()
