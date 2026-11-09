# Fails when the probe's PTX holds an fma instruction: the CUDA units are contracted (#571).
file(READ "${PTX}" ptx)
string(REGEX MATCHALL "fma\\.rn\\.f(32|64)" found "${ptx}")
if(found)
  message(
    FATAL_ERROR
    "the probe's PTX holds ${found}: CUDA units are compiled with FMA contraction"
  )
endif()
string(FIND "${ptx}" "mul.rn.f64" mul)
if(mul EQUAL -1)
  message(FATAL_ERROR "the probe's PTX holds no mul.rn.f64: it did not compile what it should")
endif()
message(STATUS "no fma in ${PTX}")
