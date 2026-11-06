#pragma once

#include <string>

namespace autan::runtime {

// results.md, run.json `solver.binary_sha256`: the SHA-256 of the running executable, into
// which the solver's code links statically, as lower-case hex. Read once per process: a running
// executable's file cannot change under it. When AUTAN_BINARY_HASH_CACHE names a directory (the
// test suites set it in the build tree), the hash is also kept there across processes, keyed by
// the file's device, inode, size, mtime and ctime. Throws std::runtime_error when the executable
// cannot be read; the next call tries again.
std::string executable_sha256();

}  // namespace autan::runtime
