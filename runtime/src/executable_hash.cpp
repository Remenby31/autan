#include "executable_hash.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <picosha2.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace autan::runtime {
namespace {

std::string hash_executable() {
#ifdef __APPLE__
  std::uint32_t size = 0;
  (void)_NSGetExecutablePath(nullptr, &size);
  std::string path(size, '\0');
  if (_NSGetExecutablePath(path.data(), &size) != 0)
    throw std::runtime_error("cannot locate the running executable");
  path.resize(std::char_traits<char>::length(path.c_str()));
#else
  const std::string path = "/proc/self/exe";
#endif
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read the running executable '" + path + "'");
  picosha2::hash256_one_by_one hasher;
  std::vector<char> buffer(1 << 20);
  while (in) {
    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    hasher.process(buffer.begin(), buffer.begin() + in.gcount());
  }
  if (in.bad()) throw std::runtime_error("cannot read the running executable '" + path + "'");
  hasher.finish();
  return picosha2::get_hash_hex_string(hasher);
}

}  // namespace

std::string executable_sha256() {
  // A static whose initialiser throws is initialised again by the next call.
  static const std::string hash = hash_executable();
  return hash;
}

}  // namespace autan::runtime
