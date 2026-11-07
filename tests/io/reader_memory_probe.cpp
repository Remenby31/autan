// Measures what reading a mesh costs in memory beyond the mesh it returns (#358): one file per
// run, in a fresh process. Linux only: the peak resident size during read_mesh (VmHWM, reset
// through /proc/self/clear_refs), and the bytes malloc holds after it (mallinfo2), which the
// returned Mesh accounts for. Prints one JSON line.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

#include <malloc.h>
#include <nlohmann/json.hpp>

#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"

namespace {

// A field of /proc/self/status, in bytes.
std::int64_t status_bytes(const std::string& key) {
  std::ifstream in("/proc/self/status");
  for (std::string line; std::getline(in, line);) {
    if (line.rfind(key + ":", 0) == 0) return std::stoll(line.substr(key.size() + 1)) * 1024;
  }
  return -1;
}

std::int64_t in_use() {
  const struct mallinfo2 m = mallinfo2();
  return static_cast<std::int64_t>(m.uordblks + m.hblkhd);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fputs("usage: reader_memory_probe MESH\n", stderr);
    return 2;
  }
  const std::int64_t bytes = autan::io::mesh_bytes(argv[1]);
  const std::int64_t rss = status_bytes("VmRSS");
  const std::int64_t held = in_use();
  std::ofstream("/proc/self/clear_refs") << "5";  // VmHWM := VmRSS
  const autan::Mesh mesh = autan::io::read_mesh(argv[1]);
  const std::int64_t peak = status_bytes("VmHWM") - rss;
  const std::int64_t kept = in_use() - held;
  const nlohmann::json out{{"mesh", argv[1]},
                           {"format", mesh.source().format},
                           {"source_bytes", bytes},
                           {"cells", mesh.n_cells()},
                           {"faces", mesh.n_faces()},
                           {"points", mesh.n_points()},
                           {"peak_bytes", peak},
                           {"mesh_bytes", kept},
                           {"beyond_mesh_bytes", peak - kept}};
  std::printf("%s\n", out.dump().c_str());
  return 0;
}
