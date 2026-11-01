#pragma once

#include <filesystem>
#include <string>

#include <cgnslib.h>

#include "autan/core/error.hpp"

namespace autan::io {

/// Throws E0130 (cannot open, read or write) with the CGNS library message.
inline void cgns_check(int status, const std::string& what, const std::string& name = "") {
  if (status != CG_OK) {
    throw Error(
        ErrorCode::io_cannot_open,
        "CGNS: cannot " + what + (name.empty() ? "" : " '" + name + "'") + ": " + cg_get_error());
  }
}

/// An open CGNS file, closed on destruction. Writing creates HDF5 files.
class CgnsFile {
 public:
  CgnsFile(const std::filesystem::path& path, int mode) : path_(path.string()) {
    if (mode == CG_MODE_WRITE) cgns_check(cg_set_file_type(CG_FILE_HDF5), "select HDF5");
    cgns_check(cg_open(path_.c_str(), mode, &handle_), "open", path_);
  }
  CgnsFile(const CgnsFile&) = delete;
  CgnsFile& operator=(const CgnsFile&) = delete;
  ~CgnsFile() {
    if (handle_ >= 0) cg_close(handle_);
  }
  [[nodiscard]] int handle() const { return handle_; }
  [[nodiscard]] const std::string& path() const { return path_; }
  void close() {
    const int h = handle_;
    handle_ = -1;
    cgns_check(cg_close(h), "close", path_);
  }

 private:
  std::string path_;
  int handle_ = -1;
};

}  // namespace autan::io
