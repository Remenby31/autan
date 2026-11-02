#pragma once

#include <filesystem>

namespace autan::io {

/// Throws E0134, `data.node` naming the first node, when the HDF5 file
/// `file` keeps data in another file: an external or user-defined link, a
/// dataset with external storage, or a virtual dataset. Reads the file's
/// structure only, before the CGNS library opens it, so that nothing is read
/// from elsewhere. A file that is not HDF5 (CGNS/ADF) passes.
void reject_outside_data(const std::filesystem::path& file);

}  // namespace autan::io
