#pragma once

#include <filesystem>

namespace autan::io {

/// Throws E0134, `data.node` naming the first node, when the CGNS file `file`
/// keeps data in another file: a CGNS link to another file (HDF5 or ADF), an
/// HDF5 external or user-defined link, a dataset with external storage, or a
/// virtual dataset. Reads the file's structure only, before the CGNS library
/// opens it, so that nothing is read from elsewhere; HDF5 plugins are not
/// loaded. A structure it cannot read in full is E0130.
void reject_outside_data(const std::filesystem::path& file);

}  // namespace autan::io
