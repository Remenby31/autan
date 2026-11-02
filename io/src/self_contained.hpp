#pragma once

#include <cstdint>
#include <filesystem>

namespace autan::io {

/// Throws E0134, `data.node` naming the first node, when the CGNS file `file`
/// keeps data in another file: a CGNS link to another file (HDF5 or ADF), an
/// HDF5 external or user-defined link, a dataset with external storage, or a
/// virtual dataset. Reads the file's structure only, before the CGNS library
/// opens it, so that nothing is read from elsewhere; HDF5 plugins are not
/// loaded. A structure it cannot read in full is E0130.
void reject_outside_data(const std::filesystem::path& file);

/// The largest twice-logical plus stored size of a filtered (compressed) dataset of `file`, 0 when
/// it has none: what HDF5 holds beside the destination while it reads one (mesh.md, "Memory of a
/// CGNS read"). Reads the structure only, plugins unloaded; E0130 when it cannot.
std::int64_t largest_filtered_bytes(const std::filesystem::path& file);

}  // namespace autan::io
