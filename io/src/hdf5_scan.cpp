#include "hdf5_scan.hpp"

#include <string>

#include <hdf5.h>
#include <nlohmann/json.hpp>

#include "autan/core/error.hpp"

namespace autan::io {

namespace {

// The first node whose data lives in another file, as an absolute path in the file.
struct Scan {
  std::string outside;
};

// H5Lvisit2 visits every link once, by name, without following soft or
// external links (HDF5 reference manual, H5Lvisit2).
herr_t visit(hid_t group, const char* name, const H5L_info2_t* info, void* data) {
  auto& scan = *static_cast<Scan*>(data);
  const auto found = [&] {
    scan.outside = std::string("/") + name;
    return 1;  // stops the iteration
  };
  if (info->type == H5L_TYPE_EXTERNAL || info->type >= H5L_TYPE_UD_MIN) return found();
  if (info->type != H5L_TYPE_HARD) return 0;
  H5O_info2_t object{};
  if (H5Oget_info_by_name3(group, name, &object, H5O_INFO_BASIC, H5P_DEFAULT) < 0) return -1;
  if (object.type != H5O_TYPE_DATASET) return 0;
  const hid_t dataset = H5Dopen2(group, name, H5P_DEFAULT);
  if (dataset < 0) return -1;
  const hid_t plist = H5Dget_create_plist(dataset);
  const bool elsewhere =
      plist >= 0 && (H5Pget_layout(plist) == H5D_VIRTUAL || H5Pget_external_count(plist) > 0);
  if (plist >= 0) H5Pclose(plist);
  H5Dclose(dataset);
  return elsewhere ? found() : 0;
}

}  // namespace

void reject_outside_data(const std::filesystem::path& file) {
  htri_t hdf5 = 0;
  H5E_BEGIN_TRY { hdf5 = H5Fis_accessible(file.string().c_str(), H5P_DEFAULT); }
  H5E_END_TRY
  if (hdf5 <= 0) return;
  Scan scan;
  herr_t status = 0;
  H5E_BEGIN_TRY {
    const hid_t f = H5Fopen(file.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    status = f < 0 ? -1 : H5Lvisit2(f, H5_INDEX_NAME, H5_ITER_NATIVE, visit, &scan);
    if (f >= 0) H5Fclose(f);
  }
  H5E_END_TRY
  if (status < 0) {
    throw Error(ErrorCode::io_cannot_open,
                "cannot read the HDF5 structure of '" + file.string() + "'");
  }
  if (!scan.outside.empty()) {
    throw Error(ErrorCode::io_external_reference,
                file.string() + ": " + scan.outside + " keeps its data in another file",
                nlohmann::json{{"node", scan.outside}}.dump());
  }
}

}  // namespace autan::io
