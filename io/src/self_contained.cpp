#include "self_contained.hpp"

#include <string>
#include <utility>
#include <vector>

#include <H5PLpublic.h>
#include <cgns_io.h>
#include <cgnslib.h>
#include <hdf5.h>
#include <nlohmann/json.hpp>

#include "autan/core/error.hpp"

namespace autan::io {

namespace {

[[noreturn]] void outside(const std::filesystem::path& file, const std::string& node) {
  // Node names are bytes: the diagnostic replaces what is not UTF-8.
  throw Error(ErrorCode::io_external_reference,
              file.string() + ": " + node + " keeps its data in another file",
              nlohmann::json{{"node", node}}.dump(-1, ' ', false,
                                                  nlohmann::json::error_handler_t::replace));
}

[[noreturn]] void unreadable(const std::filesystem::path& file) {
  throw Error(ErrorCode::io_cannot_open,
              "cannot read the structure of '" + file.string() + "' in full");
}

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
  H5Dclose(dataset);
  if (plist < 0) return -1;  // a dataset whose storage cannot be told is not certified
  const bool elsewhere = H5Pget_layout(plist) == H5D_VIRTUAL || H5Pget_external_count(plist) > 0;
  H5Pclose(plist);
  return elsewhere ? found() : 0;
}

void scan_hdf5(const std::filesystem::path& file) {
  htri_t hdf5 = 0;
  H5E_BEGIN_TRY { hdf5 = H5Fis_accessible(file.string().c_str(), H5P_DEFAULT); }
  H5E_END_TRY
  if (hdf5 <= 0) return;
  // A filtered group decodes through its filter while it is visited: no
  // plugin, code from the environment, is loaded to do it. The files the
  // readers take use no plugin filter.
  H5PLset_loading_state(0);
  Scan scan;
  herr_t status = 0;
  H5E_BEGIN_TRY {
    const hid_t f = H5Fopen(file.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    status = f < 0 ? -1 : H5Lvisit2(f, H5_INDEX_NAME, H5_ITER_NATIVE, visit, &scan);
    if (f >= 0) H5Fclose(f);
  }
  H5E_END_TRY
  if (!scan.outside.empty()) outside(file, scan.outside);
  if (status < 0) unreadable(file);
}

// CGNS links (HDF5 or ADF), through the raw node tree, before the library resolves any.
void scan_cgns_links(const std::filesystem::path& file) {
  int cgio = 0;
  if (cgio_open_file(file.string().c_str(), CGIO_MODE_READ, CGIO_FILE_NONE, &cgio) != CG_OK) {
    throw Error(ErrorCode::io_cannot_open, "cannot open '" + file.string() + "' as CGNS");
  }
  std::string found;
  bool broken = false;
  double root = 0;
  cgio_get_root_id(cgio, &root);
  std::vector<std::pair<double, std::string>> pending{{root, ""}};
  std::vector<double> children;
  while (!pending.empty() && found.empty() && !broken) {
    const auto [node, path] = pending.back();
    pending.pop_back();
    int length = 0;
    if (cgio_is_link(cgio, node, &length) != CG_OK) {
      broken = true;
      break;
    }
    if (length > 0) {
      int file_length = 0, name_length = 0;
      cgio_link_size(cgio, node, &file_length, &name_length);
      if (file_length > 0) found = path.empty() ? "/" : path;
      continue;
    }
    int count = 0;
    cgio_number_children(cgio, node, &count);
    if (count == 0) continue;
    children.resize(static_cast<std::size_t>(count));
    int got = 0;
    cgio_children_ids(cgio, node, 1, count, &got, children.data());
    for (int i = 0; i < got; ++i) {
      char name[CGIO_MAX_NAME_LENGTH + 1] = {};
      cgio_get_name(cgio, children[static_cast<std::size_t>(i)], name);
      pending.emplace_back(children[static_cast<std::size_t>(i)], path + "/" + name);
    }
  }
  cgio_close_file(cgio);
  if (!found.empty()) outside(file, found);
  if (broken) unreadable(file);
}

}  // namespace

void reject_outside_data(const std::filesystem::path& file) {
  scan_hdf5(file);
  scan_cgns_links(file);
}

}  // namespace autan::io
