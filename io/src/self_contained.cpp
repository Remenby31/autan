#include "self_contained.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <H5PLpublic.h>
#include <cgns_io.h>
#include <cgnslib.h>
#include <hdf5.h>
#include <nlohmann/json.hpp>

#include "autan/core/error.hpp"
#include "cgns_util.hpp"

namespace autan::io {

namespace {

[[noreturn]] void outside(const std::filesystem::path& file, const std::string& node) {
  // Node names are bytes: the message and the data replace what is not UTF-8.
  const auto text = [](const nlohmann::json& j) {
    return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
  };
  throw Error(ErrorCode::io_external_reference,
              file.string() + ": node " + text(node) + " keeps its data in another file",
              text(nlohmann::json{{"node", node}}));
}

[[noreturn]] void unreadable(const std::filesystem::path& file) {
  throw Error(ErrorCode::io_cannot_open,
              "cannot read the structure of '" + file.string() + "' in full");
}

// The first node whose data lives in another file, as an absolute path in the file.
struct Scan {
  bool found = false;
  std::string outside;
};

// H5Lvisit2 visits every link once, by name, without following soft or
// external links (HDF5 reference manual, H5Lvisit2).
herr_t visit(hid_t group, const char* name, const H5L_info2_t* info, void* data) {
  auto& scan = *static_cast<Scan*>(data);
  const auto found = [&] {
    // CGNS keeps a link as an HDF5 external link named " link" in the node's
    // group (CGNS ADFH): the node is that group.
    std::string node = std::string("/") + name;
    if (node.ends_with("/ link")) node.resize(node.size() - 6);
    scan.found = true;
    scan.outside = node.empty() ? "/" : node;
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
  // plugin, code from the environment, is loaded for the scan.
  unsigned int plugins = 0;
  H5PLget_loading_state(&plugins);
  H5PLset_loading_state(0);
  Scan scan;
  herr_t status = 0;
  H5E_BEGIN_TRY {
    const hid_t f = H5Fopen(file.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    status = f < 0 ? -1 : H5Lvisit2(f, H5_INDEX_NAME, H5_ITER_NATIVE, visit, &scan);
    if (f >= 0) H5Fclose(f);
  }
  H5E_END_TRY
  H5PLset_loading_state(plugins);
  if (scan.found) outside(file, scan.outside);
  if (status < 0) unreadable(file);
}

// CGNS links (HDF5 or ADF), through the raw node tree, before the library resolves any.
void scan_cgns_links(const std::filesystem::path& file) {
  int cgio = 0;
  if (cgio_open_file(file.string().c_str(), CGIO_MODE_READ, CGIO_FILE_NONE, &cgio) != CG_OK) {
    throw Error(ErrorCode::io_cannot_open, "cannot open '" + file.string() + "' as CGNS");
  }
  // Each node visited once, with the index of its parent: a path is built only for the
  // diagnostic. The walk stops after max_cgns_nodes, which the reader refuses (cgns_util.hpp).
  struct Frame {
    double id;
    std::size_t parent;
  };
  constexpr std::size_t none = static_cast<std::size_t>(-1);
  std::vector<Frame> visited;
  std::size_t found = none;
  bool broken = false;
  double root = 0;
  cgio_get_root_id(cgio, &root);
  std::vector<Frame> pending{{root, none}};
  std::vector<double> children;
  while (!pending.empty() && found == none && !broken &&
         static_cast<std::int64_t>(visited.size()) <= max_cgns_nodes) {
    const Frame frame = pending.back();
    pending.pop_back();
    const double node = frame.id;
    visited.push_back(frame);
    int length = 0;
    if (cgio_is_link(cgio, node, &length) != CG_OK) {
      broken = true;
      break;
    }
    if (length > 0) {
      int file_length = 0, name_length = 0;
      cgio_link_size(cgio, node, &file_length, &name_length);
      if (file_length > 0) found = visited.size() - 1;
      continue;
    }
    int count = 0;
    cgio_number_children(cgio, node, &count);
    if (count == 0) continue;
    if (static_cast<std::int64_t>(visited.size()) + count > max_cgns_nodes + 1) break;
    children.resize(static_cast<std::size_t>(count));
    int got = 0;
    cgio_children_ids(cgio, node, 1, count, &got, children.data());
    for (int i = 0; i < got; ++i) {
      pending.push_back({children[static_cast<std::size_t>(i)], visited.size() - 1});
    }
  }
  std::string path;
  for (std::size_t f = found; f != none && visited[f].parent != none; f = visited[f].parent) {
    char name[CGIO_MAX_NAME_LENGTH + 1] = {};
    cgio_get_name(cgio, visited[f].id, name);
    path.insert(0, std::string("/") + name);
  }
  cgio_close_file(cgio);
  if (found != none) outside(file, path.empty() ? "/" : path);
  if (broken) unreadable(file);
}

}  // namespace

void reject_outside_data(const std::filesystem::path& file) {
  scan_hdf5(file);
  scan_cgns_links(file);
}

std::int64_t largest_filtered_bytes(const std::filesystem::path& file) {
  struct Largest {
    std::int64_t bytes = 0;
    bool failed = false;
  } largest;
  const auto visit = [](hid_t object, const char* name, const H5O_info2_t* info, void* data) {
    auto& out = *static_cast<Largest*>(data);
    if (info->type != H5O_TYPE_DATASET) return 0;
    const hid_t dataset = H5Dopen2(object, name, H5P_DEFAULT);
    if (dataset < 0) {
      out.failed = true;
      return -1;
    }
    const hid_t properties = H5Dget_create_plist(dataset);
    if (properties >= 0 && H5Pget_nfilters(properties) > 0) {
      // A filter applies to whole chunks: a read inflates each chunk it touches into a buffer of
      // the chunk's logical size, whatever part of it the dataset's extent covers.
      const hid_t type = H5Dget_type(dataset);
      std::array<hsize_t, H5S_MAX_RANK> chunk{};
      const int rank = H5Pget_chunk(properties, H5S_MAX_RANK, chunk.data());
      std::int64_t logical = rank > 0 ? static_cast<std::int64_t>(H5Tget_size(type)) : 0;
      for (int d = 0; d < rank; ++d)
        logical *= static_cast<std::int64_t>(chunk[static_cast<std::size_t>(d)]);
      const auto stored = static_cast<std::int64_t>(H5Dget_storage_size(dataset));
      out.bytes = std::max(out.bytes, 2 * logical + stored);
      H5Tclose(type);
    }
    if (properties >= 0) H5Pclose(properties);
    H5Dclose(dataset);
    return 0;
  };
  unsigned int plugins = 0;
  H5PLget_loading_state(&plugins);
  H5PLset_loading_state(0);
  herr_t status = 0;
  H5E_BEGIN_TRY {
    const hid_t f = H5Fopen(file.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    status =
        f < 0 ? -1 : H5Ovisit3(f, H5_INDEX_NAME, H5_ITER_NATIVE, visit, &largest, H5O_INFO_BASIC);
    if (f >= 0) H5Fclose(f);
  }
  H5E_END_TRY
  H5PLset_loading_state(plugins);
  if (status < 0 || largest.failed) unreadable(file);
  return largest.bytes;
}

}  // namespace autan::io
