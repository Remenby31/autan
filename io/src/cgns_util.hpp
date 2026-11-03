#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cgns_io.h>
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

/// The ADF/HDF5 node tree of an open CGNS file, read through cgio: names, labels and sizes of
/// nodes without their data, and a text node's data only when asked.
class CgioTree {
 public:
  explicit CgioTree(int fn) {
    cgns_check(cg_get_cgio(fn, &cgio_), "open the node tree");
    cgns_check(cg_root_id(fn, &root_), "find the root node");
  }
  /// The file opened through cgio alone: unlike cg_open, nothing is loaded until asked for.
  explicit CgioTree(const std::filesystem::path& file) : owned_(true) {
    if (cgio_open_file(file.string().c_str(), CGIO_MODE_READ, CGIO_FILE_NONE, &cgio_) != CG_OK) {
      char message[CGIO_MAX_ERROR_LENGTH + 1] = {};
      cgio_error_message(message);
      throw Error(ErrorCode::io_cannot_open, "cannot open '" + file.string() + "': " + message);
    }
    check(cgio_get_root_id(cgio_, &root_), "find the root node");
  }
  CgioTree(const CgioTree&) = delete;
  CgioTree& operator=(const CgioTree&) = delete;
  ~CgioTree() {
    if (owned_) cgio_close_file(cgio_);
  }
  [[nodiscard]] double root() const { return root_; }

  [[nodiscard]] std::int64_t number_of_children(double node) const {
    int n = 0;
    check(cgio_number_children(cgio_, node, &n), "count children");
    return n;
  }
  [[nodiscard]] std::vector<double> children(double node) const {
    int n = 0;
    check(cgio_number_children(cgio_, node, &n), "count children");
    std::vector<double> ids(static_cast<std::size_t>(n));
    int got = 0;
    if (n > 0) check(cgio_children_ids(cgio_, node, 1, n, &got, ids.data()), "list children");
    ids.resize(static_cast<std::size_t>(got));
    return ids;
  }
  [[nodiscard]] std::string name(double node) const {
    std::array<char, CGIO_MAX_NAME_LENGTH + 1> text{};
    check(cgio_get_name(cgio_, node, text.data()), "read a node name");
    return text.data();
  }
  [[nodiscard]] std::string label(double node) const {
    std::array<char, CGIO_MAX_LABEL_LENGTH + 1> text{};
    check(cgio_get_label(cgio_, node, text.data()), "read a node label");
    return text.data();
  }
  /// The first child of `node` with this label (and name, when given).
  [[nodiscard]] std::optional<double> child(double node, std::string_view label_,
                                            std::string_view name_ = {}) const {
    for (const double id : children(node)) {
      if (label(id) == label_ && (name_.empty() || name(id) == name_)) return id;
    }
    return std::nullopt;
  }
  /// The bytes of `node`'s data as stored uncompressed: its element count times the size of its
  /// data type, from the metadata only.
  [[nodiscard]] std::int64_t logical_bytes(double node) const {
    std::array<char, CGIO_MAX_DATATYPE_LENGTH + 1> type{};
    check(cgio_get_data_type(cgio_, node, type.data()), "read a data type");
    const std::string_view t(type.data());
    const std::int64_t size = t == "C1" || t == "B1"                ? 1
                              : t == "I4" || t == "U4" || t == "R4" ? 4
                              : t == "I8" || t == "U8" || t == "R8" ? 8
                              : t == "X4"                           ? 8
                              : t == "X8"                           ? 16
                                                                    : 0;
    if (size == 0) return 0;
    int dims = 0;
    std::array<cgsize_t, CGIO_MAX_DIMENSIONS> extent{};
    check(cgio_get_dimensions(cgio_, node, &dims, extent.data()), "read dimensions");
    std::int64_t count = dims > 0 ? 1 : 0;
    for (int d = 0; d < dims; ++d)
      count *= static_cast<std::int64_t>(extent[static_cast<std::size_t>(d)]);
    return count * size;
  }
  /// The first values of an integer node, as 64-bit integers.
  template <std::size_t N>
  void read_integers(double node, std::array<std::int64_t, N>& out) const {
    std::vector<std::int64_t> all(static_cast<std::size_t>(logical_bytes(node) / 4) + N);
    check(cgio_read_all_data_type(cgio_, node, "I8", all.data()), "read an integer node");
    for (std::size_t i = 0; i < N; ++i) out[i] = all[i];
  }
  /// The number of values of `node`'s data, from the metadata only.
  [[nodiscard]] std::int64_t values(double node) const {
    int dims = 0;
    std::array<cgsize_t, CGIO_MAX_DIMENSIONS> extent{};
    check(cgio_get_dimensions(cgio_, node, &dims, extent.data()), "read dimensions");
    std::int64_t count = dims > 0 ? 1 : 0;
    for (int d = 0; d < dims; ++d)
      count *= static_cast<std::int64_t>(extent[static_cast<std::size_t>(d)]);
    return count;
  }
  /// Values [first, first + out.size()) of a one-dimensional integer node, as 64-bit integers,
  /// read alone; E0132 for a node of another rank.
  void read_integers(double node, std::int64_t first, std::span<cgsize_t> out) const {
    static_assert(sizeof(cgsize_t) == 8);
    if (out.empty()) return;
    int dims = 0;
    std::array<cgsize_t, CGIO_MAX_DIMENSIONS> extent{};
    check(cgio_get_dimensions(cgio_, node, &dims, extent.data()), "read dimensions");
    // One-dimensional, or an IndexArray_t of index dimension 1: extents (1, n).
    if (dims != 1 && !(dims == 2 && extent[0] == 1)) {
      throw Error(ErrorCode::io_malformed, "CGNS: '" + name(node) + "' is not 1-D");
    }
    // cgio reads one start, end and stride per dimension of the node: every one is given.
    std::array<cgsize_t, CGIO_MAX_DIMENSIONS> start{}, end{}, stride{};
    start.fill(1);
    end.fill(1);
    stride.fill(1);
    const std::size_t along = dims == 1 ? 0 : 1;
    start[along] = first + 1;
    end[along] = first + static_cast<cgsize_t>(out.size());
    const cgsize_t size = static_cast<cgsize_t>(out.size()), one = 1;
    check(cgio_read_data_type(cgio_, node, start.data(), end.data(), stride.data(), "I8", 1, &size,
                              &one, &size, &one, out.data()),
          "read an integer node");
  }
  /// The data of a C1 node as text.
  [[nodiscard]] std::string text(double node) const {
    std::string out(static_cast<std::size_t>(logical_bytes(node)), '\0');
    if (!out.empty())
      check(cgio_read_all_data_type(cgio_, node, "C1", out.data()), "read a text node");
    return out;
  }

 private:
  void check(int status, const char* what) const {
    if (status != CG_OK) {
      char message[CGIO_MAX_ERROR_LENGTH + 1] = {};
      cgio_error_message(message);
      throw Error(ErrorCode::io_malformed, std::string("CGNS: cannot ") + what + ": " + message);
    }
  }
  int cgio_ = 0;
  double root_ = 0.0;
  bool owned_ = false;
};

// The CGNS library allocates for every node of the tree when it opens a file, before the reader
// can drop any, and HDF5 keeps every open node's path: at most 1 MiB + 16 KiB a node, read_mesh
// included, in a tree of at most 16 levels (mesh.md, Memory of a CGNS read). A file of more nodes
// than four times the most a real grid holds (CRM-HL: 4032), or more than three times as deep as
// every real grid (5 levels), is refused.
constexpr std::int64_t max_cgns_nodes = 16384;
constexpr int max_cgns_depth = 16;

// The nodes of a tree by label, as far as the walk went: it stops once it passes max_cgns_nodes
// or max_cgns_depth, each node's children counted before their ids are read, so that walking
// costs O(max_cgns_nodes) nodes of at most max_cgns_depth levels.
struct TreeCount {
  std::map<std::string, std::int64_t> labels;
  std::int64_t nodes = 0;
  bool too_deep = false;
  [[nodiscard]] bool beyond() const { return too_deep || nodes > max_cgns_nodes; }
};

inline TreeCount count_nodes(const CgioTree& tree) {
  TreeCount out;
  std::vector<std::pair<double, int>> pending{{tree.root(), 0}};
  while (!pending.empty()) {
    const auto [node, depth] = pending.back();
    pending.pop_back();
    const std::int64_t n = tree.number_of_children(node);
    if (n == 0) continue;
    if (depth == max_cgns_depth) {
      out.too_deep = true;
      return out;
    }
    if (out.nodes + n > max_cgns_nodes) {
      out.nodes = max_cgns_nodes + 1;
      return out;
    }
    for (const double child : tree.children(node)) {
      ++out.labels[tree.label(child)];
      ++out.nodes;
      pending.emplace_back(child, depth + 1);
    }
  }
  return out;
}

}  // namespace autan::io
