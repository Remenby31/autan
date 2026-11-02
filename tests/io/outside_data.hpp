#pragma once

// The forms in which a CGNS file keeps data in another file (mesh.md, E0134),
// built on a file the solver wrote, for the readers' tests.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cgnslib.h>
#include <hdf5.h>

namespace autan_test {

enum class Outside { cgns_link, external_link, external_storage, virtual_dataset };

inline const char* outside_name(Outside form) {
  switch (form) {
    case Outside::cgns_link:
      return "CGNS link";
    case Outside::external_link:
      return "HDF5 external link";
    case Outside::external_storage:
      return "external storage";
    case Outside::virtual_dataset:
      return "virtual dataset";
  }
  return "";
}

// The HDF5 path of the first dataset of `file` whose path ends with `suffix`.
inline std::string hdf5_path(const std::filesystem::path& file, const std::string& suffix) {
  const hid_t f = H5Fopen(file.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  REQUIRE(f >= 0);
  std::vector<std::string> names;
  REQUIRE(H5Lvisit2(
              f, H5_INDEX_NAME, H5_ITER_NATIVE,
              [](hid_t, const char* name, const H5L_info2_t*, void* out) -> herr_t {
                static_cast<std::vector<std::string>*>(out)->emplace_back(name);
                return 0;
              },
              &names) >= 0);
  H5Fclose(f);
  const auto it =
      std::ranges::find_if(names, [&](const std::string& n) { return n.ends_with(suffix); });
  REQUIRE(it != names.end());
  return "/" + *it;
}

// The doubles of dataset `node` of `file`, read through HDF5, which follows
// every form: the control that the file really takes its data from `donor`.
inline std::vector<double> read_doubles(const std::filesystem::path& file,
                                        const std::string& node) {
  const hid_t f = H5Fopen(file.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  REQUIRE(f >= 0);
  const hid_t d = H5Dopen2(f, node.c_str(), H5P_DEFAULT);
  REQUIRE(d >= 0);
  const hid_t space = H5Dget_space(d);
  std::vector<double> values(static_cast<std::size_t>(H5Sget_simple_extent_npoints(space)));
  REQUIRE(H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
  H5Sclose(space);
  H5Dclose(d);
  H5Fclose(f);
  return values;
}

// Moves data of `file` (a solution.cgns, HDF5) into `donor`, filled with
// `value`, in the given form; returns the node the readers must name, and in
// `control` the dataset through which HDF5 reads the donor's values.
inline std::string move_outside(const std::filesystem::path& file, Outside form,
                                const std::filesystem::path& donor, double value,
                                std::string& control) {
  const std::string x = hdf5_path(file, "GridCoordinates/CoordinateX/ data");
  const std::string zone = x.substr(0, x.find("/GridCoordinates"));
  const hid_t probe = H5Fopen(file.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  const hid_t old = H5Dopen2(probe, x.c_str(), H5P_DEFAULT);
  const hid_t space = H5Dget_space(old);
  const auto n = static_cast<std::size_t>(H5Sget_simple_extent_npoints(space));
  H5Dclose(old);
  H5Fclose(probe);
  const std::vector<double> values(n, value);
  if (form == Outside::cgns_link) {
    H5Sclose(space);
    // The donor is a copy of the file holding `value`; the link replaces nothing the readers
    // need. HDF5 reaches the donor through the link's " link" child (CGNS ADFH).
    std::filesystem::copy_file(file, donor);
    {
      const hid_t d = H5Fopen(donor.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
      const hid_t xd = H5Dopen2(d, x.c_str(), H5P_DEFAULT);
      REQUIRE(H5Dwrite(xd, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
      H5Dclose(xd);
      H5Fclose(d);
    }
    int fn = 0;
    REQUIRE(cg_open(file.string().c_str(), CG_MODE_MODIFY, &fn) == CG_OK);
    REQUIRE(cg_goto(fn, 1, "Zone_t", 1, "end") == CG_OK);
    REQUIRE(cg_link_write("Elsewhere", donor.string().c_str(),
                          (zone + "/GridCoordinates").c_str()) == CG_OK);
    REQUIRE(cg_close(fn) == CG_OK);
    control = zone + "/Elsewhere/ link/CoordinateX/ data";
    return zone + "/Elsewhere";
  }
  if (form == Outside::external_storage) {
    std::ofstream(donor, std::ios::binary)
        .write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(n * sizeof(double)));
  } else {
    const hid_t d = H5Fcreate(donor.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    const hid_t v =
        H5Dcreate2(d, "/values", H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(H5Dwrite(v, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
    H5Dclose(v);
    H5Fclose(d);
  }
  const hid_t f = H5Fopen(file.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(f >= 0);
  std::string node = x;
  if (form == Outside::external_link) {
    node = zone + "/Elsewhere";
    REQUIRE(H5Lcreate_external(donor.string().c_str(), "/values", f, node.c_str(), H5P_DEFAULT,
                               H5P_DEFAULT) >= 0);
  } else {
    REQUIRE(H5Ldelete(f, x.c_str(), H5P_DEFAULT) >= 0);
    const hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    if (form == Outside::external_storage) {
      REQUIRE(H5Pset_external(plist, donor.string().c_str(), 0,
                              static_cast<hsize_t>(n * sizeof(double))) >= 0);
    } else {
      REQUIRE(H5Pset_virtual(plist, space, donor.string().c_str(), "/values", space) >= 0);
    }
    const hid_t moved =
        H5Dcreate2(f, x.c_str(), H5T_IEEE_F64LE, space, H5P_DEFAULT, plist, H5P_DEFAULT);
    REQUIRE(moved >= 0);
    H5Dclose(moved);
    H5Pclose(plist);
  }
  H5Sclose(space);
  H5Fclose(f);
  control = node;
  return node;
}

}  // namespace autan_test
