// Image smoke test: every library the image provides is found by CMake, links, and computes
// a result fixed by its specification. Run with mpirun -n 2.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <cgnslib.h>
#include <hdf5.h>
#include <metis.h>
#include <mpi.h>
#include <pcgnslib.h>
#include <ptscotch.h>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

// 4x4 structured grid graph in CSR form.
template <class Int>
void grid_graph(std::vector<Int>& xadj, std::vector<Int>& adj) {
  const int n = 4;
  xadj.assign(1, 0);
  adj.clear();
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      if (i > 0) adj.push_back(j * n + i - 1);
      if (i < n - 1) adj.push_back(j * n + i + 1);
      if (j > 0) adj.push_back((j - 1) * n + i);
      if (j < n - 1) adj.push_back((j + 1) * n + i);
      xadj.push_back(static_cast<Int>(adj.size()));
    }
  }
}

template <class Int>
int edge_cut(const std::vector<Int>& xadj, const std::vector<Int>& adj,
             const std::vector<Int>& part) {
  int cut = 0;
  for (std::size_t v = 0; v + 1 < xadj.size(); ++v)
    for (Int e = xadj[v]; e < xadj[v + 1]; ++e) cut += part[v] != part[adj[e]];
  return cut / 2;
}

template <class Int>
bool balanced_bisection(const std::vector<Int>& part) {
  const auto ones = std::count(part.begin(), part.end(), Int{1});
  return ones == 8 && std::all_of(part.begin(), part.end(), [](Int p) { return p == 0 || p == 1; });
}

void test_mpi(int rank, int size) {
  int sum = 0;
  MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  check(sum == size * (size - 1) / 2, "MPI_Allreduce sum of ranks");
}

void test_metis() {
  std::vector<idx_t> xadj, adj;
  grid_graph(xadj, adj);
  idx_t nv = 16, ncon = 1, nparts = 2, objval = 0;
  std::vector<idx_t> part(16);
  const int status =
      METIS_PartGraphKway(&nv, &ncon, xadj.data(), adj.data(), nullptr, nullptr, nullptr, &nparts,
                          nullptr, nullptr, nullptr, &objval, part.data());
  check(status == METIS_OK, "METIS_PartGraphKway status");
  check(balanced_bisection(part), "METIS bisection is balanced");
  check(objval == edge_cut(xadj, adj, part), "METIS reported edge cut matches partition");
}

void test_scotch() {
  check(sizeof(SCOTCH_Num) == 8, "SCOTCH_Num is 64-bit");
  std::vector<SCOTCH_Num> xadj, adj;
  grid_graph(xadj, adj);
  SCOTCH_Graph graph;
  SCOTCH_Strat strat;
  check(SCOTCH_graphInit(&graph) == 0, "SCOTCH_graphInit");
  check(SCOTCH_graphBuild(&graph, 0, 16, xadj.data(), nullptr, nullptr, nullptr,
                          static_cast<SCOTCH_Num>(adj.size()), adj.data(), nullptr) == 0,
        "SCOTCH_graphBuild");
  check(SCOTCH_graphCheck(&graph) == 0, "SCOTCH_graphCheck");
  SCOTCH_stratInit(&strat);
  std::vector<SCOTCH_Num> part(16);
  check(SCOTCH_graphPart(&graph, 2, &strat, part.data()) == 0, "SCOTCH_graphPart");
  check(balanced_bisection(part), "Scotch bisection is balanced");
  SCOTCH_stratExit(&strat);
  SCOTCH_graphExit(&graph);

  SCOTCH_Dgraph dgraph;
  check(SCOTCH_dgraphInit(&dgraph, MPI_COMM_WORLD) == 0, "SCOTCH_dgraphInit (PT-Scotch)");
  SCOTCH_dgraphExit(&dgraph);
}

// Each rank writes its rank into element [rank] of a shared dataset through MPI-IO; every
// rank reads the whole dataset back and expects 0, 1, ..., size-1.
void test_hdf5(int rank, int size) {
  const char* path = "smoke.h5";
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  hsize_t dims = static_cast<hsize_t>(size);
  hid_t space = H5Screate_simple(1, &dims, nullptr);
  hid_t dset =
      H5Dcreate2(file, "ranks", H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t offset = static_cast<hsize_t>(rank), count = 1;
  H5Sselect_hyperslab(space, H5S_SELECT_SET, &offset, nullptr, &count, nullptr);
  hid_t mem = H5Screate_simple(1, &count, nullptr);
  hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
  check(H5Dwrite(dset, H5T_NATIVE_INT, mem, space, dxpl, &rank) >= 0, "H5Dwrite collective");
  H5Pclose(dxpl);
  H5Sclose(mem);
  H5Sclose(space);
  H5Dclose(dset);
  H5Fclose(file);

  file = H5Fopen(path, H5F_ACC_RDONLY, fapl);
  dset = H5Dopen2(file, "ranks", H5P_DEFAULT);
  std::vector<int> got(size, -1), want(size);
  std::iota(want.begin(), want.end(), 0);
  check(H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, got.data()) >= 0, "H5Dread");
  check(got == want, "parallel HDF5 round trip");
  H5Dclose(dset);
  H5Fclose(file);
  H5Pclose(fapl);
}

void test_cgns() {
  const char* path = "smoke.cgns";
  int fn = 0, base = 0, zone = 0;
  cgsize_t size[3] = {8, 1, 0};
  check(cgp_mpi_comm(MPI_COMM_WORLD) == CG_OK, "cgp_mpi_comm");
  check(cgp_open(path, CG_MODE_WRITE, &fn) == CG_OK, "cgp_open write");
  check(cg_base_write(fn, "Base", 3, 3, &base) == CG_OK, "cg_base_write");
  check(cg_zone_write(fn, base, "Hex", size, CGNS_ENUMV(Unstructured), &zone) == CG_OK,
        "cg_zone_write");
  check(cgp_close(fn) == CG_OK, "cgp_close");

  char name[33] = {};
  cgsize_t got[3] = {};
  check(cgp_open(path, CG_MODE_READ, &fn) == CG_OK, "cgp_open read");
  check(cg_zone_read(fn, 1, 1, name, got) == CG_OK, "cg_zone_read");
  check(got[0] == 8 && got[1] == 1, "CGNS zone size round trip");
  check(sizeof(cgsize_t) == 8, "cgsize_t is 64-bit");
  check(cgp_close(fn) == CG_OK, "cgp_close read");
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  test_mpi(rank, size);
  test_metis();
  test_scotch();
  test_hdf5(rank, size);
  test_cgns();
  int total = 0;
  MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) std::printf(total == 0 ? "smoke: OK (%d ranks)\n" : "smoke: FAILED\n", size);
  MPI_Finalize();
  return total == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
