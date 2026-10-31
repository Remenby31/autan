// CUDA-aware MPI check: MPI_Allreduce on device buffers. Run with mpirun -n 2 on a GPU host.
// Exits 77 when no GPU is visible so callers can tell a skip from a pass.
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <mpi.h>

// Open MPI extensions (MPIX_Query_cuda_support) need mpi.h first.
#include <mpi-ext.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    if (rank == 0) std::printf("smoke_cuda: SKIPPED (no GPU visible)\n");
    MPI_Finalize();
    return 77;
  }
  if (MPIX_Query_cuda_support() != 1) {
    if (rank == 0) std::fprintf(stderr, "FAIL: MPI library reports no CUDA support at run time\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  constexpr int n = 1 << 20;
  double* buf = nullptr;
  cudaMalloc(&buf, n * sizeof(double));
  const double value = rank + 1.0;
  double* host = static_cast<double*>(std::malloc(n * sizeof(double)));
  for (int i = 0; i < n; ++i) host[i] = value;
  cudaMemcpy(buf, host, n * sizeof(double), cudaMemcpyHostToDevice);
  MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  cudaMemcpy(host, buf, n * sizeof(double), cudaMemcpyDeviceToHost);

  const double want = size * (size + 1) / 2.0;
  int bad = 0;
  for (int i = 0; i < n; ++i) bad += host[i] != want;
  int total = 0;
  MPI_Allreduce(&bad, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf(total == 0 ? "smoke_cuda: OK (%d ranks)\n" : "smoke_cuda: FAILED\n", size);
  std::free(host);
  cudaFree(buf);
  MPI_Finalize();
  return total == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
