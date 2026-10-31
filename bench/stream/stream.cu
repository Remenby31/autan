// STREAM-style device memory bandwidth benchmark (McCalpin, "Memory Bandwidth and Machine
// Balance in Current High Performance Computers", IEEE TCCA Newsletter, 1995).
// Kernels: copy c = a, scale b = q c, add c = a + b, triad a = b + q c, FP64 arrays.
// Each kernel is repeated on unchanged inputs, so the final arrays are exactly
// a = q + q (1 + q), b = q, c = 1 + q from a = 1, b = 2, c = 0; the check is bitwise.
// Prints one JSON document with per-repetition kernel times on stdout.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gpu_runtime.hpp"

namespace {

constexpr double kScalar = 3.0;

[[noreturn]] void fail(const char* code, const std::string& message) {
  std::fprintf(stderr, "error[%s]: %s\n", code, message.c_str());
  std::exit(EXIT_FAILURE);
}

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    fail("CUDA_ERROR", std::string(what) + ": " + cudaGetErrorString(status));
}

// One double2 (16-byte) load per array per thread: the widest naturally aligned access.
__global__ void init(double2* a, double2* b, double2* c, long n2) {
  const long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  if (i < n2) {
    a[i] = make_double2(1.0, 1.0);
    b[i] = make_double2(2.0, 2.0);
    c[i] = make_double2(0.0, 0.0);
  }
}

__global__ void copy(const double2* __restrict__ a, double2* __restrict__ c, long n2) {
  const long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  if (i < n2) c[i] = a[i];
}

__global__ void scale(double2* __restrict__ b, const double2* __restrict__ c, double q, long n2) {
  const long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  if (i < n2) {
    const double2 x = c[i];
    b[i] = make_double2(q * x.x, q * x.y);
  }
}

__global__ void add(const double2* __restrict__ a, const double2* __restrict__ b,
                    double2* __restrict__ c, long n2) {
  const long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  if (i < n2) {
    const double2 x = a[i], y = b[i];
    c[i] = make_double2(x.x + y.x, x.y + y.y);
  }
}

__global__ void triad(double2* __restrict__ a, const double2* __restrict__ b,
                      const double2* __restrict__ c, double q, long n2) {
  const long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  if (i < n2) {
    const double2 y = b[i], z = c[i];
    a[i] = make_double2(y.x + q * z.x, y.y + q * z.y);
  }
}

struct Options {
  long elements = 1L << 28;
  int warmup = 10;
  int reps = 100;
  int batch = 5;
  int block = 256;
};

long parse_long(const char* flag, const char* value) {
  char* end = nullptr;
  const long v = std::strtol(value, &end, 10);
  if (*value == '\0' || *end != '\0' || v <= 0)
    fail("BAD_ARGUMENT", std::string(flag) + " needs a positive integer");
  return v;
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (i + 1 >= argc) fail("BAD_ARGUMENT", flag + " needs a value");
    const long v = parse_long(argv[i], argv[i + 1]);
    ++i;
    if (flag == "--elements")
      o.elements = v;
    else if (flag == "--warmup")
      o.warmup = static_cast<int>(v);
    else if (flag == "--reps")
      o.reps = static_cast<int>(v);
    else if (flag == "--batch")
      o.batch = static_cast<int>(v);
    else if (flag == "--block")
      o.block = static_cast<int>(v);
    else
      fail("BAD_ARGUMENT", "unknown flag " + flag);
  }
  if (o.elements % 2 != 0) fail("BAD_ARGUMENT", "--elements must be even");
  return o;
}

struct Kernel {
  const char* name;
  int arrays;  // arrays touched per element: STREAM counts each read and each write once
  std::vector<double> seconds;
};

template <class Launch>
void time_kernel(Kernel& k, const Options& o, Launch launch) {
  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate");
  // One sample is the mean time of `batch` back-to-back launches between two events.
  for (int r = 0; r < o.warmup + o.reps; ++r) {
    check_cuda(cudaEventRecord(start), "cudaEventRecord");
    for (int i = 0; i < o.batch; ++i) launch();
    check_cuda(cudaEventRecord(stop), "cudaEventRecord");
    check_cuda(cudaEventSynchronize(stop), k.name);
    check_cuda(cudaGetLastError(), k.name);
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    if (r >= o.warmup) k.seconds.push_back(ms * 1e-3 / o.batch);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void verify(const double2* d, long n2, double want, const char* array) {
  std::vector<double2> h(n2);
  check_cuda(cudaMemcpy(h.data(), d, n2 * sizeof(double2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  for (long i = 0; i < n2; ++i) {
    if (h[i].x != want || h[i].y != want) {
      fail("VERIFY_FAILED", std::string("array ") + array + " element " + std::to_string(2 * i) +
                                " is " + std::to_string(h[i].x) + ", expected " +
                                std::to_string(want));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  int runtime = 0, driver = 0;
  cudaRuntimeGetVersion(&runtime);
  cudaDriverGetVersion(&driver);

  const long n2 = o.elements / 2;
  double2 *a = nullptr, *b = nullptr, *c = nullptr;
  check_cuda(cudaMalloc(&a, n2 * sizeof(double2)), "cudaMalloc a");
  check_cuda(cudaMalloc(&b, n2 * sizeof(double2)), "cudaMalloc b");
  check_cuda(cudaMalloc(&c, n2 * sizeof(double2)), "cudaMalloc c");
  const unsigned grid = static_cast<unsigned>((n2 + o.block - 1) / o.block);
  init<<<grid, o.block>>>(a, b, c, n2);
  check_cuda(cudaDeviceSynchronize(), "init");

  Kernel kernels[] = {{"copy", 2, {}}, {"scale", 2, {}}, {"add", 3, {}}, {"triad", 3, {}}};
  time_kernel(kernels[0], o, [&] { copy<<<grid, o.block>>>(a, c, n2); });
  time_kernel(kernels[1], o, [&] { scale<<<grid, o.block>>>(b, c, kScalar, n2); });
  time_kernel(kernels[2], o, [&] { add<<<grid, o.block>>>(a, b, c, n2); });
  time_kernel(kernels[3], o, [&] { triad<<<grid, o.block>>>(a, b, c, kScalar, n2); });

  verify(b, n2, kScalar, "b");
  verify(c, n2, 1.0 + kScalar, "c");
  verify(a, n2, kScalar + kScalar * (1.0 + kScalar), "a");
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  char uuid[37];
  const unsigned char* u = reinterpret_cast<const unsigned char*>(prop.uuid.bytes);
  std::snprintf(uuid, sizeof uuid,
                "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1],
                u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14],
                u[15]);
  std::printf(
      "{\"device\":{\"name\":\"%s\",\"uuid\":\"GPU-%s\",\"compute_capability\":\"%d.%d\","
      "\"sm_count\":%d,\"l2_bytes\":%d,\"memory_bytes\":%zu},",
      prop.name, uuid, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize,
      prop.totalGlobalMem);
  std::printf("\"backend\":\"" CFD_BENCH_BACKEND "\",\"runtime_version\":%d,\"driver_version\":%d,",
              runtime, driver);
  std::printf(
      "\"config\":{\"elements\":%ld,\"element_bytes\":8,\"warmup\":%d,\"reps\":%d,\"batch\":%d,"
      "\"block\":%d},",
      o.elements, o.warmup, o.reps, o.batch, o.block);
  std::printf("\"verified\":true,\"kernels\":[");
  for (std::size_t k = 0; k < std::size(kernels); ++k) {
    const Kernel& kern = kernels[k];
    std::printf("%s{\"name\":\"%s\",\"bytes_per_rep\":%ld,\"seconds\":[", k ? "," : "", kern.name,
                kern.arrays * o.elements * 8L);
    for (std::size_t r = 0; r < kern.seconds.size(); ++r)
      std::printf("%s%.9e", r ? "," : "", kern.seconds[r]);
    std::printf("]}");
  }
  std::printf("]}\n");
  return EXIT_SUCCESS;
}
