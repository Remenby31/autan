// STREAM-style memory bandwidth of an Apple GPU through Metal (McCalpin, "Memory Bandwidth and
// Machine Balance in Current High Performance Computers", IEEE TCCA Newsletter, 1995): the peak
// that the Metal loop benchmark is measured against. Kernels as bench/stream/stream.cu, on FP32
// arrays (Metal has no FP64) with one float4 (16-byte) access per array per thread, plus `read`
// (a only): Apple GPUs read faster than they copy, so the copy and read peaks together give the
// cost of a read byte and of a written byte. Inputs never change, so the final arrays are
// exactly a = q + q (1 + q), b = q, c = 1 + q; the check is bitwise. Prints the raw JSON document
// autan-bench reads.
//
//   autan_metal_stream [--elements N] [--warmup N] [--reps N] [--batch N] [--block N]
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "autan/kernels/metal.hpp"
#include "metal_bench.hpp"
#include "revision.h"

namespace {

namespace mtl = autan::kernels::metal;

constexpr float kScalar = 3.0f;

const char* const kSource[] = {R"msl(
#include <metal_stdlib>
struct Arrays {
  device float4* a;
  device float4* b;
  device float4* c;
  long n4;
  float q;
};
kernel void init_f32(constant Arrays& d [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  if (long(gid) < d.n4) {
    d.a[gid] = float4(1.0f);
    d.b[gid] = float4(2.0f);
    d.c[gid] = float4(0.0f);
  }
}
kernel void copy_f32(constant Arrays& d [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  if (long(gid) < d.n4) d.c[gid] = d.a[gid];
}
kernel void scale_f32(constant Arrays& d [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  if (long(gid) < d.n4) d.b[gid] = d.q * d.c[gid];
}
kernel void add_f32(constant Arrays& d [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  if (long(gid) < d.n4) d.c[gid] = d.a[gid] + d.b[gid];
}
kernel void triad_f32(constant Arrays& d [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  if (long(gid) < d.n4) d.a[gid] = d.b[gid] + d.q * d.c[gid];
}
kernel void read_f32(constant Arrays& d [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  if (long(gid) < d.n4) {
    const float4 v = d.a[gid];
    if (v.x == -1.0f) d.c[gid] = v;  // never true: keeps the load
  }
}
)msl",
                               nullptr};
constexpr mtl::Library kLibrary{"stream", kSource};

struct Arrays {
  float* a;
  float* b;
  float* c;
  long n4;
  float q;
};

[[noreturn]] void fail(const std::string& message) {
  std::fprintf(stderr, "error: %s\n", message.c_str());
  std::exit(EXIT_FAILURE);
}

struct Options {
  long elements = 1L << 28;
  int warmup = 10;
  int reps = 100;
  int batch = 5;
  int block = 256;
};

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    const long v = std::strtol(argv[i + 1], nullptr, 10);
    if (v <= 0) fail(flag + " needs a positive integer");
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
      fail("unknown flag " + flag);
  }
  if (o.elements % 4 != 0) fail("--elements must be a multiple of 4");
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  mtl::Context ctx;
  const auto bytes = static_cast<std::size_t>(o.elements) * sizeof(float);
  mtl::Buffer a = ctx.allocate(bytes), b = ctx.allocate(bytes), c = ctx.allocate(bytes);
  const Arrays d{a.device<float>(), b.device<float>(), c.device<float>(), o.elements / 4, kScalar};
  const auto threads = static_cast<autan::kernels::index_t>(o.elements / 4);
  ctx.dispatch(kLibrary, "init", "f32", d, threads, o.block);
  ctx.synchronise();

  struct Kernel {
    const char* name;
    int arrays;
    std::vector<double> seconds;
  };
  std::vector<Kernel> kernels{
      {"copy", 2, {}}, {"scale", 2, {}}, {"add", 3, {}}, {"triad", 3, {}}, {"read", 1, {}}};
  for (auto& k : kernels) {
    for (int r = 0; r < o.warmup + o.reps; ++r) {
      for (int i = 0; i < o.batch; ++i) {
        ctx.dispatch(kLibrary, k.name, "f32", d, threads, o.block);
      }
      const double t = ctx.synchronise();
      if (r >= o.warmup) k.seconds.push_back(t / o.batch);
    }
  }
  const float qa = kScalar + kScalar * (1.0f + kScalar);
  bool verified = true;
  for (long i = 0; i < o.elements; ++i) {
    if (a.data<float>()[i] != qa || b.data<float>()[i] != kScalar ||
        c.data<float>()[i] != 1.0f + kScalar) {
      verified = false;
      break;
    }
  }
  std::printf("{");
  autan::spike::metal::print_device(ctx);
  std::printf("\"revision\":{\"commit\":\"%s\",\"dirty\":%s},", AUTAN_BENCH_COMMIT,
              AUTAN_BENCH_DIRTY);
  std::printf(
      "\"config\":{\"elements\":%ld,\"element_bytes\":4,\"warmup\":%d,\"reps\":%d,\"batch\":%d,"
      "\"block\":%d},",
      o.elements, o.warmup, o.reps, o.batch, o.block);
  std::fprintf(stderr, "Metal compilation %.3f s\n", ctx.compile_seconds());
  std::printf("\"verified\":%s,\"kernels\":[", verified ? "true" : "false");
  for (std::size_t k = 0; k < kernels.size(); ++k) {
    std::printf("%s{\"name\":\"%s\",\"bytes_per_rep\":%zu,\"seconds\":[", k ? "," : "",
                kernels[k].name, static_cast<std::size_t>(kernels[k].arrays) * bytes);
    for (std::size_t r = 0; r < kernels[k].seconds.size(); ++r)
      std::printf("%s%.9e", r ? "," : "", kernels[k].seconds[r]);
    std::printf("]}");
  }
  std::printf("]}\n");
  return verified ? 0 : 1;
}
