// Metal runtime of the generated loops (autan/kernels/metal.hpp).
#include "autan/kernels/metal.hpp"

#include <chrono>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#if !__has_feature(objc_arc)
#error "codegen/cpp/metal.mm needs -fobjc-arc"
#endif

namespace autan::kernels::metal {

namespace {

[[noreturn]] void fail(const std::string& what, NSError* error = nil) {
  std::string msg = "E1107: " + what;
  if (error != nil) {
    msg += ": ";
    msg += error.localizedDescription.UTF8String;
  }
  throw std::runtime_error(msg);
}

}  // namespace

struct Context::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  std::map<std::string, id<MTLLibrary>> libraries;
  std::map<std::string, id<MTLComputePipelineState>> pipelines;
  // Every live buffer, made resident in each encoder: kernels reach them through GPU addresses.
  std::vector<std::weak_ptr<void>> buffers;
  id<MTLCommandBuffer> commands = nil;
  id<MTLComputeCommandEncoder> encoder = nil;
  bool first_dispatch = true;
  double compile_seconds = 0.0;

  id<MTLComputePipelineState> pipeline(const Library& lib, const std::string& function) {
    const std::string key = std::string(lib.name) + "/" + function;
    if (auto it = pipelines.find(key); it != pipelines.end()) {
      return it->second;
    }
    const auto start = std::chrono::steady_clock::now();
    auto lit = libraries.find(lib.name);
    if (lit == libraries.end()) {
      MTLCompileOptions* options = [MTLCompileOptions new];
      // No reassociation or contraction beyond the source: the double-float sums depend on it,
      // and results must equal the CPU loops to rounding.
      options.mathMode = MTLMathModeSafe;
      options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
      NSError* error = nil;
      std::string text;
      for (const char* const* piece = lib.source; *piece != nullptr; ++piece) {
        text += *piece;
        text += '\n';
      }
      id<MTLLibrary> library = [device newLibraryWithSource:@(text.c_str())
                                                    options:options
                                                      error:&error];
      if (library == nil) {
        fail(std::string("compiling the Metal library of module ") + lib.name, error);
      }
      lit = libraries.emplace(lib.name, library).first;
    }
    id<MTLFunction> fn = [lit->second newFunctionWithName:@(function.c_str())];
    if (fn == nil) {
      fail("no kernel " + function + " in module " + lib.name);
    }
    NSError* error = nil;
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:fn
                                                                              error:&error];
    if (state == nil) {
      fail("building the pipeline of " + function, error);
    }
    compile_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    pipelines.emplace(key, state);
    return state;
  }

  void open() {
    if (commands != nil) {
      return;
    }
    commands = [queue commandBuffer];
    encoder = [commands computeCommandEncoder];
    if (commands == nil || encoder == nil) {
      fail("cannot create a command buffer");
    }
    std::vector<id<MTLResource>> resident;
    std::erase_if(buffers, [](const std::weak_ptr<void>& w) { return w.expired(); });
    for (const auto& w : buffers) {
      if (auto h = w.lock()) {
        resident.push_back((__bridge id<MTLBuffer>)h.get());
      }
    }
    if (!resident.empty()) {
      [encoder useResources:resident.data()
                      count:resident.size()
                      usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    }
    first_dispatch = true;
  }
};

Context::Context() : impl_(std::make_unique<Impl>()) {
  impl_->device = MTLCreateSystemDefaultDevice();
  if (impl_->device == nil) {
    fail("no Metal device");
  }
  impl_->queue = [impl_->device newCommandQueue];
  if (impl_->queue == nil) {
    fail("cannot create a command queue");
  }
}

Context::~Context() {
  if (impl_->commands != nil) {
    [impl_->encoder endEncoding];
    [impl_->commands commit];
    [impl_->commands waitUntilCompleted];
  }
}

std::string Context::device_name() const { return impl_->device.name.UTF8String; }

Buffer Context::allocate(std::size_t bytes) {
  const std::size_t n = bytes == 0 ? 1 : bytes;
  id<MTLBuffer> buf = [impl_->device newBufferWithLength:n options:MTLResourceStorageModeShared];
  if (buf == nil) {
    fail("cannot allocate " + std::to_string(n) + " bytes");
  }
  std::memset(buf.contents, 0, n);
  Buffer b;
  b.host_ = buf.contents;
  b.gpu_ = buf.gpuAddress;
  b.bytes_ = bytes;
  b.handle_ = std::shared_ptr<void>((__bridge_retained void*)buf, [](void* p) { CFRelease(p); });
  impl_->buffers.push_back(b.handle_);
  if (impl_->encoder != nil) {
    [impl_->encoder useResource:buf usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  return b;
}

void Context::dispatch_raw(const Library& lib, std::string_view kernel, const char* tag,
                           const void* d, std::size_t d_bytes, const void* args,
                           std::size_t args_bytes, index_t threads, int block) {
  if (threads <= 0) {
    return;
  }
  if (block <= 0 || block % 32 != 0) {
    throw std::invalid_argument("E1107: the block size must be a positive multiple of 32");
  }
  std::string function(kernel);
  function += "_";
  function += tag;
  id<MTLComputePipelineState> state = impl_->pipeline(lib, function);
  if (static_cast<NSUInteger>(block) > state.maxTotalThreadsPerThreadgroup) {
    fail(function + ": block " + std::to_string(block) + " exceeds the pipeline's maximum " +
         std::to_string(state.maxTotalThreadsPerThreadgroup));
  }
  impl_->open();
  id<MTLComputeCommandEncoder> enc = impl_->encoder;
  if (!impl_->first_dispatch) {
    // Each dispatch reads what the previous ones wrote (store, then gather; loop after loop).
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  impl_->first_dispatch = false;
  [enc setComputePipelineState:state];
  [enc setBytes:d length:d_bytes atIndex:0];
  if (args != nullptr) {
    [enc setBytes:args length:args_bytes atIndex:1];
  }
  const auto width = static_cast<NSUInteger>(block);
  const auto groups = (static_cast<NSUInteger>(threads) + width - 1) / width;
  // Whole threadgroups, as CUDA grids: SIMD-group shuffles see every lane; kernels bound-check.
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

double Context::synchronise() {
  if (impl_->commands == nil) {
    return 0.0;
  }
  id<MTLCommandBuffer> cb = impl_->commands;
  [impl_->encoder endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  impl_->commands = nil;
  impl_->encoder = nil;
  if (cb.status != MTLCommandBufferStatusCompleted) {
    fail("command buffer failed", cb.error);
  }
  return cb.GPUEndTime - cb.GPUStartTime;
}

double Context::compile_seconds() const { return impl_->compile_seconds; }

}  // namespace autan::kernels::metal
