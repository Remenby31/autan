#include <cstddef>
#include <string>

#include <cuda_runtime.h>

#include "autan/core/error.hpp"
#include "halo_kernels.hpp"

namespace autan::halo {
namespace {

template <class T>
__global__ void run(Kernel kernel, Layout layout, Components<T> c, T* buffer, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  switch (kernel) {
    case Kernel::pack_forward:
      pack_forward(layout, c, buffer, i);
      break;
    case Kernel::unpack_forward:
      unpack_forward(layout, c, buffer, i);
      break;
    case Kernel::pack_reverse:
      pack_reverse(layout, c, buffer, i);
      break;
    case Kernel::unpack_reverse:
      unpack_reverse(layout, c, buffer, i);
      break;
  }
}

}  // namespace

template <class T>
void launch(Kernel kernel, const Layout& layout, Components<T> c, T* buffer, std::size_t count,
            void* stream) {
  if (count == 0) return;
  constexpr unsigned block = 256;
  const auto grid = static_cast<unsigned>((count + block - 1) / block);
  run<T><<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(kernel, layout, c, buffer, count);
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw Error(ErrorCode::halo_device_unsupported,
                std::string("halo kernel launch failed: ") + cudaGetErrorString(status));
  }
}

template void launch<float>(Kernel, const Layout&, Components<float>, float*, std::size_t, void*);
template void launch<double>(Kernel, const Layout&, Components<double>, double*, std::size_t,
                             void*);

}  // namespace autan::halo
