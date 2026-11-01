// Smallest Enzyme reverse-mode CUDA kernel (the pattern of the Enzyme CUDA guide): d(x^2)/dx on
// device. Exit 0 when the derivative is 2x on every thread.
#include <cstdio>

extern __device__ int enzyme_dup;
extern __device__ void __enzyme_autodiff(void*, ...);

__device__ void square(const double* x, double* y) { *y = *x * *x; }

__global__ void grad(const double* x, double* dx, double* y, double* dy, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    __enzyme_autodiff(reinterpret_cast<void*>(square), enzyme_dup, x + i, dx + i, enzyme_dup, y + i,
                      dy + i);
  }
}

int main() {
  const int n = 1024;
  double *x, *dx, *y, *dy;
  cudaMallocManaged(&x, n * sizeof(double));
  cudaMallocManaged(&dx, n * sizeof(double));
  cudaMallocManaged(&y, n * sizeof(double));
  cudaMallocManaged(&dy, n * sizeof(double));
  for (int i = 0; i < n; ++i) {
    x[i] = 0.5 + i;
    dx[i] = 0.0;
    dy[i] = 1.0;
  }
  grad<<<n / 128, 128>>>(x, dx, y, dy, n);
  const cudaError_t e = cudaDeviceSynchronize();
  int bad = 0;
  for (int i = 0; i < n; ++i) bad += dx[i] != 2.0 * x[i];
  std::printf("cuda=%s bad=%d dx[3]=%g\n", cudaGetErrorString(e), bad, dx[3]);
  return e == cudaSuccess && bad == 0 ? 0 : 1;
}
