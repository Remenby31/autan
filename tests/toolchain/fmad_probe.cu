// a * b + c: nvcc contracts it into fma.rn unless the unit is compiled with --fmad=false (#571).
extern "C" __global__ void autan_fmad_probe(const double* a, const double* b, const double* c,
                                            double* out) {
  out[0] = a[0] * b[0] + c[0];
}
