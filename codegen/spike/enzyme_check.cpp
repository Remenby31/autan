// Enzyme as an independent oracle (research section 4.3, item 5): Enzyme's reverse mode of the
// emitted fp64 host Roe body against the generated adjoint body, on random subsonic states that
// exercise both branches of Harten's entropy fix. Prints the largest relative difference.
//
// Built with clang and the Enzyme plugin (codegen/spike/enzyme_check.sh), outside CMake.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "roe.hpp"

extern int enzyme_dup, enzyme_const, enzyme_dupnoneed;
void __enzyme_autodiff(void*, ...);

namespace {

using P = autan::kernels::fp64;

void flux(const double* ul, const double* ur, const double* n, double area, double* out) {
  autan::kernels::roe::roe_flux<P>(ul[0], ul[1], ul[2], ul[3], ul[4], ur[0], ur[1], ur[2], ur[3],
                                   ur[4], n[0], n[1], n[2], area, out);
}

}  // namespace

int main() {
  std::mt19937_64 rng(2026);
  std::normal_distribution<double> g;
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  double worst = 0.0;
  int fixed = 0;
  const int samples = 10000;
  for (int s = 0; s < samples; ++s) {
    double ul[5], ur[5], n[3], outb[5];
    for (double* u : {ul, ur}) {
      const double rho = 1.0 + 0.2 * u01(rng), p = 1.0 + 0.2 * u01(rng);
      const double scale = s % 2 == 0 ? 0.3 : 0.01;  // half the samples near stagnation
      const double vx = scale * g(rng), vy = scale * g(rng), vz = scale * g(rng);
      u[0] = rho;
      u[1] = rho * vx;
      u[2] = rho * vy;
      u[3] = rho * vz;
      u[4] = p / 0.4 + 0.5 * rho * (vx * vx + vy * vy + vz * vz);
    }
    const double nx = g(rng), ny = g(rng), nz = g(rng), nn = std::sqrt(nx * nx + ny * ny + nz * nz);
    n[0] = nx / nn;
    n[1] = ny / nn;
    n[2] = nz / nn;
    const double area = 0.5 + u01(rng);
    for (double& b : outb) b = g(rng);
    fixed += s % 2 == 0 ? 0 : 1;

    double ulb[5] = {}, urb[5] = {}, out[5], outb_copy[5];
    std::copy(outb, outb + 5, outb_copy);
    __enzyme_autodiff(reinterpret_cast<void*>(flux), enzyme_dup, ul, ulb, enzyme_dup, ur, urb,
                      enzyme_const, n, enzyme_const, area, enzyme_dupnoneed, out, outb_copy);

    // Generated adjoint: seeds of the left and right cells; the output adjoint is
    // seed_l - seed_r (Inc signs +1, -1), so seed_l = outb, seed_r = 0.
    double ours[10];
    autan::kernels::roe::flux_adjoint_body<P>(
        ul[0], ul[1], ul[2], ul[3], ul[4], ur[0], ur[1], ur[2], ur[3], ur[4], n[0], n[1], n[2],
        area, outb[0], outb[1], outb[2], outb[3], outb[4], 0.0, 0.0, 0.0, 0.0, 0.0, ours);
    double scale = 0.0, diff = 0.0;
    bool finite = true;
    for (int k = 0; k < 5; ++k) {
      finite = finite && std::isfinite(ulb[k]) && std::isfinite(urb[k]) && std::isfinite(ours[k]) &&
               std::isfinite(ours[5 + k]);
      scale = std::max({scale, std::abs(ulb[k]), std::abs(urb[k])});
      diff = std::max({diff, std::abs(ulb[k] - ours[k]), std::abs(urb[k] - ours[5 + k])});
    }
    const double rel = finite && scale > 0.0 ? diff / scale : HUGE_VAL;
    worst = std::max(worst, rel);
  }
  std::printf("{\"samples\":%d,\"near_stagnation\":%d,\"max_rel_diff\":%.3e}\n", samples, fixed,
              worst);
  return std::isfinite(worst) && worst <= 1e-13 ? 0 : 1;
}
