#pragma once

#include <array>
#include <cmath>
#include <numbers>

namespace autan::physics {

/// The freestream W = (rho, u, v, w, p), flowing in the x-z plane at angle of attack `aoa`
/// (rad), plus the compressible point vortex of circulation `circulation` seen at (dx, dz) from
/// the vortex; positive circulation gives positive lift along (-sin aoa, 0, cos aoa).
// Thomas and Salas (1986), AIAA J. 24(7): the Prandtl-Glauert vortex; the speed of sound
// from the freestream total enthalpy, density and pressure isentropic.
[[nodiscard]] inline std::array<double, 5> point_vortex_state(const std::array<double, 5>& w,
                                                              double gamma, double aoa,
                                                              double circulation, double dx,
                                                              double dz) {
  const double speed2 = w[1] * w[1] + w[2] * w[2] + w[3] * w[3];
  const double c2 = gamma * w[4] / w[0];
  const double mach2 = speed2 / c2;
  const double theta = std::atan2(dz, dx);
  const double s = std::sin(theta - aoa);
  const double q = circulation * std::sqrt(1.0 - mach2) /
                   (2.0 * std::numbers::pi * std::hypot(dx, dz) * (1.0 - mach2 * s * s));
  const double u = w[1] + q * std::sin(theta);
  const double v = w[2];
  const double wz = w[3] - q * std::cos(theta);
  const double cb2 = c2 + 0.5 * (gamma - 1.0) * (speed2 - (u * u + v * v + wz * wz));
  const double rho = w[0] * std::pow(cb2 / c2, 1.0 / (gamma - 1.0));
  return {rho, u, v, wz, w[4] * std::pow(rho / w[0], gamma)};
}

}  // namespace autan::physics
