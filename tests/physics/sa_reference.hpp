#pragma once
// An implementation of SA-neg's source and its derivative apart from the generated loops, for the
// tests of the generated Jacobian (#163 step 7b): the host's cell block against dS/dnt, and the
// device's against the host's within an envelope made of dS/dnt's terms.

#include <cmath>

namespace autan::test {

// SA-neg's constants (kernels/spalart_allmaras.py; Spalart and Allmaras 1994, TMR's SA page).
inline constexpr double sa_cb1 = 0.1355, sa_sigma = 2.0 / 3.0, sa_cb2 = 0.622, sa_kappa = 0.41,
                        sa_cw2 = 0.3, sa_cw3 = 2.0, sa_cv1 = 7.1, sa_ct3 = 1.2, sa_ct4 = 0.5,
                        sa_c2 = 0.7, sa_c3 = 0.9, sa_rmax = 10.0;
inline constexpr double sa_cw1 = sa_cb1 / (sa_kappa * sa_kappa) + (1.0 + sa_cb2) / sa_sigma;

// A value and its derivative in nt.
struct Dual {
  double v, d;
};
inline Dual operator+(Dual a, Dual b) { return {a.v + b.v, a.d + b.d}; }
inline Dual operator-(Dual a, Dual b) { return {a.v - b.v, a.d - b.d}; }
inline Dual operator*(Dual a, Dual b) { return {a.v * b.v, a.d * b.v + a.v * b.d}; }
inline Dual operator/(Dual a, Dual b) { return {a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)}; }
inline Dual constant(double v) { return {v, 0.0}; }
inline Dual dexp(Dual a) { return {std::exp(a.v), std::exp(a.v) * a.d}; }
inline Dual dpow(Dual a, double k) { return {std::pow(a.v, k), k * std::pow(a.v, k - 1.0) * a.d}; }

// The sum of the magnitudes of the product-rule terms of dS/dnt for SA-neg's positive branch (the
// TMR SA page's source with Allmaras et al. (2012)'s S_tilde, Note 1(c)), from an implementation
// of its own: |cb1 (1-ft2) St| + |cb1 (1-ft2) nt St'| + |cb1 ft2' St nt| + |cw1 fw' (nt/d)^2| +
// |2 cw1 fw nt/d^2| + |cb1/k^2 ft2' (nt/d)^2| + |2 cb1/k^2 ft2 nt/d^2|. 1/s, as dS/dnt.
struct DerivativeTerms {
  double sum;  // sum of |t_k|
  double fw;   // |cw1 fw' (nt/d)^2|
  double ft2;  // |cb1 ft2' St nt| + |cb1/k^2 ft2' (nt/d)^2|
};

// SA-neg's positive-branch source S(nt) (the TMR SA page, with Allmaras et al. (2012)'s S_tilde,
// Note 1(c)) and its derivative in nt, computed apart from the generated loops, through the
// intermediates the envelope's terms below are made of.
struct SourceParts {
  Dual nt, st, fw, ft2;
};
inline SourceParts source_parts(double nt_v, double nu, double omega, double d) {
  const Dual nt{nt_v, 1.0};
  const Dual chi = nt / constant(nu);
  const double kd2 = sa_kappa * sa_kappa * d * d;
  const Dual c3 = chi * chi * chi;
  const Dual fv1 = c3 / (c3 + constant(sa_cv1 * sa_cv1 * sa_cv1));
  const Dual fv2 = constant(1.0) - chi / (constant(1.0) + chi * fv1);
  const Dual sbar = nt * fv2 / constant(kd2);
  const Dual om = constant(omega);
  const Dual st = sbar.v >= -sa_c2 * omega
                      ? om + sbar
                      : om + om * (constant(sa_c2 * sa_c2) * om + constant(sa_c3) * sbar) /
                                 (constant(sa_c3 - 2.0 * sa_c2) * om - sbar);
  const Dual r =
      st.v > 0.0 && nt.v / (st.v * kd2) < sa_rmax ? nt / (st * constant(kd2)) : constant(sa_rmax);
  const Dual g = r + constant(sa_cw2) * (dpow(r, 6.0) - r);
  const double c6 = std::pow(sa_cw3, 6.0);
  const Dual fw = g * dpow((constant(1.0 + c6)) / (dpow(g, 6.0) + constant(c6)), 1.0 / 6.0);
  const Dual ft2 = constant(sa_ct3) * dexp(constant(-sa_ct4) * chi * chi);
  return {nt, st, fw, ft2};
}

// dS/dnt of the positive branch, 1/s.
inline double source_derivative(double nt_v, double nu, double omega, double d) {
  const auto [nt, st, fw, ft2] = source_parts(nt_v, nu, omega, d);
  const Dual q = nt / constant(d);
  const Dual s = constant(sa_cb1) * (constant(1.0) - ft2) * st * nt -
                 (constant(sa_cw1) * fw - constant(sa_cb1 / (sa_kappa * sa_kappa)) * ft2) * q * q;
  return s.d;
}

inline DerivativeTerms source_derivative_terms(double nt_v, double nu, double omega, double d) {
  const auto [nt, st, fw, ft2] = source_parts(nt_v, nu, omega, d);
  const double q = nt.v / d;
  const double t_fw = std::abs(sa_cw1 * fw.d * q * q);
  const double t_ft2 = std::abs(sa_cb1 * ft2.d * st.v * nt.v) +
                       std::abs(sa_cb1 / (sa_kappa * sa_kappa) * ft2.d * q * q);
  const double rest = std::abs(sa_cb1 * (1.0 - ft2.v) * st.v) +
                      std::abs(sa_cb1 * (1.0 - ft2.v) * nt.v * st.d) +
                      std::abs(2.0 * sa_cw1 * fw.v * nt.v / (d * d)) +
                      std::abs(2.0 * sa_cb1 / (sa_kappa * sa_kappa) * ft2.v * nt.v / (d * d));
  return {rest + t_fw + t_ft2, t_fw, t_ft2};
}

}  // namespace autan::test
