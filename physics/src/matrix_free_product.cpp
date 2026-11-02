#include "autan/physics/matrix_free_product.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "autan/kernels/runtime.hpp"
#include "autan/runtime/exact_sum.hpp"

namespace autan::physics {

namespace {

template <class T>
bool finite(const std::vector<T>& v) {
  return std::all_of(v.begin(), v.end(),
                     [](const T& x) { return std::isfinite(static_cast<double>(x)); });
}

}  // namespace

double nt_step_fraction(std::span<const double> rho, std::span<const double> rnu,
                        std::span<const double> dnt, double nt_inf, NtBounds bounds) {
  double alpha = 1.0;
  for (std::size_t c = 0; c < dnt.size(); ++c) {
    const double scale = std::max(std::abs(rnu[c]), rho[c] * nt_inf);
    const double d = dnt[c];
    if (d < 0.0) alpha = std::min(alpha, bounds.decrease * scale / -d);
    if (d > 0.0) alpha = std::min(alpha, bounds.increase * scale / d);
    if (std::isnan(d)) alpha = 0.0;
  }
  return alpha;
}

template <class P>
MatrixFreeProduct<P>::MatrixFreeProduct(NavierStokes<P>& flow, SpalartAllmaras<P>* model,
                                        double gamma, std::array<double, 5> flow_scale,
                                        NtBounds bounds)
    : flow_(flow),
      model_(model),
      gamma_(gamma),
      bounds_(bounds),
      n_(static_cast<std::size_t>(flow.n_cells())),
      m_(model != nullptr ? 6 : 5),
      // Knoll and Keyes (2004), J. Comput. Phys. 193:357-397, sec. 2.2: sqrt of the precision
      // the state is stored in.
      eps_storage_(std::sqrt(static_cast<double>(std::numeric_limits<State>::epsilon()))) {
  for (std::size_t k = 0; k < 5; ++k) scale_[k] = flow_scale[k] > 0.0 ? flow_scale[k] : 1.0;
  scale_[5] = 1.0;
  current_.assign(m_ * n_, 0.0);
  probe_.assign(m_ * n_, 0.0);
  base_.assign(m_ * n_, 0.0);
  state_.assign(5 * n_, State(0));
  probe_state_.assign(5 * n_, State(0));
  nt_.assign(n_, State(0));
  nt_current_.assign(n_, State(0));
  res_.assign(5 * n_, Res(0));
  res_nt_.assign(n_, Res(0));
}

template <class P>
void MatrixFreeProduct<P>::start(std::span<const double> u, std::span<const State> state,
                                 std::span<const double> rnu, double nt_inf,
                                 std::span<const double> shift, std::array<bool, 6> left_out) {
  const std::size_t nf = 5 * n_;
  if (u.size() != nf || state.size() != nf || rnu.size() != (m_ - 5) * n_ || shift.size() != n_) {
    throw std::invalid_argument("MatrixFreeProduct::start: sizes do not match the mesh");
  }
  std::copy(u.begin(), u.end(), current_.begin());
  std::copy(rnu.begin(), rnu.end(), current_.begin() + static_cast<std::ptrdiff_t>(nf));
  std::copy(state.begin(), state.end(), state_.begin());
  for (std::size_t c = 0; c < rnu.size(); ++c) nt_current_[c] = State(rnu[c]);
  shift_.assign(shift.begin(), shift.end());
  nt_inf_ = nt_inf;
  left_out_ = left_out;
  halvings_ = 0;
  representation_ = 0.0;
  failed_ = false;
  flow_.freeze(state_);
  if (model_ == nullptr) {
    flow_.residual_frozen(state_, res_);
  } else {
    // rho nt's scale: its root mean square (it reaches 1e3-1e4 times the freestream's in the
    // boundary layer, which would weigh it wrongly).
    std::array<runtime::ExactSum, 2> sums;
    for (std::size_t c = 0; c < n_; ++c) sums[0].add(rnu[c] * rnu[c]);
    sums[1].add(static_cast<double>(n_));
    scale_[5] = std::sqrt(sums[0].value() / sums[1].value());
    if (!(scale_[5] > 0.0)) scale_[5] = scale_[0] * nt_inf;
    if (!(scale_[5] > 0.0)) scale_[5] = 1.0;
    model_->freeze(nt_current_);
    flow_.residual_frozen(state_, res_, [&] { model_->eddy_viscosity_frozen(nt_current_); });
    model_->residual_frozen(nt_current_, res_nt_);
    for (std::size_t c = 0; c < n_; ++c) base_[nf + c] = static_cast<double>(res_nt_[c]);
  }
  for (std::size_t i = 0; i < nf; ++i) base_[i] = static_cast<double>(res_[i]);
}

template <class P>
double MatrixFreeProduct<P>::dot(std::span<const double> a, std::span<const double> b) const {
  runtime::ExactSum sum;
  for (std::size_t k = 0; k < m_; ++k) {
    const double w = 1.0 / (scale_[k] * scale_[k]);
    for (std::size_t c = 0; c < n_; ++c) sum.add(a[k * n_ + c] * b[k * n_ + c] * w);
  }
  return sum.value();
}

template <class P>
double MatrixFreeProduct<P>::max_abs(std::span<const double> a) const {
  double m = 0.0;
  for (std::size_t k = 0; k < m_; ++k) {
    for (std::size_t c = 0; c < n_; ++c) {
      const double x = std::abs(a[k * n_ + c]) / scale_[k];
      m = std::isnan(x) ? std::numeric_limits<double>::infinity() : std::max(m, x);
    }
  }
  return m;
}

template <class P>
double MatrixFreeProduct<P>::scaled_rms(std::span<const double> x, bool skip_left_out) const {
  std::array<runtime::ExactSum, 2> sums;
  std::size_t rows = 0;
  for (std::size_t k = 0; k < m_; ++k) {
    if (skip_left_out && left_out_[k]) continue;
    rows += n_;
    for (std::size_t c = 0; c < n_; ++c) {
      const double a = x[k * n_ + c] / scale_[k];
      sums[0].add(a * a);
    }
  }
  sums[1].add(static_cast<double>(rows));
  return std::sqrt(sums[0].value() / sums[1].value());
}

template <class P>
void MatrixFreeProduct<P>::apply(std::span<const double> v, std::span<double> y) {
  const std::size_t nf = 5 * n_;
  std::fill(y.begin(), y.end(), 0.0);
  if (failed_) {
    std::fill(y.begin(), y.end(), std::numeric_limits<double>::quiet_NaN());
    return;
  }
  const double vn = scaled_rms(v, true);
  if (vn == 0.0) return;
  double eps = eps_storage_ * (1.0 + scaled_rms(current_, false)) / vn;
  auto entry = [&](std::size_t i) { return left_out_[i / n_] ? 0.0 : v[i]; };
  auto pressure = [&](const double* w) {
    return (gamma_ - 1.0) * (w[4] - 0.5 * (w[1] * w[1] + w[2] * w[2] + w[3] * w[3]) / w[0]);
  };
  auto fail = [&](int halving, double ratio) {
    halvings_ = std::max(halvings_, halving);
    representation_ = std::max(representation_, ratio);
    failed_ = true;
    std::fill(y.begin(), y.end(), std::numeric_limits<double>::quiet_NaN());
  };
  for (int halving = 0;; ++halving) {
    for (std::size_t i = 0; i < m_ * n_; ++i) probe_[i] = current_[i] + eps * entry(i);
    // Admissible and small: density and pressure within 10 % of the current state's in every
    // cell; rho nt within the step's bounds.
    double worst = 0.0;
    for (std::size_t c = 0; c < n_; ++c) {
      double w0[5], w[5];
      for (std::size_t k = 0; k < 5; ++k) {
        w0[k] = current_[k * n_ + c];
        w[k] = probe_[k * n_ + c];
      }
      const double rho = w0[0];
      const double p = pressure(w0);
      const double d = std::max(std::abs(w[0] - rho) / rho, std::abs(pressure(w) - p) / p);
      worst = std::isnan(d) ? std::numeric_limits<double>::infinity() : std::max(worst, d);
      if (m_ == 6) {
        const double nt = current_[nf + c];
        const double scale = std::max(std::abs(nt), rho * nt_inf_);
        const double dn = probe_[nf + c] - nt;
        if (!(dn >= -bounds_.decrease * scale && dn <= bounds_.increase * scale)) {
          worst = std::numeric_limits<double>::infinity();
        }
      }
    }
    flow_.state(std::span<const double>(probe_).first(nf), probe_state_);
    for (std::size_t c = 0; c < n_ && m_ == 6; ++c) nt_[c] = State(probe_[nf + c]);
    // Representable: what the stored state received, against eps v.
    std::array<runtime::ExactSum, 2> sums;
    for (std::size_t k = 0; k < m_; ++k) {
      for (std::size_t c = 0; c < n_; ++c) {
        const std::size_t i = k * n_ + c;
        const double want = eps * entry(i) / scale_[k];
        const double got =
            (k < 5 ? static_cast<double>(probe_state_[i]) - static_cast<double>(state_[i])
                   : static_cast<double>(nt_[c]) - static_cast<double>(nt_current_[c])) /
            scale_[k];
        sums[0].add((got - want) * (got - want));
        sums[1].add(want * want);
      }
    }
    const double ratio = std::sqrt(sums[0].value() / sums[1].value());
    if (worst <= 0.1) {
      if (!(ratio <= 1e-2)) {
        fail(halving, ratio);
        return;
      }
      // Also admissible: no positivity fallback of a face state switched from the held one (a
      // kink the difference would take for a derivative), and a finite residual.
      bool held;
      if (model_ == nullptr) {
        flow_.residual_frozen(probe_state_, res_);
        held = flow_.frozen_switches() == 0 && finite(res_);
      } else {
        flow_.residual_frozen(probe_state_, res_, [&] { model_->eddy_viscosity_frozen(nt_); });
        held = flow_.frozen_switches() == 0 && finite(res_);
        if (held) {
          model_->residual_frozen(nt_, res_nt_);
          held = finite(res_nt_);
        }
      }
      if (held) {
        halvings_ = std::max(halvings_, halving);
        representation_ = std::max(representation_, ratio);
        break;
      }
    }
    if (halving == 10) {
      fail(halving, ratio);
      return;
    }
    eps *= 0.5;
  }
  for (std::size_t k = 0; k < m_; ++k) {
    if (left_out_[k]) continue;
    for (std::size_t c = 0; c < n_; ++c) {
      const std::size_t i = k * n_ + c;
      const double r = k < 5 ? static_cast<double>(res_[i]) : static_cast<double>(res_nt_[c]);
      y[i] = shift_[c] * v[i] + (r - base_[i]) / eps;
    }
  }
}

template class MatrixFreeProduct<kernels::fp64>;
template class MatrixFreeProduct<kernels::fp32_acc64>;

}  // namespace autan::physics
