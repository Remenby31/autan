#pragma once

#include <array>
#include <vector>

#include "autan/runtime/exact_sum.hpp"

namespace autan_test {

/// The exact sums of `a` and of `b` (same length) by gpu::exact_sums on
/// device 0, with `blocks` blocks (0: the default); `values`, when given,
/// receives each rounded on the device by gpu::exact_value.
std::array<autan::runtime::ExactSum, 2> device_sums(const std::vector<double>& a,
                                                    const std::vector<double>& b, int blocks,
                                                    std::array<double, 2>* values = nullptr);

/// gpu::exact_sums of `a` and `b` under a device predicate holding
/// `active`, into sums whose every byte is 0x5a beforehand; `evaluated`
/// receives the number of terms the functor was asked for.
std::array<autan::runtime::ExactSum, 2> device_sums_if(const std::vector<double>& a,
                                                       const std::vector<double>& b, int active,
                                                       unsigned long long& evaluated);

/// The words of `sum` rounded on device 0 by gpu::exact_value.
double device_value(const autan::runtime::ExactSum& sum);

}  // namespace autan_test
