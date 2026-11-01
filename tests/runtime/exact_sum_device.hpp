#pragma once

#include <array>
#include <vector>

#include "autan/runtime/exact_sum.hpp"

namespace autan_test {

/// The exact sums of `a` and of `b` (same length) by gpu::exact_sums on
/// device 0, with `blocks` blocks (0: the default).
std::array<autan::runtime::ExactSum, 2> device_sums(const std::vector<double>& a,
                                                    const std::vector<double>& b, int blocks);

}  // namespace autan_test
