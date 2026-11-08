#pragma once

#include <array>
#include <vector>

#include "autan/core/wall_distance.hpp"

namespace autan::test {

/// DeviceWallSurface::distance of `points`, on a non-blocking stream.
std::vector<double> device_distances(const WallSurface& surface,
                                     const std::vector<std::array<double, 3>>& points);

}  // namespace autan::test
