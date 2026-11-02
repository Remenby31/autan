#pragma once

#include <array>
#include <span>
#include <vector>

namespace autan::physics {

/// Least-squares gradient weights of a cell over the offsets d_j from its centroid to its face
/// neighbours' (docs/design/reconstruction-jacobian.md): w_j = B+ e_j / |d_j|, e_j = d_j / |d_j|,
/// B = sum_j e_j e_j^T, B+ its pseudo-inverse with eigenvalues below 1e-10 of the largest
/// dropped. grad = sum_j w_j (phi_j - phi_c) is exact for a linear field along every direction
/// the offsets span, however stretched or rotated, and has no component along a direction they
/// do not span. A zero offset gets a zero weight.
std::vector<std::array<double, 3>> face_fit_weights(std::span<const std::array<double, 3>> offsets);

}  // namespace autan::physics
