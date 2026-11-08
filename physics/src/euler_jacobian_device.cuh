#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "autan/kernels/device.cuh"
#include "autan/physics/euler_device.hpp"

namespace autan::physics::device {

// The blocks of the implicit step in E2's layout (docs/design/multigrid-gpu.md, Data per level):
// `diag` 25 x n_cells (JacAccum, FP64), `off` 25 x n_incidences (Jac) of the device topology in
// its CSR layout, zero on boundary incidences; component q of block e at q * n + e.
template <class J>
struct JacobianBlocks {
  kernels::gpu::DeviceArray<double> diag;
  kernels::gpu::DeviceArray<J> off;
};

// Euler<P>::assemble on the device, operation for operation: each diagonal block starts from
// the shift on its diagonal, then adds +-J D for each face of the cell in cell_faces order; each
// off-diagonal block is +J_right D_right in the global left cell's row and -J_left D_left in the
// right cell's. A product J D sums its five terms from +0.0 in increasing k. Every operation is
// correctly rounded and never contracted, as the host built with -ffp-contract=off. One thread
// per cell, so each block is written once.
template <class J>
__global__ void assemble_blocks(kernels::Topology t, const J* face_jac, const J* bnd_jac,
                                const J* dwdu, const double* shift, double* diag, J* off) {
  const auto c = static_cast<kernels::index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= t.n_cells) return;
  const auto nc = static_cast<std::size_t>(t.n_cells);
  const auto nf = static_cast<std::size_t>(t.n_faces);
  const auto ninc = static_cast<std::size_t>(t.n_incidences);
  const auto ni = t.n_interior_faces;
  // Block k of a structure-of-arrays Jac dataset over n entities, entry q of entity e.
  auto at = [](const J* v, std::size_t n, std::size_t e, std::size_t k, std::size_t q) {
    return static_cast<double>(v[(25 * k + q) * n + e]);
  };
  // out = out + sign * (J D), the product's terms summed in increasing k.
  auto add_product = [&](double* out, const J* jv, std::size_t jn, std::size_t je, std::size_t jk,
                         std::size_t d, double sign) {
    for (std::size_t r = 0; r < 5; ++r) {
      for (std::size_t col = 0; col < 5; ++col) {
        double s = 0.0;
        for (std::size_t k = 0; k < 5; ++k) {
          s = __dadd_rn(s,
                        __dmul_rn(at(jv, jn, je, jk, 5 * r + k), at(dwdu, nc, d, 0, 5 * k + col)));
        }
        out[5 * r + col] = __dadd_rn(out[5 * r + col], __dmul_rn(sign, s));
      }
    }
  };
  auto left = [&](kernels::index_t f) {
    const bool rev = t.face_reversed != nullptr && t.face_reversed[f] != 0;
    return rev ? t.neighbour[f] : t.owner[f];
  };
  const auto cs = static_cast<std::size_t>(c);
  double dg[25] = {};
  for (std::size_t i = 0; i < 5; ++i) dg[6 * i] = __dadd_rn(dg[6 * i], shift[cs]);
  for (auto k = t.cell_face_offsets[c]; k < t.cell_face_offsets[c + 1]; ++k) {
    const auto slot = static_cast<std::size_t>(k);
    const auto f = t.cell_faces[slot];
    const auto fs = static_cast<std::size_t>(f);
    double o[25] = {};
    if (f >= ni) {
      add_product(dg, bnd_jac, nf, fs, 0, cs, 1.0);
    } else if (left(f) == c) {
      add_product(dg, face_jac, nf, fs, 0, cs, 1.0);
      const auto other = static_cast<std::size_t>(t.owner[f] == c ? t.neighbour[f] : t.owner[f]);
      add_product(o, face_jac, nf, fs, 1, other, 1.0);
    } else {
      add_product(dg, face_jac, nf, fs, 1, cs, -1.0);
      const auto other = static_cast<std::size_t>(t.owner[f] == c ? t.neighbour[f] : t.owner[f]);
      add_product(o, face_jac, nf, fs, 0, other, -1.0);
    }
    for (std::size_t q = 0; q < 25; ++q) off[q * ninc + slot] = static_cast<J>(o[q]);
  }
  for (std::size_t q = 0; q < 25; ++q) diag[q * nc + cs] = dg[q];
}

// The blocks in the host's terms: the diagonal block of each cell, row-major, then per cell its
// incidences in order, each with its face and its block, row-major (as converted from Jac).
template <class J>
DeviceStepperBlocks download_blocks(const kernels::Topology& t, const JacobianBlocks<J>& b,
                                    void* stream) {
  const auto nc = static_cast<std::size_t>(t.n_cells);
  const auto ninc = static_cast<std::size_t>(t.n_incidences);
  const auto s = static_cast<cudaStream_t>(stream);
  auto ok = [](cudaError_t e) {
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("download blocks: ") + cudaGetErrorString(e));
    }
  };
  auto copy = [&](auto* host, const auto* device, std::size_t n) {
    ok(kernels::gpu::detail::to_host(host, device, n * sizeof(*host), s));
  };
  std::vector<double> diag(25 * nc);
  std::vector<J> off(25 * ninc);
  std::vector<kernels::offset_t> offsets(nc + 1);
  std::vector<kernels::index_t> faces(ninc);
  copy(diag.data(), b.diag.get(), diag.size());
  copy(off.data(), b.off.get(), off.size());
  copy(offsets.data(), t.cell_face_offsets, offsets.size());
  copy(faces.data(), t.cell_faces, faces.size());
  ok(kernels::gpu::detail::synchronize(s));
  DeviceStepperBlocks out;
  out.diagonal.resize(25 * nc);
  out.offsets.assign(1, 0);
  for (std::size_t c = 0; c < nc; ++c) {
    for (std::size_t q = 0; q < 25; ++q) out.diagonal[25 * c + q] = diag[q * nc + c];
    for (auto k = offsets[c]; k < offsets[c + 1]; ++k) {
      const auto slot = static_cast<std::size_t>(k);
      out.faces.push_back(faces[slot]);
      for (std::size_t q = 0; q < 25; ++q) {
        out.off_diagonal.push_back(static_cast<double>(off[q * ninc + slot]));
      }
    }
    out.offsets.push_back(static_cast<kernels::offset_t>(out.faces.size()));
  }
  return out;
}

}  // namespace autan::physics::device
