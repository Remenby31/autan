"""A perturbed mesh is the same bits on every CPU: its seed decides it, not the hardware (#495)."""

import hashlib

import numpy as np
from autan_vv.meshes import families

# The perturbed structured hybrid n = 6, perturbation 0.3, seed 2026: the sha256 of its node
# coordinates (float64, little-endian). Measured identical on aarch64, x86-64 and macOS arm64.
HYBRID_N6_SHA256 = "a274479d5cdf7f19be7598e9c295a9ece4cd49c35cbd455c0a58a3db59b9ef85"


def test_the_generator_s_doubles_are_an_integer_times_2_to_the_minus_53() -> None:
    # The perturbation draws only these: PCG64's 64-bit outputs, their top 53 bits scaled by a
    # power of two. No libm, no vector math library, nothing a CPU could round otherwise.
    raw = np.random.default_rng(2026).bit_generator.random_raw(1000)
    expected = (raw >> np.uint64(11)).astype(np.float64) * 2.0**-53
    assert np.array_equal(np.random.default_rng(2026).random(1000), expected)


def test_a_perturbed_mesh_is_the_same_bits_on_every_cpu() -> None:
    # The structured hybrid family needs no Gmsh, so this runs on aarch64 too, where Gmsh has no
    # wheel; the input is built with the module's own helpers for that reason.
    points, cells = families._structured_hybrid(6)
    boundary = families._box_boundary(points, cells)
    moved = families.perturb(points, cells, boundary, 0.3, seed=2026)
    assert hashlib.sha256(moved.astype("<f8").tobytes()).hexdigest() == HYBRID_N6_SHA256
