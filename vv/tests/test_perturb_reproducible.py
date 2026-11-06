"""A mesh is the same bits on every CPU: its seed decides it, not the hardware or Gmsh's build (#495)."""

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


def test_canonical_tets_keep_each_tetrahedron_and_its_orientation_whatever_the_input_order() -> (
    None
):
    # Gmsh enumerates the same tetrahedra in another order on another build: any order, and any
    # orientation-keeping rotation of each, must give the same rows.
    rng = np.random.default_rng(7)
    points = rng.random((40, 3))
    tets = np.array([rng.choice(40, 4, replace=False) for _ in range(200)])

    def signed_volume(t: np.ndarray) -> np.ndarray:
        a, b, c, d = (points[t[:, i]] for i in range(4))
        return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)

    canonical = families.canonical_tets(tets)
    by_nodes = {
        tuple(sorted(t)): np.sign(v) for t, v in zip(tets, signed_volume(tets), strict=True)
    }
    for t, v in zip(canonical, signed_volume(canonical), strict=True):
        assert by_nodes[tuple(sorted(t))] == np.sign(
            v
        )  # the same tetrahedron, the same orientation
    assert len(canonical) == len(tets)
    shuffled = tets[rng.permutation(len(tets))]
    rotated = shuffled[:, [1, 2, 0, 3]]  # a 3-cycle: an even permutation
    assert np.array_equal(families.canonical_tets(rotated), canonical)
