"""A mesh is the same bits on every CPU: its seed decides it, not the hardware or Gmsh's build (#495)."""

import hashlib
import itertools
import math
from fractions import Fraction

import numpy as np
from autan_vv.meshes import families

# The perturbed structured hybrid n = 6, perturbation 0.3, seed 2026: the sha256 of its node
# coordinates (float64, little-endian). Measured identical on Linux aarch64 and x86-64 and on macOS arm64.
HYBRID_N6_SHA256 = "a274479d5cdf7f19be7598e9c295a9ece4cd49c35cbd455c0a58a3db59b9ef85"


def test_the_generator_s_doubles_are_an_integer_times_2_to_the_minus_53() -> None:
    # A guard on the dependency's contract the perturbation relies on (Generator.random, which
    # _uniform_in_ball draws): PCG64's 64-bit outputs, their top 53 bits scaled by a power of
    # two. No libm, no vector math library, nothing a CPU could round otherwise.
    raw = np.random.default_rng(2026).bit_generator.random_raw(1000)
    expected = (raw >> np.uint64(11)).astype(np.float64) * 2.0**-53
    assert np.array_equal(np.random.default_rng(2026).random(1000), expected)


def test_a_perturbed_mesh_is_the_same_bits_on_every_cpu() -> None:
    # The structured hybrid family needs no Gmsh, so this runs on aarch64 too, where Gmsh has no
    # wheel; the input is built with the module's own helpers for that reason. The file written by
    # generate_level has its own golden in test_meshes.py.
    points, cells = families._structured_hybrid(6)
    boundary = families._box_boundary(points, cells)
    moved = families.perturb(points, cells, boundary, 0.3, seed=2026)
    assert hashlib.sha256(moved.astype("<f8").tobytes()).hexdigest() == HYBRID_N6_SHA256


def test_canonical_tets_keep_each_tetrahedron_and_its_orientation_whatever_the_input_order() -> (
    None
):
    # Gmsh enumerates the same tetrahedra in another order on another build: any row order, and
    # any of the twelve orientation-keeping permutations of each row, must give the same rows.
    rng = np.random.default_rng(7)
    points = rng.random((40, 3))
    tets = np.array([rng.choice(40, 4, replace=False) for _ in range(200)])

    def signed_volume(t: np.ndarray) -> np.ndarray:
        a, b, c, d = (points[t[:, i]] for i in range(4))
        volume: np.ndarray = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
        return volume

    def is_even(p: tuple[int, ...]) -> bool:
        return sum(p[i] > p[j] for i in range(4) for j in range(i + 1, 4)) % 2 == 0

    canonical = families.canonical_tets(tets)
    by_nodes = {
        tuple(sorted(t)): np.sign(v) for t, v in zip(tets, signed_volume(tets), strict=True)
    }
    for t, v in zip(canonical, signed_volume(canonical), strict=True):
        assert by_nodes[tuple(sorted(t))] == np.sign(
            v
        )  # the same tetrahedron, the same orientation
    assert len(canonical) == len(tets)
    even = [p for p in itertools.permutations(range(4)) if is_even(p)]
    assert len(even) == 12
    for p in even:
        shuffled = tets[rng.permutation(len(tets))][:, list(p)]
        assert np.array_equal(families.canonical_tets(shuffled), canonical)
    # By hand: (1, 2, 0, 3) is (0, 1, 2, 3) rotated by a 3-cycle.
    assert families.canonical_tets(np.array([[1, 2, 0, 3]])).tolist() == [[0, 1, 2, 3]]


def test_the_manifest_s_cube_root_is_correctly_rounded() -> None:
    # h = cbrt(volume / n_cells) is written in every manifest: it must be the double nearest the
    # exact cube root, which no platform's libm guarantees for pow.
    rng = np.random.default_rng(3)
    for x in [*rng.random(300) * 1e-3, *rng.random(300), 1.0, 8.0, 0.125, 1e-12, 3.0]:
        y = families.cbrt(float(x))
        exact = Fraction(float(x))
        up, down = math.nextafter(y, math.inf), math.nextafter(y, 0.0)
        assert (
            ((Fraction(down) + Fraction(y)) / 2) ** 3
            <= exact
            <= ((Fraction(y) + Fraction(up)) / 2) ** 3
        )
    assert families.cbrt(8.0) == 2.0
