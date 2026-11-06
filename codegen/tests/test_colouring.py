"""Face colourings of the coloured strategy (kernels::colour_faces): the greedy colouring by
definition, and its memory on a cell with many faces (docs/design/setup-memory.md)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import autan_spike as s
import meshes
import pytest
from conftest import KINDS


def _greedy(m: s.Mesh, first: int, last: int) -> list[int]:
    """Faces in order, each the least colour no earlier face of the range at its cells has."""
    taken: dict[int, set[int]] = {}
    out = []
    for f in range(first, last):
        cells = [int(m.owner[f])] + ([int(m.neighbour[f])] if f < m.n_interior else [])
        used = set().union(*(taken.get(c, set()) for c in cells))
        k = 0
        while k in used:
            k += 1
        for c in cells:
            taken.setdefault(c, set()).add(k)
        out.append(k)
    return out


@pytest.mark.parametrize("kind", KINDS)
def test_faces_take_the_least_colour_free_at_their_cells(kind: str) -> None:
    m = s.mesh(meshes.build(kind, 4, 0.15, seed=3))
    assert s.face_colours(m).tolist() == _greedy(m, 0, m.n_interior)
    assert s.face_colours(m, interior=False).tolist() == _greedy(m, m.n_interior, m.n_faces)


_HUB = """
import json, resource, sys
import autan_spike as s, meshes
m = s.mesh(meshes.hub_row(int(sys.argv[1])))
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
colours = s.face_colours(m)
grown = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
unit = 1 if sys.platform == "darwin" else 1024
print(json.dumps({"grown": grown * unit, "colours": int(colours.max()) + 1}))
"""


def test_a_cell_with_many_faces_colours_in_memory_of_its_faces() -> None:
    # A hub cell with d faces gives its d neighbours colours up to d: the colouring holds the
    # faces and colours (docs/design/setup-memory.md), not a set of colours per cell, which
    # grew the peak by 77 MB here (macOS, measured).
    d = 32768
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    out = subprocess.run(
        [sys.executable, "-c", _HUB, str(d)],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    found = json.loads(out.stdout)
    assert found["colours"] >= d  # the hub's faces all differ
    assert found["grown"] < 8 << 20
