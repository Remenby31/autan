"""Reader of Gmsh MSH 4.1 ASCII files, linear quadrangles and hexahedra only.

Follows the format description of the Gmsh reference manual, section "MSH file format" (version
4.1): sections `$MeshFormat`, `$PhysicalNames`, `$Entities`, `$Nodes` and `$Elements` are read;
any other section is skipped, as the manual specifies for unrecognised headers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from autan_vv.errors import VVError

if TYPE_CHECKING:
    from pathlib import Path

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

QUAD, HEX = 3, 5
_NODES_PER_TYPE = {QUAD: 4, HEX: 8}


@dataclass(frozen=True)
class Msh:
    """Points (n, 3) in node-tag order and, per physical group name, its elements as zero-based
    point indices: quadrangles (k, 4) for 2D groups, hexahedra (k, 8) for 3D groups."""

    points: FloatArray
    groups: dict[str, IntArray]
    dims: dict[str, int]


def _bad(path: Path, what: str) -> VVError:
    return VVError(f"{path}: {what}", code="V0901")


def _sections(path: Path, lines: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    i = 0
    while i < len(lines):
        head = lines[i].strip()
        if not head:
            i += 1
            continue
        if not head.startswith("$") or head.startswith("$End"):
            raise _bad(path, f"expected a section header at line {i + 1}, got {head[:40]!r}")
        name = head[1:]
        end = f"$End{name}"
        try:
            j = lines.index(end, i + 1)
        except ValueError:
            raise _bad(path, f"section ${name} is not closed by {end}") from None
        if name in out:
            raise _bad(path, f"section ${name} appears twice")
        out[name] = lines[i + 1 : j]
        i = j + 1
    return out


def read(path: Path) -> Msh:
    """Read `path`. A file that breaks the format, or holds anything but linear quadrangles and
    hexahedra in its physical groups, raises V0901."""
    lines = [line.rstrip("\r") for line in path.read_text().split("\n")]
    sec = _sections(path, lines)
    for need in ("MeshFormat", "PhysicalNames", "Entities", "Nodes", "Elements"):
        if need not in sec:
            raise _bad(path, f"no ${need} section")
    fmt = sec["MeshFormat"][0].split()
    if fmt[:2] != ["4.1", "0"]:
        raise _bad(path, f"not MSH 4.1 ASCII (header {' '.join(fmt)!r})")

    names: dict[tuple[int, int], str] = {}
    body = sec["PhysicalNames"]
    for line in body[1 : 1 + int(body[0])]:
        d, t, rest = line.split(maxsplit=2)
        names[int(d), int(t)] = rest.strip().strip('"')

    entity_groups: dict[tuple[int, int], list[int]] = {}
    ent = sec["Entities"]
    counts = [int(v) for v in ent[0].split()]
    row = 1
    for dim, n in enumerate(counts):
        for _ in range(n):
            v = ent[row].split()
            row += 1
            tag = int(v[0])
            k = 4 if dim == 0 else 7  # tag, then a point or a bounding box
            entity_groups[dim, tag] = [int(t) for t in v[k + 1 : k + 1 + int(v[k])]]

    nodes = sec["Nodes"]
    n_blocks, n_nodes, _, max_tag = (int(v) for v in nodes[0].split())
    tags = np.empty(n_nodes, dtype=np.int64)
    xyz = np.empty((n_nodes, 3))
    row, filled = 1, 0
    for _ in range(n_blocks):
        _, _, parametric, count = (int(v) for v in nodes[row].split())
        if parametric:
            raise _bad(path, "parametric nodes are not read")
        tags[filled : filled + count] = [int(t) for t in nodes[row + 1 : row + 1 + count]]
        block = " ".join(nodes[row + 1 + count : row + 1 + 2 * count]).split()
        xyz[filled : filled + count] = np.asarray(block, dtype=np.float64).reshape(count, 3)
        row += 1 + 2 * count
        filled += count
    if filled != n_nodes or len(np.unique(tags)) != n_nodes:
        raise _bad(path, "node count or tags inconsistent with the $Nodes header")
    index = np.full(max_tag + 1, -1, dtype=np.int64)
    order = np.argsort(tags)
    index[tags[order]] = np.arange(n_nodes)
    points = xyz[order]

    groups: dict[str, list[IntArray]] = {}
    dims: dict[str, int] = {}
    elements = sec["Elements"]
    n_blocks = int(elements[0].split()[0])
    row = 1
    for _ in range(n_blocks):
        dim, tag, etype, count = (int(v) for v in elements[row].split())
        block = elements[row + 1 : row + 1 + count]
        row += 1 + count
        physical = entity_groups.get((dim, tag))
        if physical is None:
            raise _bad(path, f"element block on undeclared entity ({dim}, {tag})")
        if not physical:
            continue
        if etype not in _NODES_PER_TYPE:
            raise _bad(path, f"element type {etype} in a physical group; only 3 and 5 are read")
        width = _NODES_PER_TYPE[etype] + 1
        conn = np.asarray(" ".join(block).split(), dtype=np.int64).reshape(count, width)[:, 1:]
        if conn.min() < 1 or conn.max() > max_tag or np.any(index[conn] < 0):
            raise _bad(path, "an element references an undeclared node")
        for p in physical:
            name = names.get((dim, p))
            if name is None:
                raise _bad(path, f"physical group ({dim}, {p}) has no name")
            groups.setdefault(name, []).append(index[conn])
            dims[name] = dim
    return Msh(points, {k: np.concatenate(v) for k, v in groups.items()}, dims)
