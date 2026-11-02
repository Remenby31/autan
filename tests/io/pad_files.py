"""Pads a file the runtime accepts to about BYTES bytes without changing what it describes, for
tests that compare a memory estimate with what reading the file costs (#358).

    python tests/io/pad_files.py KIND SOURCE OUT BYTES [--empty-zone NAME] [--skipped-elements N]

KIND is one of:
  gmsh       a Gmsh 4.1 file, ASCII or binary: a $NodeData section the reader skips, at its end
  polymesh   an ASCII OpenFOAM polyMesh directory: points that no face uses, then whitespace
  cgns       a CGNS/HDF5 file: grid points that no element uses, and a base Descriptor_t; with
             --skipped-elements N, also a BAR_2 section of N edges after its last element id,
             which the reader skips (mesh.md, Readers)
  json       a JSON document (MMS, initial or face states): whitespace after its opening brace;
             with --empty-zone, a face-state document also gets a zone with no faces
The mesh and state hashes of OUT differ from SOURCE's: compute them again.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import NoReturn

import h5py
import numpy as np


def fail(message: str) -> NoReturn:
    sys.exit(message)


def gmsh(source: Path, out: Path, size: int) -> None:
    text = source.read_bytes()
    line = b"1 0.0\n"
    count = max(0, (size - len(text) - 64) // len(line))
    # One scalar view (Gmsh reference manual, $NodeData): string tag, real tag time 0, integer
    # tags step 0, 1 component, `count` values.
    header = b'$NodeData\n1\n"padding"\n1\n0.0\n3\n0\n1\n' + str(count).encode() + b"\n"
    out.write_bytes(text + header + line * count + b"$EndNodeData\n")


def polymesh(source: Path, out: Path, size: int) -> None:
    shutil.copytree(source, out)
    points = (out / "points").read_text()
    if "format      binary" in points or "format binary" in points:
        fail("polymesh: only ASCII points can be padded")
    head = re.search(r"\n(\d+)\n\(\n", points)
    if head is None:
        fail("polymesh: no point count found")
    n = int(head.group(1))
    used = sum(p.stat().st_size for p in out.iterdir())
    line = "(9 9 9)\n"
    extra = max(0, (size - used) // 2 // len(line))
    end = points.rindex("\n)\n")
    padded = points[:end] + "\n" + (line * extra).rstrip("\n") + points[end:]
    padded = padded.replace(head.group(0), f"\n{n + extra}\n(\n", 1)
    (out / "points").write_text(padded)
    used = sum(p.stat().st_size for p in out.iterdir())
    with (out / "owner").open("a") as f:
        f.write("\n" * max(0, size - used))


def bar_section(zone: h5py.Group, count: int) -> None:
    """A BAR_2 section of `count` edges between the zone's first two grid points, numbered after
    its last element id, copied from the layout of one of its sections."""
    sections = [k for k in zone if zone[k].attrs.get("label") == b"Elements_t"]
    if not sections:
        fail("cgns: an element section to copy the node layout from is expected")
    last = max(int(zone[k]["ElementRange"][" data"][()].max()) for k in sections)
    zone.copy(sections[0], "SkippedEdges")
    node = zone["SkippedEdges"]
    name = node.attrs["name"]
    node.attrs.modify("name", np.array(b"SkippedEdges", dtype=name.dtype))
    header = node[" data"][()]
    header[0] = 3  # ElementType_t BAR_2 (CGNS SIDS)
    header[1] = 0
    node[" data"][...] = header
    ids = node["ElementRange"][" data"]
    node["ElementRange"][" data"][...] = np.array([last + 1, last + count], dtype=ids.dtype)
    connectivity = node["ElementConnectivity"][" data"]
    edges = np.tile(np.array([1, 2], dtype=connectivity.dtype), count)
    del node["ElementConnectivity"][" data"]
    node["ElementConnectivity"].create_dataset(" data", data=edges)
    for child in list(node):
        if child not in (" data", "ElementRange", "ElementConnectivity"):
            del node[child]


def cgns(source: Path, out: Path, size: int, skipped: int = 0) -> None:
    shutil.copyfile(source, out)
    with h5py.File(out, "r+") as f:
        bases = [k for k in f if f[k].attrs.get("label") == b"CGNSBase_t"]
        if len(bases) != 1:
            fail("cgns: one base expected")
        base = f[bases[0]]
        zones = [k for k in base if base[k].attrs.get("label") == b"Zone_t"]
        if len(zones) != 1:
            fail("cgns: one zone expected")
        zone = base[zones[0]]
        grid = zone["GridCoordinates"]
        extra = max(0, (size - out.stat().st_size) // 2 // 24)
        for name in ("CoordinateX", "CoordinateY", "CoordinateZ"):
            old = grid[name][" data"][()]
            del grid[name][" data"]
            grid[name].create_dataset(
                " data", data=np.concatenate([old, np.full(extra, 9.0, old.dtype)])
            )
        dims = zone[" data"][()]
        dims[0, 0] += extra
        zone[" data"][...] = dims
        descriptor = next((k for k in base if base[k].attrs.get("label") == b"Descriptor_t"), None)
        if descriptor is None:
            fail("cgns: a base Descriptor_t to copy the node layout from is expected")
        base.copy(descriptor, "Padding")
        node = base["Padding"]
        name = node.attrs["name"]
        node.attrs.modify("name", np.array(b"Padding", dtype=name.dtype))
        del node[" data"]
        text = max(1, size - out.stat().st_size)
        node.create_dataset(" data", data=np.full(text, ord(" "), dtype=np.int8))
        if skipped > 0:
            bar_section(zone, skipped)


def json_document(source: Path, out: Path, size: int, empty_zone: str | None) -> None:
    text = source.read_text()
    if empty_zone is not None:
        doc = json.loads(text)
        doc["zones"][empty_zone] = {"faces": 0, "centroids": [], "values": []}
        text = json.dumps(doc)
    brace = text.index("{") + 1
    out.write_text(text[:brace] + " " * max(0, size - len(text.encode())) + text[brace:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=["gmsh", "polymesh", "cgns", "json"])
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("bytes", type=int)
    parser.add_argument("--empty-zone")
    parser.add_argument("--skipped-elements", type=int, default=0)
    a = parser.parse_args()
    if a.kind == "gmsh":
        gmsh(a.source, a.out, a.bytes)
    elif a.kind == "polymesh":
        polymesh(a.source, a.out, a.bytes)
    elif a.kind == "cgns":
        cgns(a.source, a.out, a.bytes, a.skipped_elements)
    else:
        json_document(a.source, a.out, a.bytes, a.empty_zone)


if __name__ == "__main__":
    main()
