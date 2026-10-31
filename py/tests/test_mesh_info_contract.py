"""The real producer through the real consumer: `autan-mesh-info` output read by the SDK.

Expected facts come from tests/data/mesh/README.md, which describes meshes written by
Gmsh and OpenFOAM, not by our readers.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import autan
import pytest
from autan.meshinfo import mesh_sha256

from tests.conftest import ROOT

MESHES = ROOT / "tests" / "data" / "mesh"
CELLS = {"hex": 27, "prism": 54, "pyramid": 48, "tet": 48, "hanging": 101}


def _producer() -> Path | None:
    candidates = [
        os.environ.get("AUTAN_MESH_INFO"),
        ROOT / "build" / "io" / "autan-mesh-info",
        shutil.which("autan-mesh-info"),
    ]
    return next((Path(c) for c in candidates if c and Path(c).is_file()), None)


PRODUCER = _producer()
pytestmark = pytest.mark.skipif(
    PRODUCER is None,
    reason="autan-mesh-info not built (cmake --build build --target autan-mesh-info, "
    "or set AUTAN_MESH_INFO)",
)
SOURCES = sorted(
    p for p in MESHES.glob("*/*") if p.name.startswith("polyMesh") or p.suffix == ".msh"
)


def mesh_info(source: Path) -> autan.MeshInfo:
    assert PRODUCER is not None
    done = subprocess.run(
        [str(PRODUCER), "--json", str(source)], capture_output=True, text=True, check=True
    )
    return autan.MeshInfo.from_document(json.loads(done.stdout), autan.Location(file=str(source)))


@pytest.mark.parametrize("source", SOURCES, ids=[str(p.relative_to(MESHES)) for p in SOURCES])
def test_producer_output_is_a_valid_mesh_info(source: Path) -> None:
    info = mesh_info(source)
    assert info.source.sha256 == mesh_sha256(source)
    mesh = source.parent.name
    if mesh in CELLS:
        assert info.counts.cells == CELLS[mesh]
    types = {z.name: z.type for z in info.boundary_zones}
    fmt = info.source.format
    if mesh != "hanging" and fmt in ("polymesh", "fluent"):
        assert (types["ymin"], types["zmax"]) == ("wall", "symmetry")
    if fmt == "gmsh":
        assert set(types.values()) == {None}
    if fmt == "fluent":
        assert [(z.name, z.type) for z in info.cell_zones] == [("fluid-1", "fluid")]
        assert {t for n, t in types.items() if n not in ("ymin", "zmax", "walls")} == {"outlet"}
    if mesh == "hybrid" and fmt == "polymesh":
        assert sorted(z.name for z in info.cell_zones) == ["lower", "upper"]
        assert [i.zones for i in info.interfaces] == [("lower", "upper")]


def test_plan_checks_the_hash_of_a_real_mesh(tmp_path: Path) -> None:
    source = MESHES / "hybrid" / "polyMesh"
    info = mesh_info(source)
    case = tmp_path / "case.py"
    case.write_text(
        "import autan\nfrom autan.units import K, Pa\n"
        f"case = autan.Case('hybrid', mesh={str(source)!r})\n"
        "case.physics = autan.Compressible(turbulence=autan.Laminar())\n"
        "case.freestream(mach=0.2, pressure=101325 * Pa, temperature=300 * K)\n"
        "for zone in ('xmin', 'xmax', 'ymax', 'zmin'):\n"
        "    case.boundary(zone).farfield()\n"
    )
    result = autan.load(case).plan(info)
    assert result.diagnostics == ()
    assert {b.zone: b.provenance for b in result.boundaries} == {
        "xmin": "code",
        "xmax": "code",
        "ymax": "code",
        "zmin": "code",
        "ymin": "mesh",
        "zmax": "mesh",
    }
