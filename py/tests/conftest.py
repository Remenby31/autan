import contextlib
import copy
import hashlib
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = ROOT / "py" / "examples"
CRM_HL = (EXAMPLES / "crm_hl.py").read_text()
HEAT_EXCHANGER = (EXAMPLES / "heat_exchanger.py").read_text()

# Stand-ins for the example meshes: their SHA-256 is the source.sha256 of the fixtures.
MESH_FILES = {"crm-hl.cgns": b"crm-hl fixture", "heat-exchanger.msh": b"heat-exchanger fixture"}


def identity_fields() -> list[str]:
    page = (ROOT / "docs" / "contracts" / "results.md").read_text()
    section = page.split("### Identity\n", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^\d+\. `([a-z0-9_.]+)`$", section, flags=re.MULTILINE)


def identity_of(run: dict[str, Any]) -> str:
    parts = []
    for path in identity_fields():
        if path.startswith("start.") and run["start"] is None:
            continue
        value: Any = run
        for key in path.split("."):
            value = value[key]
        parts.append("" if value is None else str(value))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def mesh_document(name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((FIXTURES / f"{name}.meshinfo.json").read_text())
    return document


def ghosts_of(document: dict[str, Any]) -> str:
    """The ghost stencil of ``document``'s partition, ``"face2"`` without one."""
    return str((document.get("partition") or {}).get("ghost_stencil", "face2"))


def one_part(document: dict[str, Any]) -> dict[str, Any]:
    """The one-part partition autan-mesh-info writes without --ranks (mesh-info.md): the mesh
    itself, its periodic ghosts the part's ghosts and the cells its self-exchange sends, with
    the ghost stencil of ``document``'s partition."""
    counts = document["counts"]
    interior = counts["faces"] - sum(z["faces"] for z in document["boundary_zones"])
    ghosts = counts["periodic_ghosts"]
    return {
        "ranks": 1,
        "groups": "none",
        "ghost_stencil": ghosts_of(document),
        "cells_by_rank": [counts["cells"]],
        "ghosts_by_rank": [ghosts],
        "faces_by_rank": [counts["faces"]],
        "interior_faces_by_rank": [interior],
        "stencil_entries_by_rank": [counts["stencil_entries"]],
        "vertex_stencil_entries_by_rank": [counts["vertex_stencil_entries"]],
        "band_entries_by_rank": [counts["band_entries"]],
        "boundary_fit_entries_by_rank": [counts["boundary_fit_entries"]],
        "nodes_by_rank": [counts["nodes"]],
        "face_nodes_by_rank": [counts["face_nodes"]],
        "sent_cells_by_rank": [ghosts],
    }


def prepared(
    document: dict[str, Any], *, scale: float = 1.0, ranks: int = 1, ghost_stencil: str = ""
) -> dict[str, Any]:
    """``document`` as autan-mesh-info writes it for a run at ``scale`` metres per mesh unit on
    ``ranks`` ranks (mesh-info.md): its lines' scale and, on several ranks, a partition. The
    partition holds ``ghost_stencil`` ghosts, by default those of ``document``'s partition; on
    several ranks it is an even split without ghosts or groups: these tests plan something
    else."""
    document = copy.deepcopy(document)
    ghost_stencil = ghost_stencil or ghosts_of(document)
    if document.get("partition"):
        document["partition"]["ghost_stencil"] = ghost_stencil
    document["quality"]["lines"]["scale"] = scale
    if ranks == 1:
        document["partition"] = one_part(document)
    if ranks > 1:
        counts = document["counts"]
        interior = counts["faces"] - sum(z["faces"] for z in document["boundary_zones"])

        def split(total: int) -> list[int]:
            return [total // ranks + (r < total % ranks) for r in range(ranks)]

        document["partition"] = {
            "ranks": ranks,
            "groups": "none",
            "ghost_stencil": ghost_stencil,
            "cells_by_rank": split(counts["cells"]),
            "ghosts_by_rank": [0] * ranks,
            "faces_by_rank": split(counts["faces"]),
            "interior_faces_by_rank": split(interior),
            "stencil_entries_by_rank": split(counts["stencil_entries"]),
            "vertex_stencil_entries_by_rank": split(counts["vertex_stencil_entries"]),
            "band_entries_by_rank": split(counts["band_entries"]),
            "boundary_fit_entries_by_rank": split(counts["boundary_fit_entries"]),
            "nodes_by_rank": split(counts["nodes"]),
            "face_nodes_by_rank": split(counts["face_nodes"]),
            "sent_cells_by_rank": [0] * ranks,
        }
    return document


def place_meshes(directory: Path) -> None:
    for name, content in MESH_FILES.items():
        (directory / name).write_bytes(content)


@pytest.fixture
def crm_mesh() -> autan.MeshInfo:
    return autan.MeshInfo.load(FIXTURES / "crm-hl.meshinfo.json")


@pytest.fixture
def mesh_variant() -> Callable[..., autan.MeshInfo]:
    """A fixture MeshInfo after ``edit`` mutates its JSON document."""

    def build(name: str, edit: Callable[[dict[str, Any]], None]) -> autan.MeshInfo:
        document = copy.deepcopy(mesh_document(name))
        edit(document)
        if (document.get("partition") or {}).get("ranks") == 1:
            document["partition"] = one_part(document)
        # The finest multigrid level's edges are the interior faces the edit leaves, when the
        # edit keeps what they depend on (some edits break the document on purpose).
        with contextlib.suppress(KeyError, TypeError, IndexError):
            boundary = sum(zone["faces"] for zone in document["boundary_zones"])
            document["quality"]["multigrid"]["edges_by_level"][0] = (
                document["counts"]["faces"] - boundary
            )
        return autan.MeshInfo.from_document(document, autan.Location(file=name))

    return build


@pytest.fixture
def write_case(tmp_path: Path) -> Callable[..., Path]:
    """Write a case file next to the example mesh stand-ins."""
    place_meshes(tmp_path)

    def write(body: str, name: str = "case.py") -> Path:
        path = tmp_path / name
        path.write_text(body)
        return path

    return write


@pytest.fixture
def example(tmp_path: Path) -> Callable[[str], Path]:
    """Copy an example case next to the example mesh stand-ins."""
    place_meshes(tmp_path)

    def copy_example(name: str) -> Path:
        return Path(shutil.copy(EXAMPLES / name, tmp_path / name))

    return copy_example
