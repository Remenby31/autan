import copy
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import autan
from autan.errors import CaseError, Location
from autan.schema import CaseSpec

from autan_ui.confine import Folder

log = logging.getLogger(__name__)

MESH_INFO = ".meshinfo.json"


# The MeshInfo files (`*.meshinfo.json`) of one directory, which the case editor plans
# against; found by listing the directory, never by a path a request names.
class Meshes:
    def __init__(self, root: Path | None) -> None:
        self.root = root

    def names(self) -> list[str]:
        if self.root is None or not self.root.is_dir():
            return []
        return sorted(
            p.name[: -len(MESH_INFO)]
            for p in self.root.iterdir()
            if p.name.endswith(MESH_INFO) and not p.name.startswith(".")
        )

    def read(self, name: str) -> autan.MeshInfo:
        if self.root is None or name not in self.names():
            msg = f"no MeshInfo {name!r}"
            raise LookupError(msg)
        with Folder(self.root) as folder:
            document = json.loads(folder.read(name + MESH_INFO).decode())
        return autan.MeshInfo.from_document(document, Location(file=name + MESH_INFO))

    def summaries(self) -> list[dict[str, Any]]:
        out = []
        for name in self.names():
            try:
                info = self.read(name).document()
            except (LookupError, OSError, ValueError, CaseError) as error:
                log.warning("MeshInfo %s skipped: %s", name, error)
                continue
            out.append(
                {
                    "name": name,
                    "source": info["source"],
                    "counts": info["counts"],
                    "boundary_zones": info["boundary_zones"],
                    "cell_zones": info["cell_zones"],
                }
            )
        return out


def render(document: dict[str, Any], mesh_info: autan.MeshInfo | None) -> dict[str, Any]:
    where = Location(file="case.json")
    try:
        case = autan.Case.from_spec(CaseSpec.from_document(document, where), file="case.json")
        plan = _plan(document, mesh_info) if mesh_info is not None else None
        if plan is not None:
            # The plan of the copy, told as the plan of the case returned.
            plan["case"] = {"name": case.spec.name, "sha256": case.spec.hash()}
            plan["mesh"]["path"] = case.spec.mesh
    except CaseError as error:
        return {
            "ok": False,
            "diagnostics": [d.model_dump(mode="json") for d in error.diagnostics],
        }
    return {
        "ok": True,
        "python": case.to_python(),
        "toml": case.to_toml(),
        "plan": plan,
        "diagnostics": [],
    }


def _plan(document: dict[str, Any], mesh_info: autan.MeshInfo) -> dict[str, Any]:
    # Planned from a copy in an empty scratch directory, so the SDK reads no file the case
    # names: the mesh is a name that does not exist there (plan reads only the MeshInfo) and
    # every file a case names by `path` points at one that does not exist either, which plan
    # reports as unreadable.
    planned = _repointed(copy.deepcopy(document))
    planned["mesh"] = "mesh-not-read"
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "case.json"
        path.write_text(json.dumps(planned))
        return autan.load(path).plan(mesh_info).document()


def _repointed(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: "file-not-read.json" if k == "path" else _repointed(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_repointed(v) for v in value]
    return value
