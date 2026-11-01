import copy
import json
import logging
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import autan
from autan.errors import CaseError, Location
from autan.schema import CaseSpec

from autan_ui.confine import Folder

log = logging.getLogger(__name__)

MESH_INFO = ".meshinfo.json"


class Meshes:
    """The MeshInfo files (`*.meshinfo.json`) of one directory, which the case editor plans
    against. Files are found by listing the directory, never by a path a request names."""

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
        """The MeshInfo listed as `name`; LookupError when there is none."""
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
    """A case document as the SDK reads it: its case.toml and, against `mesh_info`, its plan;
    or the SDK's diagnostics when it refuses the document."""
    where = Location(file="case.json")
    try:
        case = autan.Case.from_spec(CaseSpec.from_document(document, where), file="case.json")
        plan = _plan(document, mesh_info) if mesh_info is not None else None
    except CaseError as error:
        return {
            "ok": False,
            "diagnostics": [d.model_dump(mode="json") for d in error.diagnostics],
        }
    return {"ok": True, "toml": case.to_toml(), "plan": plan, "diagnostics": []}


def _plan(document: dict[str, Any], mesh_info: autan.MeshInfo) -> dict[str, Any]:
    # Planned from a copy in an empty scratch directory, so the SDK reads no file the case
    # names: the mesh path is cut to its name (plan reads only the MeshInfo) and every state
    # file points at a name that does not exist there, which plan reports as unreadable.
    planned = copy.deepcopy(document)
    planned["mesh"] = PurePosixPath(str(planned.get("mesh", ""))).name or "mesh"
    specs = [r.get("condition") for r in planned.get("boundaries", []) if isinstance(r, dict)]
    for spec in [*specs, planned.get("initial_state")]:
        if isinstance(spec, dict) and "path" in spec:
            spec["path"] = "states.json"
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "case.json"
        path.write_text(json.dumps(planned))
        return autan.load(path).plan(mesh_info).document()
