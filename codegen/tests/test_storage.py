"""Storage types of geometry in the generated Data<P> struct (kernel-generator.md, Precision)."""

from __future__ import annotations

import re
from pathlib import Path

from autan_codegen.generate import generate

ROOT = Path(__file__).resolve().parents[2]


def test_geometry_read_by_rel_keeps_the_geom_type_and_the_rest_is_stored_for_compute(
    tmp_path: Path,
) -> None:
    generate(ROOT / "kernels" / "advection_diffusion.py", tmp_path, ["cpu"])
    header = (tmp_path / "advection_diffusion.hpp").read_text()
    stored = {name: kind for kind, name in re.findall(r"const typename P::(\w+)\* (\w+);", header)}
    # The interior and Dirichlet fluxes take rel(x_f, x_c) of the centroids, in FP64.
    assert stored["cell_centroid"] == "geom"
    assert stored["face_centroid"] == "geom"
    # Everything else reaches bodies converted to the compute type: stored in its precision.
    for name in ("face_area", "face_moment", "cell_volume", "lsq_weight"):
        assert stored[name] == "geom_value", name
