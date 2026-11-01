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


MODULE = """
from autan_codegen.dsl import Geom, Grad, Sum, cell_reduce, cells, geometry, pointwise, rel

area_sum = cells.field("area_sum", Geom)
offsets = cells.field("offsets", Grad.Vec3)


@pointwise
def offset(xc: Geom.Vec3, xn: Geom.Vec3) -> Grad.Vec3:
    return rel(xn, xc)


# rel() reads the boundary dataset of other(boundary=...) too.
spread = cell_reduce(offset, xc=geometry.cell_centroid.self_,
                     xn=geometry.cell_centroid.other(boundary=geometry.face_centroid), acc=Sum(offsets))


@pointwise
def area(a: Geom) -> Geom:
    return a


# A Geom dataset a kernel writes keeps the Geom type.
surface = cell_reduce(area, a=geometry.face_area_magnitude, acc=Sum(area_sum))
"""


def test_boundary_sources_of_rel_and_written_geom_keep_the_geom_type(tmp_path: Path) -> None:
    src = tmp_path / "geom_storage.py"
    src.write_text(MODULE)
    generate(src, tmp_path / "gen", ["cpu"])
    header = (tmp_path / "gen" / "geom_storage.hpp").read_text()
    stored = {name: kind for kind, name in re.findall(r"typename P::(\w+)\* (\w+);", header)}
    assert stored["face_centroid"] == "geom"
    assert stored["cell_centroid"] == "geom"
    assert stored["area_sum"] == "geom"
    assert stored["face_area_magnitude"] == "geom_value"
