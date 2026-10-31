"""Kernel DSL: precision roles, value types, mesh data, pointwise bodies and loop declarations.

A kernel module declares data (`cells.field`, `faces.field`, `param`), pointwise bodies
(`@pointwise`) and loops (`face_loop`, `cell_loop`, `cell_reduce`). The generator reads every loop
bound to a module-level name and emits one C++ function per loop, backend and race strategy.
"""

from __future__ import annotations

import enum
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from autan_codegen.errors import Code, CodegenError

if TYPE_CHECKING:
    from collections.abc import Callable


class Role(enum.Enum):
    """Precision role of a value. A policy maps each role to a storage type."""

    STATE = "state"
    GRAD = "grad"
    RESIDUAL = "residual"
    GEOM = "geom"
    PARAM = "param"


@dataclass(frozen=True)
class Type:
    """A role and a shape: () scalar, (n,) vector, (3, 3) matrix."""

    role: Role
    shape: tuple[int, ...] = ()

    @property
    def size(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


class RoleAnnotation:
    """Annotation object: `State` is a scalar, `State.Vec3` a 3-vector, `Geom.Mat3` a 3x3 matrix."""

    def __init__(self, role: Role) -> None:
        self.scalar = Type(role)
        self.Vec3 = Type(role, (3,))
        self.Vec5 = Type(role, (5,))
        self.Mat3 = Type(role, (3, 3))
        self.Mat5 = Type(role, (5, 5))


State = RoleAnnotation(Role.STATE)
Grad = RoleAnnotation(Role.GRAD)
Residual = RoleAnnotation(Role.RESIDUAL)
Geom = RoleAnnotation(Role.GEOM)
Param = RoleAnnotation(Role.PARAM)


def as_type(annotation: object) -> Type:
    if isinstance(annotation, RoleAnnotation):
        return annotation.scalar
    if isinstance(annotation, Type):
        return annotation
    raise CodegenError(Code.NOT_A_DSL_TYPE, f"not a DSL type: {annotation!r}")


# ---------------------------------------------------------------------------- mesh data


class MeshSet(enum.Enum):
    CELLS = "cells"
    FACES = "faces"
    INCIDENCES = "incidences"


@dataclass(eq=False)
class Dataset:
    """An array over a mesh set. Geometry datasets are filled by the runtime and are passive."""

    name: str
    set: MeshSet
    type: Type
    geometry: bool = False

    @property
    def left(self) -> Binding:
        return Binding(self, "left")

    @property
    def right(self) -> Binding:
        return Binding(self, "right")

    @property
    def self_(self) -> Binding:
        return Binding(self, "self")

    def other(self, boundary: Dataset) -> Binding:
        """Value across each incidence of a cell; `boundary` (a face dataset) on boundary faces."""
        return Binding(self, "other", boundary)


@dataclass(eq=False)
class Parameter:
    name: str
    type: Type


@dataclass(frozen=True, eq=False)
class Binding:
    """How a body argument is loaded: directly, or through the face/incidence -> cell maps."""

    dataset: Dataset | Parameter
    via: str = "direct"  # direct | left | right | self | other
    boundary: Dataset | None = None


class _SetDecl:
    def __init__(self, s: MeshSet) -> None:
        self._set = s

    def field(self, name: str, annotation: object) -> Dataset:
        return Dataset(name, self._set, as_type(annotation))


cells = _SetDecl(MeshSet.CELLS)
faces = _SetDecl(MeshSet.FACES)


def param(name: str, annotation: object) -> Parameter:
    t = as_type(annotation)
    if t.role is not Role.PARAM:
        raise CodegenError(Code.NOT_A_DSL_TYPE, f"parameter {name} must have role Param")
    return Parameter(name, t)


class geometry:
    """FP64 geometry datasets, computed by the runtime from the core's Geometry."""

    face_area = Dataset("face_area", MeshSet.FACES, Geom.Vec3, geometry=True)
    face_centroid = Dataset("face_centroid", MeshSet.FACES, Geom.Vec3, geometry=True)
    # Sum over the face's fan triangles of (triangle centroid - face centroid) (x) area vector:
    # zero on planar faces, makes the flux of a linear field through a warped face exact.
    face_moment = Dataset("face_moment", MeshSet.FACES, Geom.Mat3, geometry=True)
    face_normal = Dataset("face_normal", MeshSet.FACES, Geom.Vec3, geometry=True)
    face_area_magnitude = Dataset("face_area_magnitude", MeshSet.FACES, Geom.scalar, geometry=True)
    # Face centroid minus owner (left) and neighbour (right) centroid; right is zero on boundaries.
    face_r_left = Dataset("face_r_left", MeshSet.FACES, Geom.Vec3, geometry=True)
    face_r_right = Dataset("face_r_right", MeshSet.FACES, Geom.Vec3, geometry=True)
    cell_centroid = Dataset("cell_centroid", MeshSet.CELLS, Geom.Vec3, geometry=True)
    cell_volume = Dataset("cell_volume", MeshSet.CELLS, Geom.scalar, geometry=True)
    lsq_weight = Dataset("lsq_weight", MeshSet.INCIDENCES, Geom.Vec3, geometry=True)


# ---------------------------------------------------------------------------- bodies


@dataclass(eq=False)
class Body:
    """A pointwise function in the restricted subset; parsed lazily into the IR."""

    fn: Callable[..., Any]

    @property
    def name(self) -> str:
        return self.fn.__name__

    @property
    def source_file(self) -> str:
        return inspect.getsourcefile(self.fn) or "<unknown>"

    def __call__(self, *_args: object) -> object:
        raise CodegenError(
            Code.UNSUPPORTED_CONSTRUCT,
            f"{self.name} is a DSL body; it can only be called from another",
        )


def pointwise(fn: Callable[..., Any]) -> Body:
    """Declare a pointwise body. Its source is parsed into the generator's IR."""
    return Body(fn)


# ---------------------------------------------------------------------------- loops


@dataclass(frozen=True, eq=False)
class Inc:
    """Increment a cell dataset through the face's owner (`left`) and neighbour (`right`)."""

    dataset: Dataset
    left: int = 1
    right: int | None = None


@dataclass(frozen=True, eq=False)
class Write:
    dataset: Dataset


@dataclass(frozen=True, eq=False)
class Sum:
    dataset: Dataset


@dataclass(frozen=True, eq=False)
class Min:
    dataset: Dataset


@dataclass(frozen=True, eq=False)
class Max:
    dataset: Dataset


Accumulator = Sum | Min | Max


@dataclass(eq=False)
class Loop:
    kind: str  # face | boundary | cell | reduce
    body: Body
    bindings: dict[str, Binding]
    outputs: list[Inc | Write | Accumulator]
    finalise: Body | None = None
    finalise_bindings: dict[str, Binding] = field(default_factory=dict)
    name: str = ""


def _bindings(raw: dict[str, object]) -> dict[str, Binding]:
    out: dict[str, Binding] = {}
    for k, v in raw.items():
        if isinstance(v, Binding):
            out[k] = v
        elif isinstance(v, Dataset | Parameter):
            out[k] = Binding(v)
        else:
            raise CodegenError(
                Code.INVALID_BINDING, f"argument {k}: not a dataset, binding or parameter: {v!r}"
            )
    return out


def _outs(out: object) -> list[Any]:
    return list(out) if isinstance(out, list | tuple) else [out]


def face_loop(body: Body, *, out: object, **args: object) -> Loop:
    """Loop over interior faces; `out` increments owner and neighbour cells."""
    outs = _outs(out)
    for o in outs:
        if not isinstance(o, Inc) or o.right is None:
            raise CodegenError(
                Code.INVALID_LOOP_OUTPUT, "interior face loop outputs are Inc with both signs"
            )
    return Loop("face", body, _bindings(args), outs)


def boundary_loop(body: Body, *, out: object, **args: object) -> Loop:
    """Loop over boundary faces; `out` increments the owner cell."""
    outs = _outs(out)
    for o in outs:
        if not isinstance(o, Inc) or o.right is not None:
            raise CodegenError(
                Code.INVALID_LOOP_OUTPUT, "boundary loop outputs are Inc of the owner only"
            )
    return Loop("boundary", body, _bindings(args), outs)


def cell_loop(body: Body, *, out: object, **args: object) -> Loop:
    """Loop over owned cells; `out` writes cell datasets."""
    outs = _outs(out)
    for o in outs:
        if not isinstance(o, Write):
            raise CodegenError(Code.INVALID_LOOP_OUTPUT, "cell loop outputs are Write")
    return Loop("cell", body, _bindings(args), outs)


def cell_reduce(
    body: Body,
    *,
    acc: object,
    finalise: Body | None = None,
    finalise_args: dict[str, object] | None = None,
    **args: object,
) -> Loop:
    """Reduce `body` over the faces of each owned cell into accumulators, then write them.

    The body's outputs feed `acc` in order (Sum, Min or Max, each naming its target cell dataset).
    With `finalise`, the accumulated values are passed to it as its first arguments, followed by
    `finalise_args`, and its outputs are written to the accumulators' datasets instead.
    """
    accs = _outs(acc)
    for a in accs:
        if not isinstance(a, Sum | Min | Max):
            raise CodegenError(
                Code.INVALID_LOOP_OUTPUT, "cell_reduce accumulators are Sum, Min or Max"
            )
    return Loop(
        "reduce", body, _bindings(args), accs, finalise, _bindings(dict(finalise_args or {}))
    )


# ---------------------------------------------------------------------------- functions


def _intrinsic(*_: object) -> object:
    raise CodegenError(Code.UNSUPPORTED_CONSTRUCT, "DSL intrinsics only exist inside parsed bodies")


dot = cross = norm = sqrt = exp = log = sin = cos = rel = transpose = Vec3 = _intrinsic
