"""Load a kernel module and check each loop's bindings against its body: the backend-neutral model."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from autan_codegen import dsl
from autan_codegen.errors import Code, CodegenError
from autan_codegen.frontend import Signature, parse

if TYPE_CHECKING:
    from pathlib import Path

    from autan_codegen.ir import Function

ROLE_TYPE = {
    dsl.Role.STATE: "state",
    dsl.Role.GRAD: "grad",
    dsl.Role.RESIDUAL: "residual",
    dsl.Role.GEOM: "geom",
    dsl.Role.PARAM: "param",
}


@dataclass
class LoopModel:
    name: str
    loop: dsl.Loop
    fn: Function
    sig: Signature
    fin_fn: Function | None
    fin_sig: Signature | None
    # Finalise call order: ("acc", k) or ("bind", name); default: every accumulator, then bindings.
    fin_args: list[tuple[str, int | str]] | None = None
    # Adjoint of a cell reduction (kind "reduce_adjoint"): see derive.reduce_adjoint.
    radj: object = None

    @property
    def kind(self) -> str:
        return self.loop.kind

    def reads(self) -> list[dsl.Dataset | dsl.Parameter]:
        out: list[dsl.Dataset | dsl.Parameter] = []
        for b in [*self.loop.bindings.values(), *self.loop.finalise_bindings.values()]:
            out.append(b.dataset)
            if b.boundary is not None:
                out.append(b.boundary)
        return out

    def writes(self) -> list[dsl.Dataset]:
        return [o.dataset for o in self.loop.outputs]


@dataclass
class Module:
    name: str
    path: Path
    loops: list[LoopModel]
    datasets: list[dsl.Dataset]
    params: list[dsl.Parameter]
    bodies: list[tuple[Function, Signature]]
    extra_bodies: list[tuple[Function, list[dsl.Type]]] = field(default_factory=list)


ALLOWED_VIA = {
    "face": {
        dsl.MeshSet.CELLS: {"left", "right"},
        dsl.MeshSet.FACES: {"direct"},
    },
    "reduce_stencil": {
        dsl.MeshSet.CELLS: {"self", "other"},
        dsl.MeshSet.STENCIL: {"direct"},
    },
    "boundary": {
        dsl.MeshSet.CELLS: {"left"},
        dsl.MeshSet.FACES: {"direct"},
    },
    "cell": {dsl.MeshSet.CELLS: {"direct"}},
    "reduce": {
        dsl.MeshSet.CELLS: {"self", "other"},
        dsl.MeshSet.INCIDENCES: {"direct"},
    },
}


def _check_bindings(
    where: str, kind: str, sig: Signature, bindings: dict[str, dsl.Binding], skip: int = 0
) -> None:
    names = sig.names[skip:]
    if set(names) != set(bindings):
        missing = sorted(set(names) - set(bindings))
        extra = sorted(set(bindings) - set(names))
        raise CodegenError(
            Code.INVALID_BINDING,
            f"bindings do not match arguments: missing {missing}, extra {extra}",
            where,
        )
    for name, t in zip(sig.names[skip:], sig.types[skip:], strict=True):
        b = bindings[name]
        ds = b.dataset
        if ds.type.shape != t.shape or ds.type.role is not t.role:
            raise CodegenError(
                Code.INVALID_BINDING,
                f"argument {name} is {t.role.value}{t.shape}, data {ds.name} is {ds.type.role.value}{ds.type.shape}",
                where,
            )
        if isinstance(ds, dsl.Parameter):
            if b.via != "direct":
                raise CodegenError(
                    Code.INVALID_BINDING, f"parameter {ds.name} is read directly", where
                )
            continue
        allowed = ALLOWED_VIA[kind].get(ds.set, set())
        if b.via not in allowed:
            raise CodegenError(
                Code.INVALID_BINDING,
                f"{ds.set.value} data {ds.name} cannot be read as {b.via} in a {kind} loop",
                where,
            )
        if b.via == "other":
            bd = b.boundary
            if bd is None or bd.set is not dsl.MeshSet.FACES or bd.type != ds.type:
                raise CodegenError(
                    Code.INVALID_BINDING,
                    f"{name}: other() needs a face dataset of the same type",
                    where,
                )


def _check_outputs(where: str, lm: LoopModel) -> None:
    loop = lm.loop
    sig = lm.fin_sig if lm.fin_sig is not None else lm.sig
    rets = sig.returns
    if rets is None:
        raise CodegenError(Code.INVALID_BINDING, "loop bodies annotate their return types", where)
    if len(rets) != len(loop.outputs):
        raise CodegenError(
            Code.INVALID_BINDING,
            f"body returns {len(rets)} values, loop has {len(loop.outputs)} outputs",
            where,
        )
    for t, o in zip(rets, loop.outputs, strict=True):
        ds = o.dataset
        if ds.set is not dsl.MeshSet.CELLS or ds.geometry:
            raise CodegenError(
                Code.INVALID_BINDING, f"output {ds.name} must be a cell dataset", where
            )
        if ds.type.shape != t.shape:
            raise CodegenError(
                Code.INVALID_BINDING,
                f"output {ds.name} has shape {ds.type.shape}, body gives {t.shape}",
                where,
            )
        if isinstance(o, dsl.Inc) and ds.type.role is not dsl.Role.RESIDUAL:
            # Inc targets hold the accumulation type of the policy (section 6, generator rules).
            raise CodegenError(
                Code.INC_TARGET_NOT_ACCUMULATION,
                f"Inc target {ds.name} must have role Residual",
                where,
            )
    if lm.fin_sig is not None:
        rets_body = lm.sig.returns or []
        for k, t in enumerate(rets_body):
            ft = lm.fin_sig.types[k]
            if ft.shape != t.shape or ft.role is not dsl.Role.RESIDUAL:
                raise CodegenError(
                    Code.INVALID_BINDING,
                    f"finalise argument {lm.fin_sig.names[k]} must be Residual{t.shape}",
                    where,
                )


def build_loop(name: str, loop: dsl.Loop) -> LoopModel:
    where = f"{loop.body.source_file}: loop {name}"
    fn, sig = parse(loop.body)
    fin_fn = fin_sig = None
    kind = "reduce_stencil" if loop.kind == "reduce" and loop.over == dsl.STENCIL else loop.kind
    _check_bindings(where, kind, sig, loop.bindings)
    if loop.finalise is not None:
        fin_fn, fin_sig = parse(loop.finalise)
        n_acc = len(loop.outputs)
        _check_bindings(where, "cell", fin_sig, loop.finalise_bindings, skip=n_acc)
    lm = LoopModel(name, loop, fn, sig, fin_fn, fin_sig)
    _check_outputs(where, lm)
    read = {id(d) for d in lm.reads()}
    for d in lm.writes():
        if id(d) in read:
            # In place updates would race between strategies and lose the adjoint of the input.
            raise CodegenError(Code.INVALID_BINDING, f"{d.name} is both read and written", where)
    return lm


def load(path: Path) -> Module:
    """Import a kernel module and build the model of every loop it binds to a global name."""
    name = path.stem
    spec = importlib.util.spec_from_file_location(f"autan_kernel_{name}", path)
    if spec is None or spec.loader is None:
        raise CodegenError(Code.INVALID_KERNEL_MODULE, f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    loops = []
    for gname, obj in vars(mod).items():
        if isinstance(obj, dsl.Loop):
            obj.name = gname
            loops.append(build_loop(gname, obj))
    if not loops:
        raise CodegenError(Code.INVALID_KERNEL_MODULE, f"{path} declares no loop")
    datasets: list[dsl.Dataset] = []
    params: list[dsl.Parameter] = []
    for lm in loops:
        for d in [*lm.reads(), *lm.writes()]:
            target = params if isinstance(d, dsl.Parameter) else datasets
            if all(d is not x for x in target):
                if any(d.name == x.name for x in target):
                    raise CodegenError(
                        Code.INVALID_BINDING, f"two different data named {d.name}", str(path)
                    )
                target.append(d)  # type: ignore[arg-type]
    bodies: list[tuple[Function, Signature]] = []
    for lm in loops:
        bodies.append((lm.fn, lm.sig))
        if lm.fin_fn is not None and lm.fin_sig is not None:
            bodies.append((lm.fin_fn, lm.fin_sig))
    return Module(name, path, loops, datasets, params, bodies)
