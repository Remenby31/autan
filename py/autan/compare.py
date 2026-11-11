"""Two run directories compared: record, history and solution (docs/sdk/diff.md)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel

from autan.errors import CaseError, Location, diagnostic
from autan.results import IDENTITY, Results

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from autan import cgns

RTOL = 1e-6
# results.md: under reproducibility "ranks", runs whose identities differ only by these give
# the same bits, so the record does not compare them.
_LEFT_OUT = ("devices.count", "cell_order")
COMPARED = (*(p for p in IDENTITY if p not in _LEFT_OUT), "status", "iterations")
REPORTED = (
    "run_id",
    "devices.count",
    "cell_order",
    "solver.binary_sha256",
    "started",
    "finished",
    "timing",
)
_SOLUTIONS = ("FlowSolution", "FlowSolutionMean")


class Entry(BaseModel):
    """One record entry: both runs' values, whether it is compared and whether they agree."""

    name: str
    a: Any
    b: Any
    compared: bool
    agree: bool


class Series(BaseModel):
    """A compared series (a history key, a solution array): whether it agrees, its largest
    absolute and relative difference, and the first iteration (history) or index (solution)
    where it differs."""

    name: str
    agree: bool
    max_abs: float
    max_rel: float
    first: int | None


class Report(BaseModel):
    """The comparison of two runs (docs/sdk/diff.md). ``agree`` is the verdict; ``differences``
    names every record entry, series and array that differs; ``solution`` says whether the
    solutions were compared and why not."""

    agree: bool
    bitwise: bool
    rtol: float
    record: list[Entry]
    history: list[Series]
    history_first_difference: int | None
    solution: str
    fields: list[Series]
    differences: list[str]


def _value(document: dict[str, Any], path: str) -> Any:
    value: Any = document
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def agree(
    a: NDArray[np.float64], b: NDArray[np.float64], *, bitwise: bool, rtol: float
) -> NDArray[np.bool_]:
    """Whether each pair of values agrees: equal bytes with ``bitwise``; otherwise equal or
    adjacent FP64 numbers, or |a - b| <= ``rtol`` * max(|a|, |b|)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if bitwise:
        return np.asarray(a.view(np.uint64) == b.view(np.uint64))
    with np.errstate(invalid="ignore"):
        close = np.abs(a - b) <= rtol * np.maximum(np.abs(a), np.abs(b))
        return np.asarray((a == b) | (np.nextafter(a, b) == b) | close)


def _series(
    name: str,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    bitwise: bool,
    rtol: float,
    at: NDArray[np.int64] | None = None,
) -> Series:
    ok = agree(a, b, bitwise=bitwise, rtol=rtol)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
        scale = np.maximum(np.abs(a), np.abs(b))
        rel = np.where(scale > 0, gap / np.where(scale > 0, scale, 1.0), 0.0)
    bad = np.flatnonzero(~ok)
    first = None if bad.size == 0 else int(at[bad[0]] if at is not None else bad[0])
    return Series(
        name=name,
        agree=bad.size == 0,
        max_abs=float(np.nanmax(gap)) if gap.size else 0.0,
        max_rel=float(np.nanmax(rel)) if rel.size else 0.0,
        first=first,
    )


def _flat(record: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, f"{name}."))
        elif isinstance(value, bool | int | float):
            out[name] = float(value)
    return out


def _record(a: Results, b: Results, *, bitwise: bool, rtol: float) -> list[Entry]:
    da, db = a.run.document(), b.run.document()
    entries = [
        Entry(
            name=p,
            a=_value(da, p),
            b=_value(db, p),
            compared=True,
            agree=_value(da, p) == _value(db, p),
        )
        for p in COMPARED
    ]
    ma, mb = _flat(da.get("monitors") or {}), _flat(db.get("monitors") or {})
    for key in sorted(set(ma) | set(mb)):
        same = key in ma and key in mb
        if same:
            same = bool(
                agree(np.array([ma[key]]), np.array([mb[key]]), bitwise=bitwise, rtol=rtol)[0]
            )
        entries.append(
            Entry(name=f"monitors.{key}", a=ma.get(key), b=mb.get(key), compared=True, agree=same)
        )
    entries.extend(
        Entry(name=p, a=_value(da, p), b=_value(db, p), compared=False, agree=True)
        for p in REPORTED
    )
    return entries


def _history(
    a: Results, b: Results, *, bitwise: bool, rtol: float
) -> tuple[list[Series], int | None]:
    ra = {int(r["iteration"]): _flat(r) for r in a.history._records}
    rb = {int(r["iteration"]): _flat(r) for r in b.history._records}
    common = sorted(set(ra) & set(rb))
    only = sorted(set(ra) ^ set(rb))
    out: list[Series] = []
    firsts: list[int] = [only[0]] if only else []
    if only:
        out.append(Series(name="iterations", agree=False, max_abs=0.0, max_rel=0.0, first=only[0]))
    at = np.array(common, dtype=np.int64)
    keys = sorted({k for i in common for k in (*ra[i], *rb[i])} - {"iteration"})
    for key in keys:
        present_a = np.array([key in ra[i] for i in common])
        present_b = np.array([key in rb[i] for i in common])
        va = np.array([ra[i].get(key, 0.0) for i in common], dtype=np.float64)
        vb = np.array([rb[i].get(key, 0.0) for i in common], dtype=np.float64)
        series = _series(key, va, vb, bitwise=bitwise, rtol=rtol, at=at)
        missing = np.flatnonzero(present_a != present_b)
        if missing.size:
            first = int(at[missing[0]])
            series = series.model_copy(
                update={"agree": False, "first": min(first, series.first or first)}
            )
        out.append(series)
        if series.first is not None and not series.agree:
            firsts.append(series.first)
    return out, (min(firsts) if firsts else None)


def _array(results: Results, name: str, node: cgns.Node, dtype: type) -> NDArray[Any]:
    data = np.asarray(node.data)
    if data.dtype != dtype:
        where = Location(file=str(results.path / "solution.cgns"))
        message = f"{name} is {data.dtype}, results.md writes it as {np.dtype(dtype)}"
        raise CaseError([diagnostic("E0701", message, location=where)])
    return data.ravel()


def _arrays(results: Results) -> dict[str, NDArray[Any]]:
    zone: cgns.Node = results._zone()
    out: dict[str, NDArray[Any]] = {}
    for group in _SOLUTIONS:
        if group in zone.children:
            for node in zone.child(group).with_label("DataArray_t"):
                name = f"{group}/{node.name}"
                out[name] = _array(results, name, node, np.float64)
    if "WallSolution" in zone.children:
        region = zone.child("WallSolution")
        out["WallSolution/PointList"] = np.asarray(region.child("PointList").data).ravel()
        for node in region.with_label("DataArray_t"):
            name = f"WallSolution/{node.name}"
            out[name] = _array(results, name, node, np.float64)
    return out


def _solution(
    a: Results, b: Results, *, bitwise: bool, rtol: float, fields: list[str] | None
) -> tuple[str, list[Series]]:
    if a.run.mesh.sha256 != b.run.mesh.sha256:
        return "not compared: the meshes differ", []
    # results.md: readers do not use solution.cgns while the run is running.
    has_a, has_b = (
        r.run.status != "running" and (r.path / "solution.cgns").exists() for r in (a, b)
    )
    if not has_a and not has_b:
        return "neither run has a solution", []
    if has_a != has_b:
        return ("only the first run has a solution" if has_a else "only the second has one"), [
            Series(name="solution.cgns", agree=False, max_abs=0.0, max_rel=0.0, first=None)
        ]
    xa, xb = _arrays(a), _arrays(b)
    names = sorted(set(xa) | set(xb))
    if fields is not None:
        names = [n for n in names if n.split("/", 1)[1] in fields or n.endswith("/PointList")]
    out: list[Series] = []
    for name in names:
        if name not in xa or name not in xb or xa[name].shape != xb[name].shape:
            out.append(Series(name=name, agree=False, max_abs=0.0, max_rel=0.0, first=None))
        elif name.endswith("/PointList"):
            same = bool(np.array_equal(xa[name], xb[name]))
            out.append(Series(name=name, agree=same, max_abs=0.0, max_rel=0.0, first=None))
        else:
            out.append(_series(name, xa[name], xb[name], bitwise=bitwise, rtol=rtol))
    return "compared", out


def diff(
    a: str | Path,
    b: str | Path,
    *,
    bitwise: bool = False,
    rtol: float = RTOL,
    fields: list[str] | None = None,
) -> Report:
    """Compare the run directories ``a`` and ``b`` as docs/sdk/diff.md specifies: the record,
    the history line by line and, on the same mesh, the solution arrays. ``fields`` restricts
    the solution to those field names. A path that is not a run directory raises
    ``CaseError`` ``E0701``."""
    ra, rb = Results.open(a), Results.open(b)
    record = _record(ra, rb, bitwise=bitwise, rtol=rtol)
    history, first = _history(ra, rb, bitwise=bitwise, rtol=rtol)
    solution, arrays = _solution(ra, rb, bitwise=bitwise, rtol=rtol, fields=fields)
    differences = [
        *(e.name for e in record if e.compared and not e.agree),
        *(f"history.{s.name}" for s in history if not s.agree),
        *(s.name for s in arrays if not s.agree),
    ]
    return Report(
        agree=not differences,
        bitwise=bitwise,
        rtol=rtol,
        record=record,
        history=history,
        history_first_difference=first,
        solution=solution,
        fields=arrays,
        differences=differences,
    )
