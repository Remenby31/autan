"""The exported C++ header and JSON document reproduce the symbolic solution through real consumers."""

import ctypes
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from autan_vv.errors import VVError
from autan_vv.mms import CATALOG, dump_json, parse_json_expressions, to_cpp
from autan_vv.mms.solutions import ManufacturedSolution, X

GROUPS = ("fields", "gradients", "conserved", "sources", "auxiliary")

WRAPPER = """
#include "solution.hpp"
namespace m = autan::mms::{name};
{wrappers}
"""


def _points() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.uniform(0.02, 0.98, (16, 3))


def _expected(ms: ManufacturedSolution, pts: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for group in GROUPS:
        if group == "gradients":
            grads = ms.gradients()
            cols = [sp.lambdify(X, g[k], "numpy") for g in grads.values() for k in range(3)]
            out[group] = np.stack([np.broadcast_to(f(*pts.T), pts.shape[:1]) for f in cols], axis=1)
        else:
            values = ms.evaluator(group)(pts)
            out[group] = (
                np.stack(list(values.values()), axis=1) if values else np.zeros((len(pts), 0))
            )
    return out


def _compile(ms: ManufacturedSolution, tmp: Path) -> ctypes.CDLL:
    """Build the header into a shared library with one extern "C" entry point per group and precision."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("no C++ compiler on PATH: the C++ export is not verified")
    (tmp / "solution.hpp").write_text(to_cpp(ms))
    wrappers = "\n".join(
        f'extern "C" void {g}_{real}({real} x, {real} y, {real} z, {real}* o) {{ m::{g}<{real}>(x, y, z, o); }}'
        for g in GROUPS
        if _size(ms, g) > 0
        for real in ("double", "float")
    )
    (tmp / "wrapper.cpp").write_text(WRAPPER.format(name=ms.name, wrappers=wrappers))
    lib = tmp / "libmms.so"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-o",
            str(lib),
            str(tmp / "wrapper.cpp"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return ctypes.CDLL(str(lib))


def _run_cpp(
    lib: ctypes.CDLL, ms: ManufacturedSolution, real: str, pts: np.ndarray
) -> dict[str, np.ndarray]:
    ctype = ctypes.c_double if real == "double" else ctypes.c_float
    out = {}
    for g in GROUPS:
        n = _size(ms, g)
        values = np.zeros((len(pts), n))
        if n:
            fn = getattr(lib, f"{g}_{real}")
            fn.argtypes = [ctype, ctype, ctype, ctypes.POINTER(ctype)]
            buf = (ctype * n)()
            for i, p in enumerate(pts):
                fn(*(ctype(float(c)) for c in p), buf)
                values[i] = np.frombuffer(buf, dtype=np.float64 if real == "double" else np.float32)
        out[g] = values
    return out


def _size(ms: ManufacturedSolution, group: str) -> int:
    return 3 * len(ms.fields) if group == "gradients" else len(getattr(ms, group))


@pytest.mark.parametrize("name", sorted(CATALOG))
def test_cpp_header_matches_symbolic_values_in_fp64_and_fp32(name: str, tmp_path: Path) -> None:
    ms = CATALOG[name]()
    pts = _points()
    expected = _expected(ms, pts)
    lib = _compile(ms, tmp_path)
    fp64 = _run_cpp(lib, ms, "double", pts)
    fp32 = _run_cpp(lib, ms, "float", pts)
    for group in GROUPS:
        if _size(ms, group) == 0:
            continue
        scale = np.max(np.abs(expected[group]), axis=0) + 1e-300
        # Relative to each output's magnitude over the domain, in units of the precision's epsilon.
        for got, real in ((fp64, np.float64), (fp32, np.float32)):
            assert np.max(np.abs(got[group] - expected[group]) / scale) < 32 * np.finfo(real).eps, (
                group
            )


@pytest.mark.parametrize("name", sorted(CATALOG))
def test_json_document_round_trips_through_a_parser(name: str) -> None:
    ms = CATALOG[name]()
    doc = json.loads(dump_json(ms))
    assert doc["schema"] == "autan.mms/1"
    assert doc["parameters"] == dict(ms.parameters)
    pts = _points()
    expected = _expected(ms, pts)
    for group in GROUPS:
        if _size(ms, group) == 0:
            continue
        parsed = parse_json_expressions(doc, group)
        got = np.stack(
            [
                np.broadcast_to(sp.lambdify(X, e, "numpy")(*pts.T), pts.shape[:1])
                for e in parsed.values()
            ],
            axis=1,
        )
        scale = np.max(np.abs(expected[group]), axis=0) + 1e-300
        assert np.max(np.abs(got - expected[group]) / scale) < 1e-12, group


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').getcwd()",
        "x.real",
        "[x]",
        "lambda: 1",
        "sin(x, y=1)",
        "unknown(x)",
        "x ** 2",
        "1 +",
    ],
)
def test_json_reader_rejects_anything_outside_the_grammar(text: str) -> None:
    doc = {"g": {"subexpressions": [], "outputs": {"f": text}}}
    with pytest.raises(VVError) as err:
        parse_json_expressions(doc, "g")
    assert err.value.code == "V0202"
