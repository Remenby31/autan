"""`autan diff` through its output and exit status, on runs of the shipped runtime
(docs/sdk/diff.md; acceptance.md, M4)."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import autan
import h5py
import numpy as np
import pytest
from autan.cli import main

from tests.test_solver_backend import CASE, MESH, MESH_INFO, RUNTIME, needs_mpi

pytestmark = pytest.mark.skipif(
    None in (RUNTIME, MESH_INFO), reason="autan-run and autan-mesh-info not built"
)

INVISCID = CASE.replace("autan.Laminar()", "autan.Inviscid()")
SOLUTIONS = ("FlowSolution", "FlowSolutionMean", "WallSolution")


def _run(
    tmp_path: Path, name: str, text: str, *, ranks: int = 1, mesh_info: tuple[str, ...] = ()
) -> Path:
    assert MESH_INFO is not None
    assert RUNTIME is not None
    case = tmp_path / f"{name}.py"
    case.write_text(text)
    meshinfo = tmp_path / f"{name}.meshinfo.json"
    meshinfo.write_text(
        subprocess.run(
            [str(MESH_INFO), "--json", *mesh_info, str(MESH)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    result = autan.load(case).run(
        autan.MeshInfo.load(meshinfo),
        backend=autan.SolverBackend(
            devices=autan.Devices(kind="cpu", count=ranks), executable=RUNTIME
        ),
        runs=tmp_path / name,
    )
    return result.path


def _report(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    status = main(["diff", *argv, "--json"])
    return status, json.loads(capsys.readouterr().out)


def _label(node: Any) -> str:
    label = None if node is None else node.attrs.get("label")
    return "" if label is None else bytes(np.asarray(label)).rstrip(b"\0 ").decode()


# The solution arrays that differ between two solution.cgns, read with h5py: bit for bit, or by
# docs/sdk/diff.md's rule at `rtol` (equal or adjacent FP64 numbers, or within rtol relatively).
def _differing_arrays(a: Path, b: Path, rtol: float | None = None) -> set[str]:
    out = set()
    with h5py.File(a / "solution.cgns", "r") as fa, h5py.File(b / "solution.cgns", "r") as fb:
        for group in SOLUTIONS:
            path = f"Base/Zone/{group}"
            if path not in fa and path not in fb:
                continue
            names = {n for n in (*fa[path], *fb[path]) if _label(fa[path].get(n)) == "DataArray_t"}
            for name in names:
                xa, xb = fa[f"{path}/{name}/ data"][...], fb[f"{path}/{name}/ data"][...]
                if rtol is None:
                    differs = xa.tobytes() != xb.tobytes()
                else:
                    close = np.abs(xa - xb) <= rtol * np.maximum(np.abs(xa), np.abs(xb))
                    differs = not np.all((xa == xb) | (np.nextafter(xa, xb) == xb) | close)
                if differs:
                    out.add(f"{group}/{name}")
    return out


def _changed_copy(run: Path, to: Path, change: Any) -> Path:
    copy = to / run.name
    shutil.copytree(run, copy)
    with h5py.File(copy / "solution.cgns", "r+") as handle:
        data = handle["Base/Zone/FlowSolution/Pressure/ data"]
        values = data[...]
        index = int(np.argmax(np.abs(values)))
        values.flat[index] = change(values.flat[index])
        data[...] = values
    return copy


@pytest.mark.mpi
@needs_mpi
def test_ranks_runs_on_one_and_two_ranks_agree_bit_for_bit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # runtime.md: an inviscid steady run with gauss-seidel under "ranks" is bit for bit across
    # rank counts.
    text = (
        INVISCID.replace("autan.Inviscid())", 'autan.Inviscid(), reproducibility="ranks")')
        + "case.time = autan.Steady(max_iterations=5, linear_solver='gauss-seidel')\n"
    )
    one = _run(tmp_path, "one", text)
    two = _run(tmp_path, "two", text, ranks=2, mesh_info=("--ranks", "2"))
    for flags in (("--bitwise",), ()):
        status, report = _report(capsys, str(one), str(two), *flags)
        assert (status, report["agree"], report["differences"]) == (0, True, [])
    assert report["solution"] == "compared"
    count = {e["name"]: e for e in report["record"]}["devices.count"]
    assert (count["a"], count["b"], count["compared"]) == (1, 2, False)


def test_a_difference_is_named_wherever_it_is(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = INVISCID + "case.time = autan.Steady(max_iterations=5, linear_solver='gauss-seidel')\n"
    base = _run(tmp_path, "base", text)
    # A perturbed freestream: the record (the case) and exactly the arrays that differ, in each mode.
    perturbed = _run(tmp_path, "mach", text.replace("mach=0.2", "mach=0.2 * 1.001"))
    for flags, rtol in ((("--bitwise",), None), ((), 1e-6)):
        status, report = _report(capsys, str(base), str(perturbed), *flags)
        assert status == 1
        assert "case.sha256" in report["differences"]
        named = {d for d in report["differences"] if d.split("/")[0] in SOLUTIONS}
        oracle = _differing_arrays(base, perturbed, rtol)
        assert oracle
        assert named == oracle
    stopped = tmp_path / "stopped" / base.name
    shutil.copytree(base, stopped)
    info = json.loads((stopped / "run.json").read_text())
    (stopped / "run.json").write_text(json.dumps({**info, "status": "stopped"}, indent=2))
    autan.Results.open(stopped)
    status, report = _report(capsys, str(base), str(stopped))
    assert (status, report["differences"]) == (1, ["status"])
    flipped = _changed_copy(
        base,
        tmp_path / "flip",
        lambda v: np.frombuffer((np.float64(v).view(np.uint64) ^ np.uint64(1)).tobytes())[0],
    )
    status, report = _report(capsys, str(base), str(flipped), "--bitwise")
    assert (status, report["differences"]) == (1, ["FlowSolution/Pressure"])
    assert _report(capsys, str(base), str(flipped))[0] == 0
    scaled = _changed_copy(base, tmp_path / "scale", lambda v: v * 1.001)
    status, report = _report(capsys, str(base), str(scaled))
    assert (status, report["differences"]) == (1, ["FlowSolution/Pressure"])
    # The history alone: one residual of iteration 3 scaled by 1.001.
    changed = tmp_path / "history" / base.name
    shutil.copytree(base, changed)
    lines = [json.loads(line) for line in (changed / "history.jsonl").read_text().splitlines()]
    line = next(x for x in lines if x["iteration"] == 3)
    density = line["residuals"]["density"]
    line["residuals"]["density"] = density * 1.001
    (changed / "history.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    status, report = _report(capsys, str(base), str(changed))
    assert (status, report["differences"]) == (1, ["history.residuals.density"])
    assert report["history_first_difference"] == 3
    [series] = [s for s in report["history"] if s["name"] == "residuals.density"]
    assert series["max_rel"] == pytest.approx(1 - 1 / 1.001, rel=1e-9)


def test_a_run_that_cannot_be_read_exits_2_with_e0701(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(["diff", str(tmp_path / "a"), str(tmp_path / "b"), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert (status, [d["code"] for d in out["diagnostics"]]) == (2, ["E0701"])
