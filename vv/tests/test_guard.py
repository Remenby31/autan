"""The nightly guard's verdict (#611) on histories built so that each measure follows by hand."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from autan_vv import guard
from autan_vv.cli import main

COMMIT = "c" * 40
STEPS = guard.STEPS
PLATEAU = 16000


def _series(points: list[tuple[int, float]]) -> np.ndarray:
    """Residuals at steps 1..STEPS, log-linear between (step, value) points."""
    steps, values = zip(*points, strict=True)
    return 10.0 ** np.interp(np.arange(1, STEPS + 1), steps, np.log10(values))


def _green() -> dict[str, np.ndarray]:
    # Main's shape (#611): energy peaks at step 7, is 10 orders below the peak by step 1 500 and
    # stays there; momentum-y starts small, so only the shared momentum reference gives it 10 orders.
    # Every residual reaches its final level by step 16 000 and stays there, as main's runs do.
    return {
        "density": _series([(1, 1e-15), (STEPS, 1e-15)]),
        "momentum-x": _series([(1, 4.0), (PLATEAU, 4e-13), (STEPS, 4e-13)]),
        "momentum-y": _series([(1, 1.5e-5), (2, 1e-3), (PLATEAU, 1.6e-13), (STEPS, 1.6e-13)]),
        "momentum-z": _series([(1, 1e-38), (2, 1e-19), (STEPS, 1e-19)]),
        "energy": _series(
            [(1, 12.0), (7, 400.0), (1500, 3e-8), (PLATEAU, 5.5e-10), (STEPS, 5.5e-10)]
        ),
        "nu-tilde": _series([(1, 6.8e-5), (PLATEAU, 1.1e-18), (STEPS, 1.1e-18)]),
    }


def _run(path: Path, series: dict[str, np.ndarray], **info: Any) -> Path:
    path.mkdir(parents=True)
    lines = [
        json.dumps(
            {
                "iteration": i + 1,
                "time": None,
                "provisional": i < 101,
                "monitors": {},
                "residuals": {k: float(v[i]) for k, v in series.items()},
            }
        )
        for i in range(STEPS)
    ]
    (path / "history.jsonl").write_text("\n".join(lines) + "\n")
    run = {"solver": {"commit": COMMIT, "binary_sha256": "b" * 64}, "start": None,
           "iterations": STEPS, "status": "completed"} | info  # fmt: skip
    (path / "run.json").write_text(json.dumps(run))
    return path


def _night(tmp_path: Path, one: dict[str, np.ndarray], two: dict[str, np.ndarray] | None = None,
           **kw: Any) -> dict[str, Any]:  # fmt: skip
    r1 = _run(tmp_path / "run1", one)
    r2 = _run(tmp_path / "run2", one if two is None else two)
    return guard.verdict(r1, r2, COMMIT, has_544=kw.pop("has_544", False), **kw)


def test_main_like_night_is_green_and_its_measures_follow_from_the_history(tmp_path: Path) -> None:
    v = _night(tmp_path, _green())
    assert v["outcome"] == "green", v["failing"]
    m = v["measures"]
    assert m["drops"]["energy"]["peak"] == pytest.approx(math.log10(400.0 / 5.5e-10))
    assert m["drops"]["momentum-y"]["shared"] == pytest.approx(math.log10(4.0 / 1.6e-13))
    assert m["rule_own"] is None  # momentum-y is 7.97 orders below its own step-1 value
    assert m["rule_shared"] is not None
    energy = _green()["energy"]
    below = energy <= energy.max() * 1e-10
    assert m["m3f"] == int(np.argmax(below)) + 1  # first step below
    assert m["m3s"] == m["m3f"]  # monotone after the peak: the crossing is sustained
    # Once #544 is in the night's commit, the shared rule must be met: here it is.
    assert _night(tmp_path / "n544", _green(), has_544=True)["outcome"] == "green"


def _worse(edit: str) -> dict[str, np.ndarray]:
    s = _green()
    if edit == "energy-final":  # (b) and (c): energy ends an order higher
        s["energy"] = _series(
            [(1, 12.0), (7, 400.0), (1500, 3e-8), (PLATEAU, 5.5e-9), (STEPS, 5.5e-9)]
        )
    elif edit == "m3f":  # (d): the first crossing comes late
        s["energy"] = _series(
            [(1, 12.0), (7, 400.0), (5000, 3e-8), (PLATEAU, 5.5e-10), (STEPS, 5.5e-10)]
        )
    elif edit == "tail":  # (d): energy back above 1e-10 x peak on 6 % of steps 10 001-20 000
        e = s["energy"].copy()
        e[14000:14600] = 1e-6
        s["energy"] = e
    elif edit == "peak":  # (d): a peak 100 times higher, the rest unchanged
        e = s["energy"].copy()
        e[6] *= 100.0
        s["energy"] = e
    elif edit == "momentum-z":  # (b): the slab's zero equation leaves its floor
        s["momentum-z"] = _series([(1, 1e-38), (2, 1e-19), (STEPS, 1e-9)])
    elif edit == "nu-tilde":  # (b) and (c)
        s["nu-tilde"] = _series([(1, 6.8e-5), (PLATEAU, 1.1e-16), (STEPS, 1.1e-16)])
    return s


@pytest.mark.parametrize(
    ("edit", "gate", "what"),
    [
        ("energy-final", "b", "energy"),
        ("m3f", "d", "m3f"),
        ("tail", "d", "tail_fraction"),
        ("peak", "d", "peak_ratio"),
        ("momentum-z", "b", "momentum-z"),
        ("nu-tilde", "c", "nu-tilde"),
    ],
)
def test_a_night_worse_than_main_is_red_on_its_gate(
    tmp_path: Path, edit: str, gate: str, what: str
) -> None:
    v = _night(tmp_path, _worse(edit))
    assert v["outcome"] == "red"
    failing = {(c["gate"], c.get("equation") or c.get("measure")) for c in v["failing"]}
    assert (gate, what) in failing
    if edit == "tail":
        assert ("d", "m3f") not in failing
        assert v["measures"]["tail_fraction"] == pytest.approx(0.06)


def test_a_short_late_excursion_moves_m3s_but_no_longer_fails_the_night(tmp_path: Path) -> None:
    # m3s is reported, not gated (#611: round-off-level inputs alone move it past any useful bound).
    s = _green()
    e = s["energy"].copy()
    e[13999] = 1e-6
    s["energy"] = e
    v = _night(tmp_path, s)
    assert v["outcome"] == "green", v["failing"]
    assert v["measures"]["m3s"] == 14001  # the step after the excursion at step 14 000
    assert v["measures"]["tail_fraction"] == pytest.approx(1e-4)


def test_the_gate_measures_are_the_frozen_evaluators(tmp_path: Path) -> None:
    # The gates read the measures the null search selected them on (vv/guard/null_eval.py, frozen
    # before any null history was read): the same values from both implementations.
    spec = importlib.util.spec_from_file_location(
        "null_eval", Path(__file__).resolve().parents[1] / "guard" / "null_eval.py"
    )
    assert spec is not None
    assert spec.loader is not None
    null_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(null_eval)
    sloped = (
        _green()
    )  # energy still falling across steps 18 001-20 000: median and last step differ
    sloped["energy"] = _series([(1, 12.0), (7, 400.0), (1500, 3e-8), (STEPS, 5.5e-10)])
    for name, series in (
        ("green", _green()),
        ("tail", _worse("tail")),
        ("peak", _worse("peak")),
        ("sloped", sloped),
    ):
        _, lines = guard.read_run(_run(tmp_path / name, series), COMMIT)
        m, c = guard.measure(lines), null_eval.candidates(lines)
        assert m["tail_fraction"] == c["T"]
        assert m["peak_ratio"] == c["G_P"]
        assert m["median_drops"] == c["median_drop"]


def test_an_excursion_of_the_zero_equation_is_red_even_if_it_recovers(tmp_path: Path) -> None:
    s = _green()
    z = s["momentum-z"].copy()
    z[9999] = 1e-9  # above its floor (1e-12 x 12) at step 10 000, back at 1e-19 after
    s["momentum-z"] = z
    v = _night(tmp_path, s)
    assert ("b", "momentum-z") in {(c["gate"], c.get("equation")) for c in v["failing"]}


def test_a_run_that_exits_non_zero_is_red_whatever_its_history(tmp_path: Path) -> None:
    v = _night(tmp_path, _green(), exit_codes=(0, 1))
    assert v["outcome"] == "red"
    assert "exit" in {c["gate"] for c in v["failing"]}


def test_after_544_the_shared_rule_must_be_met(tmp_path: Path) -> None:
    # Energy ends at 2e-9, above 1e-10 x its step-1 value 12: the rule is never met.
    s = _green()
    s["energy"] = _series([(1, 12.0), (7, 400.0), (1500, 3e-8), (STEPS, 2e-9)])
    v = _night(tmp_path, s, has_544=True)
    assert v["measures"]["rule_shared"] is None
    assert v["outcome"] == "red"
    assert "a" in {c["gate"] for c in v["failing"]}


def test_two_runs_that_differ_make_the_night_red(tmp_path: Path) -> None:
    two = _green()
    two["energy"] = two["energy"].copy()
    two["energy"][STEPS - 1] = np.nextafter(two["energy"][STEPS - 1], 1.0)
    v = _night(tmp_path, _green(), two)
    assert v["outcome"] == "red"
    assert "repeat" in {c["gate"] for c in v["failing"]}


def test_same_binary_as_the_last_green_night_needs_the_same_history(tmp_path: Path) -> None:
    previous = _night(tmp_path / "prev", _green())
    assert _night(tmp_path / "same", _green(), previous=previous)["outcome"] == "green"
    moved = dict(previous, history_sha256="0" * 64)
    assert "e" in {c["gate"] for c in _night(tmp_path / "e", _green(), previous=moved)["failing"]}
    other = dict(moved, binary_sha256="d" * 64)  # another binary: the difference is reported only
    v = _night(tmp_path / "other", _green(), previous=other)
    assert v["outcome"] == "green"
    assert v["across_nights"]["same_binary"] is False


@pytest.mark.parametrize("fault", ["truncated", "foreign commit", "non-finite"])
def test_a_run_the_guard_cannot_measure_is_red_not_void(tmp_path: Path, fault: str) -> None:
    r1 = _run(tmp_path / "run1", _green())
    r2 = _run(tmp_path / "run2", _green())
    if fault == "truncated":
        lines = (r1 / "history.jsonl").read_text().splitlines()[:-1]
        (r1 / "history.jsonl").write_text("\n".join(lines) + "\n")
    elif fault == "foreign commit":
        info = json.loads((r1 / "run.json").read_text())
        info["solver"]["commit"] = "f" * 40
        (r1 / "run.json").write_text(json.dumps(info))
    else:
        s = _green()
        s["energy"] = s["energy"].copy()
        s["energy"][100] = math.nan
        _run(tmp_path / "nan", s)
        r1 = tmp_path / "nan"
    v = guard.verdict(r1, r2, COMMIT, has_544=False)
    assert v["outcome"] == "red"
    assert "unmeasurable" in v["reason"]


def test_void_only_with_evidenced_infrastructure_and_missing_without_a_verdict(
    tmp_path: Path,
) -> None:
    v = guard.verdict(tmp_path / "none", tmp_path / "none", COMMIT, has_544=False,
                      infra="cgroup OOM kill (memory.events oom_kill 1)")  # fmt: skip
    assert v["outcome"] == "void"
    night = tmp_path / "2026-11-09"
    night.mkdir()
    assert guard.night_status(night) == "MISSING"
    (night / "verdict.json").write_text(json.dumps(v))
    assert guard.night_status(night) == "void"


def test_cli_writes_strict_json_and_exits_1_on_red(tmp_path: Path) -> None:
    night = tmp_path / "night"
    night.mkdir()
    r1 = _run(night / "run1", _worse("energy-final"))
    r2 = _run(night / "run2", _worse("energy-final"))
    with pytest.raises(SystemExit) as err:
        main(["guard", "verdict", str(night), "--run1", str(r1), "--run2", str(r2),
              "--commit", COMMIT])  # fmt: skip
    assert err.value.code == 1
    assert json.loads((night / "verdict.json").read_text())["outcome"] == "red"
    assert main(["guard", "status", str(night)]) == 0


def test_an_infinite_drop_passes_and_is_written_as_strict_json(tmp_path: Path) -> None:
    s = _green()
    x = s["momentum-x"].copy()
    x[-1] = 0.0  # exactly zero at the last step: an infinite drop
    s["momentum-x"] = x
    night = tmp_path / "night"
    night.mkdir()
    r1, r2 = _run(night / "run1", s), _run(night / "run2", s)
    assert main(["guard", "verdict", str(night), "--run1", str(r1), "--run2", str(r2),
                 "--commit", COMMIT]) == 0  # fmt: skip

    def refuse(name: str) -> None:
        raise ValueError(name)

    report = json.loads((night / "verdict.json").read_text(), parse_constant=refuse)
    assert report["outcome"] == "green"
    assert report["measures"]["drops"]["momentum-x"]["first"] is None  # infinity, written as null


WATCHDOG = Path(__file__).resolve().parents[1] / "guard" / "watchdog.sh"


@pytest.mark.skipif(shutil.which("flock") is None, reason="watchdog.sh needs flock (util-linux)")
def test_the_watchdog_voids_a_drained_night_left_without_a_verdict(tmp_path: Path) -> None:
    # The night the watchdog closes now: the day before the latest 07:00 UTC slot.
    now = datetime.now(UTC)
    if now.strftime("%H%M") < "0700":
        now -= timedelta(days=1)
    night = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    root, home = tmp_path / "guard", tmp_path / "home"
    (home / "cfd" / "queue" / "pending").mkdir(parents=True)
    marker = "drained by the orchestrator at 2026-11-09T14:30Z for an ML4 relaunch"

    def status(drained: bool) -> str:
        shutil.rmtree(root, ignore_errors=True)
        (root / night).mkdir(parents=True)
        (root / night / "COMMIT").write_text(COMMIT + "\n")
        if drained:
            (root / night / "DRAINED").write_text(marker + "\n")
        env = {**os.environ, "HOME": str(home), "AUTAN_GUARD_ROOT": str(root)}
        subprocess.run(["bash", str(WATCHDOG)], env=env, check=True)
        return (root / "STATUS").read_text()

    line = status(drained=True)
    assert line.split()[:3] == [night, COMMIT[:12], "void"]
    assert marker in line  # the drain's who, when and why reach STATUS
    assert json.loads((root / night / "verdict.json").read_text())["outcome"] == "void"
    assert status(drained=False).split()[:3] == [night, COMMIT[:12], "MISSING"]


def test_gate_c_reads_the_median_of_the_last_2000_steps_not_the_last_step(tmp_path: Path) -> None:
    window = slice(18000, 20000)  # steps 18 001-20 000
    # Energy high over the window but low at its last step: (c) fails on the median, (b) passes.
    high = _green()
    e = high["energy"].copy()
    e[window] = 1e-8
    e[-1] = 5.5e-10
    high["energy"] = e
    v = _night(tmp_path / "high", high)
    failing = {(c["gate"], c.get("equation") or c.get("measure")) for c in v["failing"]}
    assert ("c", "energy") in failing
    assert ("b", "energy") not in failing
    assert v["measures"]["median_drops"]["energy/peak"] == pytest.approx(
        math.log10(400.0 / float(np.median(e[window])))
    )
    # The opposite: a spike at the last step only. (b) fails on the final residual, (c) passes.
    spike = _green()
    e = spike["energy"].copy()
    e[-1] = 1e-6
    spike["energy"] = e
    v = _night(tmp_path / "spike", spike)
    failing = {(c["gate"], c.get("equation") or c.get("measure")) for c in v["failing"]}
    assert ("b", "energy") in failing
    assert ("c", "energy") not in failing
