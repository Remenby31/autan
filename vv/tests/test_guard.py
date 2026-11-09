"""The nightly guard's verdict (#611) on histories built so that each measure follows by hand."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from autan_vv import guard
from autan_vv.cli import main

if TYPE_CHECKING:
    from pathlib import Path

COMMIT = "c" * 40
STEPS = guard.STEPS


def _series(points: list[tuple[int, float]]) -> np.ndarray:
    """Residuals at steps 1..STEPS, log-linear between (step, value) points."""
    steps, values = zip(*points, strict=True)
    return 10.0 ** np.interp(np.arange(1, STEPS + 1), steps, np.log10(values))


def _green() -> dict[str, np.ndarray]:
    # Main's shape (#611): energy peaks at step 7, is 10 orders below the peak by step 1 500 and
    # stays there; momentum-y starts small, so only the shared momentum reference gives it 10 orders.
    return {
        "density": _series([(1, 1e-15), (STEPS, 1e-15)]),
        "momentum-x": _series([(1, 4.0), (STEPS, 4e-13)]),
        "momentum-y": _series([(1, 1.5e-5), (2, 1e-3), (STEPS, 1.6e-13)]),
        "momentum-z": _series([(1, 1e-38), (2, 1e-19), (STEPS, 1e-19)]),
        "energy": _series([(1, 12.0), (7, 400.0), (1500, 3e-8), (STEPS, 5.5e-10)]),
        "nu-tilde": _series([(1, 6.8e-5), (STEPS, 1.1e-18)]),
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
        s["energy"] = _series([(1, 12.0), (7, 400.0), (1500, 3e-8), (STEPS, 5.5e-9)])
    elif edit == "m3f":  # (d): the first crossing comes late
        s["energy"] = _series([(1, 12.0), (7, 400.0), (5000, 3e-8), (STEPS, 5.5e-10)])
    elif edit == "m3s":  # (d): an excursion above 1e-10 x peak at step 14 000
        e = s["energy"].copy()
        e[13999] = 1e-6
        s["energy"] = e
    elif edit == "momentum-z":  # (b): the slab's zero equation leaves its floor
        s["momentum-z"] = _series([(1, 1e-38), (2, 1e-19), (STEPS, 1e-9)])
    elif edit == "nu-tilde":  # (b) and (c)
        s["nu-tilde"] = _series([(1, 6.8e-5), (STEPS, 1.1e-16)])
    return s


@pytest.mark.parametrize(
    ("edit", "gate", "what"),
    [
        ("energy-final", "b", "energy"),
        ("m3f", "d", "m3f"),
        ("m3s", "d", "m3s"),
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
    if edit == "m3s":
        assert ("d", "m3f") not in failing
        assert v["measures"]["m3s"] == 14001  # the step after the excursion at step 14 000


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
