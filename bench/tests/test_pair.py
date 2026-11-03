import fcntl
import functools
import json
import os
import random
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from autan_bench import gpu, pair, runner
from autan_bench.cli import main
from autan_bench.errors import BenchError, Code
from conftest import bench, make_result
from test_run import FAKE_IOREG, FAKE_SMI, fake_runnable, raw_output


def noisy(center: float, rng: random.Random, n: int = 20, spread: float = 0.005) -> list[float]:
    """n samples within ±spread of center."""
    return [center * (1 + rng.uniform(-spread, spread)) for _ in range(n)]


def test_the_order_alternates_which_side_goes_first():
    assert pair.order(5) == ["change", "base", "base", "change", "change", "base",
                             "base", "change", "change", "base"]  # fmt: skip


def test_no_change_is_ok_with_an_interval_containing_one():
    rng = random.Random(1)
    row = pair.judge("s.triad", "GB/s", "higher", [noisy(220, rng) for _ in range(5)],
                     [noisy(220, rng) for _ in range(5)])  # fmt: skip
    assert row.verdict == pair.OK
    assert row.ci[0] < 1 < row.ci[1]
    assert row.ratio == pytest.approx(1, abs=0.005)


@pytest.mark.parametrize(("better", "change_center"), [("higher", 200.0), ("lower", 1.1)])
def test_a_known_ten_percent_loss_is_a_regression(better, change_center):
    rng = random.Random(2)
    base_center = 220.0 if better == "higher" else 1.0
    row = pair.judge("s.k", "u", better, [noisy(change_center, rng) for _ in range(5)],
                     [noisy(base_center, rng) for _ in range(5)])  # fmt: skip
    assert row.verdict == pair.REGRESSION
    # R < 1 means worse whichever the direction: 200/220 and 1/1.1.
    assert row.ratio == pytest.approx(1 / 1.1, rel=0.01)
    assert row.ci[1] < 1


@pytest.mark.parametrize(("better", "factor"), [("higher", 1.15), ("lower", 0.85)])
def test_an_improvement_is_ok(better, factor):
    rng = random.Random(3)
    row = pair.judge("s.k", "u", better, [noisy(100 * factor, rng) for _ in range(5)],
                     [noisy(100, rng) for _ in range(5)])  # fmt: skip
    assert row.verdict == pair.OK
    assert row.ratio > 1


@pytest.mark.parametrize(
    ("better", "change", "verdict"),
    [
        ("higher", 95.0, pair.OK),  # exactly 5 % less bandwidth passes
        ("higher", 94.9, pair.REGRESSION),
        ("lower", 105.0, pair.OK),  # exactly 5 % more time passes: R = 1/1.05, not 0.95
        ("lower", 105.1, pair.REGRESSION),  # 5.1 % slower fails, as compare judges it
    ],
)
def test_the_threshold_is_five_percent_in_the_metric_s_own_terms(better, change, verdict):
    row = pair.judge("s.k", "u", better, [[change, change]] * 5, [[100.0, 100.0]] * 5)
    assert row.verdict == verdict


def series(sides: list[str], value, rng) -> tuple[list[list[float]], list[list[float]]]:
    """Each side's runs, run i of the series measuring value(i, side)."""
    runs = {"change": [], "base": []}
    for i, side in enumerate(sides):
        runs[side].append(noisy(value(i, side), rng, spread=0.002))
    return runs["change"], runs["base"]


def test_a_drift_of_the_machine_is_cancelled_by_the_alternation():
    """The machine slows by 1 % per run over the series, the code is the same on both sides: the
    first run of each pair is favoured, and that alternates between the sides, so at most 1/5 of
    one run's drift is left; all base runs then all change runs would read a 5 % regression."""
    rng = random.Random(4)

    def drift(i: int, _: str) -> float:
        return 220 * (1 - 0.01 * i)

    alternated = pair.judge("s.k", "GB/s", "higher", *series(pair.order(5), drift, rng))
    assert alternated.verdict == pair.OK
    assert alternated.ratio == pytest.approx(1, abs=0.004)
    blocked = pair.judge("s.k", "GB/s", "higher",
                         *series(["base"] * 5 + ["change"] * 5, drift, rng))  # fmt: skip
    assert blocked.ratio < 0.95


def test_a_real_loss_is_caught_under_a_drift():
    """A 6 % loss while the machine slows by 1.1 % per run (codex on #428: pooled medians read
    R = 0.951 there, ok)."""
    rng = random.Random(7)

    def value(i: int, side: str) -> float:
        return 100 * (1 - 0.011 * i) * (0.94 if side == "change" else 1.0)

    row = pair.judge("s.k", "GB/s", "higher", *series(pair.order(5), value, rng))
    assert row.verdict == pair.REGRESSION
    assert row.ratio == pytest.approx(0.94, abs=0.004)


def test_a_slowdown_shared_by_both_runs_of_a_pair_cancels():
    """Runs ±8 % apart, the machine's doing, the change 6 % worse in every pair: a regression
    with a tight interval (codex on #428: drawing each side's runs independently gave
    [0.827, 1.069], inconclusive)."""
    rng = random.Random(8)
    offsets = [0.92, 1.08, 1.0, 0.95, 1.05]
    change = [noisy(94 * o, rng, spread=0.001) for o in offsets]
    base = [noisy(100 * o, rng, spread=0.001) for o in offsets]
    row = pair.judge("s.k", "GB/s", "higher", change, base)
    assert row.verdict == pair.REGRESSION
    assert row.ci[1] - row.ci[0] < 0.01
    # The run CV still reports the machine's spread: medians 92, 108, 100, 95, 105.
    assert row.base_run_cv == pytest.approx(44.5**0.5 / 100, rel=0.01)


def test_noise_within_pairs_enters_the_interval():
    """Each run is tight but the pairs disagree, from 20 % worse to 10 % better: a 6 % mean
    loss is not established."""
    rng = random.Random(5)
    ratios = [0.80, 1.10, 0.93, 0.85, 1.05]
    change = [noisy(100 * r, rng, spread=0.001) for r in ratios]
    base = [noisy(100, rng, spread=0.001) for _ in ratios]
    row = pair.judge("s.k", "GB/s", "higher", change, base)
    assert row.ratio < 0.95
    assert row.verdict == pair.INCONCLUSIVE
    assert row.ci[1] >= 1


@pytest.mark.parametrize("pairs", [1, 3, 4])
def test_fewer_than_the_declared_five_pairs_are_insufficient(pairs):
    with pytest.raises(BenchError) as err:
        pair.judge("s.k", "u", "higher", [[1.0, 1.0]] * pairs, [[1.0, 1.0]] * pairs)
    assert err.value.code == Code.INSUFFICIENT_PAIRS


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_median_that_gives_no_ratio_is_refused(bad):
    with pytest.raises(BenchError) as err:
        pair.judge("s.k", "u", "lower", [[bad, bad]] * 5, [[1.0, 1.0]] * 5)
    assert err.value.code == Code.NON_POSITIVE_MEDIAN


def test_an_upper_bound_of_one_in_round_off_is_not_a_regression():
    """Codex on #428: these pairs' exact upper bound is 1, computed as 0.9999999999999999."""
    change = [[15625.0, 15625.0]] * 4 + [[676.0, 676.0]]
    base = [[17576.0, 17576.0]] * 4 + [[625.0, 625.0]]
    row = pair.judge("s.k", "u", "higher", change, base)
    assert row.ratio < 0.95
    assert row.ci[1] == pytest.approx(1.0, abs=1e-12)
    assert row.verdict == pair.INCONCLUSIVE


def test_extreme_medians_are_a_coded_error_not_a_crash():
    """Codex on #428: 1e200/1e-200 overflowed as a quotient (math domain error). As logarithms
    the ratio is 1, but resamples drawing that pair twice leave the float range: an explicit
    error, not a crash."""
    change = [[1e200, 1e200], [1e-200, 1e-200]] + [[1.0, 1.0]] * 3
    base = [[1e-200, 1e-200], [1e200, 1e200]] + [[1.0, 1.0]] * 3
    with pytest.raises(BenchError) as err:
        pair.judge("s.k", "u", "higher", change, base)
    assert err.value.code == Code.INCOMPATIBLE
    assert "the interval beyond the float range" in str(err.value)


def test_unpaired_runs_are_refused():
    with pytest.raises(BenchError) as err:
        pair.judge("s.k", "u", "higher", [[1.0]] * 5, [[1.0]] * 4)
    assert err.value.code == Code.INCOMPATIBLE


def test_the_bootstrap_is_reproducible():
    rng = random.Random(6)
    change, base = [noisy(97, rng) for _ in range(5)], [noisy(100, rng) for _ in range(5)]
    assert pair.judge("s", "u", "higher", change, base) == pair.judge("s", "u", "higher", change,
                                                                       base)  # fmt: skip


def test_runs_from_another_gpu_or_config_are_refused():
    ok = make_result([bench("stream.triad", [1.0, 2.0])])
    with pytest.raises(BenchError) as err:
        pair.judge_all([ok], [make_result([bench("stream.triad", [1.0, 2.0])], gpu="Other")])
    assert err.value.code == Code.GPU_MISMATCH
    with pytest.raises(BenchError) as err:
        pair.judge_all([make_result([bench("stream.triad", [1.0, 2.0])], config={"n": 2})], [ok])
    assert err.value.code == Code.INCOMPATIBLE


def test_a_benchmark_missing_from_a_run_is_refused():
    both = make_result([bench("stream.triad", [1.0, 2.0]), bench("stream.copy", [1.0, 2.0])])
    with pytest.raises(BenchError) as err:
        pair.judge_all([make_result([bench("stream.triad", [1.0, 2.0])])], [both])
    assert err.value.code == Code.MISSING_BENCHMARK


def test_a_benchmark_whose_direction_differs_is_refused():
    lower = make_result([bench("stream.triad", [1.0, 2.0], better="lower")])
    with pytest.raises(BenchError) as err:
        pair.judge_all([lower], [make_result([bench("stream.triad", [1.0, 2.0])])])
    assert err.value.code == Code.INCOMPATIBLE


# The command, with fake binaries: each prints the raw format with its kernel times scaled by a
# factor per run (read from a file, one line per run), appends its side to a log, and records
# whether the GPU lock was held by someone else while it ran.
PAIR_BINARY = """#!{python}
import fcntl, json, pathlib, sys, time
log = pathlib.Path({log!r})
runs = [line for line in log.read_text().splitlines()] if log.exists() else []
factors = pathlib.Path({factors!r}).read_text().split()
factor = factors[sum(1 for r in runs if r.startswith({side!r}))]
if factor == "crash":
    raise SystemExit(1)
factor = 0 if factor == "hang" else float(factor)
def held():
    with open({lock!r}) as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True
setup = ""
if {gated!r}:
    # A gated binary: its setup (here, nothing) before the go, then ready and the go line.
    setup = f" setup-held={{held()}}"
    print("autan-bench: ready", file=sys.stderr, flush=True)
    sys.stdin.readline()
with log.open("a") as out:
    out.write(f"{side} {{held()}}{{setup}}\\n")
if factor == 0:
    pathlib.Path({log!r} + ".timed").write_text(str(__import__("os").getpid()))
    time.sleep(60)
busy = pathlib.Path({busy!r})
busy.touch()
time.sleep(0.3)
busy.unlink()
raw = json.loads(pathlib.Path({raw!r}).read_text())
raw["kernels"][0]["seconds"] = [s * factor * (1 + 0.001 * i) for i, s in
                                enumerate([0.010] * 12)]
print(json.dumps(raw))
"""


@pytest.fixture
def paired(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    fake_runnable(tmp_path, monkeypatch, 3)
    busy = tmp_path / "busy"
    # Both platforms' telemetry: a harness run as a subprocess keeps its own sys.platform.
    for tool, text in (("nvidia-smi", FAKE_SMI), ("ioreg", FAKE_IOREG)):
        path = tmp_path / tool
        path.write_text(text.format(busy=busy))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    # The load and idle checks are test_run's; here they are made quick, not skipped.
    monkeypatch.setattr(gpu, "require_idle", functools.partial(gpu.require_idle, interval_s=0.02))
    monkeypatch.setattr(gpu, "runnable_threads", lambda: 2.0)
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(raw_output(True, "cuda")))
    lock = tmp_path / "lock"
    lock.touch()

    def make(change: list[float], base: list[float], gated: bool = False) -> list[str]:
        binaries = {}
        for side, factors in (("change", change), ("base", base)):
            (tmp_path / f"{side}.factors").write_text(" ".join(map(str, factors)))
            path = tmp_path / f"{side}-bench"
            path.write_text(PAIR_BINARY.format(
                python=sys.executable, log=str(tmp_path / "log"), side=side, lock=str(lock),
                busy=str(busy), raw=str(raw), factors=str(tmp_path / f"{side}.factors"),
                gated=gated))  # fmt: skip
            path.chmod(path.stat().st_mode | stat.S_IEXEC)
            binaries[side] = str(path)
        return ["pair", "--suite", "codegen", "--precision", "fp32", "--change",
                binaries["change"], "--base", binaries["base"], "--out", str(tmp_path / "out"),
                "--lock", str(lock)]  # fmt: skip

    return make


def log(tmp_path: Path) -> list[str]:
    return (tmp_path / "log").read_text().splitlines()


def test_pair_alternates_the_sides_in_one_hold_and_records_each_run(
    paired, tmp_path, capsys, monkeypatch
):
    holds = []
    take = gpu.gpu_lock

    @contextmanager
    def counted(*args, **kwargs):
        holds.append(1)
        with take(*args, **kwargs) as waited:
            yield waited

    monkeypatch.setattr(gpu, "gpu_lock", counted)
    assert main([*paired([1.0] * 10, [1.0] * 10), "--confirm"]) == 0
    # One hold for the series; an ok first series needs no confirmation.
    assert holds == [1]
    # Each run saw the lock held (by the gate), change first, then base first, then change first.
    assert log(tmp_path) == [f"{s} True" for s in pair.order(5)]
    out = tmp_path / "out"
    runs = sorted(p.name for p in (out / "series-1").iterdir())
    assert runs == [f"{i + 1:02d}-{s}.json" for i, s in enumerate(pair.order(5))]
    summary = json.loads((out / "pair.json").read_text())
    assert summary["schema"] == pair.SCHEMA_ID
    assert summary["outcome"] == pair.OK
    assert summary["state"] == "complete"
    (series,) = summary["series"]
    assert series["order"] == pair.order(5)
    assert series["hold_s"] > 0
    names = {r["name"] for r in series["rows"]}
    assert names == {"codegen.flux", "codegen.flux.time_per_cell"}
    for row in series["rows"]:
        assert row["ratio"] == pytest.approx(1)
        assert row["change_run_cv"] == row["base_run_cv"] == pytest.approx(0)
    assert "lock held" in capsys.readouterr().out


def test_a_regression_is_confirmed_by_a_second_series_which_decides(paired, tmp_path, capsys):
    # The change's kernels take 10 % longer in every run of both series.
    assert main([*paired([1.1] * 10, [1.0] * 10), "--confirm"]) == 1
    err = capsys.readouterr().err
    assert "error[REGRESSION]" in err
    assert "confirmed by a second series" in err
    summary = json.loads((tmp_path / "out" / "pair.json").read_text())
    assert [s["outcome"] for s in summary["series"]] == [pair.REGRESSION, pair.REGRESSION]
    assert summary["outcome"] == pair.REGRESSION
    assert len(log(tmp_path)) == 20


def test_a_regression_not_confirmed_passes(paired, tmp_path):
    # The first series reads the change 10 % slower; the second, the same as the base.
    assert main([*paired([1.1] * 5 + [1.0] * 5, [1.0] * 10), "--confirm"]) == 0
    summary = json.loads((tmp_path / "out" / "pair.json").read_text())
    assert [s["outcome"] for s in summary["series"]] == [pair.REGRESSION, pair.OK]
    assert summary["outcome"] == pair.OK


def test_inconclusive_twice_is_noise_too_high(paired, tmp_path, capsys):
    # A 7 % loss on average, but runs ±10 % apart: the interval contains 1 in both series.
    change = [0.97, 1.17, 1.07, 1.20, 0.90] * 2
    base = [1.10, 0.90, 1.0, 1.0, 1.0] * 2
    summary_md = tmp_path / "summary.md"
    assert main([*paired(change, base), "--confirm", "--summary", str(summary_md)]) == 3
    assert "### Paired gate, codegen: **INCONCLUSIVE: noise too high**" in summary_md.read_text()
    assert summary_md.read_text().count("| codegen.flux |") == 2
    err = capsys.readouterr().err
    assert "error[INCONCLUSIVE]" in err
    assert "noise too high" in err
    summary = json.loads((tmp_path / "out" / "pair.json").read_text())
    assert [s["outcome"] for s in summary["series"]] == [pair.INCONCLUSIVE] * 2
    assert len(summary["series"]) == 2


def test_without_confirm_one_series_decides(paired, tmp_path, capsys):
    summary_md = tmp_path / "summary.md"
    assert main([*paired([1.1] * 5, [1.0] * 5), "--summary", str(summary_md)]) == 1
    text = summary_md.read_text()
    assert "### Paired gate, codegen: **REGRESSION**" in text
    assert "Series 1: 10 runs (C B B C C B B C C B)" in text
    assert "Series 2" not in text
    assert "confirmed" not in capsys.readouterr().err
    assert len(log(tmp_path)) == 10


def test_runs_measured_before_a_failure_are_kept(paired, tmp_path, capsys):
    # The tenth run, the base's fifth, crashes: the nine before it are on disk.
    assert main(paired(["1.0"] * 5, ["1.0", "1.0", "1.0", "1.0", "crash"])) == 2
    assert "error[BINARY_FAILED]" in capsys.readouterr().err
    kept = sorted(p.name for p in (tmp_path / "out" / "series-1").iterdir())
    assert kept == [f"{i + 1:02d}-{s}.json" for i, s in enumerate(pair.order(5)[:9])]


def test_a_failed_confirmation_keeps_the_first_series_and_no_verdict(paired, tmp_path):
    # The first series reads a regression; the confirmation's last run crashes.
    summary_md = tmp_path / "summary.md"
    argv = [
        *paired(["1.1"] * 10, ["1.0"] * 9 + ["crash"]),
        "--confirm",
        "--summary",
        str(summary_md),
    ]
    assert main(argv) == 2
    summary = json.loads((tmp_path / "out" / "pair.json").read_text())
    assert [s["outcome"] for s in summary["series"]] == [pair.REGRESSION]
    # The gate did not conclude: no verdict, the error recorded and in the job summary.
    assert summary["state"] == "failed"
    assert "outcome" not in summary
    assert "BINARY_FAILED" in summary["error"]
    assert "DID NOT COMPLETE" in summary_md.read_text()


def test_an_earlier_run_s_files_never_pass_for_this_one_s(paired, tmp_path):
    out = tmp_path / "out"
    (out / "series-2").mkdir(parents=True)
    (out / "series-2" / "01-change.json").write_text("{}")
    (out / "pair.json").write_text('{"outcome": "ok"}')
    # This run crashes at once: nothing of the earlier one is left to read as its result.
    assert main(paired(["crash"] * 5, ["1.0"] * 5)) == 2
    assert json.loads((out / "pair.json").read_text())["state"] == "failed"
    assert not (out / "series-2").exists()


def test_pair_refuses_fewer_than_five_pairs(paired, capsys):
    assert main([*paired([1.0] * 8, [1.0] * 8), "--pairs", "4"]) == 2
    assert "error[INSUFFICIENT_PAIRS]" in capsys.readouterr().err


def test_the_stream_suite_measures_with_the_stream_command_s_arguments(monkeypatch, tmp_path):
    seen = []

    def series(*args, **_):
        seen.append(args[2])
        raise BenchError(Code.NO_STREAM, "stop here")

    monkeypatch.setattr(runner, "pair_series", series)
    argv = ["pair", "--suite", "stream", "--precision", "fp64", "--change", "c", "--base", "b",
            "--out", str(tmp_path / "out")]  # fmt: skip
    assert main(argv) == 2
    assert seen == [["--elements", str(1 << 28), "--warmup", "10", "--reps", "100",
                     "--batch", "5", "--block", "256"]]  # fmt: skip


def test_gated_binaries_take_one_hold_per_pair_and_set_up_outside_it(paired, tmp_path, monkeypatch):
    holds = []
    take = gpu.gpu_lock

    @contextmanager
    def counted(*args, **kwargs):
        holds.append(1)
        with take(*args, **kwargs) as waited:
            yield waited

    monkeypatch.setattr(gpu, "gpu_lock", counted)
    assert main([*paired([1.0] * 5, [1.0] * 5, gated=True), "--ready", "30"]) == 0
    # Five pairs, five holds: other jobs may run between pairs, never inside one.
    assert len(holds) == 5
    runs = log(tmp_path)
    assert [r.split()[0] for r in runs] == pair.order(5)
    # Every run measured under the lock, every setup done without it.
    assert all(r.split()[1] == "True" and r.split()[2] == "setup-held=False" for r in runs)
    summary = json.loads((tmp_path / "out" / "pair.json").read_text())
    assert summary["outcome"] == pair.OK
    assert summary["series"][0]["hold_s"] > 0


def test_a_gated_pair_whose_second_process_fails_leaves_nothing_running(paired, tmp_path):
    # The base's first run crashes before its ready line: the change's process is killed too.
    assert main([*paired(["1.0"] * 5, ["crash"] * 5, gated=True), "--ready", "30"]) == 2
    left = subprocess.run(["pgrep", "-f", str(tmp_path / "change-bench")], capture_output=True,
                          text=True, check=False)  # fmt: skip
    assert left.stdout.strip() == ""


def test_only_the_tracked_benchmarks_are_judged(paired, tmp_path, write_json):
    # The loop benchmark times every strategy; the gate judges the baseline's tracked set.
    tracked = write_json("tracked.json", make_result([bench("codegen.flux", [1.0, 2.0])]))
    argv = [*paired([1.0] * 5, [1.0] * 5), "--tracked", str(tracked)]
    assert main(argv) == 0
    (series,) = json.loads((tmp_path / "out" / "pair.json").read_text())["series"]
    assert [r["name"] for r in series["rows"]] == ["codegen.flux"]


def test_a_tracked_benchmark_that_was_not_measured_is_an_error():
    runs = [make_result([bench("stream.triad", [1.0, 2.0])])] * 5
    with pytest.raises(BenchError) as err:
        pair.judge_all(runs, runs, names={"stream.triad", "stream.copy"})
    assert err.value.code == Code.MISSING_BENCHMARK


def test_a_terminated_harness_leaves_no_gated_binary_running_after_the_lock(paired, tmp_path):
    """Codex on #445: a TERM to the harness used to skip its cleanup, and a gated binary, in a
    session of its own, kept timing with the lock released."""
    argv = [*paired(["hang"] * 5, ["1.0"] * 5, gated=True), "--ready", "30"]
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    code = "import sys; from autan_bench.cli import main; sys.exit(main(sys.argv[1:]))"
    harness = subprocess.Popen([sys.executable, "-c", code, *argv], env=env)
    timed = tmp_path / "log.timed"
    for _ in range(300):
        if timed.exists():
            break
        time.sleep(0.1)
    child = int(timed.read_text())
    harness.terminate()
    assert harness.wait(timeout=30) == 143
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)
    summary = json.loads((tmp_path / "out" / "pair.json").read_text())
    assert summary["state"] == "failed"


def test_readiness_is_timed_from_each_process_s_launch(tmp_path):
    """Codex on #445: two processes started together were timed from each wait in turn."""
    script = tmp_path / "gated"
    script.write_text(f"#!{sys.executable}\nimport sys, time\ntime.sleep(float(sys.argv[1]))\n"
                      "print('autan-bench: ready', file=sys.stderr, flush=True)\n"
                      "sys.stdin.readline()\n")  # fmt: skip
    script.chmod(0o755)
    first, second = runner._Gated(script, ["0.6"]), runner._Gated(script, ["0.6"])
    try:
        # Both set up in parallel for 0.6 s (plus the interpreter's start): the second, waited
        # for after the first, still reads its own setup, not the few ms left after the first.
        a, b = first.wait_ready(5), second.wait_ready(5)
        assert a >= 0.6
        assert b >= 0.6
        assert abs(a - b) < 0.3
    finally:
        first.kill()
        second.kill()
    slow_a, slow_b = runner._Gated(script, ["0.2"]), runner._Gated(script, ["1.5"])
    try:
        slow_a.wait_ready(1)
        with pytest.raises(BenchError) as err:
            slow_b.wait_ready(1)
        assert err.value.code == Code.BINARY_FAILED
    finally:
        slow_a.kill()
        slow_b.kill()


def test_a_bad_tracked_baseline_fails_this_run_s_record(paired, tmp_path):
    """Codex on #445: a bad --tracked left an earlier run's ok pair.json in place."""
    argv = paired([1.0] * 5, [1.0] * 5)
    out = tmp_path / "out"
    out.mkdir()
    (out / "pair.json").write_text('{"state": "complete", "outcome": "ok"}')
    assert main([*argv, "--tracked", str(tmp_path / "absent.json")]) == 2
    assert json.loads((out / "pair.json").read_text())["state"] == "failed"


def test_hold_per_pair_releases_the_lock_between_pairs(paired, tmp_path, monkeypatch):
    # STREAM's series: one hold per pair (#403), so other jobs can take the GPU between pairs.
    events = []
    take = gpu.gpu_lock

    @contextmanager
    def watched(path, *args, **kwargs):
        with take(path, *args, **kwargs) as waited:
            events.append("hold")
            yield waited
        with Path(path).open() as handle:  # free once released: another job could take it now
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
        events.append("free")

    monkeypatch.setattr(gpu, "gpu_lock", watched)
    assert main([*paired([1.0] * 10, [1.0] * 10), "--hold-per-pair"]) == 0
    assert events == ["hold", "free"] * 5
    runs = log(tmp_path)
    assert [r.split()[0] for r in runs] == pair.order(5)
    assert all(r.split()[1] == "True" for r in runs)  # every run measured under the lock
    summary = json.loads((tmp_path / "out" / "pair.json").read_text())
    assert summary["outcome"] == pair.OK
