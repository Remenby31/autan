"""Measure, validate and compare Autan benchmark results."""

import argparse
import json
import os
import shutil
import sys
from fnmatch import fnmatch
from pathlib import Path

from autan_bench import gpu, pair, result, runner
from autan_bench.compare import (
    DEFAULT_THRESHOLD,
    HEALTH_MINIMUM,
    INCONCLUSIVE,
    REGRESSION,
    compare,
    health,
)
from autan_bench.errors import BenchError, Code

MAX_BASELINE_CV = 0.02
MIN_ARRAY_OVER_L2 = 4


def _env_seconds(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value else None


def _say(text: str, *, err: bool = False) -> None:
    print(text, file=sys.stderr if err else sys.stdout)


def _table(doc: dict) -> str:
    width = max(len(b["name"]) for b in doc["benchmarks"])
    lines = [f"{'benchmark':<{width}} {'median':>10} {'IQR':>8} {'CV':>7}  unit"]
    for b in doc["benchmarks"]:
        s = b["stats"]
        lines.append(
            f"{b['name']:<{width}} {s['median']:>10.2f} {s['iqr']:>8.2f} {s['cv']:>6.2%}  {b['unit']}"
        )
    return "\n".join(lines)


def _report(doc: dict, out: Path) -> int:
    result.write(doc, out)
    _say(_table(doc))
    _say(f"wrote {out}")
    return 0


def _stream_args(args: argparse.Namespace) -> list[str]:
    binary_args = ["--elements", args.elements, "--warmup", args.warmup, "--reps", args.reps]
    binary_args += ["--batch", args.batch, "--block", args.block]
    return [str(a) for a in binary_args]


def _stream(args: argparse.Namespace) -> int:
    doc = runner.run(
        args.binary,
        _stream_args(args),
        suite="stream",
        precision="fp64",
        lock=args.lock,
        lock_timeout_s=args.lock_timeout,
        max_load_per_cpu=args.max_load,
    )
    return _report(doc, args.out)


def _run(args: argparse.Namespace) -> int:
    binary_args = args.binary_args[1:] if args.binary_args[:1] == ["--"] else args.binary_args
    doc = runner.run(
        args.binary,
        binary_args,
        suite=args.suite,
        precision=args.precision,
        lock=args.lock,
        lock_timeout_s=args.lock_timeout,
        max_load_per_cpu=args.max_load,
        ready_timeout_s=args.ready,
    )
    return _report(doc, args.out)


def _validate(args: argparse.Namespace) -> int:
    for path in args.results:
        result.load(path)
        _say(f"{path}: valid {result.SCHEMA_ID}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    rows = compare(result.load(args.result), result.load(args.baseline), args.threshold)
    width = max(len(r.name) for r in rows)
    _say(
        f"{'benchmark':<{width}} {'baseline [95 % CI]':>26} {'current [95 % CI]':>26} {'change':>8}"
    )
    for r in rows:
        base = f"{r.baseline:.2f} [{r.baseline_ci[0]:.2f}, {r.baseline_ci[1]:.2f}]"
        now = f"{r.current:.2f} [{r.current_ci[0]:.2f}, {r.current_ci[1]:.2f}]"
        flag = "" if r.verdict == "ok" else f"  {r.verdict.upper()}"
        _say(f"{r.name:<{width}} {base:>26} {now:>26} {r.change:>+8.2%}  {r.unit}{flag}")
    bad = [r.name for r in rows if r.verdict == REGRESSION]
    if bad:
        _say(
            f"error[REGRESSION]: {', '.join(bad)} worse than the baseline by more than "
            f"{args.threshold:.0%}, with disjoint 95 % intervals of the medians",
            err=True,
        )
        return 1
    doubt = [r.name for r in rows if r.verdict == INCONCLUSIVE]
    if doubt:
        _say(
            f"error[INCONCLUSIVE]: {', '.join(doubt)} worse by more than {args.threshold:.0%} "
            "but within noise; measure again",
            err=True,
        )
        return 3
    return 0


def _health(args: argparse.Namespace) -> int:
    rows = health(result.load(args.result), result.load(args.baseline), args.minimum)
    width = max(len(name) for name, _, _ in rows)
    for name, share, ok in rows:
        _say(f"{name:<{width}} {share:>7.1%}{'' if ok else '  BELOW'}")
    if all(ok for _, _, ok in rows):
        return 0
    _say(
        f"error[UNFIT]: below {args.minimum:.0%} of {args.baseline}: the machine is not fit to "
        "measure",
        err=True,
    )
    return 4


def _pair_table(rows: list[pair.Row]) -> str:
    width = max(len(r.name) for r in rows)
    lines = [
        (
            f"{'benchmark':<{width}} {'change':>10} {'base':>10} {'ratio [95 % CI]':>24}"
            f" {'run CV c/b':>14}"
        )
    ]
    for r in rows:
        ratio = f"{r.ratio:.4f} [{r.ci[0]:.4f}, {r.ci[1]:.4f}]"
        cv = f"{r.change_run_cv:.2%}/{r.base_run_cv:.2%}"
        flag = "" if r.verdict == pair.OK else f"  {r.verdict.upper()}"
        lines.append(
            f"{r.name:<{width}} {r.change_median:>10.2f} {r.base_median:>10.2f} {ratio:>24}"
            f" {cv:>14}  {r.unit}{flag}"
        )
    return "\n".join(lines)


def _pair(args: argparse.Namespace) -> int:
    if args.pairs < pair.MIN_PAIRS:
        raise BenchError(
            Code.INSUFFICIENT_PAIRS,
            f"{args.pairs} pair(s) asked, the paired gate needs {pair.MIN_PAIRS}",
        )
    binary_args = args.binary_args[1:] if args.binary_args[:1] == ["--"] else args.binary_args
    if args.suite == "stream" and not binary_args:
        # The stream command's defaults, so that both sides measure what the absolute gate does.
        defaults = parser().parse_args(["stream", "--binary", "-", "--out", "-"])
        binary_args = _stream_args(defaults)
    args.out.mkdir(parents=True, exist_ok=True)
    # Nothing of an earlier run in this directory may pass for this one's.
    (args.out / "pair.json").unlink(missing_ok=True)
    for old in args.out.glob("series-*"):
        shutil.rmtree(old)
    summary: dict = {
        "schema": pair.SCHEMA_ID,
        "suite": args.suite,
        "threshold": args.threshold,
        "pairs": args.pairs,
        "state": "running",
        "series": [],
    }

    def save() -> None:
        (args.out / "pair.json").write_text(json.dumps(summary, indent=2) + "\n")

    def report() -> None:
        if args.summary:
            with args.summary.open("a") as out:
                out.write(pair.markdown(summary))

    save()
    final = pair.OK
    try:
        # A first series that is not ok is confirmed by a second, in a hold of its own, which
        # decides (as gate.sh does for the absolute gate); never a third.
        for number in (1, 2) if args.confirm else (1,):
            directory = args.out / f"series-{number}"
            directory.mkdir()

            def keep(i: int, side: str, doc: dict, directory: Path = directory) -> None:
                result.write(doc, directory / f"{i + 1:02d}-{side}.json")

            series = runner.pair_series(
                args.change,
                args.base,
                binary_args,
                pair.order(args.pairs),
                suite=args.suite,
                precision=args.precision,
                lock=args.lock,
                lock_timeout_s=args.lock_timeout,
                max_load_per_cpu=args.max_load,
                on_run=keep,
            )
            sides = series.sides
            change = [d for s, d in zip(sides, series.docs, strict=True) if s == "change"]
            base = [d for s, d in zip(sides, series.docs, strict=True) if s == "base"]
            rows = pair.judge_all(change, base, args.threshold)
            final = pair.outcome(rows)
            summary["series"].append(pair.series_record(sides, series.lock_wait_s,
                                                        series.hold_s, rows, final))  # fmt: skip
            # Saved after each series: a failed confirmation keeps the first series' record,
            # with the gate's state still "running", never a verdict.
            save()
            _say(f"series {number}: {len(sides)} runs, lock held {series.hold_s:.0f} s")
            _say(_pair_table(rows))
            if final == pair.OK:
                break
    except BenchError as err:
        summary["state"] = "failed"
        summary["error"] = str(err)
        save()
        report()
        raise
    summary["state"] = "complete"
    summary["outcome"] = final
    save()
    report()
    _say(f"wrote {args.out / 'pair.json'}")
    confirmed = len(summary["series"]) == 2
    names = [r["name"] for r in summary["series"][-1]["rows"] if r["verdict"] == final]
    if final == pair.REGRESSION:
        _say(
            f"error[REGRESSION]: {', '.join(names)} worse than the base by more than "
            f"{args.threshold:.0%}, the ratio's 95 % interval below 1"
            + (", confirmed by a second series" if confirmed else ""),
            err=True,
        )
        return 1
    if final == pair.INCONCLUSIVE:
        _say(
            f"error[INCONCLUSIVE]: {', '.join(names)} worse by more than {args.threshold:.0%} "
            + (
                "with an interval containing 1 in both series: noise too high; measure again "
                "at another time"
                if confirmed
                else "but within noise"
            ),
            err=True,
        )
        return 3
    return 0


def _baseline(args: argparse.Namespace) -> int:
    if len(args.results) < args.min_runs:
        raise BenchError(
            Code.TOO_FEW_RUNS,
            f"{len(args.results)} run(s) given, the baseline needs {args.min_runs}",
        )
    runs = [result.load(path) for path in args.results]
    doc = runs[0] if len(runs) == 1 else result.pool(runs)
    if args.select:
        unmatched = [
            p for p in args.select if not any(fnmatch(b["name"], p) for b in doc["benchmarks"])
        ]
        if unmatched:
            raise BenchError(Code.MISSING_BENCHMARK, f"no benchmark matches {', '.join(unmatched)}")
        doc["benchmarks"] = [
            b for b in doc["benchmarks"] if any(fnmatch(b["name"], p) for p in args.select)
        ]
    if doc["provenance"]["solver"]["dirty"]:
        raise BenchError(
            Code.DIRTY_TREE, "a baseline must come from a binary built from a clean checkout"
        )
    # The environment a baseline was measured in: the container image on Linux; on macOS, where
    # Metal runs without a container, the OS version.
    provenance = doc["provenance"]
    if provenance["backend"] == "metal":
        if not provenance.get("os_version"):
            raise BenchError(Code.ENVIRONMENT_UNKNOWN, "a Metal baseline must record os_version")
    elif not provenance["image"]:
        raise BenchError(
            Code.ENVIRONMENT_UNKNOWN, "a baseline must record its image digest (AUTAN_IMAGE)"
        )
    if doc["suite"] == "stream":
        array_bytes = doc["config"]["elements"] * doc["config"]["element_bytes"]
        l2 = doc["provenance"]["gpu"].get("l2_bytes")
        if l2 is None:
            raise BenchError(
                Code.FOOTPRINT_TOO_SMALL,
                "the result records no L2 size to check the footprint against",
            )
        if array_bytes < MIN_ARRAY_OVER_L2 * l2:
            raise BenchError(
                Code.FOOTPRINT_TOO_SMALL,
                f"each array is {array_bytes} B; a DRAM baseline needs {MIN_ARRAY_OVER_L2} x L2 ({l2} B)",
            )
    noisy = [
        f"{b['name']} ({b['stats']['cv']:.2%})"
        for b in doc["benchmarks"]
        if b["stats"]["cv"] >= args.max_cv
    ]
    if noisy:
        raise BenchError(
            Code.CV_TOO_HIGH, f"coefficient of variation >= {args.max_cv:.0%}: {', '.join(noisy)}"
        )
    result.write(doc, args.out)
    tracked = ", ".join(b["name"] for b in doc["benchmarks"])
    _say(f"wrote {args.out}: {tracked}")
    if doc["suite"] == "stream":
        _say(f"peak {result.peak_bandwidth(doc):.1f} GB/s")
    return 0


def _peak(args: argparse.Namespace) -> int:
    _say(f"{result.peak_bandwidth(result.load(args.baseline)):.2f}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autan-bench", description=__doc__)
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("stream", help="measure device memory bandwidth with the STREAM kernels")
    s.add_argument(
        "--binary", type=Path, required=True, help="the stream executable built from bench/"
    )
    s.add_argument("--out", type=Path, required=True, help="result JSON to write")
    s.add_argument("--elements", type=int, default=1 << 28, help="FP64 elements per array")
    s.add_argument("--warmup", type=int, default=10)
    s.add_argument("--reps", type=int, default=100)
    s.add_argument("--batch", type=int, default=5, help="launches averaged into one sample")
    s.add_argument("--block", type=int, default=256, help="threads per block")
    s.add_argument(
        "--lock", type=Path, default=gpu.DEFAULT_LOCK, help="flock file serialising GPU jobs"
    )
    s.add_argument(
        "--lock-timeout",
        type=float,
        metavar="SECONDS",
        default=_env_seconds("AUTAN_BENCH_LOCK_TIMEOUT"),
        help="fail (LOCK_TIMEOUT) if the GPU lock is still held after this long; default: wait"
        " ($AUTAN_BENCH_LOCK_TIMEOUT)",
    )
    s.add_argument(
        "--max-load",
        type=float,
        metavar="PER_CPU",
        help="refuse to run when the host's runnable threads per CPU exceed this (HOST_BUSY)",
    )
    s.set_defaults(func=_stream)

    r = sub.add_parser(
        "run",
        help="time any benchmark binary that prints the raw result format",
        description=runner.run.__doc__,
    )
    r.add_argument("--suite", required=True, help="suite name, the prefix of every benchmark")
    r.add_argument("--binary", type=Path, required=True)
    r.add_argument("--precision", required=True, choices=["fp32", "fp32+acc64", "fp64"])
    r.add_argument("--out", type=Path, required=True, help="result JSON to write")
    r.add_argument(
        "--lock", type=Path, default=gpu.DEFAULT_LOCK, help="flock file serialising GPU jobs"
    )
    r.add_argument(
        "--lock-timeout",
        type=float,
        metavar="SECONDS",
        default=_env_seconds("AUTAN_BENCH_LOCK_TIMEOUT"),
        help="fail (LOCK_TIMEOUT) if the GPU lock is still held after this long; default: wait"
        " ($AUTAN_BENCH_LOCK_TIMEOUT)",
    )
    r.add_argument(
        "--max-load",
        type=float,
        metavar="PER_CPU",
        help="refuse to run when the host's runnable threads per CPU exceed this (HOST_BUSY)",
    )
    r.add_argument(
        "--ready",
        type=float,
        nargs="?",
        const=1800.0,
        metavar="TIMEOUT_S",
        help="gated binary: start it without the lock, wait (default 1800 s) for its"
        " 'autan-bench: ready' line on stderr, then take the lock and send 'go' on its stdin",
    )
    r.add_argument("binary_args", nargs=argparse.REMAINDER, help="-- then the binary's arguments")
    r.set_defaults(func=_run)

    v = sub.add_parser("validate", help="check result files against the result schema")
    v.add_argument("results", type=Path, nargs="+")
    v.set_defaults(func=_validate)

    c = sub.add_parser("compare", help="fail when a result regresses against a baseline")
    c.add_argument("result", type=Path)
    c.add_argument("baseline", type=Path)
    c.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="tolerated relative loss"
    )
    c.set_defaults(func=_compare)

    h = sub.add_parser(
        "health",
        help="exit 4 unless every baseline benchmark reaches a share of its median (machine health)",
    )
    h.add_argument("result", type=Path)
    h.add_argument("baseline", type=Path)
    h.add_argument(
        "--min", type=float, default=HEALTH_MINIMUM, dest="minimum", help="share of the baseline"
    )
    h.set_defaults(func=_health)

    b = sub.add_parser("baseline", help="promote a clean, low-noise result to a baseline")
    b.add_argument("results", type=Path, nargs="+", help="one run, or several runs to pool")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="refuse fewer runs (tracked loops: 5, spaced in time)",
    )
    b.add_argument("--max-cv", type=float, default=MAX_BASELINE_CV)
    b.add_argument(
        "--select",
        action="append",
        metavar="PATTERN",
        help="keep only the benchmarks whose name matches this glob (repeatable): the tracked set",
    )
    b.set_defaults(func=_baseline)

    a = sub.add_parser(
        "pair",
        help="judge a change against its base, measured alternately in one hold of the lock",
        description=pair.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    a.add_argument("--suite", required=True, help="suite name, the prefix of every benchmark")
    a.add_argument("--change", type=Path, required=True, help="the change's benchmark binary")
    a.add_argument("--base", type=Path, required=True, help="the base's, built from its own tree")
    a.add_argument("--precision", required=True, choices=["fp32", "fp32+acc64", "fp64"])
    a.add_argument("--out", type=Path, required=True, help="directory for the runs and pair.json")
    a.add_argument(
        "--pairs",
        type=int,
        default=pair.DEFAULT_PAIRS,
        help="pairs per series, each side first in turn",
    )
    a.add_argument(
        "--confirm",
        action="store_true",
        help="when the first series is not ok, measure a second series, which decides",
    )
    a.add_argument(
        "--threshold", type=float, default=pair.DEFAULT_THRESHOLD, help="tolerated relative loss"
    )
    a.add_argument(
        "--summary", type=Path, help="append the verdicts as Markdown (a CI job summary)"
    )
    a.add_argument(
        "--lock", type=Path, default=gpu.DEFAULT_LOCK, help="flock file serialising GPU jobs"
    )
    a.add_argument(
        "--lock-timeout",
        type=float,
        metavar="SECONDS",
        default=_env_seconds("AUTAN_BENCH_LOCK_TIMEOUT"),
        help="fail (LOCK_TIMEOUT) if the GPU lock is still held after this long; default: wait"
        " ($AUTAN_BENCH_LOCK_TIMEOUT)",
    )
    a.add_argument(
        "--max-load",
        type=float,
        metavar="PER_CPU",
        help="refuse to run when the host's runnable threads per CPU exceed this (HOST_BUSY)",
    )
    a.add_argument(
        "binary_args",
        nargs=argparse.REMAINDER,
        help="-- then the binaries' arguments; none for the stream suite: the stream command's",
    )
    a.set_defaults(func=_pair)

    k = sub.add_parser("peak", help="print the peak STREAM bandwidth (GB/s) of a baseline")
    k.add_argument("baseline", type=Path)
    k.set_defaults(func=_peak)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``autan-bench`` command; returns 0 on success, 1 on regression, 2 on error, 3 when a comparison is inconclusive, 4 when ``health`` finds the machine unfit."""
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except BenchError as err:
        _say(str(err), err=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
