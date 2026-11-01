"""Measure, validate and compare Autan benchmark results."""

import argparse
import sys
from fnmatch import fnmatch
from pathlib import Path

from autan_bench import gpu, result, runner
from autan_bench.compare import DEFAULT_THRESHOLD, compare
from autan_bench.errors import BenchError, Code

MAX_BASELINE_CV = 0.02
MIN_ARRAY_OVER_L2 = 4


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


def _stream(args: argparse.Namespace) -> int:
    binary_args = ["--elements", args.elements, "--warmup", args.warmup, "--reps", args.reps]
    binary_args += ["--batch", args.batch, "--block", args.block]
    doc = runner.run(
        args.binary,
        [str(a) for a in binary_args],
        suite="stream",
        precision="fp64",
        lock=args.lock,
    )
    return _report(doc, args.out)


def _run(args: argparse.Namespace) -> int:
    binary_args = args.binary_args[1:] if args.binary_args[:1] == ["--"] else args.binary_args
    doc = runner.run(
        args.binary, binary_args, suite=args.suite, precision=args.precision, lock=args.lock
    )
    return _report(doc, args.out)


def _validate(args: argparse.Namespace) -> int:
    for path in args.results:
        result.load(path)
        _say(f"{path}: valid {result.SCHEMA_ID}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    rows = compare(result.load(args.result), result.load(args.baseline), args.threshold)
    _say(f"{'benchmark':<16} {'baseline':>10} {'current':>10} {'change':>8}")
    for r in rows:
        flag = "  REGRESSION" if r.regression else ""
        _say(
            f"{r.name:<16} {r.baseline:>10.2f} {r.current:>10.2f} {r.change:>+8.2%}  {r.unit}{flag}"
        )
    bad = [r.name for r in rows if r.regression]
    if bad:
        _say(
            f"error[REGRESSION]: {', '.join(bad)} worse than the baseline by more than {args.threshold:.0%}",
            err=True,
        )
        return 1
    return 0


def _baseline(args: argparse.Namespace) -> int:
    doc = result.load(args.result)
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
    if not doc["provenance"]["image"]:
        raise BenchError(
            Code.IMAGE_UNKNOWN, "a baseline must record its image digest (AUTAN_IMAGE)"
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

    b = sub.add_parser("baseline", help="promote a clean, low-noise result to a baseline")
    b.add_argument("result", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--max-cv", type=float, default=MAX_BASELINE_CV)
    b.add_argument(
        "--select",
        action="append",
        metavar="PATTERN",
        help="keep only the benchmarks whose name matches this glob (repeatable): the tracked set",
    )
    b.set_defaults(func=_baseline)

    k = sub.add_parser("peak", help="print the peak STREAM bandwidth (GB/s) of a baseline")
    k.add_argument("baseline", type=Path)
    k.set_defaults(func=_peak)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``autan-bench`` command; returns 0 on success, 1 on regression, 2 on error."""
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except BenchError as err:
        _say(str(err), err=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
