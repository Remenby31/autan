import json
import math
from pathlib import Path
from typing import Any

HISTORY = "history.json"


class PerfHistoryError(ValueError):
    def __init__(self, where: str, problem: str) -> None:
        super().__init__(f"{where} {problem}")


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise PerfHistoryError(where, "is not a finite number")
    return float(value)


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise PerfHistoryError(where, "is not a string")
    return value


def _culprit(value: object, where: str) -> dict[str, str]:
    """A commit a bisection blamed, with the range it searched."""
    if not isinstance(value, dict) or value.get("status") != "culprit":
        raise PerfHistoryError(where, "is not a culprit")
    return {
        "status": "culprit",
        "good": _text(value.get("good"), f"{where}.good"),
        "bad": _text(value.get("bad"), f"{where}.bad"),
    }


def read_perf_history(folder: Path) -> dict[str, Any]:
    """main's performance history as bench/history/history.py writes it in the `perf-history`
    artifact of gpu-bench: the GB10 peak, the benchmarks gated against a baseline, and one record
    per measured commit of main, oldest first. Raises PerfHistoryError on any other shape."""
    try:
        doc = json.loads((folder / HISTORY).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PerfHistoryError(HISTORY, f"cannot be read: {error}") from error
    if not isinstance(doc, dict):
        raise PerfHistoryError(HISTORY, "is not an object")
    peak = _number(doc.get("peak_GBps"), "peak_GBps")
    for key in ("tracked", "records"):
        if not isinstance(doc.get(key), list):
            raise PerfHistoryError(key, "is not a list")
    tracked, records = doc["tracked"], doc["records"]
    out = []
    for i, rec in enumerate(records):
        where = f"records[{i}]"
        if not isinstance(rec, dict) or not isinstance(rec.get("benchmarks"), dict):
            raise PerfHistoryError(where, "has no benchmarks")
        benchmarks = {}
        for name, b in rec["benchmarks"].items():
            at = f"{where}.benchmarks.{name}"
            if not isinstance(b, dict):
                raise PerfHistoryError(at, "is not an object")
            entry: dict[str, Any] = {
                "median": _number(b.get("median"), f"{at}.median"),
                "cv": _number(b.get("cv"), f"{at}.cv"),
                "unit": _text(b.get("unit"), f"{at}.unit"),
            }
            # history.py gives a % of peak to every bandwidth and to nothing else.
            if (entry["unit"] == "GB/s") != ("percent_of_peak" in b):
                raise PerfHistoryError(at, "has a percent_of_peak if and only if it is in GB/s")
            if "percent_of_peak" in b:
                entry["percent_of_peak"] = _number(b["percent_of_peak"], f"{at}.percent_of_peak")
            benchmarks[name] = entry
        regression = (
            _culprit(rec["regression"], f"{where}.regression") if "regression" in rec else None
        )
        host = rec.get("host")
        if host is not None:
            host = _text(host, f"{where}.host")
        out.append(
            {
                "commit": _text(rec.get("commit"), f"{where}.commit"),
                "measured": _text(rec.get("measured"), f"{where}.measured"),
                "host": host,
                "gpu": _text(rec.get("gpu"), f"{where}.gpu"),
                "benchmarks": benchmarks,
                **({"regression": regression} if regression is not None else {}),
            }
        )
    return {
        "peak_GBps": peak,
        "tracked": [_text(t, "tracked[]") for t in tracked],
        "records": out,
    }
