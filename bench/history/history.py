#!/usr/bin/env python3
"""Performance history of main from gpu-bench's result artifacts.

    bench/history/history.py ARTIFACTS_DIR --baselines DIR --out DIR

ARTIFACTS_DIR holds one directory per measured commit, as `gh run download` writes the
`gpu-bench-<sha>` artifacts (bench/history/fetch.sh does it in CI): autan-bench result files,
and `bisect/summary.json` when that run bisected a regression. Writes `history.json` (one record
per commit, oldest first), `history.csv` (one row per commit and benchmark) and `history.html`
(one chart per tracked benchmark, commit by commit, culprits of bisected regressions marked).
Tracked benchmarks are those of the baselines; every benchmark is kept in the data. "% of
peak" divides a bandwidth by the GB10 peak, the highest STREAM median of `gb10.json`.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path


def load_results(commit_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(commit_dir.glob("*.json")):
        doc = json.loads(path.read_text())
        if isinstance(doc, dict) and doc.get("schema", "").startswith("autan.bench-result/"):
            docs.append(doc)
    return docs


def culprits(artifacts: Path) -> dict[str, dict]:
    """Commits a bisection named, with the range it searched."""
    out = {}
    for summary in artifacts.glob("*/bisect/summary.json"):
        s = json.loads(summary.read_text())
        if s.get("status") == "culprit":
            out[s["culprit"]] = {"status": "culprit", "good": s["good"], "bad": s["bad"]}
    return out


def record(docs: list[dict], peak: float) -> dict | None:
    commits = {d["provenance"]["solver"]["commit"] for d in docs}
    if len(commits) != 1:
        return None
    first = min(docs, key=lambda d: d["created"])
    benchmarks = {}
    for doc in docs:
        for b in doc["benchmarks"]:
            entry = {"median": b["stats"]["median"], "cv": b["stats"]["cv"], "unit": b["unit"]}
            if b["unit"] == "GB/s":
                entry["percent_of_peak"] = 100 * b["stats"]["median"] / peak
            benchmarks[b["name"]] = entry
    return {
        "commit": commits.pop(),
        "measured": first["created"],
        "host": first["provenance"].get("host"),
        "gpu": first["provenance"]["gpu"]["name"],
        "image": first["provenance"].get("image"),
        "benchmarks": benchmarks,
    }


def build(artifacts: Path, baselines: Path) -> dict:
    gb10 = json.loads((baselines / "gb10.json").read_text())
    peak = max(b["stats"]["median"] for b in gb10["benchmarks"])
    tracked = []
    for path in sorted(baselines.glob("*.json")):
        tracked += [b["name"] for b in json.loads(path.read_text())["benchmarks"]]
    records = []
    for commit_dir in sorted(p for p in artifacts.iterdir() if p.is_dir()):
        docs = load_results(commit_dir)
        rec = record(docs, peak) if docs else None
        if rec:
            records.append(rec)
    records.sort(key=lambda r: r["measured"])
    blamed = culprits(artifacts)
    for rec in records:
        if rec["commit"] in blamed:
            rec["regression"] = blamed[rec["commit"]]
    return {"peak_GBps": peak, "tracked": tracked, "records": records}


def write_csv(history: dict, path: Path) -> None:
    with path.open("w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["measured", "commit", "host", "benchmark", "unit", "median", "cv",
                      "percent_of_peak", "tracked", "culprit"])  # fmt: skip
        for rec in history["records"]:
            for name, b in sorted(rec["benchmarks"].items()):
                out.writerow([rec["measured"], rec["commit"], rec["host"], name, b["unit"],
                              b["median"], b["cv"], b.get("percent_of_peak", ""),
                              name in history["tracked"], "regression" in rec])  # fmt: skip


INK, MUTED, GRID, SERIES, CRITICAL = "#1f1f1e", "#6b6a63", "#e6e5e0", "#2a78d6", "#d03b3b"


def chart(history: dict, name: str) -> str:
    points = [(i, r) for i, r in enumerate(history["records"]) if name in r["benchmarks"]]
    if not points:
        return ""
    w, h, left, right, top, bottom = 640, 180, 56, 12, 12, 28
    values = [r["benchmarks"][name]["median"] for _, r in points]
    # At least +-5 % around the data (the gate's threshold): run-to-run noise must not look
    # like a trend.
    mid = (min(values) + max(values)) / 2
    half = max((max(values) - min(values)) / 2 * 1.2, 0.05 * abs(mid)) or 1.0
    lo, hi = mid - half, mid + half
    decimals = max(0, math.ceil(-math.log10((hi - lo) / 3)) + 1)
    n = max(len(history["records"]) - 1, 1)

    def x(i: int) -> float:
        return left + (w - left - right) * i / n

    def y(v: float) -> float:
        return top + (h - top - bottom) * (hi - v) / (hi - lo)

    unit = points[0][1]["benchmarks"][name]["unit"]
    path = " ".join(f"{x(i):.1f},{y(r['benchmarks'][name]['median']):.1f}" for i, r in points)
    marks = []
    for i, r in points:
        b = r["benchmarks"][name]
        culprit = "regression" in r
        tip = (f"{r['commit'][:10]} {r['measured']} {b['median']:.4g} {unit} "
               f"(CV {100 * b['cv']:.1f} %)" + (" culprit of a regression" if culprit else ""))  # fmt: skip
        color = CRITICAL if culprit else SERIES
        marks.append(f'<circle cx="{x(i):.1f}" cy="{y(b["median"]):.1f}" r="{5 if culprit else 4}" '
                     f'fill="{color}" stroke="#fcfcfb" stroke-width="2"><title>'
                     f"{html.escape(tip)}</title></circle>")  # fmt: skip
    grid = "".join(
        f'<line x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}"/>'
        f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end" font-size="11" '
        f'fill="{MUTED}">{v:.{decimals}f}</text>'
        for v in (lo + (hi - lo) * k / 3 for k in range(4))
    )  # fmt: skip
    return (f'<figure><figcaption>{html.escape(name)} ({html.escape(unit)})</figcaption>'
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(name)} by commit">'
            f'{grid}<polyline points="{path}" fill="none" stroke="{SERIES}" stroke-width="2"/>'
            f'{"".join(marks)}<text x="{left}" y="{h - 8}" font-size="11" fill="{MUTED}">'
            f'oldest</text><text x="{w - right}" y="{h - 8}" font-size="11" fill="{MUTED}" '
            f'text-anchor="end">newest</text></svg></figure>')  # fmt: skip


def write_html(history: dict, path: Path) -> None:
    rows = "".join(
        f"<tr><td>{html.escape(r['measured'])}</td><td><code>{r['commit'][:10]}</code></td>"
        f"<td>{html.escape(str(r['host']))}</td><td>{'culprit' if 'regression' in r else ''}</td></tr>"
        for r in reversed(history["records"])
    )  # fmt: skip
    charts = "".join(chart(history, name) for name in history["tracked"])
    path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Autan performance history</title>
<style>body{{font-family:system-ui,sans-serif;color:{INK};background:#fcfcfb;margin:16px;max-width:680px}}
figure{{margin:16px 0}} figcaption{{font-size:13px;margin-bottom:4px}} svg{{width:100%;height:auto}}
table{{border-collapse:collapse;font-size:12px}} td{{padding:2px 8px;border-bottom:1px solid {GRID}}}</style>
</head><body><h1>Performance of main</h1>
<p>Medians of each tracked benchmark, one point per measured commit (gpu-bench on the bench Spark).
Hover a point for its commit, value and CV; red points are commits a bisection blamed.
GB10 peak: {history["peak_GBps"]:.2f} GB/s.</p>{charts}
<h2>Measured commits</h2><table><tr><td>Measured</td><td>Commit</td><td>Host</td><td></td></tr>{rows}</table>
</body></html>
""")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    history = build(args.artifacts, args.baselines)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "history.json").write_text(json.dumps(history, indent=1) + "\n")
    write_csv(history, args.out / "history.csv")
    write_html(history, args.out / "history.html")
    sys.stdout.write(f"{len(history['records'])} measured commits\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
