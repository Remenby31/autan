#!/usr/bin/env python3
"""Report a bisected regression of main: an issue labelled perf-regression in every case, and a
comment on the culprit's pull request when one commit is to blame.

    bench/ci/report_regression.py BISECT_DIR BASELINE RUN_URL

Reads BISECT_DIR/summary.json (bench/ci/bisect.py) and the probes' tracked-loop results; the
benchmarks of BASELINE measured in ns are the ones tabulated. Uses gh (GH_TOKEN,
GITHUB_REPOSITORY); needs issues: write and pull-requests: write.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

LABEL = "perf-regression"
RESULT = "codegen-hex-n128-fp32.json"


def gh(*args: str, body: str | None = None) -> str:
    out = subprocess.run(["gh", *args], input=body, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def medians(path: Path) -> dict[str, float]:
    doc = json.loads(path.read_text())
    return {b["name"]: b["stats"]["median"] for b in doc["benchmarks"]}


def table(bisect: Path, tracked: list[str], new: str, old: str) -> str:
    after, before = medians(bisect / new / RESULT), medians(bisect / old / RESULT)
    rows = ["| Loop | ns / cell before | after | change |", "|---|---|---|---|"]
    for name in tracked:
        if name in after and name in before:
            change = 100 * (after[name] - before[name]) / before[name]
            rows.append(f"| `{name}` | {before[name]:.3f} | {after[name]:.3f} | {change:+.1f} % |")
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    bisect, baseline, run_url = Path(argv[0]), Path(argv[1]), argv[2]
    summary = json.loads((bisect / "summary.json").read_text())
    tracked = [b["name"] for b in json.loads(baseline.read_text())["benchmarks"]
               if b["unit"] == "ns"]  # fmt: skip
    repo = os.environ["GITHUB_REPOSITORY"]
    status = summary["status"]
    probes = "\n".join(f"- `{p['commit'][:10]}`: {p['verdict']}" for p in summary["probes"])
    span = f"`{summary['good'][:10]}..{summary['bad'][:10]}` ({len(summary['candidates'])} commits)"
    gate = f"Gate (`autan-bench compare`, culprit against its parent):\n\n```\n{summary.get('comparison', '').strip()}\n```"

    if status == "culprit":
        culprit = summary["culprit"]
        prs = gh("api", f"repos/{repo}/commits/{culprit}/pulls", "--jq", ".[].number").split()
        where = f"#{prs[0]}" if prs else f"`{culprit[:10]}`"
        title = f"perf-regression: {where} slows the tracked loops"
        body = (
            f"Bisection of main over {span} on the GB10 bench runner ({run_url}) names "
            f"`{culprit}` ({where}): it fails the gate and its parent `{summary['parent'][:10]}` "
            f"passes.\n\n{table(bisect, tracked, culprit, summary['parent'])}\n\n{gate}\n\n"
            f"Probes (each after a STREAM check of the machine):\n{probes}\n"
        )
    elif status == "spread":
        title = f"perf-regression: tracked loops slower over {span}, no single culprit"
        body = (
            f"Bisection of main over {span} ({run_url}): `{summary['first_failing'][:10]}` is "
            f"the first commit that fails the gate, but it is not a regression from its parent "
            f"`{summary['last_passing'][:10]}`. The shift is spread over several commits; no "
            f"commit is blamed.\n\n"
            f"{table(bisect, tracked, summary['first_failing'], summary['last_passing'])}\n\n"
            f"{gate}\n\nProbes:\n{probes}\n"
        )
    else:
        title = f"perf-regression: bisection over {span} stopped ({status})"
        body = (
            f"A regression of main was confirmed ({run_url}), but its bisection stopped at "
            f"`{summary.get('at', '?')[:10]}`: {status}. Nothing is interpreted from an "
            f"unhealthy machine or a failed probe.\n\nProbes:\n{probes}\n"
        )

    gh("label", "create", LABEL, "--force", "--color", "B60205",
       "--description", "A regression of main found by gpu-bench")  # fmt: skip
    issue = gh("issue", "create", "--title", title, "--label", LABEL, "--body-file", "-",
               body=body)  # fmt: skip
    if status == "culprit" and prs:
        gh("pr", "comment", prs[0], "--body-file", "-", body=f"{body}\nTracked in {issue}.\n")
    sys.stdout.write(f"{issue}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
