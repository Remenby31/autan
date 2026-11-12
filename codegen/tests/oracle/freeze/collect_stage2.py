"""ADR 0003 stage 2: call heldout.collect on every attempt, with what the watchdog says of each.

    python -m oracle.freeze.collect_stage2 MANIFEST MANIFEST_SHA256 LIBRARY_SHA256 RECORDS AUXILIARY \\
        --queue QUEUE_DIR --job-prefix PREFIX

It refuses to collect while any job whose ID starts with PREFIX is still pending or running in
QUEUE_DIR, or when that directory lacks `pending/` or `running/`: a record read before its watchdog
had finished could be accepted before the watchdog says how it ended.

Every `*.json` file in RECORDS (instances, `<id>.attempt-<...>.json`) and AUXILIARY is passed; an
unreadable one is passed as its raw text, so that collect counts it as malformed (never dropped).
Each attempt's `<file>.watchdog` marker must say `"exited"` with status 0; otherwise (missing,
unreadable, `"running"`, `"timeout"`, another status) the attempt gets a `watchdog_issue`, which
stage 2's collect makes decisive. A marker without its record becomes an attempt of its own,
invalid."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from oracle import heldout

MARKER = ".watchdog"


def watchdog_issue(marker: Path) -> str | None:
    """What is wrong with an attempt according to its watchdog marker (None: exited 0)."""
    try:
        doc = json.loads(marker.read_text())
    except (OSError, ValueError):
        return f"no readable marker {marker.name}"
    if not isinstance(doc, dict) or doc.get("status") != "exited":
        return f"marker says {doc.get('status') if isinstance(doc, dict) else doc!r}"
    if doc.get("exit") != 0:
        return f"the command exited {doc.get('exit')!r}"
    elapsed, bound = doc.get("elapsed_s"), doc.get("bound_s")
    if not (
        isinstance(elapsed, (int, float))
        and isinstance(bound, (int, float))
        and 0 <= elapsed <= bound
    ):
        return f"the marker's elapsed time {elapsed!r} is not within its bound {bound!r}"
    return None


def load(p: Path) -> Any:
    text = p.read_text()
    try:
        rec = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(rec, dict) and (issue := watchdog_issue(p.with_name(p.name + MARKER))):
        rec["watchdog_issue"] = issue
    return rec


def orphan(marker: Path, sid: str | None) -> dict[str, Any]:
    return {
        "schema": heldout.SCHEMA if sid else heldout.AUX_SCHEMA,
        "id": sid,
        "stage": "marker without a record",
        "watchdog_issue": f"{marker.name} has no record",
        "invalid": [f"invalid: the marker {marker.name} has no record"],
    }


def jobs_in_flight(queue: Path, prefix: str) -> list[str]:
    for state in ("pending", "running"):
        if not (queue / state).is_dir():
            msg = f"no queue directory {queue / state}: collection refused"
            raise SystemExit(msg)
    return sorted(
        p.name
        for state in ("pending", "running")
        for p in (queue / state).glob("*.job")
        if p.name.split("-", 1)[-1].startswith(prefix)
    )


def _orphans(directory: Path, with_id: bool) -> list[tuple[str | None, dict[str, Any]]]:
    out = []
    for marker in sorted(directory.glob(f"*.json{MARKER}")):
        if not marker.with_name(marker.name.removesuffix(MARKER)).exists():
            sid = marker.name.split(".attempt-")[0] if with_id else None
            out.append((sid, orphan(marker, sid)))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    for name in ("manifest", "manifest_sha256", "library_sha256", "records", "auxiliary"):
        ap.add_argument(name)
    ap.add_argument("--queue", required=True, type=Path)
    ap.add_argument("--job-prefix", required=True)
    args = ap.parse_args(argv)
    if flying := jobs_in_flight(args.queue, args.job_prefix):
        print(f"not collected: jobs still pending or running: {flying}")
        return 3
    records, aux_dir = Path(args.records), Path(args.auxiliary)
    attempts: dict[str, list[Any]] = defaultdict(list)
    for p in sorted(records.glob("*.json")):
        attempts[p.name.split(".attempt-")[0]].append(load(p))
    for sid, rec in _orphans(records, with_id=True):
        attempts[str(sid)].append(rec)
    auxiliary = [load(p) for p in sorted(aux_dir.glob("*.json"))]
    auxiliary += [rec for _, rec in _orphans(aux_dir, with_id=False)]
    result = heldout.collect(
        Path(args.manifest).read_bytes(),
        args.manifest_sha256,
        args.library_sha256,
        dict(attempts),
        auxiliary,
    )
    issues = {
        k: [r["watchdog_issue"] for r in v if isinstance(r, dict) and r.get("watchdog_issue")]
        for k, v in sorted(attempts.items())
    }
    print(json.dumps({
        "watchdog_issues": {k: v for k, v in issues.items() if v},
        "attempts_per_instance": {k: len(v) for k, v in sorted(attempts.items())},
        "auxiliary_attempts": len(auxiliary),
        "result": result,
    }, indent=1))  # fmt: skip
    return 0 if not result["failures"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
