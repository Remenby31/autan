"""ADR 0003 stage 2: call heldout.collect on every attempt, with the watchdog's latches.

    python -m oracle.freeze.collect_stage2 MANIFEST MANIFEST_SHA256 LIBRARY_SHA256 RECORDS AUXILIARY \\
        --queue QUEUE_DIR --job-prefix PREFIX

It refuses to collect while any job whose ID starts with PREFIX is still pending or running in
QUEUE_DIR: a record read before its watchdog had finished could miss a latch written after.

Every `*.json` file in RECORDS (instances, `<id>.attempt-<...>.json`) and AUXILIARY is passed; an
unreadable one is passed as its raw text, so that collect counts it as malformed (never dropped).
An attempt whose `<file>.timeout` latch exists is marked `timeout_latched`, which stage 2's collect
makes decisive. A latch without its record becomes an attempt of its own, latched and invalid."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from oracle import heldout

LATCH = ".timeout"


def load(p: Path) -> Any:
    text = p.read_text()
    try:
        rec = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(rec, dict) and p.with_name(p.name + LATCH).exists():
        rec["timeout_latched"] = True
    return rec


def orphan(latch: Path, sid: str | None) -> dict[str, Any]:
    return {"schema": heldout.SCHEMA if sid else heldout.AUX_SCHEMA, "id": sid,
            "stage": "latched without a record", "timeout_latched": True,
            "invalid": [f"invalid: the latch {latch.name} has no record"]}  # fmt: skip


def jobs_in_flight(queue: Path, prefix: str) -> list[str]:
    return sorted(
        p.name
        for state in ("pending", "running")
        for p in (queue / state).glob("*.job")
        if p.name.split("-", 1)[-1].startswith(prefix)
    )


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
    attempts: dict[str, list[Any]] = defaultdict(list)
    records = Path(args.records)
    for p in sorted(records.glob("*.json")):
        attempts[p.name.split(".attempt-")[0]].append(load(p))
    for latch in sorted(records.glob(f"*.json{LATCH}")):
        if not latch.with_name(latch.name.removesuffix(LATCH)).exists():
            sid = latch.name.split(".attempt-")[0]
            attempts[sid].append(orphan(latch, sid))
    aux_dir = Path(args.auxiliary)
    auxiliary = [load(p) for p in sorted(aux_dir.glob("*.json"))]
    auxiliary += [
        orphan(latch, None)
        for latch in sorted(aux_dir.glob(f"*.json{LATCH}"))
        if not latch.with_name(latch.name.removesuffix(LATCH)).exists()
    ]
    result = heldout.collect(
        Path(args.manifest).read_bytes(), args.manifest_sha256, args.library_sha256,
        dict(attempts), auxiliary,
    )  # fmt: skip
    print(json.dumps({
        "latched": sorted(p.name for p in records.glob(f"*{LATCH}")),
        "attempts_per_instance": {k: len(v) for k, v in sorted(attempts.items())},
        "auxiliary_attempts": len(auxiliary),
        "result": result,
    }, indent=1))  # fmt: skip
    return 0 if not result["failures"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
