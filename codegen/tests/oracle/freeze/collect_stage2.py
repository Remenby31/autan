"""ADR 0003 stage 2: call heldout.collect on every attempt, with the run script's latches.

    python -m oracle.freeze.collect_stage2 MANIFEST MANIFEST_SHA256 LIBRARY_SHA256 RECORDS AUXILIARY

Every `*.json` file in RECORDS (instances, `<id>.attempt-<...>.json`) and AUXILIARY is passed;
an unreadable one is passed as its raw text, so that collect counts it as malformed (never
dropped). An attempt whose `<file>.timeout` latch exists is marked `timeout_latched`, which stage
2's collect makes decisive."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from oracle import heldout


def load(p: Path) -> Any:
    text = p.read_text()
    try:
        rec = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(rec, dict) and p.with_name(p.name + ".timeout").exists():
        rec["timeout_latched"] = True
    return rec


def main(argv: list[str]) -> int:
    manifest, manifest_sha256, library_sha256, records, auxiliary = argv
    attempts: dict[str, list[Any]] = defaultdict(list)
    files = sorted(Path(records).glob("*.json"))
    for p in files:
        attempts[p.name.split(".attempt-")[0]].append(load(p))
    aux_files = sorted(Path(auxiliary).glob("*.json"))
    result = heldout.collect(
        Path(manifest).read_bytes(), manifest_sha256, library_sha256, dict(attempts),
        [load(p) for p in aux_files],
    )  # fmt: skip
    print(json.dumps({
        "record_files": len(files),
        "latched": sorted(p.name for p in Path(records).glob("*.timeout")),
        "attempts_per_instance": {k: len(v) for k, v in sorted(attempts.items())},
        "auxiliary_files": len(aux_files),
        "result": result,
    }, indent=1))  # fmt: skip
    return 0 if not result["failures"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
