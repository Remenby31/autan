#!/usr/bin/env python3
"""Fails unless the vendored libraries are exactly the files third_party/SHA256SUMS lists, each with
the SHA-256 listed there: upstream's bytes, never edited here (third_party/README.md). A file
under a vendored directory that the list omits fails too, as does an empty list."""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "third_party"
VENDORED = ("nlohmann_json", "picosha2")


def main() -> int:
    listed: dict[str, str] = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        listed[name] = digest
    present = {
        p.relative_to(ROOT).as_posix()
        for d in VENDORED
        for p in (ROOT / d).rglob("*")
        if p.is_file()
    }
    errors = [] if listed else ["SHA256SUMS lists no file"]
    for name, digest in sorted(listed.items()):
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f"third_party/{name}: missing, or not the bytes SHA256SUMS lists")
    errors += [
        f"third_party/{name}: not listed in SHA256SUMS" for name in sorted(present - set(listed))
    ]
    for e in errors:
        sys.stderr.write(e + "\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
