#!/usr/bin/env python3
"""Runs the benchmark mesh's generator and checks the repository modules it loaded against a list.

    python bench/ci/mesh-modules.py (--check LIST | --write LIST) -- AUTAN_VV_ARGS...

gpu-bench's mesh job generates its mesh with this (`autan-vv meshes hex --levels 128`), in the
generator's own environment (uv run --project vv). The mesh depends on the repository files the
command loads, nothing else of vv/: bench/ci/measured-mesh-files.txt lists them, changes.sh
measures a change to one of them, and the mesh's cache key hashes them (bench/ci/mesh-key.sh). A
module the command loads but the list lacks, or a listed one it no longer loads, fails --check
(exit 1), so that the list follows the code; --write rewrites it. Third-party modules are left to
vv/uv.lock and vv/pyproject.toml, which changes.sh measures as files.
"""

import sys

# bench/ci holds bisect.py: as the script's directory it would shadow the standard library's.
if sys.path and sys.path[0] and __file__.startswith(sys.path[0]):
    del sys.path[0]

import argparse
import importlib
from pathlib import Path

HEADER = (
    "# Files the benchmark mesh's generator loads: bench/ci/mesh-modules.py --write rewrites it.\n"
)


def loaded(source_root: Path) -> list[str]:
    """Repository-relative paths of the loaded modules under source_root."""
    repo = Path.cwd().resolve()
    root = source_root.resolve()
    files = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).resolve().is_relative_to(root):
            files.add(Path(path).resolve().relative_to(repo).as_posix())
    return sorted(files)


def listed(path: Path) -> list[str]:
    lines = path.read_text().splitlines() if path.is_file() else []
    return sorted(line for line in lines if line and not line.startswith("#"))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ours, theirs = _split(argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", type=Path)
    mode.add_argument("--write", type=Path)
    parser.add_argument("--entry", default="autan_vv.cli:main", help="MODULE:FUNCTION to run")
    parser.add_argument("--source-root", type=Path, default=Path("vv/src"))
    args = parser.parse_args(ours)
    module, _, function = args.entry.partition(":")
    status = getattr(importlib.import_module(module), function)(theirs)
    if status not in (0, None):
        return int(status)
    files = loaded(args.source_root)
    if args.write:
        args.write.write_text(HEADER + "".join(f"{f}\n" for f in files))
        sys.stdout.write(f"wrote {len(files)} files to {args.write}\n")
        return 0
    want = listed(args.check)
    errors = [
        f"{f} is loaded by the mesh generator but not listed: add it"
        for f in files
        if f not in want
    ]
    errors += [f"{f} is listed but the mesh generator no longer loads it: remove it"
               for f in want if f not in files]  # fmt: skip
    for error in errors:
        sys.stdout.write(f"::error file={args.check}::{error}\n")
    if errors:
        sys.stdout.write(f"rewrite it with: python bench/ci/mesh-modules.py --write {args.check} -- "
                         + " ".join(theirs) + "\n")  # fmt: skip
    sys.stdout.write(f"{len(files)} repository modules loaded by the mesh generator\n")
    return 1 if errors else 0


def _split(argv: list[str]) -> tuple[list[str], list[str]]:
    """This script's arguments, then the generator's, after `--`."""
    if "--" not in argv:
        return argv, []
    i = argv.index("--")
    return argv[:i], argv[i + 1 :]


if __name__ == "__main__":
    sys.exit(main())
