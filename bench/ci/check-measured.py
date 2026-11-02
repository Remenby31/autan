#!/usr/bin/env python3
"""Checks that changes.sh's measured directories are exactly those of the benchmarks' build graph.

    bench/ci/check-measured.py BUILD_DIR:TARGET...   (from the repository's root, once built)

gpu-bench runs only when a change touches a measured path (bench/ci/changes.sh). The source
directories among them must be every top-level directory that a measured binary is built from,
sources and the headers they included (ninja's graph and deps log), and no other: a missing one
lets a change to measured code go unmeasured, an extra one makes every change there wait hours
for a measurement of nothing it changed (#360). Exits 1, with the directories to add or remove.
"""

import re
import subprocess
import sys
from pathlib import Path

CHANGES = Path(__file__).with_name("changes.sh")


def measured() -> tuple[list[str], str]:
    """changes.sh's measured directories and its whole measured-paths pattern."""
    out = subprocess.run(["bash", str(CHANGES), "--measured"], capture_output=True, text=True,
                         check=True).stdout.splitlines()  # fmt: skip
    return out[0].split(), out[1]


def ninja(build: Path, *args: str) -> list[str]:
    return subprocess.run(["ninja", "-C", str(build), "-t", *args], capture_output=True,
                          text=True, check=True).stdout.splitlines()  # fmt: skip


def graph_files(build: Path, target: str, repo: Path) -> set[str]:
    """The repository files `target` is built from: every input of its graph and every header
    its objects included when they were compiled; generated files in the build tree excluded."""
    if not (build / "build.ninja").is_file():
        sys.exit(f"check-measured: {build} is not a Ninja build tree (CMAKE_GENERATOR=Ninja)")
    inputs = ninja(build, "inputs", target)
    objects = [i for i in inputs if i.endswith((".o", ".obj"))]
    if not objects:
        sys.exit(f"check-measured: {target} in {build} has no object: was it built?")
    deps = [line.strip() for line in ninja(build, "deps", *objects) if line.startswith("    ")]
    files = set()
    for name in [i for i in inputs if i not in objects] + deps:
        path = (build / name).resolve()
        if path.is_relative_to(repo) and not path.is_relative_to(build.resolve()):
            files.add(path.relative_to(repo).as_posix())
    return files


def verdict(files: set[str], dirs: list[str], pattern: str) -> list[str]:
    """The errors: graph files outside the measured paths, measured directories outside the
    graph."""
    errors = [f"{f} is built into a benchmark but is not a measured path: add its directory"
              for f in sorted(files) if not re.search(pattern, f)]  # fmt: skip
    errors += [f"{d}/ is measured but no benchmark is built from it: remove it"
               for d in dirs if not any(f.startswith(f"{d}/") for f in files)]  # fmt: skip
    return errors


def main(specs: list[str]) -> int:
    if not specs:
        sys.exit(__doc__)
    repo = Path.cwd().resolve()
    files: set[str] = set()
    for spec in specs:
        build, _, target = spec.partition(":")
        files |= graph_files(Path(build), target, repo)
    dirs, pattern = measured()
    errors = verdict(files, dirs, pattern)
    for error in errors:
        sys.stdout.write(f"::error file=bench/ci/changes.sh::{error}\n")
    tops = sorted({f.split("/", 1)[0] for f in files})
    sys.stdout.write(f"{len(files)} repository files in the graph, under {' '.join(tops)}; "
                     f"measured directories: {' '.join(dirs)}\n")  # fmt: skip
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
