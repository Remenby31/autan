#!/usr/bin/env python3
"""Checks that bench/ci/measured-files.txt is exactly the set of files the benchmarks are built from.

    bench/ci/check-measured.py [--write] BUILD_DIR:TARGET...   (from the repository's root, built)

gpu-bench runs only when a change touches a measured file (bench/ci/changes.sh): a line of
measured-files.txt, or a path its pattern names (image, method, build files, mesh generator). The
list must be every repository file a measured binary is built from, sources, the headers they
included (ninja's deps log) and the inputs of their generated code, and no other: a missing one
lets a change to measured code go unmeasured, an extra one makes a change there wait hours for a
measurement of nothing it changed (#360). Build files are left to the pattern. Exits 1 with the
files to add or remove; --write rewrites the list from the graph instead.
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

CHANGES = Path(__file__).with_name("changes.sh")
LIST = Path(
    os.environ.get("AUTAN_MEASURED_FILES") or Path(__file__).with_name("measured-files.txt")
)
HEADER = "# Files the benchmarks are built from: bench/ci/check-measured.py --write rewrites it.\n"


def listed(path: Path) -> set[str]:
    """measured-files.txt's paths, comments and blank lines left out."""
    lines = path.read_text().splitlines() if path.is_file() else []
    return {line for line in lines if line and not line.startswith("#")}


def ninja(build: Path, *args: str) -> list[str]:
    return subprocess.run(["ninja", "-C", str(build), "-t", *args], capture_output=True,
                          text=True, check=True).stdout.splitlines()  # fmt: skip


def unquote(name: str) -> str:
    """A path as `ninja -t inputs` prints it: shell-quoted when it holds a space or a quote."""
    return shlex.split(name)[0] if name[:1] in "'\"" or "\\" in name else name


def object_deps(build: Path, objects: list[str]) -> list[str]:
    """The headers each object included, from ninja's deps log; an object without a valid
    record (never built, or rebuilt since) fails, rather than passing with its headers unseen."""
    out = ninja(build, "deps", *objects)
    deps: list[str] = []
    seen = set()
    for line in out:
        if line.startswith("    "):
            deps.append(line.strip())
        elif line.strip():
            head, _, state = line.partition(": ")
            if "(VALID)" not in state:
                sys.exit(f"check-measured: {build}/{head}: {state or 'no dependency record'}")
            seen.add(head)
    if missing := sorted(set(objects) - seen):
        sys.exit(f"check-measured: no dependency record for {', '.join(missing)} in {build}")
    return deps


def graph_files(build: Path, target: str, repo: Path) -> set[str]:
    """The repository files `target` is built from: every input of its graph and every header
    its objects included when they were compiled; generated files in the build tree excluded.
    A symlink counts under both its own path and its target's."""
    if not (build / "build.ninja").is_file():
        sys.exit(f"check-measured: {build} is not a Ninja build tree (CMAKE_GENERATOR=Ninja)")
    inputs = [unquote(i) for i in ninja(build, "inputs", target)]
    objects = [i for i in inputs if i.endswith((".o", ".obj"))]
    if not objects:
        sys.exit(f"check-measured: {target} in {build} has no object: was it built?")
    tree = build.resolve()
    files = set()
    for name in [i for i in inputs if i not in objects] + object_deps(build, objects):
        lexical = Path(os.path.normpath(build.absolute() / name))
        for path in (lexical, lexical.resolve()):
            if path.is_relative_to(repo) and not path.is_relative_to(tree):
                files.add(path.relative_to(repo).as_posix())
    return files


BUILD_FILES = re.compile(r"(^|/)CMakeLists\.txt$|\.cmake$")


def verdict(files: set[str], measured: set[str]) -> list[str]:
    """The errors: graph files missing from the list, listed files outside the graph (build
    files left to changes.sh's pattern on both sides)."""
    code = {f for f in files if not BUILD_FILES.search(f)}
    errors = [f"{f} is built into a benchmark but is not in measured-files.txt: add it"
              for f in sorted(code - measured)]  # fmt: skip
    errors += [f"{f} is in measured-files.txt but no benchmark is built from it: remove it"
               for f in sorted(measured - code)]  # fmt: skip
    return errors


def main(argv: list[str]) -> int:
    write = "--write" in argv
    specs = [a for a in argv if a != "--write"]
    if not specs:
        sys.exit(__doc__)
    repo = Path.cwd().resolve()
    files: set[str] = set()
    for spec in specs:
        build, _, target = spec.partition(":")
        files |= graph_files(Path(build), target, repo)
    code = sorted(f for f in files if not BUILD_FILES.search(f))
    if write:
        LIST.write_text(HEADER + "".join(f"{f}\n" for f in code))
        sys.stdout.write(f"wrote {len(code)} files to {LIST}\n")
        return 0
    errors = verdict(files, listed(LIST))
    for error in errors:
        sys.stdout.write(f"::error file=bench/ci/measured-files.txt::{error}\n")
    if errors:
        sys.stdout.write(
            "rewrite it with: python3 bench/ci/check-measured.py --write " + " ".join(specs) + "\n"
        )
    tops = sorted({f.split("/", 1)[0] for f in code})
    sys.stdout.write(f"{len(code)} repository files in the graph, under {' '.join(tops)}\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
