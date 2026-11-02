import os
import subprocess
import sys
from pathlib import Path

import pytest

CHECK = Path(__file__).parents[1] / "ci" / "check-measured.py"
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash and a fake ninja")


def fake_build(tmp_path: Path, inputs: list[str], deps: list[str]) -> dict[str, str]:
    """A repository, a Ninja build tree inside it, and a ninja answering for that tree as the
    real one does: `-t inputs` relative to the tree or absolute, `-t deps` indented."""
    repo = tmp_path / "repo"
    build = repo / "build"
    build.mkdir(parents=True)
    (build / "build.ninja").write_text("")
    deps_out = "obj/a.o: #deps 2, deps mtime 1 (VALID)\n" + "".join(f"    {d}\n" for d in deps)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ninja = bin_dir / "ninja"
    ninja.write_text(
        "#!/bin/sh\n"
        'case "$4" in\n'
        f"  inputs) printf '%s\\n' {' '.join([*inputs, 'obj/a.o'])} ;;\n"
        f"  deps) printf '%s' '{deps_out}' ;;\n"
        "esac\n"
    )
    ninja.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "REPO": str(repo)}


def check(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CHECK), "build:bench"], cwd=env["REPO"], env=env,
                          capture_output=True, text=True, check=False)  # fmt: skip


MEASURED = ("codegen", "core", "io", "kernels", "partition")


def test_a_graph_of_exactly_the_measured_directories_passes(tmp_path):
    # Sources relative to the tree, a header absolute; system headers and generated files in the
    # build tree are not the repository's.
    sources = [f"../{d}/src/x.cpp" for d in MEASURED]
    header = str(tmp_path / "repo" / "core" / "include" / "x.hpp")
    env = fake_build(tmp_path, sources, [header, "/usr/include/stdio.h", "gen/revision.h"])
    proc = check(env)
    assert proc.returncode == 0, proc.stdout
    assert "6 repository files in the graph, under codegen core io kernels partition" in proc.stdout


def test_a_header_from_an_unmeasured_directory_fails(tmp_path):
    # A measured source including runtime/'s header: a change there changes the benchmark.
    sources = [f"../{d}/src/x.cpp" for d in MEASURED]
    env = fake_build(tmp_path, sources, ["../runtime/include/run.hpp"])
    proc = check(env)
    assert proc.returncode == 1
    assert (
        "runtime/include/run.hpp is built into a benchmark but is not a measured path"
        in proc.stdout
    )


def test_a_measured_directory_no_benchmark_is_built_from_fails(tmp_path):
    env = fake_build(tmp_path, ["../codegen/a.cpp", "../core/b.cpp", "../io/c.cpp"], [])
    proc = check(env)
    assert proc.returncode == 1
    assert "kernels/ is measured but no benchmark is built from it" in proc.stdout
    assert "partition/ is measured but no benchmark is built from it" in proc.stdout


def test_a_tree_that_is_not_built_is_refused(tmp_path):
    env = fake_build(tmp_path, [], [])
    (Path(env["REPO"]) / "build" / "build.ninja").unlink()
    proc = check(env)
    assert proc.returncode != 0
    assert "not a Ninja build tree" in proc.stderr
