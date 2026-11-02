import importlib.machinery
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CHECK = Path(__file__).parents[1] / "ci" / "check-measured.py"
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash and a fake ninja")


def fake_build(
    tmp_path: Path, inputs: list[str], deps: list[str], state: str = "#deps 2, deps mtime 1 (VALID)"
) -> dict[str, str]:
    """A repository, a Ninja build tree inside it, and a ninja answering for that tree as the
    real one does: `-t inputs` relative to the tree or absolute, `-t deps` indented."""
    repo = tmp_path / "repo"
    build = repo / "build"
    build.mkdir(parents=True)
    (build / "build.ninja").write_text("")
    deps_out = f"obj/a.o: {state}\n" + "".join(f"    {d}\n" for d in deps)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ninja = bin_dir / "ninja"
    ninja.write_text(
        "#!/bin/sh\n"
        'case "$4" in\n'
        f"  inputs) cat <<'EOF'\n{chr(10).join([*inputs, 'obj/a.o'])}\nEOF\n  ;;\n"
        f"  deps) printf '%s' '{deps_out}' ;;\n"
        "esac\n"
    )
    ninja.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "REPO": str(repo)}


def check(env: dict[str, str], listed: list[str], *flags: str) -> subprocess.CompletedProcess:
    """The check against a measured-files.txt holding `listed` (its comment line included)."""
    files = Path(env["REPO"]).parent / "measured-files.txt"
    files.write_text("# Files the benchmarks are built from\n" + "".join(f"{f}\n" for f in listed))
    env = {**env, "AUTAN_MEASURED_FILES": str(files)}
    return subprocess.run([sys.executable, str(CHECK), *flags, "build:bench"], cwd=env["REPO"],
                          env=env, capture_output=True, text=True, check=False)  # fmt: skip


SOURCES = ["../codegen/src/a.cpp", "../io/src/b.cpp", "../kernels/euler.py"]
LISTED = ["codegen/src/a.cpp", "io/src/b.cpp", "kernels/euler.py"]


def test_a_list_of_exactly_the_graph_s_files_passes(tmp_path):
    # Sources relative to the tree, a header absolute; system headers and generated files in the
    # build tree are not the repository's.
    header = str(tmp_path / "repo" / "core" / "include" / "x.hpp")
    env = fake_build(tmp_path, SOURCES, [header, "/usr/include/stdio.h", "gen/revision.h"])
    proc = check(env, [*LISTED, "core/include/x.hpp"])
    assert proc.returncode == 0, proc.stdout
    assert "4 repository files in the graph, under codegen core io kernels" in proc.stdout


def test_a_header_missing_from_the_list_fails(tmp_path):
    # A measured source including runtime/'s header: a change there changes the benchmark, while
    # runtime/'s other files stay unmeasured.
    proc = check(fake_build(tmp_path, SOURCES, ["../runtime/include/run.hpp"]), LISTED)
    assert proc.returncode == 1
    assert (
        "runtime/include/run.hpp is built into a benchmark but is not in measured-files.txt"
        in proc.stdout
    )
    assert "rewrite it with: python3 bench/ci/check-measured.py --write build:bench" in proc.stdout


def test_a_listed_file_no_benchmark_is_built_from_fails(tmp_path):
    proc = check(fake_build(tmp_path, SOURCES, []), [*LISTED, "physics/src/old.cpp"])
    assert proc.returncode == 1
    assert (
        "physics/src/old.cpp is in measured-files.txt but no benchmark is built from it"
        in proc.stdout
    )


def test_write_lists_the_graph_and_the_check_then_passes(tmp_path):
    env = fake_build(tmp_path, [*SOURCES, "../io/CMakeLists.txt"], ["../core/include/x.hpp"])
    assert check(env, [], "--write").returncode == 0
    files = Path(env["REPO"]).parent / "measured-files.txt"
    lines = files.read_text().splitlines()
    assert lines[0].startswith("# ")  # how to rewrite it
    assert lines[1:] == sorted([*LISTED, "core/include/x.hpp"])  # build files left to the pattern
    proc = subprocess.run([sys.executable, str(CHECK), "build:bench"], cwd=env["REPO"],
                          env={**env, "AUTAN_MEASURED_FILES": str(files)}, capture_output=True,
                          text=True, check=False)  # fmt: skip
    assert proc.returncode == 0, proc.stdout


def test_a_tree_that_is_not_built_is_refused(tmp_path):
    env = fake_build(tmp_path, [], [])
    (Path(env["REPO"]) / "build" / "build.ninja").unlink()
    proc = check(env, LISTED)
    assert proc.returncode != 0
    assert "not a Ninja build tree" in proc.stderr


@pytest.mark.parametrize("state", ["deps not found", "#deps 2, deps mtime 1 (STALE)"])
def test_an_object_without_a_valid_dependency_record_fails(tmp_path, state):
    # Its headers would go unseen: the check refuses instead of passing.
    proc = check(fake_build(tmp_path, SOURCES, [], state=state), LISTED)
    assert proc.returncode != 0
    assert "obj/a.o" in proc.stderr


def test_a_quoted_input_is_read_as_its_path(tmp_path):
    # ninja shell-quotes a path with a space: it is still a repository file.
    proc = check(fake_build(tmp_path, [*SOURCES, "'../runtime/gen script.py'"], []), LISTED)
    assert proc.returncode == 1
    assert "runtime/gen script.py is built into a benchmark" in proc.stdout


def test_a_symlinked_source_counts_where_it_lies_too(tmp_path):
    # runtime/link.cpp -> ../codegen/real.cpp: retargeting the link changes the benchmark.
    env = fake_build(tmp_path, [*SOURCES, "../runtime/link.cpp"], [])
    repo = Path(env["REPO"])
    (repo / "codegen").mkdir()
    (repo / "codegen" / "real.cpp").write_text("")
    (repo / "runtime").mkdir()
    (repo / "runtime" / "link.cpp").symlink_to("../codegen/real.cpp")
    proc = check(env, [*LISTED, "codegen/real.cpp"])
    assert proc.returncode == 1
    assert "runtime/link.cpp is built into a benchmark" in proc.stdout


def test_build_files_are_left_to_the_pattern(tmp_path):
    # A .cmake input of the graph needs no line in the list: changes.sh's pattern measures it.
    proc = check(fake_build(tmp_path, [*SOURCES, "../kernels/config.cmake"], []), LISTED)
    assert proc.returncode == 0, proc.stdout


def load_check():
    loader = importlib.machinery.SourceFileLoader("check_measured", str(CHECK))
    spec = importlib.util.spec_from_loader("check_measured", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


# A compilation: never beside a measurement on a Spark, where it would need the GPU lock; the
# hosted "Bench harness" job and workstations run it.
ON_SPARK = (
    os.environ.get("RUNNER_ENVIRONMENT") == "self-hosted" or Path("/home/admin/autan-gpu").exists()
)


HOSTED_CI = os.environ.get("GITHUB_ACTIONS") == "true" and not ON_SPARK


# Hosted CI must run it (a missing tool then fails it): elsewhere, without the tools, it skips.
@pytest.mark.skipif(not HOSTED_CI and not (shutil.which("cmake") and shutil.which("ninja")
                    and shutil.which("cc")), reason="needs cmake, ninja and a C compiler")  # fmt: skip
@pytest.mark.skipif(ON_SPARK, reason="compiles: hosted CI and workstations only, not on a Spark")
def test_each_target_s_graph_is_its_own_sources_and_headers(tmp_path, monkeypatch):
    # A real build: two measured targets and one that is not; each target's graph holds its
    # own sources and the headers they include, the other target's never.
    repo = tmp_path / "repo"
    for path, text in {
        "codegen/a.c": '#include "h.h"\nint main(void) { return H; }\n',
        "core/include/h.h": "#define H 0\n",
        "io/b.c": "int main(void) { return 0; }\n",
        "runtime/c.c": "int main(void) { return 0; }\n",
        "CMakeLists.txt": "cmake_minimum_required(VERSION 3.20)\nproject(t C)\n"
        "add_executable(a codegen/a.c)\ntarget_include_directories(a PRIVATE core/include)\n"
        "add_executable(b io/b.c)\nadd_executable(c runtime/c.c)\n",
    }.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    build = repo / "build"
    subprocess.run(["cmake", "-S", repo, "-B", build, "-G", "Ninja"], check=True,
                   capture_output=True)  # fmt: skip
    subprocess.run(["cmake", "--build", build], check=True, capture_output=True)
    check_measured = load_check()
    monkeypatch.chdir(repo)
    a = check_measured.graph_files(Path("build"), "a", repo.resolve())
    b = check_measured.graph_files(Path("build"), "b", repo.resolve())
    assert a == {"codegen/a.c", "core/include/h.h"}
    assert b == {"io/b.c"}
