import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "ci" / "mesh-modules.py"


def package(tmp_path: Path) -> Path:
    """A repository with a generator package: its entry loads `used`, never `unused`."""
    src = tmp_path / "src" / "gen"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "used.py").write_text("VALUE = 1\n")
    (src / "unused.py").write_text("VALUE = 2\n")
    (src / "cli.py").write_text(
        "def main(argv):\n    from gen import used\n    return 0 if argv == ['ok'] else 3\n"
    )
    return tmp_path


def run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args[:-1], "--entry", "gen.cli:main", "--source-root", "src",
         "--", args[-1]],
        cwd=repo, env={"PYTHONPATH": str(repo / "src")}, capture_output=True, text=True, check=False,
    )  # fmt: skip


def test_write_lists_exactly_the_modules_the_generator_loads(tmp_path):
    repo = package(tmp_path)
    assert run(repo, "--write", "list.txt", "ok").returncode == 0
    lines = (repo / "list.txt").read_text().splitlines()
    assert lines[0].startswith("# ")
    assert lines[1:] == ["src/gen/__init__.py", "src/gen/cli.py", "src/gen/used.py"]
    assert run(repo, "--check", "list.txt", "ok").returncode == 0


def test_a_module_loaded_but_not_listed_or_listed_but_not_loaded_fails(tmp_path):
    repo = package(tmp_path)
    (repo / "list.txt").write_text("src/gen/__init__.py\nsrc/gen/cli.py\nsrc/gen/unused.py\n")
    proc = run(repo, "--check", "list.txt", "ok")
    assert proc.returncode == 1
    assert "src/gen/used.py is loaded by the mesh generator but not listed" in proc.stdout
    assert "src/gen/unused.py is listed but the mesh generator no longer loads it" in proc.stdout


def test_a_failing_generator_fails_before_any_check(tmp_path):
    repo = package(tmp_path)
    (repo / "list.txt").write_text("")
    assert run(repo, "--check", "list.txt", "bad").returncode == 3


KEY = SCRIPT.with_name("mesh-key.sh")


@pytest.mark.skipif(shutil.which("sha256sum") is None, reason="needs sha256sum (coreutils)")
def test_the_key_follows_the_listed_modules_not_the_other_generators(tmp_path):
    root = SCRIPT.parents[2]
    listed = [line for line in (root / "bench/ci/measured-mesh-files.txt").read_text().splitlines()
              if line and not line.startswith("#")]  # fmt: skip
    files = [*listed, "bench/ci/measured-mesh-files.txt", "bench/ci/mesh-modules.py",
             "bench/ci/mesh-key.sh", "vv/pyproject.toml", "vv/uv.lock", "vv/.python-version",
             "vv/src/autan_vv/meshes/naca0012.py"]  # fmt: skip
    for f in files:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / f, tmp_path / f)

    def key() -> str:
        return subprocess.run(["bash", "bench/ci/mesh-key.sh"], cwd=tmp_path, check=True,
                              capture_output=True, text=True).stdout  # fmt: skip

    first = key()
    assert first.startswith("key=mesh-hex-n128-")
    (tmp_path / "vv/src/autan_vv/meshes/naca0012.py").write_text("# another mesh's generator\n")
    assert key() == first
    with (tmp_path / listed[-1]).open("a") as f:
        f.write("# changed\n")
    assert key() != first
