import subprocess
import sys
from pathlib import Path

GUARD = Path(__file__).parents[1] / "ci" / "require-ran.py"


def report(tmp_path: Path, body: str) -> Path:
    """A JUnit report written by pytest itself for a small test file."""
    tests = tmp_path / "test_sample.py"
    tests.write_text("import pytest\n" + body)
    out = tmp_path / "report.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tests),
                    f"--junitxml={out}"], capture_output=True, check=False, cwd=tmp_path)  # fmt: skip
    return out


def guard(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(GUARD), str(path)], capture_output=True,
                          text=True, check=False)  # fmt: skip


def test_a_report_where_every_test_was_skipped_fails_with_the_reasons(tmp_path):
    out = report(tmp_path, "def test_a():\n    pytest.skip('the GPU is used by another tenant')\n"
                           "def test_b():\n    pytest.skip('no binary')\n")  # fmt: skip
    proc = guard(out)
    assert proc.returncode == 1
    assert proc.stdout.startswith("::error::")
    assert "another tenant" in proc.stdout
    assert "no binary" in proc.stdout


def test_a_report_where_one_test_ran_passes(tmp_path):
    out = report(tmp_path, "def test_a():\n    pytest.skip('busy')\ndef test_b():\n    pass\n")
    assert guard(out).returncode == 0
