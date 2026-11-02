import json
import subprocess
import sys
from pathlib import Path

HEALTH = Path(__file__).parents[1] / "ci" / "health.py"
BASELINE = {
    "stream.copy": 234.88,
    "stream.scale": 236.24,
    "stream.add": 235.53,
    "stream.triad": 235.66,
}


def result(path: Path, medians: dict[str, float]) -> Path:
    doc = {"benchmarks": [{"name": n, "stats": {"median": m}} for n, m in medians.items()]}
    path.write_text(json.dumps(doc))
    return path


def health(tmp_path: Path, probe: dict[str, float]) -> subprocess.CompletedProcess:
    base = result(tmp_path / "baseline.json", BASELINE)
    got = result(tmp_path / "probe.json", probe)
    return subprocess.run([sys.executable, str(HEALTH), str(got), str(base)],
                          capture_output=True, text=True, check=False)  # fmt: skip


def test_a_probe_within_3_percent_of_the_baseline_is_healthy(tmp_path):
    proc = health(tmp_path, {n: m * 0.971 for n, m in BASELINE.items()})
    assert proc.returncode == 0, proc.stdout
    assert "machine healthy" in proc.stdout


def test_the_2026_11_02_probe_is_degraded(tmp_path):
    # spark-5359 at 11:48Z (#284): main's own STREAM, 219-224 GB/s against 235.
    proc = health(tmp_path, {"stream.copy": 222.4, "stream.scale": 221.8, "stream.add": 224.2,
                             "stream.triad": 221.2})  # fmt: skip
    assert proc.returncode == 3
    assert "MACHINE DEGRADED" in proc.stdout
    assert "stream.copy" in proc.stdout


def test_one_low_kernel_is_enough(tmp_path):
    probe = dict(BASELINE, **{"stream.triad": BASELINE["stream.triad"] * 0.969})
    proc = health(tmp_path, probe)
    assert proc.returncode == 3
    assert "stream.triad" in proc.stdout.splitlines()[-1]


def test_a_probe_missing_a_kernel_cannot_be_judged(tmp_path):
    probe = {n: m for n, m in BASELINE.items() if n != "stream.add"}
    proc = health(tmp_path, probe)
    assert proc.returncode == 2
    assert "lacks stream.add" in proc.stdout
