import importlib.machinery
import importlib.util
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "memprof" / "autan-memprof"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="reads Linux's /proc")

# Real memory, touched: 1 GiB in four steps, then exit 3. The process lets any other one
# trace it (prctl PR_SET_PTRACER_ANY), so that gdb can read its stack without root.
GROW = """
import ctypes, time
ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)
chunks = []
for _ in range(4):
    chunks.append(b"\\1" * (256 << 20))
    time.sleep(0.6)
raise SystemExit(3)
"""


def profile(tmp_path: Path, name: str, *options: str) -> tuple[int, Path]:
    script = tmp_path / "grow.py"
    script.write_text(GROW)
    out = tmp_path / name
    # Through a shell: the memory is in a descendant of the command.
    proc = subprocess.run([str(SCRIPT), *options, "--out", str(out), "--", "sh", "-c",
                           f"{sys.executable} {script}; exit $?"], check=False, timeout=60,
                          capture_output=True, text=True)  # fmt: skip
    return proc.returncode, out


def test_a_command_s_memory_is_sampled_with_its_descendants_and_its_status_kept(tmp_path):
    status, out = profile(tmp_path, "case.fp32", "--every", "0.1", "--stacks", "0")
    _, other = profile(tmp_path, "case.fp64", "--every", "0.1", "--stacks", "0")
    assert status == 3
    rows = [line.split("\t") for line in Path(f"{out}.tsv").read_text().splitlines()[1:]]
    assert float(rows[0][1]) < 0.5
    assert max(float(r[2]) for r in rows) >= 1.0
    # Two prefixes differing after a dot keep their own files.
    assert Path(f"{other}.tsv").exists()
    assert Path(f"{out}.tsv").read_text() != Path(f"{other}.tsv").read_text()
    meta = json.loads(Path(f"{out}.meta.json").read_text())
    assert meta["command"][0] == "sh"
    assert meta["started"]


@pytest.mark.skipif(not shutil.which("gdb"), reason="needs gdb")
def test_each_stage_line_carries_the_memory_after_its_stack_and_the_frames(tmp_path):
    status, out = profile(tmp_path, "staged", "--every", "0.1", "--stacks", "0.5", "--frames",
                          r"^_?Py")  # fmt: skip
    assert status == 3
    lines = Path(f"{out}.table").read_text().splitlines()
    assert len(lines) >= 3
    assert any("Py" in line.split("GiB", 1)[1] for line in lines)
    rss = [float(line.split("RSS")[1].split("GiB")[0]) for line in lines]
    nets = [float(line.split("net")[1].split()[0]) for line in lines]
    # Each net change is the step between consecutive lines' memory.
    assert nets[1:] == pytest.approx([b - a for a, b in itertools.pairwise(rss)], abs=0.011)
    assert max(rss) >= 0.7


def test_a_stage_line_reports_the_net_change_and_the_peak_in_between():
    loader = importlib.machinery.SourceFileLoader("autan_memprof", str(SCRIPT))
    spec = importlib.util.spec_from_loader("autan_memprof", loader)
    memprof = importlib.util.module_from_spec(spec)
    loader.exec_module(memprof)
    # 1 -> 10 -> 1 GiB between two stacks: no net change, a 10 GiB peak.
    line = memprof.table_line(12.0, 1.0, 1.0, 10.0, ["autan::kernels::cell_stencil"])
    assert "net   +0.00" in line
    assert "peak since   10.00" in line
    assert line.endswith("kernels::cell_stencil")
