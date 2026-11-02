import importlib.machinery
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "memprof" / "autan-memprof"


def load():
    loader = importlib.machinery.SourceFileLoader("autan_memprof", str(SCRIPT))
    spec = importlib.util.spec_from_loader("autan_memprof", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.mark.skipif(sys.platform != "linux", reason="reads Linux's /proc")
def test_a_command_s_memory_is_sampled_as_it_grows_and_its_status_kept(tmp_path):
    # Real memory, touched: 1 GiB in four steps, then the command exits 3.
    grow = ("import time\nchunks = []\nfor _ in range(4):\n"
            "    chunks.append(b'\\1' * (256 << 20)); time.sleep(0.4)\nraise SystemExit(3)\n")  # fmt: skip
    out = tmp_path / "prof"
    proc = subprocess.run([str(SCRIPT), "--every", "0.1", "--stacks", "0", "--out", str(out),
                           "--", sys.executable, "-c", grow], check=False, timeout=60)  # fmt: skip
    assert proc.returncode == 3
    rows = [line.split("\t") for line in out.with_suffix(".tsv").read_text().splitlines()[1:]]
    rss = [float(r[1]) for r in rows]
    peak = [float(r[2]) for r in rows]
    assert rss[0] < 0.5
    assert max(peak) >= 1.0


def test_each_stack_sample_is_tied_to_the_memory_step_it_saw():
    memprof = load()
    samples = [
        (0.0, 0.1, 0.1, None, 50.0),
        (5.0, 2.1, 2.1, None, 48.0),
        (10.0, 6.1, 6.2, None, 44.0),
    ]
    stacks = [(0.1, []), (5.2, ["autan::io::read_cgns"]),
              (9.9, ["autan::kernels::cell_stencil", "autan::physics::Euler::Impl::Impl"])]  # fmt: skip
    lines = memprof.table(samples, stacks)
    assert len(lines) == 3
    assert "step   +2.00" in lines[1]
    assert lines[1].endswith("io::read_cgns")
    assert "step   +4.00" in lines[2]
    assert lines[2].endswith("kernels::cell_stencil < physics::Euler::Impl::Impl")
