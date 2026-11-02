import sys

import pytest
from autan_bench import gpu


@pytest.fixture
def proc(tmp_path, monkeypatch):
    """A /proc as Linux writes it, the counters #400 records and others it must leave out."""
    monkeypatch.setattr(sys, "platform", "linux")
    (tmp_path / "loadavg").write_text("4.36 4.07 3.90 2/1234 5678\n")
    (tmp_path / "stat").write_text("cpu  1 2 3 4\nprocs_running 3\nprocs_blocked 1\nctxt 99\n")
    (tmp_path / "pressure").mkdir()
    (tmp_path / "pressure" / "memory").write_text(
        "some avg10=12.50 avg60=3.00 avg300=1.00 total=123456\n"
        "full avg10=1.25 avg60=0.30 avg300=0.10 total=4567\n"
    )
    (tmp_path / "vmstat").write_text(
        "nr_free_pages 100\npgscan_kswapd 10\npgsteal_direct 5\ncompact_stall 7\n"
        "pgmigrate_success 3\nthp_fault_alloc 2\npgfault 1000\npgmajfault 4\nnr_dirty 8\n"
    )
    (tmp_path / "meminfo").write_text(
        "MemTotal: 100 kB\nMemFree: 10 kB\nMemAvailable: 50 kB\nCached: 20 kB\nDirty: 1 kB\n"
        "Writeback: 0 kB\nAnonPages: 30 kB\nShmem: 5 kB\n"
    )
    monkeypatch.setattr(gpu, "PROC", tmp_path)
    return tmp_path


@pytest.mark.usefixtures("proc")
def test_the_host_s_own_counters_are_read():
    a = gpu.host_activity()
    assert a["loadavg"] == [4.36, 4.07, 3.90]
    assert (a["procs_running"], a["procs_blocked"]) == (3, 1)
    assert a["pressure"] == {"memory": {"some": {"avg10": 12.5, "avg60": 3.0, "avg300": 1.0,
                                                 "total": 123456.0},
                                        "full": {"avg10": 1.25, "avg60": 0.3, "avg300": 0.1,
                                                 "total": 4567.0}}}  # fmt: skip
    assert a["vmstat"] == {"pgscan_kswapd": 10, "pgsteal_direct": 5, "compact_stall": 7,
                           "pgmigrate_success": 3, "thp_fault_alloc": 2, "pgfault": 1000,
                           "pgmajfault": 4}  # fmt: skip
    assert a["meminfo_bytes"] == {k: v * 1024 for k, v in (
        ("MemFree", 10), ("MemAvailable", 50), ("Cached", 20), ("Dirty", 1), ("Writeback", 0),
        ("AnonPages", 30))}  # fmt: skip


def test_an_unreadable_counter_is_left_out_not_fatal(proc):
    (proc / "vmstat").unlink()
    (proc / "loadavg").write_text("garbage\n")
    a = gpu.host_activity()
    assert "vmstat" not in a
    assert "loadavg" not in a
    assert "pressure" in a  # the rest is still read


def test_off_linux_there_is_no_reading(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert gpu.host_activity() is None
