import copy
import sys

import jsonschema
import pytest
from autan_bench import gpu, result


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
    assert isinstance(a.pop("monotonic_s"), float)
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


@pytest.mark.usefixtures("proc")
def test_each_reading_carries_its_monotonic_time():
    first, second = gpu.host_activity(), gpu.host_activity()
    assert second["monotonic_s"] >= first["monotonic_s"]


def test_a_malformed_counter_is_left_out(proc):
    # A truncated PSI line, a short loadavg: left out, the reading still made.
    (proc / "pressure" / "memory").write_text("some avg10=1.0 avg60=1.0 avg300=1.0 total=\n")
    (proc / "loadavg").write_text("1.0 2.0\n")
    a = gpu.host_activity()
    assert "loadavg" not in a
    assert "memory" not in a.get("pressure", {})
    assert a["vmstat"]["compact_stall"] == 7


def test_lxcfs_views_are_left_out_and_flagged(proc):
    # Under LXCFS, /proc/meminfo and /proc/pressure describe a container, not the host.
    (proc / "self").mkdir()
    (proc / "self" / "mounts").write_text(
        "lxcfs /proc/meminfo fuse.lxcfs rw 0 0\nlxcfs /proc/pressure fuse.lxcfs rw 0 0\n"
    )
    a = gpu.host_activity()
    assert a["lxcfs"] == ["/proc/meminfo", "/proc/pressure"]
    assert "meminfo_bytes" not in a
    assert "pressure" not in a
    assert "vmstat" in a  # the host's own


@pytest.mark.parametrize(
    "spoil",
    [
        lambda a: a["pressure"]["memory"].__setitem__("some", {"av10": 1.0}),
        lambda a: a["pressure"].__setitem__("memory2", a["pressure"]["memory"]),
        lambda a: a["vmstat"].__setitem__("pgscan_direct", -1),
        lambda a: a["vmstat"].__setitem__("nr_free_pages", 3),
        lambda a: a["meminfo_bytes"].__setitem__("MemAvailable2", 1),
    ],
    ids=["psi field", "psi resource", "negative counter", "other counter", "meminfo key"],
)
@pytest.mark.usefixtures("proc")
def test_the_schema_refuses_misspelled_or_impossible_readings(spoil):
    reading = gpu.host_activity()
    schema = result.schema()["properties"]["conditions"]["properties"]["host_activity"]
    jsonschema.validate({"before": reading, "after": reading}, schema)
    bad = copy.deepcopy(reading)
    spoil(bad)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"before": bad, "after": reading}, schema)
