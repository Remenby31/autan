import socket
from pathlib import Path

import pytest
from autan_bench import provenance

PERSONAL = "MacBook-Pro-Someone"


@pytest.fixture
def personal_host(monkeypatch, tmp_path):
    # The hostname and the container marker files belong to the operating system: faked there.
    monkeypatch.setattr(socket, "gethostname", lambda: PERSONAL)
    monkeypatch.delenv("AUTAN_BENCH_HOST", raising=False)
    monkeypatch.setattr(provenance, "CONTAINER_MARKERS", (tmp_path / "no-dockerenv",))
    return tmp_path


def test_the_host_is_never_recorded_by_its_hostname(personal_host, monkeypatch):
    label = provenance.host_label()
    assert label != PERSONAL
    assert label == label.lower()
    assert all(c.isalnum() or c == "-" for c in label)

    monkeypatch.setenv("AUTAN_BENCH_HOST", "spark-5359")
    assert provenance.host_label() == "spark-5359"


def test_inside_a_container_the_host_is_its_short_id(personal_host, monkeypatch):
    marker = personal_host / "dockerenv"
    marker.touch()
    monkeypatch.setattr(provenance, "CONTAINER_MARKERS", (marker,))
    monkeypatch.setattr(socket, "gethostname", lambda: "a91236083dec")
    assert provenance.host_label() == "a91236083dec"


def test_paths_under_home_are_written_with_a_tilde(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert provenance.path_label(tmp_path / ".autan-gpu.lock") == "~/.autan-gpu.lock"
    assert provenance.path_label(Path("/home/admin/autan-gpu/gpu.lock")) == "/home/admin/autan-gpu/gpu.lock"
