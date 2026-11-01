import hashlib
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from autan_bench.errors import BenchError, Code


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _machine_model() -> str:
    """The machine's model as a neutral label, e.g. ``apple-m4-pro``; never its hostname."""
    if sys.platform == "darwin":
        try:
            brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            brand = ""
        if brand.strip():
            return _slug(brand)
    for source in (Path("/proc/device-tree/model"), Path("/sys/class/dmi/id/product_name")):
        try:
            model = source.read_text(errors="replace").strip("\x00\n ")
        except OSError:
            continue
        if model:
            return _slug(model)
    return _slug(f"{platform.system()}-{platform.machine()}")


# Files Docker and Podman create in every container.
CONTAINER_MARKERS = (Path("/.dockerenv"), Path("/run/.containerenv"))


def _in_container() -> bool:
    return any(marker.exists() for marker in CONTAINER_MARKERS)


def host_label() -> str:
    """Where the run happened, without personal data.

    ``$AUTAN_BENCH_HOST`` when set (the Spark runners set their machine name); inside a container,
    its short id, which is the container's hostname; elsewhere the machine model.
    """
    if label := os.environ.get("AUTAN_BENCH_HOST"):
        return label
    if _in_container():
        return socket.gethostname()
    return _machine_model()


def path_label(path: Path) -> str:
    """``path`` with the home directory written as ``~``, so no user name is recorded."""
    try:
        return "~/" + str(path.resolve().relative_to(Path.home().resolve()))
    except ValueError:
        return str(path)


def collect(binary: Path, revision: dict[str, Any]) -> dict[str, object]:
    """Solver revision the binary was built from, the binary's digest, host and image."""
    commit = revision.get("commit", "")
    if len(commit) != 40:
        raise BenchError(Code.NO_COMMIT, f"{binary} records no git commit ({commit!r})")
    return {
        "solver": {"commit": commit, "dirty": revision["dirty"]},
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "host": host_label(),
        "image": os.environ.get("AUTAN_IMAGE"),
        "image_revision": os.environ.get("AUTAN_IMAGE_REVISION"),
    }
