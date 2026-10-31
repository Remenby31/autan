import hashlib
import os
import socket
from pathlib import Path
from typing import Any

from autan_bench.errors import BenchError, Code


def collect(binary: Path, revision: dict[str, Any]) -> dict[str, object]:
    """Solver revision the binary was built from, the binary's digest, host and image."""
    commit = revision.get("commit", "")
    if len(commit) != 40:
        raise BenchError(Code.NO_COMMIT, f"{binary} records no git commit ({commit!r})")
    return {
        "solver": {"commit": commit, "dirty": revision["dirty"]},
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "host": os.environ.get("AUTAN_HOST") or socket.gethostname(),
        "image": os.environ.get("AUTAN_IMAGE"),
        "image_revision": os.environ.get("AUTAN_IMAGE_REVISION"),
    }
