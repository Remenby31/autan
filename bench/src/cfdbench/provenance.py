import os
import socket
import subprocess
from pathlib import Path

from cfdbench.errors import BenchError


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BenchError(
            "NO_COMMIT", f"cannot read the solver commit from {repo}; run inside a git checkout"
        ) from exc


def collect(repo: Path) -> dict[str, object]:
    """Commit, dirty flag, host and container image of the current run."""
    return {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "dirty": bool(_git(repo, "status", "--porcelain", "--untracked-files=no")),
        "host": os.environ.get("CFD_HOST") or socket.gethostname(),
        "image": os.environ.get("CFD_IMAGE"),
        "image_revision": os.environ.get("CFD_IMAGE_REVISION"),
    }
