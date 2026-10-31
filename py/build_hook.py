"""Hatch build hook: record the git commit the package is built from."""

import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _git(root: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", root, *args], check=True, capture_output=True, text=True
    ).stdout.strip()


class BuildInfoHook(BuildHookInterface):  # type: ignore[type-arg]
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        commit = _git(self.root, "rev-parse", "HEAD")
        dirty = bool(_git(self.root, "status", "--porcelain", "--untracked-files=no"))
        target = Path(self.root, "py", "cfd", "_build_info.py")
        target.write_text(f'COMMIT = "{commit}"\nDIRTY = {dirty}\n')
        build_data["artifacts"].append("py/cfd/_build_info.py")
