"""Hatch build hook: record the git commit the package is built from.

In a git checkout the commit is read from git. A source distribution has no git
metadata; it carries the file written when it was built, which is used as is.
"""

import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

BUILD_INFO = "py/autan/_build_info.py"


def _git(root: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", root, *args], check=True, capture_output=True, text=True
    ).stdout.strip()


class BuildInfoHook(BuildHookInterface):  # type: ignore[type-arg]
    PLUGIN_NAME = "custom"

    def initialize(self, _version: str, build_data: dict[str, Any]) -> None:
        target = Path(self.root, BUILD_INFO)
        if Path(self.root, ".git").exists():
            commit = _git(self.root, "rev-parse", "HEAD")
            dirty = bool(_git(self.root, "status", "--porcelain", "--untracked-files=no"))
            target.write_text(f'COMMIT = "{commit}"\nDIRTY = {dirty}\n')
        elif not target.exists():
            message = f"{BUILD_INFO} is missing and {self.root} is not a git checkout"
            raise RuntimeError(message)
        build_data["artifacts"].append(BUILD_INFO)
