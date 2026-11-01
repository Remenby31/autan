import os
from pathlib import Path

NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def is_link(path: Path) -> bool:
    """A symbolic link, or a Windows junction (which `is_symlink` does not report)."""
    return path.is_symlink() or path.is_junction()


def real_file(path: Path, root: Path) -> bool:
    """A regular file, not a link, whose resolved location lies under `root` (resolved)."""
    return path.is_file() and not is_link(path) and path.resolve().is_relative_to(root)


def read_bytes(path: Path, offset: int = 0) -> bytes:
    """The bytes of `path` from `offset`, refusing to open it through a symbolic link."""
    descriptor = os.open(path, os.O_RDONLY | NOFOLLOW)
    with os.fdopen(descriptor, "rb") as file:
        file.seek(offset)
        return file.read()


def read_text(path: Path) -> str:
    return read_bytes(path).decode("utf-8")
