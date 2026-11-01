import os
import stat
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Self

NOFOLLOW = os.O_NOFOLLOW
DIRECTORY = os.O_DIRECTORY
# Search-only access where the platform has it, so a directory without read permission still
# opens; O_RDONLY elsewhere.
SEARCH = getattr(os, "O_PATH", getattr(os, "O_SEARCH", os.O_RDONLY))


def is_link(path: Path) -> bool:
    """A symbolic link, or a Windows junction (which `is_symlink` does not report)."""
    return path.is_symlink() or path.is_junction()


class Folder:
    """A directory opened once. Files are read relative to its descriptor, opening every path
    component without following links, so replacing the directory or anything inside it by a
    link after the check cannot redirect a read. POSIX only (`dir_fd`, `O_NOFOLLOW`)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd = os.open(path, SEARCH | DIRECTORY | NOFOLLOW)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        os.close(self.fd)

    def open(self, relative: str) -> int:
        """Descriptor of the regular file at `relative` (POSIX separators); OSError otherwise."""
        *folders, name = PurePosixPath(relative).parts
        current = self.fd
        try:
            for part in folders:
                if part in ("..", "/"):
                    raise FileNotFoundError(relative)
                parent = current
                current = os.open(part, SEARCH | DIRECTORY | NOFOLLOW, dir_fd=parent)
                if parent != self.fd:
                    os.close(parent)
            # Non-blocking so that a FIFO is rejected below instead of hanging the open.
            descriptor = os.open(name, os.O_RDONLY | NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
        finally:
            if current != self.fd:
                os.close(current)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise IsADirectoryError(relative)
        return descriptor

    def read(self, relative: str, offset: int = 0) -> bytes:
        with os.fdopen(self.open(relative), "rb") as file:
            file.seek(offset)
            return file.read()

    def stat(self, name: str) -> os.stat_result | None:
        """Status of the entry `name` of this directory, without following a link; None when
        there is none or `name` is not a single entry name."""
        if name in ("", ".", "..") or "/" in name or "\0" in name:
            return None
        try:
            return os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except OSError:
            return None
