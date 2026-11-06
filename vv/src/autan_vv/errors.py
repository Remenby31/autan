"""Coded errors raised by the V&V tools."""

from typing import Any


class VVError(Exception):
    """An error with a stable code `V####` and, for some codes, `data` naming what it is about;
    callers match on `code` and `data`, never on the message."""

    def __init__(self, message: str, *, code: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.data = data or {}
