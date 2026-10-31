"""Coded errors raised by the V&V tools."""


class VVError(Exception):
    """An error with a stable code `V####`; callers match on `code`, never on the message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
