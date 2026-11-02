"""The bound on every free text of a case and of the files it names (case-schema.md#free-texts)."""

from __future__ import annotations

# The solver allocates a text before it can check it; 4 096 bytes is a path's limit on Linux.
TEXT_BYTES = 4096


def bounded(text: str) -> str:
    """``text`` if it is at most ``TEXT_BYTES`` bytes in UTF-8; ``ValueError`` otherwise."""
    if len(text.encode("utf-8", "surrogatepass")) > TEXT_BYTES:
        message = f"longer than {TEXT_BYTES} bytes in UTF-8"
        raise ValueError(message)
    return text
