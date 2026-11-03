"""The bounds on every free text and number of a case and of the files it names
(case-schema.md#free-texts)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# The solver allocates a text before it can check it; 4 096 bytes is a path's limit on Linux.
TEXT_BYTES = 4096


def bounded(text: str) -> str:
    """``text`` if it is at most ``TEXT_BYTES`` bytes in UTF-8; ``ValueError`` otherwise."""
    if len(text.encode("utf-8", "surrogatepass")) > TEXT_BYTES:
        message = f"longer than {TEXT_BYTES} bytes in UTF-8"
        raise ValueError(message)
    return text


# A JSON number's text: a double needs at most 24 characters; 64 also covers integers and other
# spellings. A JSON parser holds the number's text twice before it converts it.
NUMBER_BYTES = 64


def number(convert: type[float | int]) -> Callable[[str], float | int]:
    """A ``json.loads`` hook that converts a number's text no longer than ``NUMBER_BYTES``."""

    def checked(text: str) -> float | int:
        if len(text) > NUMBER_BYTES:
            message = f"a number of {len(text)} characters, more than {NUMBER_BYTES}"
            raise ValueError(message)
        return convert(text)

    return checked


def load_json(text: str | bytes, **hooks: Any) -> Any:
    """``json.loads(text, **hooks)``, a document nested deeper than Python can parse raising
    ``ValueError`` like any other invalid document."""
    try:
        return json.loads(text, **hooks)
    except RecursionError:
        message = "nested too deeply"
        raise ValueError(message) from None
