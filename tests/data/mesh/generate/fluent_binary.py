"""Re-encodes an ASCII Fluent mesh with binary data sections.

Nodes become section 3010 (little-endian float64), cells 2012 and faces 2013
(little-endian int32), following the binary layout of the Fluent file format
(ANSYS Fluent User's Guide, "Mesh File Format"): the header stays text, the
payload sits between "(" and ")End of Binary Section <id>)". Zone sections 39
are rewritten as 45 so that both spellings are covered. Values are copied, not
recomputed, so the binary file describes exactly the ASCII mesh.

Usage: python fluent_binary.py INPUT.msh OUTPUT.msh
"""

import re
import struct
import sys
from pathlib import Path


def sections(text):
    """Yields the top-level parenthesised sections of a Fluent file."""
    depth, start, in_string = 0, None, False
    for i, ch in enumerate(text):
        if in_string:
            in_string = ch != '"'
        elif ch == '"':
            in_string = True
        elif ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                yield text[start : i + 1]


def encode(section):
    match = re.match(r"\((\d+)\s*\(([^)]*)\)\s*(\((.*)\))?\s*\)$", section, re.DOTALL)
    if not match:
        return section
    sid, header, body = int(match[1]), match[2].split(), match[4]
    if sid == 39:
        return section.replace("(39", "(45", 1)
    if body is None or sid not in (10, 12, 13):
        return section
    tokens = body.split()
    if sid == 10:
        payload = struct.pack(f"<{len(tokens)}d", *map(float, tokens))
    else:
        payload = struct.pack(f"<{len(tokens)}i", *(int(t, 16) for t in tokens))
    bid = {10: 3010, 12: 2012, 13: 2013}[sid]
    return (
        f"({bid} ({' '.join(header)})(".encode()
        + payload
        + f")End of Binary Section {bid})".encode()
    )


def main():
    text = Path(sys.argv[1]).read_text(encoding="ascii")
    with Path(sys.argv[2]).open("wb") as out:
        for section in sections(text):
            encoded = encode(section)
            out.write(encoded if isinstance(encoded, bytes) else encoded.encode())
            out.write(b"\n")


if __name__ == "__main__":
    main()
