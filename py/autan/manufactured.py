"""The bounds of an ``autan.mms/1`` document (case-schema.md#verification)."""

from __future__ import annotations

import re
from typing import Any

# The catalogue's largest document is 18,387 bytes and its deepest expression 44 levels
# (autan-vv, rans_sa); the bounds leave about 57x and 23x.
MMS_BYTES = 1 << 20
MMS_DEPTH = 1024

GROUPS = ("fields", "gradients", "conserved", "sources", "auxiliary")
FUNCTIONS = frozenset({"pow", "sqrt", "sin", "cos", "exp", "log"})
_TOKEN = re.compile(
    r"\s*(?:(?P<number>[0-9.](?:[0-9.]|[eE][+-]?)*)|(?P<name>[A-Za-z0-9_]+)|(?P<op>[-+*/(),]))"
)
_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "neg": 3}


def depth(text: str, limit: int = MMS_DEPTH) -> int | None:
    """Depth of the tree the solver's expression grammar builds from ``text``: 1 for a number
    or a name, plus one for each parenthesis, unary minus, binary operator and function above
    it. ``None`` if ``text`` is not an expression; stops at ``limit + 1``."""
    values: list[int] = []
    ops: list[Any] = []

    def reduce() -> None:
        op = ops.pop()
        if op == "neg":
            values.append(1 + values.pop())
        else:
            b, a = values.pop(), values.pop()
            values.append(1 + max(a, b))

    position, operand = 0, True
    text = text.rstrip()
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            return None
        position = match.end()
        number, name, op = match.group("number", "name", "op")
        if operand and number is not None:
            try:
                float(number)
            except ValueError:
                return None
            values.append(1)
            operand = False
        elif operand and name is not None and name in FUNCTIONS:
            following = _TOKEN.match(text, position)
            if following is None or following.group("op") != "(":
                return None
            position = following.end()
            ops.append([name, 1])
        elif operand and name is not None:
            values.append(1)
            operand = False
        elif operand and op == "-":
            ops.append("neg")
        elif operand and op == "(":
            ops.append("(")
        elif not operand and op in ("+", "-", "*", "/"):
            while ops and isinstance(ops[-1], str) and ops[-1] != "(":
                if _PRECEDENCE[ops[-1]] < _PRECEDENCE[op]:
                    break
                reduce()
            ops.append(op)
            operand = True
        elif not operand and op in (")", ","):
            while ops and isinstance(ops[-1], str) and ops[-1] != "(":
                reduce()
            if not ops:
                return None
            if op == ",":
                if ops[-1] == "(" or ops[-1][0] != "pow" or ops[-1][1] != 1:
                    return None
                ops[-1][1] = 2
                operand = True
            elif ops[-1] == "(":
                ops.pop()
                values.append(1 + values.pop())
            else:
                name, arguments = ops.pop()
                if arguments != (2 if name == "pow" else 1):
                    return None
                if arguments == 2:
                    b, a = values.pop(), values.pop()
                    values.append(1 + max(a, b))
                else:
                    values.append(1 + values.pop())
        else:
            return None
        if values and values[-1] > limit:
            return limit + 1
    if operand:
        return None
    while ops:
        if not isinstance(ops[-1], str) or ops[-1] == "(":
            return None
        reduce()
    return values[0] if len(values) == 1 else None


def expressions(document: dict[str, Any]) -> list[Any]:
    """Every expression string of the document's groups, as the solver reads them."""
    found: list[Any] = []
    for group in (g for g in GROUPS if g in document):
        content = document[group]
        pairs = content.get("subexpressions") if isinstance(content, dict) else None
        outputs = content.get("outputs") if isinstance(content, dict) else None
        if not isinstance(pairs, list) or not isinstance(outputs, dict):
            found.append(None)  # the solver refuses a group that is not this shape
            continue
        found.extend(p[1] if isinstance(p, list) and len(p) == 2 else None for p in pairs)
        found.extend(outputs.values())
    return found


def too_deep(document: dict[str, Any]) -> str | None:
    """Why the document's expressions break the contract's bound, or ``None``."""
    for text in expressions(document):
        if not isinstance(text, str):
            return "an expression is not a string"
        found = depth(text)
        if found is None:
            return f"{text[:40]!r} is not an expression"
        if found > MMS_DEPTH:
            return f"an expression is deeper than {MMS_DEPTH} levels"
    return None
