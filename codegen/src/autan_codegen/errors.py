"""Coded generator errors, reported with the DSL source location when one is known."""

from __future__ import annotations

import enum


class Code(enum.StrEnum):
    """Stable error codes of the generator (docs/design/kernel-generator.md, "Errors")."""

    UNSUPPORTED_CONSTRUCT = "E1001"
    NOT_A_DSL_TYPE = "E1002"
    INVALID_BINDING = "E1003"
    INVALID_LOOP_OUTPUT = "E1004"
    INVALID_CONTROL_FLOW = "E1005"
    TYPE_MISMATCH = "E1006"
    REL_OF_COMPUTED_VALUE = "E1007"
    INC_TARGET_NOT_ACCUMULATION = "E1008"
    INVALID_KERNEL_MODULE = "E1009"
    UNKNOWN_BACKEND = "E1010"


class CodegenError(Exception):
    """A kernel source the generator rejects. `code` is stable; the message is for humans."""

    def __init__(self, code: Code, message: str, where: str | None = None) -> None:
        self.code = code
        self.where = where
        super().__init__(f"{where}: {code}: {message}" if where else f"{code}: {message}")
