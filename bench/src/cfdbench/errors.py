from enum import StrEnum


class Code(StrEnum):
    """Stable error codes of the harness; scripts and tests match on these, not on messages."""

    BINARY_FAILED = "BINARY_FAILED"
    CV_TOO_HIGH = "CV_TOO_HIGH"
    DIRTY_TREE = "DIRTY_TREE"
    GPU_BUSY = "GPU_BUSY"
    GPU_CONTENDED = "GPU_CONTENDED"
    GPU_MISMATCH = "GPU_MISMATCH"
    INCOMPATIBLE = "INCOMPATIBLE"
    MISSING_BENCHMARK = "MISSING_BENCHMARK"
    NO_COMMIT = "NO_COMMIT"
    NO_SAMPLES = "NO_SAMPLES"
    NO_STREAM = "NO_STREAM"
    NVIDIA_SMI_FAILED = "NVIDIA_SMI_FAILED"
    RESULT_UNREADABLE = "RESULT_UNREADABLE"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    VERIFY_FAILED = "VERIFY_FAILED"


class BenchError(Exception):
    """A harness failure carrying a stable :class:`Code`."""

    def __init__(self, code: Code, message: str) -> None:
        super().__init__(f"error[{code}]: {message}")
        self.code = code
