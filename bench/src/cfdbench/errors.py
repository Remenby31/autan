class BenchError(Exception):
    """A harness failure with a stable code (for example ``GPU_BUSY``) callers can match on."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"error[{code}]: {message}")
        self.code = code
