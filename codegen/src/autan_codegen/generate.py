"""Generate the C++ files of one kernel module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from autan_codegen import model
from autan_codegen.emit_cpp import emit_common_header
from autan_codegen.emit_cpu import emit_cpu_header
from autan_codegen.emit_gpu import emit_gpu_header
from autan_codegen.errors import Code, CodegenError

if TYPE_CHECKING:
    from pathlib import Path

BACKENDS = ("cpu", "gpu")


def generate(
    path: Path, out: Path, backends: list[str], line_directives: bool = True
) -> list[Path]:
    """Write `<module>.hpp` and one file per backend into `out`; return the paths written."""
    for b in backends:
        if b not in BACKENDS:
            raise CodegenError(Code.UNKNOWN_BACKEND, f"unknown backend {b!r}")
    m = model.load(path.resolve())
    files = {f"{m.name}.hpp": emit_common_header(m, line_directives)}
    if "cpu" in backends:
        files[f"{m.name}_cpu.hpp"] = emit_cpu_header(m)
    if "gpu" in backends:
        files[f"{m.name}_gpu.cuh"] = emit_gpu_header(m)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in files.items():
        p = out / name
        # Unchanged files keep their timestamp so that the build does not recompile them.
        if not p.exists() or p.read_text() != text:
            p.write_text(text)
        written.append(p)
    return written
