"""`python -m autan_codegen KERNEL.py --out DIR [--backends cpu,gpu,metal] [--source-root DIR]
[--face-values]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autan_codegen.errors import CodegenError
from autan_codegen.generate import BACKENDS, generate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="autan_codegen", description=__doc__)
    ap.add_argument("kernels", nargs="+", type=Path, help="kernel modules (.py)")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument(
        "--backends", default="cpu", help=f"comma-separated subset of {','.join(BACKENDS)}"
    )
    ap.add_argument(
        "--no-line", action="store_true", help="omit #line directives to the DSL source"
    )
    ap.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="#line directives name the DSL sources relative to this directory",
    )
    ap.add_argument(
        "--policies",
        default=None,
        help="policies to instantiate (default: every policy the backends support)",
    )
    ap.add_argument(
        "--face-values",
        action="store_true",
        help="also write <module>_values_cpu.hpp: each face loop's outputs per face (measurements)",
    )
    ns = ap.parse_args(argv)
    try:
        for path in ns.kernels:
            generate(
                path,
                ns.out,
                ns.backends.split(","),
                line_directives=not ns.no_line,
                policies=ns.policies.split(",") if ns.policies else None,
                source_root=ns.source_root,
                face_values=ns.face_values,
            )
    except CodegenError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
