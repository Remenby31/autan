"""`python -m autan_codegen KERNEL.py --out DIR [--backends cpu,cuda,hip]`."""

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
    ap.add_argument("--policies", default="fp64,fp32+acc64,fp32", help="policies to instantiate")
    ns = ap.parse_args(argv)
    try:
        for path in ns.kernels:
            generate(
                path,
                ns.out,
                ns.backends.split(","),
                line_directives=not ns.no_line,
                policies=ns.policies.split(","),
            )
    except CodegenError:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
