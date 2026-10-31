"""Check a repaired Linux wheel before it can be published.

Fails when the wheel exceeds the size budget, bundles an NVIDIA library, or lacks native code
(SASS) for an architecture listed in ``AUTAN_CUDA_ARCHITECTURES``.

Usage: python check_wheel.py WHEEL_DIR
"""

import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# PyPI's default per-file limit is 100 MB; keep headroom for growth between two releases.
BUDGET_MB = 90
# Driver libraries break on every other driver version; NVIDIA runtime libraries carry a
# proprietary licence and come from the nvidia-* wheels instead. Neither may be bundled.
NVIDIA_LIBRARIES = re.compile(r"lib(cuda|nvidia-|cudart|nvrtc|nvJitLink|cublas|cusparse)")


def check(wheel: Path) -> list[str]:
    problems = []
    size_mb = wheel.stat().st_size / 1e6
    sys.stdout.write(f"{wheel.name}: {size_mb:.1f} MB\n")
    if size_mb > BUDGET_MB:
        problems.append(f"{size_mb:.1f} MB exceeds the {BUDGET_MB} MB budget")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        problems += [f"bundles NVIDIA library {n}" for n in names if NVIDIA_LIBRARIES.search(n)]
        modules = [n for n in names if n.endswith(".so")]
        with tempfile.TemporaryDirectory() as tmp:
            archive.extractall(tmp, members=modules)
            expected = os.environ.get("AUTAN_CUDA_ARCHITECTURES", "")
            for module in modules:
                problems += check_fatbin(Path(tmp, module), expected)
    return problems


def check_fatbin(module: Path, expected: str) -> list[str]:
    listing = subprocess.run(
        ["cuobjdump", "--list-elf", "--list-ptx", str(module)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    sass = set(re.findall(r"\.sm_(\d+)\.cubin", listing))
    ptx = set(re.findall(r"\.sm_(\d+)\.ptx", listing))
    sys.stdout.write(f"  {module.name}: SASS {sorted(sass, key=int)}, PTX {sorted(ptx, key=int)}\n")
    problems = []
    for arch in filter(None, expected.split(";")):
        number, _, kind = arch.partition("-")
        found = ptx if kind == "virtual" else sass
        if number not in found:
            problems.append(f"{module.name} lacks {arch}")
    return problems


def main() -> int:
    problems = []
    for wheel in sorted(Path(sys.argv[1]).glob("*.whl")):
        problems += [f"{wheel.name}: {p}" for p in check(wheel)]
    for problem in problems:
        sys.stderr.write(f"error: {problem}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
