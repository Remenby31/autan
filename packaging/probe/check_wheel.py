"""Check a repaired Linux wheel before it can be published.

Fails when the wheel exceeds the size budget, bundles any shared library besides our own
extension modules, lacks the licence of a statically linked dependency, or lacks native code
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
# The wheel holds our code only. Driver libraries break on every other driver version, and NVIDIA
# runtime libraries carry a proprietary licence (AGENTS.md allows them as dependencies only), so
# any shared library auditwheel grafted into <name>.libs/ is refused, whatever its name.
OUR_MODULES = re.compile(r"^autan[a-z_]*/[^/]+\.abi3\.so$")
SHARED_LIBRARY = re.compile(r"\.so(\.\d+)*$")
# Licences of code linked statically into the extension modules.
REQUIRED_LICENCES = ("LICENSE", "LICENSES/nanobind.txt")


def check(wheel: Path) -> list[str]:
    problems = []
    size_mb = wheel.stat().st_size / 1e6
    sys.stdout.write(f"{wheel.name}: {size_mb:.1f} MB\n")
    if size_mb > BUDGET_MB:
        problems.append(f"{size_mb:.1f} MB exceeds the {BUDGET_MB} MB budget")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        libraries = [n for n in names if SHARED_LIBRARY.search(n)]
        problems += [f"bundles {n}" for n in libraries if not OUR_MODULES.match(n)]
        modules = [n for n in libraries if OUR_MODULES.match(n)]
        licences = {
            n.split(".dist-info/licenses/", 1)[1] for n in names if ".dist-info/licenses/" in n
        }
        problems += [f"lacks licence file {f}" for f in REQUIRED_LICENCES if f not in licences]
        with tempfile.TemporaryDirectory() as tmp:
            archive.extractall(tmp, members=modules)
            expected = os.environ.get("AUTAN_CUDA_ARCHITECTURES", "")
            for module in modules if expected else []:
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
