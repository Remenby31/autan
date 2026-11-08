"""The spike libraries are test tooling: no product target links or depends on them, so read-only
getters added to them (ADR 0003) never change the product's binaries."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPIKE = re.compile(r"\bautan_spike(_gpu|_metal)?\b")
COMMAND = re.compile(r"(\w+)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", re.DOTALL)


def test_no_product_target_depends_on_the_spike() -> None:
    files = [
        p
        for p in [*ROOT.rglob("CMakeLists.txt"), *ROOT.rglob("*.cmake")]
        if not any(part.startswith(("build", ".")) for part in p.relative_to(ROOT).parts)
    ]
    assert files
    seen = 0
    for path in files:
        text = re.sub(r"#[^\n]*", "", path.read_text())
        for command, args in COMMAND.findall(text):
            if not SPIKE.search(args):
                continue
            seen += 1
            first = args.split()[0]
            assert SPIKE.fullmatch(first), f"{path}: {command}({args.strip()[:80]}) uses the spike"
    assert seen


TEST_ONLY = re.compile(r"\bautan_spike_test_only_\w+|\bFp32Acc32TestOnly\b|fp32\+acc32-test-only")
SOURCES = {".cpp", ".hpp", ".h", ".cu", ".cuh", ".mm", ".metal", ".py", ".cmake", ".txt"}


def test_no_product_file_names_the_test_only_entry_points() -> None:
    # The defect controls of ADR 0003 corrupt what the kernels read through these; only the spike
    # and its tests may name them.
    allowed = (ROOT / "codegen" / "spike", ROOT / "codegen" / "tests")
    seen = 0
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if path.suffix not in SOURCES or not path.is_file():
            continue
        if any(
            part.startswith(("build", ".")) or part in ("venv", "node_modules")
            for part in rel.parts
        ):
            continue
        found = TEST_ONLY.search(path.read_text(errors="ignore"))
        if any(path.is_relative_to(a) for a in allowed):
            seen += bool(found)
            continue
        assert found is None, f"{rel} names {found.group(0)}"
    assert seen  # the spike does define them


IMPORT = re.compile(r"\bautan_spike\b")  # any mention, e.g. `import os, autan_spike as s`


def test_no_python_outside_the_spike_and_its_tests_imports_it() -> None:
    # The CMake test above is the boundary for compiled targets; this one covers Python callers,
    # which could reach the test-only entry points without naming them. Both are needed: a
    # spelling scan alone does not prove isolation.
    allowed = (ROOT / "codegen" / "spike", ROOT / "codegen" / "tests")
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(
            part.startswith(("build", ".")) or part in ("venv", "node_modules")
            for part in rel.parts
        ):
            continue
        if any(path.is_relative_to(a) for a in allowed):
            continue
        assert IMPORT.search(path.read_text(errors="ignore")) is None, f"{rel} names autan_spike"
