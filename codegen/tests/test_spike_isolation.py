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
