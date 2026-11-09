"""The freeze tooling of ADR 0003 (#43): node ids are frozen whole, and every frozen id is one a
run of the auxiliary suite can produce. Stage 1 froze a node id truncated at its first space, which
no run could report, so stage 1 failed on the manifest rather than on any check."""

import json
import os
import subprocess
import sys
from pathlib import Path

from oracle.freeze.make_manifest import AUX_FILES, node_ids

CODEGEN = Path(__file__).resolve().parents[1]
MANIFEST = CODEGEN / "tests" / "oracle" / "freeze" / "manifest-stage1.json"


def test_node_ids_are_kept_whole_with_their_spaces() -> None:
    collected = (
        "tests/test_a.py::test_one\n"
        "tests/test_a.py::test_two[a=9.0 G=[0, 0, 0] fp64 lsq]\n"
        "\n"
        "2 tests collected in 0.01s\n"
    )
    assert node_ids(collected) == [
        "tests/test_a.py::test_one",
        "tests/test_a.py::test_two[a=9.0 G=[0, 0, 0] fp64 lsq]",
    ]


def test_every_frozen_node_id_is_one_the_auxiliary_suite_collects() -> None:
    frozen = json.loads(MANIFEST.read_text())["auxiliary_tests"]["node_ids"]
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *AUX_FILES,
        ],
        cwd=CODEGEN,
        capture_output=True,
        text=True,
        env=dict(os.environ),
        check=False,
    )
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    collected = set(node_ids(run.stdout))
    missing = [n for n in frozen if n not in collected]
    assert not missing, f"frozen node ids no run can produce: {missing}"
