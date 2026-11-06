import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

MUTATE = pathlib.Path(__file__).resolve().parents[1] / "mutate.py"

# The "binary" is value.txt copied into the build; building fails on a value holding "broken".
CMAKE = """cmake_minimum_required(VERSION 3.20)
project(probe NONE)
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/value.txt
  COMMAND ${CMAKE_COMMAND} -DIN=${CMAKE_SOURCE_DIR}/value.txt -P ${CMAKE_SOURCE_DIR}/check.cmake
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/value.txt ${CMAKE_BINARY_DIR}/value.txt
  DEPENDS ${CMAKE_SOURCE_DIR}/value.txt
)
add_custom_target(probe ALL DEPENDS ${CMAKE_BINARY_DIR}/value.txt)
"""
CHECK = """file(READ ${IN} v)
if(v MATCHES "broken")
  message(FATAL_ERROR "broken")
endif()
"""
# A test reading the built value: "bad" fails it, "hang" never ends.
TEST = """#!/usr/bin/env bash
v=$(cat "$(dirname "$0")/value.txt")
printf '%s\\n' "$1" >> "$SPECS"
case "$v" in *hang*) sleep 600 ;; esac
case "$v" in *bad*) echo "FAILED"; exit 1 ;; esac
echo "All tests passed"
"""


def project(root: pathlib.Path, value: str) -> tuple[pathlib.Path, pathlib.Path]:
    src, build = root / "src", root / "build"
    src.mkdir()
    (src / "CMakeLists.txt").write_text(CMAKE)
    (src / "check.cmake").write_text(CHECK)
    (src / "value.txt").write_text(value)
    subprocess.run(["cmake", "-S", src, "-B", build], check=True, capture_output=True)
    subprocess.run(["cmake", "--build", build], check=True, capture_output=True)
    (build / "test.sh").write_text(TEST)
    (build / "test.sh").chmod(0o755)
    return src, build


def mutant(name: str, old: str, new: str, test: str = "*") -> dict:
    return {
        "name": name,
        "file": "value.txt",
        "from": old,
        "to": new,
        "target": "probe",
        "exe": "test.sh",
        "test": test,
    }


def mutate(
    src: pathlib.Path, build: pathlib.Path, mutants: list[dict], results: pathlib.Path | None = None
) -> subprocess.CompletedProcess:
    spec = build.parent / "mutants.json"
    spec.write_text(json.dumps(mutants))
    args = [sys.executable, MUTATE, "--min-timeout", "2", src, build, spec]
    return subprocess.run(
        args + ([results] if results else []), check=False, capture_output=True, text=True
    )


@pytest.fixture(autouse=True)
def specs(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    log = tmp_path / "specs.log"
    monkeypatch.setenv("SPECS", str(log))
    return log


def test_each_mutant_gets_its_verdict_and_the_source_comes_back(tmp_path: pathlib.Path) -> None:
    src, build = project(tmp_path, "alpha good\n")
    results = tmp_path / "results.jsonl"
    mutants = [
        mutant("bad", "good", "bad"),
        mutant("harmless", "alpha", "beta"),
        mutant("broken", "alpha", "broken"),
        mutant("hang", "good", "hang"),
    ]
    r = mutate(src, build, mutants, results)
    assert r.returncode == 1, r.stdout
    verdicts = {
        json.loads(line)["name"]: json.loads(line)["verdict"]
        for line in results.read_text().splitlines()
    }
    assert verdicts == {
        "bad": "caught",
        "harmless": "survived",
        "broken": "not built",
        "hang": "caught by timeout",
    }
    assert (src / "value.txt").read_text() == "alpha good\n"
    assert (build / "value.txt").read_text() == "alpha good\n"
    # Started again, it repeats nothing.
    assert mutate(src, build, mutants, results).returncode == 0
    assert len(results.read_text().splitlines()) == 4


def test_a_build_of_another_tree_is_refused(tmp_path: pathlib.Path) -> None:
    src, build = project(tmp_path, "alpha good\n")
    copy = tmp_path / "copy"
    shutil.copytree(src, copy)
    r = mutate(copy, build, [mutant("bad", "good", "bad")])
    assert r.returncode == 2
    assert (copy / "value.txt").read_text() == "alpha good\n"


def test_a_comma_in_a_test_name_reaches_catch2_escaped(
    tmp_path: pathlib.Path, specs: pathlib.Path
) -> None:
    src, build = project(tmp_path, "alpha good\n")
    assert mutate(src, build, [mutant("bad", "good", "bad", "cells, faces")]).returncode == 0
    assert set(specs.read_text().splitlines()) == {"cells\\, faces"}


def test_a_test_failing_on_the_original_stops_the_run(tmp_path: pathlib.Path) -> None:
    src, build = project(tmp_path, "alpha bad\n")
    r = mutate(src, build, [mutant("harmless", "alpha", "beta")])
    assert r.returncode == 2
    assert (src / "value.txt").read_text() == "alpha bad\n"


def test_a_binary_left_by_an_interrupted_run_is_rebuilt_first(tmp_path: pathlib.Path) -> None:
    # A run stopped mid-build: the mutant compiled after its source was restored, so the binary
    # is newer than the original source and a build alone would keep it.
    src, build = project(tmp_path, "alpha bad\n")
    (src / "value.txt").write_text("alpha good\n")
    old = (build / "value.txt").stat().st_mtime - 60
    os.utime(src / "value.txt", (old, old))
    assert (
        subprocess.run(["cmake", "--build", build], check=False, capture_output=True).returncode
        == 0
    )
    assert (build / "value.txt").read_text() == "alpha bad\n"
    assert mutate(src, build, [mutant("bad", "good", "bad")]).returncode == 0
    assert (build / "value.txt").read_text() == "alpha good\n"
