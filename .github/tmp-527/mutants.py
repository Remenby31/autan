"""Noyau's six mutants against the macOS Debug build, with (A) and without (B) libc++'s hardening."""

import json
import os
import pathlib
import subprocess
import sys
import time

SRC = pathlib.Path.cwd()
PREFIX = os.environ["CGNS_PREFIX"]
BUILDS = {"A-hardened": SRC / "b-hardened", "B-plain": SRC / "b-plain"}
TARGETS = [
    "autan_partition_tests",
    "autan_runtime_tests",
    "autan_euler_run_tests",
    "autan_io_tests",
    "autan_core_tests",
    "autan-run",
    "autan-run-relaxation",
]
CI = [
    "-G",
    "Ninja",
    f"-DCMAKE_PREFIX_PATH={PREFIX}",
    "-DCMAKE_BUILD_TYPE=Debug",
    "-DAUTAN_WARNINGS_AS_ERRORS=ON",
    "-DAUTAN_ENABLE_METIS=ON",
]


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build(b):
    r = sh(["cmake", "--build", str(b), "--target", *TARGETS])
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-2000:])
        sys.exit(f"build failed in {b}")


def run(b, exes):
    for exe in exes:
        r = sh([str(b / exe)], cwd=b, timeout=900)
        if r.returncode:
            out = r.stdout + r.stderr
            how = (
                "HARDENING-ABORT"
                if (
                    "libc++ Hardening" in out
                    or "assertion" in out.lower()
                    or "SIGABRT" in out
                    or r.returncode in (-6, 134)
                )
                else f"exit {r.returncode}"
            )
            return f"FAIL in {exe.split('/')[-1]} ({how})"
    return "pass"


for name, b in BUILDS.items():
    flags = (
        ["-DCMAKE_CXX_FLAGS=-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG"]
        if name.startswith("A")
        else []
    )
    t = time.time()
    r = sh(["cmake", "-S", str(SRC), "-B", str(b), *CI, *flags])
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit("configure failed")
    build(b)
    print(f"{name}: configured+built in {time.time() - t:.0f} s", flush=True)
mutants = json.load(open(SRC / ".github/tmp-527/mutants.json"))
allexes = sorted({e for m in mutants for e in m["exes"]})
for name, b in BUILDS.items():
    print(f"original, {name}: {run(b, allexes)}", flush=True)
for m in mutants:
    f = SRC / m["file"]
    orig = f.read_text()
    assert orig.count(m["from"]) == 1, f"{m['name']}: 'from' occurs {orig.count(m['from'])} times"
    try:
        f.write_text(orig.replace(m["from"], m["to"]))
        res = {}
        for name, b in BUILDS.items():
            build(b)
            res[name] = run(b, m["exes"])
        print(f"MUTANT {m['name']}: " + "; ".join(f"{k}: {v}" for k, v in res.items()), flush=True)
    finally:
        f.write_text(orig)
