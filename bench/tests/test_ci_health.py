import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CI = Path(__file__).parents[1] / "ci"
HEALTH = CI / "health.py"
COMMIT = "2" * 40
BASELINE = {
    "stream.copy": 234.88,
    "stream.scale": 236.24,
    "stream.add": 235.53,
    "stream.triad": 235.66,
}


def result(path: Path, medians: dict[str, float], commit: str = COMMIT) -> Path:
    doc = {"provenance": {"solver": {"commit": commit, "dirty": False}},
           "benchmarks": [{"name": n, "stats": {"median": m}} for n, m in medians.items()]}  # fmt: skip
    path.write_text(json.dumps(doc))
    return path


def health(tmp_path: Path, probe: dict[str, float], base: dict[str, float] = BASELINE,
           built: str = COMMIT) -> tuple[int, dict | None]:  # fmt: skip
    """Exit status and the JSON verdict (its last line), None when there is none."""
    args = [str(result(tmp_path / "probe.json", probe, built)),
            str(result(tmp_path / "baseline.json", base)), "--commit", COMMIT]  # fmt: skip
    proc = subprocess.run([sys.executable, str(HEALTH), *args], capture_output=True, text=True,
                          check=False)  # fmt: skip
    last = proc.stdout.strip().splitlines()[-1]
    return proc.returncode, json.loads(last) if last.startswith("{") else None


def test_a_probe_within_3_percent_of_the_baseline_is_healthy(tmp_path):
    status, verdict = health(tmp_path, {n: m * 0.971 for n, m in BASELINE.items()})
    assert (status, verdict) == (0, {"state": "healthy", "low": []})


def test_exactly_97_percent_is_healthy_whatever_the_round_off(tmp_path):
    # 223.391 / 230.3 is 0.97 in decimal, a hair below it in binary.
    status, verdict = health(tmp_path, {"stream.copy": 223.391}, base={"stream.copy": 230.3})
    assert (status, verdict["state"]) == (0, "healthy")


def test_the_2026_11_02_probe_is_degraded(tmp_path):
    # spark-5359 at 11:48Z (#284): main's own STREAM, 219-224 GB/s against 235.
    probe = {
        "stream.copy": 222.4,
        "stream.scale": 221.8,
        "stream.add": 224.2,
        "stream.triad": 221.2,
    }
    status, verdict = health(tmp_path, probe)
    assert status == 3
    assert verdict == {"state": "degraded", "low": sorted(BASELINE)}


def test_one_low_kernel_is_enough(tmp_path):
    probe = dict(BASELINE, **{"stream.triad": BASELINE["stream.triad"] * 0.969})
    assert health(tmp_path, probe) == (3, {"state": "degraded", "low": ["stream.triad"]})


@pytest.mark.parametrize(
    ("probe", "built"),
    [({n: m for n, m in BASELINE.items() if n != "stream.add"}, COMMIT), (BASELINE, "3" * 40)],
    ids=["a kernel missing", "built from another commit"],
)
def test_a_probe_that_cannot_be_judged_gives_no_state(tmp_path, probe, built):
    assert health(tmp_path, probe, built=built) == (2, None)


@pytest.fixture
def repo(tmp_path: Path) -> dict[str, str]:
    """A repository root as health.sh sees it, the probe already built: a fake lock wrapper that
    runs its command, and a fake `uv` writing the probe's result (medians from $PROBE)."""
    root = tmp_path / "root"
    (root / "bench" / "ci").mkdir(parents=True)
    (root / "bench" / "lock").mkdir()
    shutil.copy(CI / "health.sh", root / "bench" / "ci")
    shutil.copy(HEALTH, root / "bench" / "ci")
    lock = root / "bench" / "lock" / "autan-gpu-lock"
    lock.write_text('#!/bin/sh\nwhile [ "$1" != "--" ]; do shift; done; shift; exec "$@"\n')
    bindir = tmp_path / "bin"
    bindir.mkdir()
    uv = bindir / "uv"
    uv.write_text('#!/bin/sh\nout=; while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done\n'
                  'cp "$PROBE" "$out"\n')  # fmt: skip
    out = tmp_path / "out"
    probe_bin = out / f"reference-{COMMIT}" / "build" / "stream"
    probe_bin.parent.mkdir(parents=True)
    probe_bin.write_text("")
    for f in (lock, uv, probe_bin):
        f.chmod(0o755)
    result(root / "baseline.json", BASELINE)
    return {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "ROOT": str(root),
            "OUT": str(out)}  # fmt: skip


@pytest.mark.parametrize(
    ("factor", "status", "state"), [(0.99, 0, "healthy"), (0.95, 3, "degraded")]
)
def test_health_sh_states_its_verdict_for_github_output(repo, tmp_path, factor, status, state):
    probe = result(tmp_path / "measured.json", {n: m * factor for n, m in BASELINE.items()})
    proc = subprocess.run(["bash", "bench/ci/health.sh", "baseline.json", repo["OUT"]],
                          cwd=repo["ROOT"], env={**repo, "PROBE": str(probe)},
                          capture_output=True, text=True, check=False)  # fmt: skip
    assert proc.returncode == status, proc.stderr
    assert proc.stdout == f"state={state}\n"  # stdout is what $GITHUB_OUTPUT receives


def test_health_sh_gives_no_state_for_a_probe_of_another_commit(repo, tmp_path):
    probe = result(tmp_path / "measured.json", BASELINE, commit="3" * 40)
    proc = subprocess.run(["bash", "bench/ci/health.sh", "baseline.json", repo["OUT"]],
                          cwd=repo["ROOT"], env={**repo, "PROBE": str(probe)},
                          capture_output=True, text=True, check=False)  # fmt: skip
    assert proc.returncode == 2
    assert proc.stdout == ""


def test_health_sh_builds_the_probe_and_leaves_the_repository_as_it_was(repo, tmp_path):
    # main's checkout keeps its whole history (the bisection walks it): with the baseline's
    # commit present, the probe is built without any fetch, the repository stays complete, and
    # the worktree it was built in is removed, so a cleaned RUNNER_TEMP leaves nothing stale.
    root = Path(repo["ROOT"])
    env = {**repo, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}  # fmt: skip
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "base"]):
        subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, env=env, check=True,
                            capture_output=True, text=True).stdout.strip()  # fmt: skip
    base = json.loads((root / "baseline.json").read_text())
    base["provenance"]["solver"]["commit"] = commit
    (root / "baseline.json").write_text(json.dumps(base))
    # The fake lock "builds": it creates the binary the cmake command would (-B DIR).
    (root / "bench" / "lock" / "autan-gpu-lock").write_text(
        '#!/bin/sh\nwhile [ "$1" != "--" ]; do shift; done; shift\n'
        'case "$*" in *cmake*) d=$(printf %s "$*" | sed "s/.*-B .\\([^ \']*\\).*/\\1/");'
        ' mkdir -p "$d" && : > "$d/stream" && chmod +x "$d/stream"; exit 0;; esac\nexec "$@"\n'
    )
    probe = result(tmp_path / "measured.json", BASELINE, commit=commit)
    out = tmp_path / "out2"
    proc = subprocess.run(["bash", "bench/ci/health.sh", "baseline.json", str(out)], cwd=root,
                          env={**env, "PROBE": str(probe)}, capture_output=True, text=True,
                          check=False)  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert (out / f"reference-{commit}" / "build" / "stream").is_file()
    assert not (out / f"reference-{commit}" / "src").exists()
    shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=root, env=env,
                             check=True, capture_output=True, text=True).stdout.strip()  # fmt: skip
    assert shallow == "false"
    worktrees = subprocess.run(["git", "worktree", "list"], cwd=root, env=env, check=True,
                               capture_output=True, text=True).stdout.splitlines()  # fmt: skip
    assert len(worktrees) == 1
