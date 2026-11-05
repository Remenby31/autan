import json
import os
import subprocess
import time
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "job-started.sh"
REPO = "Remenby31/autan"


def run_hook(
    tmp_path: Path, event: str, payload: dict, repository: str = REPO, **beside: str
) -> subprocess.CompletedProcess:
    # The hook reads the allowed repository, and its memory admission settings, from files beside
    # it. Outside a runner it has no Runner.Worker ancestor to kill; that part is checked on a
    # real runner.
    hook = tmp_path / "job-started.sh"
    hook.write_text(HOOK.read_text())
    (tmp_path / "repository").write_text(REPO + "\n")
    smi = beside.pop("nvidia_smi", HEALTHY_SMI)
    for name, text in beside.items():
        (tmp_path / name.replace("_", "-")).write_text(text)
    # nvidia-smi at the boundary, healthy unless a test says otherwise: never this machine's.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "nvidia-smi").write_text(smi)
    (bin_dir / "nvidia-smi").chmod(0o755)
    path = tmp_path / "event.json"
    path.write_text(json.dumps(payload))
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GITHUB_REPOSITORY": repository,
        "GITHUB_EVENT_NAME": event,
        "GITHUB_EVENT_PATH": str(path),
    }
    return subprocess.run(["bash", str(hook)], check=False, env=env, capture_output=True, text=True)


HEALTHY_SMI = "#!/bin/sh\necho 'GPU 0: NVIDIA GB10'\n"


def pull_request(head: str) -> dict:
    return {
        "repository": {"full_name": REPO},
        "pull_request": {"head": {"repo": {"full_name": head}}},
    }


@pytest.mark.parametrize(
    ("event", "payload", "accepted"),
    [
        ("push", {"repository": {"full_name": REPO}}, True),
        ("workflow_dispatch", {"repository": {"full_name": REPO}}, True),
        ("schedule", {"repository": {"full_name": REPO}}, True),
        ("pull_request", pull_request(REPO), True),
        ("pull_request", pull_request("attacker/autan"), False),
        (
            "pull_request",
            {"repository": {"full_name": REPO}, "pull_request": {"head": {"repo": None}}},
            False,
        ),
        ("pull_request_target", pull_request(REPO), False),
        ("issue_comment", {"repository": {"full_name": REPO}}, False),
        ("workflow_run", {"repository": {"full_name": REPO}}, False),
        ("push", {"repository": {"full_name": "attacker/autan"}}, False),
    ],
)
def test_hook_accepts_only_trusted_triggers_of_this_repository(tmp_path, event, payload, accepted):
    assert (run_hook(tmp_path, event, payload).returncode == 0) == accepted


def test_hook_refuses_another_repository(tmp_path):
    assert (
        run_hook(
            tmp_path, "push", {"repository": {"full_name": REPO}}, repository="attacker/x"
        ).returncode
        != 0
    )


def meminfo(available_gib: float) -> str:
    return f"MemTotal: {128 << 20} kB\nMemAvailable: {int(available_gib * (1 << 20))} kB\n"


PUSH = {"repository": {"full_name": REPO}}


@pytest.mark.parametrize(("available", "accepted"), [(30.0, True), (29.9, False)])
def test_a_job_waits_for_the_host_memory_floor_then_is_refused(tmp_path, available, accepted):
    start = time.monotonic()
    out = run_hook(tmp_path, "push", PUSH, min_available_gib="30\n", admit_wait_s="2\n",
                   meminfo=meminfo(available))  # fmt: skip
    assert (out.returncode == 0) == accepted
    if not accepted:
        assert time.monotonic() - start >= 2  # it waited before refusing


def test_a_job_starts_once_memory_comes_back(tmp_path):
    # Memory comes back only once the hook is seen waiting for it.
    hook = tmp_path / "job-started.sh"
    hook.write_text(HOOK.read_text())
    (tmp_path / "repository").write_text(REPO + "\n")
    (tmp_path / "min-available-gib").write_text("30\n")
    (tmp_path / "admit-wait-s").write_text("60\n")
    (tmp_path / "meminfo").write_text(meminfo(10))
    event = tmp_path / "event.json"
    event.write_text(json.dumps(PUSH))
    env = {"PATH": os.environ["PATH"], "GITHUB_REPOSITORY": REPO, "GITHUB_EVENT_NAME": "push",
           "GITHUB_EVENT_PATH": str(event)}  # fmt: skip
    proc = subprocess.Popen(["bash", str(hook)], env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert "WAITING FOR MEMORY" in proc.stdout.readline()
        (tmp_path / "meminfo").write_text(meminfo(40))
        assert proc.wait(timeout=30) == 0
    finally:
        proc.kill()


def test_a_fork_is_refused_without_waiting_for_memory(tmp_path):
    start = time.monotonic()
    out = run_hook(tmp_path, "pull_request", pull_request("attacker/autan"),
                   min_available_gib="30\n", admit_wait_s="30\n", meminfo=meminfo(1))  # fmt: skip
    assert out.returncode != 0
    assert time.monotonic() - start < 5


def test_no_floor_no_wait(tmp_path):
    assert run_hook(tmp_path, "push", PUSH, meminfo=meminfo(1)).returncode == 0


@pytest.mark.parametrize("smi", [
    # As nvidia-smi answers in a container whose device rules a host daemon-reload revoked.
    "#!/bin/sh\necho 'Failed to initialize NVML: Unknown Error'\nexit 255\n",
    # As it hangs on a wedged driver.
    "#!/bin/sh\nsleep 60\n",
])  # fmt: skip
def test_a_runner_that_lost_its_gpu_refuses_the_job(tmp_path, smi):
    start = time.monotonic()
    out = run_hook(tmp_path, "push", PUSH, nvidia_smi=smi, gpu_probe_s="1\n")
    assert out.returncode != 0
    assert time.monotonic() - start < 15
