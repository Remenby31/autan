import json
import os
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "job-started.sh"
REPO = "Remenby31/autan"


def run_hook(
    tmp_path: Path, event: str, payload: dict, repository: str = REPO
) -> subprocess.CompletedProcess:
    path = tmp_path / "event.json"
    path.write_text(json.dumps(payload))
    env = {
        "PATH": os.environ["PATH"],
        "AUTAN_RUNNER_REPOSITORY": REPO,
        "GITHUB_REPOSITORY": repository,
        "GITHUB_EVENT_NAME": event,
        "GITHUB_EVENT_PATH": str(path),
    }
    return subprocess.run(["bash", str(HOOK)], check=False, env=env, capture_output=True, text=True)


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
