#!/usr/bin/env python3
"""Whether the CPU tests are healthy on GB10, for gpu-tests' hosted CPU shards. Fails closed.

    GH_TOKEN=... GH_REPO=owner/repo bench/ci/gb10-health.py

Unhealthy, exit 1, when either holds:
- an issue labelled gb10-cpu-difference is open;
- the last completed scheduled gpu-tests run on main did not pass its step "CPU tests on GB10"
  (failed, cancelled, timed out, skipped after an earlier failure, or never reached), and no issue
  labelled gb10-cpu-difference was closed after that run ended. Closing such an issue is how a
  person records that a manual gpu-tests run checked the fix on GB10 (docs/infra/runners.md);
  it is also the way out when the nightly's report could not open one.

Exit 2 when GitHub's API cannot be read: the shards then fail too. A last nightly from before the
step existed (its commit has no bench/ci/gb10-health.py) counts as healthy, with a notice; one
whose commit has it but whose job lists no such step is unhealthy.
"""

import json
import os
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
LABEL = "gb10-cpu-difference"
STEP = "CPU tests on GB10"


def get(path: str, *, missing_ok: bool = False) -> object:
    request = urllib.request.Request(
        API + path,
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        if missing_ok and e.code == 404:
            return None
        sys.stdout.write(f"::error::GB10 health: cannot read {path}: {e}\n")
        raise SystemExit(2) from e
    except Exception as e:
        sys.stdout.write(f"::error::GB10 health: cannot read {path}: {e}\n")
        raise SystemExit(2) from e


def labelled(repo: str, state: str) -> list[dict]:
    """Every issue (not pull request) with the label in this state, all pages."""
    out: list[dict] = []
    page = 1
    while True:
        items = get(f"/repos/{repo}/issues?labels={LABEL}&state={state}&per_page=100&page={page}")
        assert isinstance(items, list)
        out += [i for i in items if "pull_request" not in i]
        if len(items) < 100:
            return out
        page += 1


def main() -> int:
    repo = os.environ["GH_REPO"]
    open_ = [i["number"] for i in labelled(repo, "open")]
    if open_:
        numbers = ", ".join(f"#{n}" for n in open_)
        sys.stdout.write(
            f"::error::the CPU tests failed on GB10 ({numbers}, label {LABEL}). Fix it, check the "
            "fix with a manual gpu-tests run on its branch (it runs the CPU tests on GB10), then "
            "close the issue.\n"
        )
        return 1

    found = get(
        f"/repos/{repo}/actions/workflows/gpu-tests.yml/runs"
        "?event=schedule&branch=main&status=completed&per_page=1"
    )
    assert isinstance(found, dict)
    runs = found["workflow_runs"]
    if not runs:
        sys.stdout.write("::notice::GB10 health: no completed nightly yet\n")
        return 0
    run = runs[0]
    listed = get(f"/repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100")
    assert isinstance(listed, dict)
    steps = [s for j in listed["jobs"] if j["name"] == "tests" for s in j.get("steps", [])]
    gb10 = [s for s in steps if s["name"].startswith(STEP)]
    if not gb10:
        # Only a nightly of a commit without this script may lack the step.
        has_step = get(
            f"/repos/{repo}/contents/bench/ci/gb10-health.py?ref={run['head_sha']}",
            missing_ok=True,
        )
        if has_step is None:
            sys.stdout.write(
                f"::notice::GB10 health: the last nightly ({run['html_url']}) predates the GB10 "
                "CPU step\n"
            )
            return 0
    outcome = gb10[0].get("conclusion") if gb10 else None
    if outcome == "success":
        sys.stdout.write(
            f"GB10 health: the last nightly passed the CPU tests on GB10 ({run['html_url']})\n"
        )
        return 0

    ended = run["updated_at"]
    cleared = [
        i["number"]
        for i in labelled(repo, "closed")
        if i.get("closed_at") and i["closed_at"] > ended
    ]
    if cleared:
        sys.stdout.write(
            f"GB10 health: the last nightly's GB10 CPU step was {outcome or 'not reached'}, "
            f"cleared by #{cleared[0]}, closed after it\n"
        )
        return 0
    sys.stdout.write(
        f"::error::the last nightly ({run['html_url']}) did not pass the CPU tests on GB10 "
        f"({outcome or 'not reached'}) and no issue labelled {LABEL} was closed since. Check the "
        "fix with a manual gpu-tests run, then open and close an issue with that label to record it.\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
