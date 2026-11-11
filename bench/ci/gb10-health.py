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

Exit 2 when GitHub's API cannot be read: the shards then fail too. A last nightly from before
the step existed (its job lists no such step) counts as healthy, with a notice.
"""

import json
import os
import sys
import urllib.request

API = "https://api.github.com"
LABEL = "gb10-cpu-difference"
STEP = "CPU tests on GB10"


def get(path: str) -> object:
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
    except Exception as e:
        sys.stdout.write(f"::error::GB10 health: cannot read {path}: {e}\n")
        raise SystemExit(2) from e


def main() -> int:
    repo = os.environ["GH_REPO"]
    issues = get(f"/repos/{repo}/issues?labels={LABEL}&state=open&per_page=10")
    open_ = [i["number"] for i in issues if "pull_request" not in i]  # type: ignore[union-attr]
    if open_:
        numbers = ", ".join(f"#{n}" for n in open_)
        sys.stdout.write(
            f"::error::the CPU tests failed on GB10 ({numbers}, label {LABEL}). Fix it, check the "
            "fix with a manual gpu-tests run on its branch (it runs the CPU tests on GB10), then "
            "close the issue.\n"
        )
        return 1

    runs = get(
        f"/repos/{repo}/actions/workflows/gpu-tests.yml/runs"
        "?event=schedule&branch=main&status=completed&per_page=1"
    )["workflow_runs"]  # type: ignore[index]
    if not runs:
        sys.stdout.write("::notice::GB10 health: no completed nightly yet\n")
        return 0
    run = runs[0]
    jobs = get(f"/repos/{repo}/actions/runs/{run['id']}/jobs?per_page=50")["jobs"]  # type: ignore[index]
    steps = [s for j in jobs if j["name"] == "tests" for s in j.get("steps", [])]
    gb10 = [s for s in steps if s["name"].startswith(STEP)]
    if not gb10 and any(j["name"] == "tests" and j.get("steps") for j in jobs):
        sys.stdout.write(
            f"::notice::GB10 health: the last nightly ({run['html_url']}) predates the GB10 CPU step\n"
        )
        return 0
    outcome = gb10[0].get("conclusion") if gb10 else None
    if outcome == "success":
        sys.stdout.write(
            f"GB10 health: the last nightly passed the CPU tests on GB10 ({run['html_url']})\n"
        )
        return 0

    ended = run["updated_at"]
    closed = get(f"/repos/{repo}/issues?labels={LABEL}&state=closed&sort=updated&per_page=20")
    cleared = [
        i["number"]
        for i in closed  # type: ignore[union-attr]
        if "pull_request" not in i and i.get("closed_at") and i["closed_at"] > ended
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
