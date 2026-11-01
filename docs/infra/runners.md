# GPU runners on the DGX Spark

Each Spark (`spark`, `spark-5359`) runs one GitHub Actions runner registered on `Remenby31/autan` with the labels `self-hosted, linux, ARM64, gpu, gb10`. GPU workflows (`.github/workflows/gpu-*.yml`) target them with `runs-on: [self-hosted, linux, ARM64, gpu, gb10]`. Timed benchmarks add `bench`, a label only spark-5359 carries: spark's GPU is used continuously by another team's process, which the idle check rightly refuses. Register with `AUTAN_RUNNER_LABELS=gpu,gb10,bench` to give a runner that label, or add it with `gh api -X POST repos/Remenby31/autan/actions/runners/<id>/labels -f 'labels[]=bench'`. spark-5359 runs a second runner process for `main`, labelled `bench-main` (`AUTAN_RUNNER_INSTANCE=main AUTAN_RUNNER_LABELS=gpu,gb10,bench-main`, container `autan-runner-main`): `gpu-bench` runs on `main` target it, so they never queue behind pull requests in GitHub, and on the machine the GPU lock orders them, `main` first ([bench.md](bench.md#the-gpu-lock)).

## Threat model

The repository is public, so anyone can open a pull request, and a pull request from a fork can rewrite any workflow file, including its `runs-on` and `if:`. A job that reaches a self-hosted runner executes arbitrary code on a machine that sits on the company tailnet, shares its GPU and Docker daemon with other teams, and keeps state between jobs. The assets at risk are, in order: the tailnet and LAN behind the Spark, the other teams' containers and data on the host, the runner's registration credential, and the integrity of our benchmark results.

## Protections

| Layer | Protection | Enforced by |
|---|---|---|
| Trigger | Only `push`, `workflow_dispatch`, `schedule`, and `pull_request` whose head is a branch of this repository run. Fork pull requests, `pull_request_target`, `workflow_run`, `issue_comment` and everything else are refused before the first step: the hook kills the job's `Runner.Worker`, because a failing hook alone would still let `if: always()` steps run. The allowed repository comes from a root-owned file in the image, never from the environment a workflow controls. | the job-started hook `docker/runner/job-started.sh`, baked into the runner image; a workflow cannot change or skip it |
| Trigger | GPU jobs carry `if: github.event_name != 'pull_request' \|\| github.event.pull_request.head.repo.full_name == github.repository`, so fork runs are skipped rather than failed. | the workflow (convenience only: a fork can edit it) |
| Trigger | Workflows from first-time and external contributors wait for a maintainer's approval (`fork-pr-contributor-approval: all_external_contributors`). Never approve a fork run that touches `.github/`. | repository setting |
| Host | The runner runs in a container as an unprivileged user (uid 1001), with every capability dropped, `no-new-privileges`, 64 GiB memory, 8192 processes and rotated logs. No Docker socket, no host mount except its own volumes and, read-only, the GPU lock's directory. A job cannot see or touch other containers. | `docker/runner/deploy.sh` |
| Network | The runner sits on an internal Docker network with no route out and an isolated gateway (the host has no address on it). Its only egress is a Squid proxy that allows ports 80 and 443 to public addresses and refuses, after DNS resolution, loopback, RFC 1918, Tailscale (`100.64.0.0/10`, `fd7a:115c:a1e0::/48`), link-local and multicast destinations. `deploy.sh` replaces a proxy whose image is out of date and refuses an internal network without these properties. | `docker/runner/proxy/squid.conf`, `deploy.sh` |
| Token | Workflows run with `permissions: contents: read` and `actions/checkout` with `persist-credentials: false`. The registration token is single-use and expires after one hour; it is piped on stdin and handed to the runner through its environment, never on a command line or on disk. | workflows, `deploy.sh register` |
| Supply chain | Third-party actions are pinned by commit SHA; images by digest; downloads by SHA-256. | workflow and Dockerfile review |

Checked on the Sparks: a temporary workflow whose job had an `if: always()`, an `if: failure()` and a plain step ran on a runner configured to refuse it; the job failed with no step executed. The same workflow on a runner that accepts it ran its steps (positive control). From a container on the runner's network: GitHub and PyPI answer through the proxy (200); direct HTTPS and the host's addresses are unreachable; through the proxy, a Tailscale peer, `100.100.100.100`, the LAN, `169.254.169.254`, `localhost`, IPv6 and IPv4-mapped literals, public names that resolve to private addresses (`*.nip.io`) and non-web ports are refused. The hook's decisions are tested in `docker/runner/tests/` with event payloads for every trigger.

Residual risks, accepted because only trusted triggers run code: jobs run as the same user as the runner, so a job can read the runner's credentials and alter the runner installation or the workspace that later jobs reuse; anyone who can push a branch can therefore act as the runner. Ephemeral just-in-time runners would remove this, at the price of a repository-administration token on a machine that brokers registrations. Disk use is not capped (no quota without root): a job can fill the shared Docker storage. The proxy lets a job reach any public host, so a malicious maintainer could exfiltrate what the runner sees; the repository is public and the runner holds no secret beyond its own registration.

## Writing GPU jobs

Jobs run directly in the runner container, which is the dev image with the GPU attached: no `docker run` inside a job. Wrap GPU work in `flock /home/admin/autan-gpu/gpu.lock <cmd>`; `autan-bench` takes the lock itself. The runner refuses to start if it cannot open the lock.

## Install or update a runner

Run from a workstation with `gh` logged in as the repository owner and SSH access to the Spark. Replace `spark` with `spark-5359` for the second one.

```sh
rsync -a --delete --exclude .git ./ spark:~/autan-hpc/src/
ssh spark 'cd ~/autan-hpc/src && docker build -t autan-dev:local docker/dev && docker/runner/deploy.sh build autan-dev:local'
gh api -X POST repos/Remenby31/autan/actions/runners/registration-token --jq .token \
  | ssh spark 'cd ~/autan-hpc/src && docker/runner/deploy.sh register'
ssh spark 'cd ~/autan-hpc/src && docker/runner/deploy.sh start'
```

For the `main` runner of spark-5359, prefix `register` and `start` with `AUTAN_RUNNER_INSTANCE=main AUTAN_RUNNER_LABELS=gpu,gb10,bench-main`. `register` is needed once per runner: the credentials live in the `autan-runner-state` volume and survive image updates. To update the dev image a runner uses, rebuild and `start` again (it replaces the `autan-runner` container; a running job is cancelled). Containers restart with the Docker daemon, so runners come back after a reboot without `systemd --user` lingering or sudo. `deploy.sh status` and `deploy.sh logs` show the state; `deploy.sh stop` removes the containers. To retire a runner, also delete it in the repository settings (or `gh api -X DELETE repos/Remenby31/autan/actions/runners/<id>`).
