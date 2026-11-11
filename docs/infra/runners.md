# GPU runners on the DGX Spark

Each Spark (`spark`, `spark-5359`) runs one GitHub Actions runner registered on `Remenby31/autan` with the labels `self-hosted, linux, ARM64, gpu, gb10`. GPU workflows (`.github/workflows/gpu-*.yml`) target them with `runs-on: [self-hosted, linux, ARM64, gpu, gb10]`. Timed benchmarks add `bench`, a label only spark-5359 carries: spark's GPU is used continuously by another team's process, which the idle check rightly refuses. Register with `AUTAN_RUNNER_LABELS=gpu,gb10,bench` to give a runner that label, or add it with `gh api -X POST repos/Remenby31/autan/actions/runners/<id>/labels -f 'labels[]=bench'`. spark-5359 runs a second runner process for `main`, labelled `bench-main` (`AUTAN_RUNNER_INSTANCE=main AUTAN_RUNNER_LABELS=bench-main`, container `autan-runner-main`; `bench-main` is its only label, so no other job lands on it): `gpu-bench` runs on `main` target it, so they never queue behind pull requests in GitHub, and on the machine the GPU lock orders them, `main` first ([bench.md](bench.md#the-gpu-lock)).

## Threat model

The repository is public, so anyone can open a pull request, and a pull request from a fork can rewrite any workflow file, including its `runs-on` and `if:`. A job that reaches a self-hosted runner executes arbitrary code on a machine that sits on the company tailnet, shares its GPU and Docker daemon with other teams, and keeps state between jobs. The assets at risk are, in order: the tailnet and LAN behind the Spark, the other teams' containers and data on the host, the runner's registration credential, and the integrity of our benchmark results.

## Protections

| Layer | Protection | Enforced by |
|---|---|---|
| Trigger | Only `push`, `workflow_dispatch`, `schedule`, and `pull_request` whose head is a branch of this repository run. Fork pull requests, `pull_request_target`, `workflow_run`, `issue_comment` and everything else are refused before the first step: the hook kills the job's `Runner.Worker`, because a failing hook alone would still let `if: always()` steps run. The allowed repository comes from a root-owned file in the image, never from the environment a workflow controls. | the job-started hook `docker/runner/job-started.sh`, baked into the runner image; a workflow cannot change or skip it |
| Trigger | GPU jobs carry `if: github.event_name != 'pull_request' \|\| github.event.pull_request.head.repo.full_name == github.repository`, so fork runs are skipped rather than failed. | the workflow (convenience only: a fork can edit it) |
| Trigger | Workflows from first-time and external contributors wait for a maintainer's approval (`fork-pr-contributor-approval: all_external_contributors`). Never approve a fork run that touches `.github/`. | repository setting |
| Host | The runner runs in a container as an unprivileged user (uid 1001), with every capability dropped, `no-new-privileges`, 64 GiB memory (less on a shared Spark, below), 8192 processes and rotated logs. No Docker socket, no host mount except its own volumes and, read-only, the GPU lock's directory. A job cannot see or touch other containers. | `docker/runner/deploy.sh` |
| Network | The runner sits on an internal Docker network with no route out and an isolated gateway (the host has no address on it). Its only egress is a Squid proxy that allows ports 80 and 443 to public addresses and refuses, after DNS resolution, loopback, RFC 1918, Tailscale (`100.64.0.0/10`, `fd7a:115c:a1e0::/48`), link-local and multicast destinations. `deploy.sh` replaces a proxy whose image is out of date and refuses an internal network without these properties. | `docker/runner/proxy/squid.conf`, `deploy.sh` |
| Token | Workflows run with `permissions: contents: read` and `actions/checkout` with `persist-credentials: false`. The registration token is single-use and expires after one hour; it is piped on stdin and handed to the runner through its environment, never on a command line or on disk. | workflows, `deploy.sh register` |
| Supply chain | Third-party actions are pinned by commit SHA; images by digest; downloads by SHA-256. | workflow and Dockerfile review |

Checked on the Sparks: a temporary workflow whose job had an `if: always()`, an `if: failure()` and a plain step ran on a runner configured to refuse it; the job failed with no step executed. The same workflow on a runner that accepts it ran its steps (positive control). From a container on the runner's network: GitHub and PyPI answer through the proxy (200); direct HTTPS and the host's addresses are unreachable; through the proxy, a Tailscale peer, `100.100.100.100`, the LAN, `169.254.169.254`, `localhost`, IPv6 and IPv4-mapped literals, public names that resolve to private addresses (`*.nip.io`) and non-web ports are refused. The hook's decisions are tested in `docker/runner/tests/` with event payloads for every trigger.

Residual risks, accepted because only trusted triggers run code: jobs run as the same user as the runner, so a job can read the runner's credentials and alter the runner installation or the workspace that later jobs reuse; anyone who can push a branch can therefore act as the runner. Ephemeral just-in-time runners would remove this, at the price of a repository-administration token on a machine that brokers registrations. Disk use is not capped (no quota without root): a job can fill the shared Docker storage. The proxy lets a job reach any public host, so a malicious maintainer could exfiltrate what the runner sees; the repository is public and the runner holds no secret beyond its own registration.

## Writing GPU jobs

Jobs run directly in the runner container, which is the dev image with the GPU attached: no `docker run` inside a job. Wrap GPU work in `flock /home/admin/autan-gpu/gpu.lock <cmd>`; `autan-bench` takes the lock itself. The runner refuses to start if it cannot open the lock.

## Where gpu-tests runs its CPU tests

The Spark runs gpu-tests' build, GPU tests and generator only. The CPU tests (`-LE 'gpu|mpi'` with `-j 4`, then `-L mpi -LE gpu` with `-j 1`) run on hosted `ubuntu-24.04-arm` runners, in job `cpu-tests`:
- **Image and build.** Each runner builds the tree itself with the Spark job's configuration (the workflow's `CONFIGURE`). It runs in the dev image `gpu-image.yml` published, pinned by digest.
- **Shards.** `bench/ci/cpu-test-shards.py` deals the tests into three shards by name, and job `cpu-tests-check` checks across the three runners that each CPU test ran exactly once.
- **MPI settings.** Open MPI runs as root with oversubscription, through environment variables, so the test commands are the Spark's.

The Spark still runs the whole CPU suite on GB10, last, on the nightly and on a manual run (`workflow_dispatch`). When that step does not pass on the nightly (failed, cancelled, timed out, or never reached), the report opens an issue labelled `gb10-cpu-difference`; later nights comment on the open one.

Every hosted CPU shard then fails, as a red gate does, and so does every merge. `bench/ci/gb10-health.py` blocks the shards while any of these holds:
- the issue is open;
- the last nightly did not pass its GB10 CPU step and no such issue was closed after it (this covers a report that could not open the issue);
- GitHub's API cannot be read.

To clear it:
1. Fix the difference.
2. Run gpu-tests by hand on the fix's branch: that runs the CPU tests on GB10.
3. Close the issue once that step is green. If the report opened none, open one with the label and close it, to record the check.

When `docker/dev` changes, the shards warn that the pinned image is older. Once `gpu-image.yml` has published the new image on `main`, pin its arm64 digest in `cpu-tests`; the Spark runner's image is redeployed the same way (below).

## On a Spark shared with other teams

The runner of spark-5512 (`spark`) is a guest on a host whose other teams come first. Its placement comes from the environment of every `deploy.sh` command:

| Variable | spark-5512 | Effect |
|---|---|---|
| `AUTAN_RUNNER_HOME` | `~/cfd/autan-runner` | the lock's directory (`autan-gpu/`) under it, mounted where jobs expect it (`/home/admin/autan-gpu`); no guard cron, the host's crontab is not ours |
| `AUTAN_RUNNER_CPUSET` | `10-18` | `--cpuset-cpus` for the runner and its proxy: cores 0–9 belong to another team, core 19 to a campaign |
| `AUTAN_RUNNER_MEMORY` | `16g` | `--memory` and `--memory-swap` |
| `AUTAN_RUNNER_NICE` | `10` | the runner, and so every job step, runs at this niceness; without `CAP_SYS_NICE` no job can lower it |
| `AUTAN_RUNNER_BUILD_JOBS` | `6` | `CMAKE_BUILD_PARALLEL_LEVEL` for the jobs: a CUDA compiler process takes up to 2 GB, so 6 fit in 16 GiB |
| `AUTAN_RUNNER_MIN_AVAILABLE_GIB` (`build`) | `30` | the job-started hook holds each job until the host's `MemAvailable` reaches 30 GiB, at most 15 minutes, then refuses it before its first step; the value is baked root-owned into the image, so no workflow changes it |

On spark-5512 every `deploy.sh` command goes through `~/cfd/autan-runner/runner.sh`, which exports these values and runs `deploy.sh` from its checkout in `~/cfd/autan-runner/src`: `deploy.sh` run directly there would start an unconfined runner (all cores, 64 GiB, nice 0, no memory floor). `deploy.sh` refuses `AUTAN_RUNNER_HOME` on a host that already has `/home/admin/autan-gpu/gpu.lock`: two locks would let two jobs hold the GPU. The hook's wait counts against the job's `timeout-minutes` (90 for `gpu-tests`, whose jobs take 10 to 25 minutes). The floor binds every job as GitHub starts it; like every runner setting, it does not survive a job that rewrites the runner's installation, which the threat model's residual risks already accept for anyone who can push a branch.

**Stacks of a hung test.** The hang watchdog (`bench/ci/ctest-watchdog.sh`) attaches `gdb` to every process of a test that hangs. The Sparks' Yama `ptrace_scope` is 1, so a debugger may attach only to its own descendants, and the runner user has no `sudo`. The runner's image therefore installs `gdb` and builds `libautan_ptracer.so` (`docker/runner/ptracer.c`), which the watchdog preloads into the test command it starts, and only there: the test's processes let a process of the same user trace them. The runner agent never preloads it, so a job step gains no new way to trace the agent; what a job can already reach as the same user is the threat model above.

**A runner that lost its GPU.** A reload of the host's systemd (any package or unit install) can revoke a running container's GPU: `nvidia-smi` in it answers `Failed to initialize NVML: Unknown Error`. `deploy.sh` passes the GPU's device nodes explicitly to prevent it (NVIDIA's workaround; some `runc` versions still lose them), and the job-started hook refuses every job of such a runner with "RUNNER LOST THE GPU" before its first step. Restart the runner: on spark-5512 and spark-75d6 `~/cfd/autan-runner/runner.sh start`, elsewhere `deploy.sh start` with the instance's variables (`AUTAN_RUNNER_INSTANCE=main …` for a `main` runner).

Everything it creates is named `autan-runner*`: the containers, the networks, the images, and its two volumes (`autan-runner-state`, `autan-runner-work`), which stay Docker volumes because the runner's uid cannot write a directory of the host's user without root. Removing it: `deploy.sh stop`, `docker volume rm autan-runner-state autan-runner-work`, `docker network rm autan-runner-internal autan-runner-egress`, `docker image rm autan-runner:local autan-runner-proxy:local`, `rm -rf ~/cfd/autan-runner`, and deleting the runner in the repository settings.

## Install or update a runner

Run from a workstation with `gh` logged in as the repository owner and SSH access to the Spark. Replace `spark` with `spark-5359` for the second one.

```sh
rsync -a --delete --exclude .git ./ spark:~/autan-hpc/src/
ssh spark 'cd ~/autan-hpc/src && docker build -t autan-dev:local docker/dev && docker/runner/deploy.sh build autan-dev:local'
gh api -X POST repos/Remenby31/autan/actions/runners/registration-token --jq .token \
  | ssh spark 'cd ~/autan-hpc/src && docker/runner/deploy.sh register'
ssh spark 'cd ~/autan-hpc/src && docker/runner/deploy.sh start'
```

On spark-5512, run `~/cfd/autan-runner/runner.sh build autan-dev:local`, `register` and `start` instead (above). For the `main` runner of spark-5359, prefix `register` and `start` with `AUTAN_RUNNER_INSTANCE=main AUTAN_RUNNER_LABELS=bench-main`. `register` is needed once per runner: the credentials live in the `autan-runner-state` volume and survive image updates. To update the dev image a runner uses, rebuild and `start` again (it replaces the `autan-runner` container; a running job is cancelled). Containers restart with the Docker daemon, so runners come back after a reboot without `systemd --user` lingering or sudo. `deploy.sh status` and `deploy.sh logs` show the state; `deploy.sh stop` removes the containers. To retire a runner, also delete it in the repository settings (or `gh api -X DELETE repos/Remenby31/autan/actions/runners/<id>`).
