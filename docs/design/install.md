# Installing Autan

How a user goes from nothing to a working `autan` in one command, what we support, and what we refuse. The evidence behind each choice is in [docs/research/distribution.md](../research/distribution.md). Status: the installer, the wheel pipeline and the clean-machine tests exist as a prototype (`install.sh`, `packaging/`, `.github/workflows/wheels.yml`); the solver core is not in the wheel yet.

## The commands

```sh
curl -LsSf https://remenby31.github.io/autan/install.sh | sh     # Linux, macOS
uv tool install autan                                          # or: pipx install autan, pip install autan
autan doctor
```

The only system requirement is the GPU driver: NVIDIA driver ≥ 580 on Linux for CUDA. No CUDA toolkit, no ROCm, no MPI, no compiler, no root, no system Python.

## The first five minutes

| Time | The user sees | What happens |
|---|---|---|
| 0:00 | `curl … \| sh` | The script is one function called on its last line: a truncated download runs nothing. |
| 0:01 | `autan-install: NVIDIA GPU: NVIDIA GB10, 580.173.02` | Platform checks run before any download: glibc ≥ 2.28, x86_64 or aarch64, macOS ≥ 14 on Apple Silicon. Refusals are one line saying what to use instead. The GPU is reported, never installed. |
| 0:03 | `downloading uv 0.13.6` | A pinned uv is fetched over HTTPS (TLS ≥ 1.2) into `~/.local/share/autan/uv-<version>/` and checked against the SHA-256 written in the script. No checksum tool means no install. The user's own uv, if any, is left alone. |
| 0:05 | `installing autan` | `uv tool install autan` into an isolated environment; uv downloads a managed Python 3.13 if the system lacks it. The command lands in `~/.local/bin`. |
| 0:40 | `restart your shell, or run: export PATH=…` | `uv tool update-shell` adds `~/.local/bin` to the shell profile, unless it is on `PATH` already or `AUTAN_NO_MODIFY_PATH=1`. In GitHub Actions it appends to `$GITHUB_PATH`. |
| 0:45 | `autan doctor` output | Every check prints `ok`, `warn` or `fail` with the fix. A failure never undoes the install. |
| 1:00 | `autan plan case.py` | The user starts work. |

Measured on the prototype: about 40 s on a fresh `ubuntu:24.04` container on the DGX Spark, most of it the Python and NumPy downloads.

Options, all environment variables so `curl | sh` can pass them: `AUTAN_VERSION`, `AUTAN_PYTHON` (default 3.13), `AUTAN_GPU` (`auto`, `cuda`, `cpu`; `cuda` fails without a driver), `AUTAN_FIND_LINKS` (extra wheel directory), `AUTAN_OFFLINE=1` (wheelhouse only, see below), `AUTAN_NO_MODIFY_PATH=1`. Upgrade: rerun the script. Uninstall, with the uv the script installed (users with their own uv can call it instead):

```sh
~/.local/share/autan/uv-*/uv tool uninstall autan     # $XDG_DATA_HOME/autan when it is set
rm -rf ~/.local/share/autan
```

## What `autan doctor` checks

| Check | `fail` / `warn` when | Fix it prints |
|---|---|---|
| python | — | — |
| native | the extension does not import (wrong platform, broken install) | `uv tool install --reinstall autan` |
| driver | driver supports less than CUDA 13.0 | upgrade the NVIDIA driver to ≥ 580 |
| gpuN | the fatbin has no SASS this GPU runs (warn: PTX JIT), or the probe kernel fails | report with the output |
| gpu | AMD GPU found but no ROCm build installed (warn) | the `autan[rocm]` command once published |
| path | the command's directory is not on `PATH` (warn) | the `export` line, or `uv tool update-shell` |

No GPU is information, not a warning: the CPU backend runs and says so, as AGENTS.md requires of every fallback. MPI is checked only when the user asks for more than one node. The prototype implements these checks in `autan-probe`; they move into `autan doctor` when the core ships (see [coordination](#coordination)).

## Platforms

| Platform | Wheel | GPU backend | Status |
|---|---|---|---|
| Linux x86_64, glibc ≥ 2.28 | `cp312-abi3-manylinux_2_28_x86_64` | CUDA, SASS sm_75–sm_120 + PTX | prototype, CI |
| Linux aarch64, glibc ≥ 2.28 (DGX Spark, GH200, GB200) | `cp312-abi3-manylinux_2_28_aarch64` | CUDA, same list | prototype, CI, GPU-tested on GB10 |
| macOS 14+, Apple Silicon | `cp312-abi3-macosx_14_0_arm64` | CPU (NEON); Metal in M1b | prototype, CI |
| Linux x86_64 + AMD Instinct | base wheel + `autan[rocm]` | HIP, per-gfx code objects | after M1 |
| Windows | — | — | via WSL2 only |

One wheel per platform serves Python 3.12 and later (stable ABI). The architecture list is set once, in `packaging/probe/pyproject.toml` (`AUTAN_CUDA_ARCHITECTURES`), which feeds both the build and the check that refuses a wheel missing one of them.

## How the wheel finds CUDA

```
site-packages/
  autan/_native.abi3.so        RUNPATH $ORIGIN/../nvidia/cu13/lib, NEEDED libcudart.so.13
  nvidia/cu13/lib/libcudart.so.13   from the nvidia-cuda-runtime wheel (Linux only)
/usr/lib/…/libcuda.so.1        from the driver
```

`auditwheel repair --exclude` keeps every NVIDIA library out of our wheel; `packaging/probe/check_wheel.py` fails the build if the wheel holds any shared library besides our extension modules, lacks the licence of nanobind (linked statically), exceeds 90 MB, or misses a listed architecture in the fatbin. Later CUDA libraries (NVRTC for user kernels, `@autan.kernel`) arrive the same way, as `nvidia-*` dependencies with a `sys_platform == 'linux'` marker.

## Release pipeline

1. `wheels.yml` builds the SDK wheel and sdist, and the native wheels with cibuildwheel on `ubuntu-24.04`, `ubuntu-24.04-arm` and `macos-15`.
2. The same workflow installs them with `install.sh` on fresh `ubuntu:24.04`, `debian:12`, `rockylinux:8` containers and a macOS runner with an empty home, pinned to the version under test and checked after install; builds the offline wheelhouse and installs it with networking disabled; checks that `alpine` and `ubuntu:18.04` are refused before any download; and, on a self-hosted Spark, runs the GPU probe in a container that sees only the driver, under the GPU lock.
3. `release-testpypi.yml` (manual) publishes a `.devN` build to TestPyPI with Trusted Publishing and PEP 740 attestations, attests every file and `install.sh` with GitHub artifact attestations, then installs it from TestPyPI on Linux and macOS.
4. The real release (not written yet, lead's approval required) follows the same path to PyPI on a release-please tag, uploads `install.sh`, the wheels and their `SHA256SUMS` to an immutable GitHub release, and publishes the runtime image.

A cautious user verifies before running:

```sh
curl -LsSfO https://github.com/Remenby31/autan/releases/latest/download/install.sh
gh attestation verify install.sh -R Remenby31/autan
sh install.sh
```

## Secondary channels

| Channel | Shape | When |
|---|---|---|
| Container | `ghcr.io/remenby31/autan-runtime:<version>`, `ubuntu:24.04` + uv + the wheel, amd64 and arm64, attested; GPU through `--gpus all`; no `nvidia/cuda` base needed | with the first PyPI release |
| Homebrew | tap `Remenby31/homebrew-autan`, `Language::Python::Virtualenv` formula built from the sdist; macOS only (CPU, later Metal) | with the first PyPI release; homebrew-core once stable |
| Offline wheelhouse | `packaging/wheelhouse.sh` builds, per platform: `install.sh`, the pinned uv archive, the Python build that uv picks laid out as a uv Python mirror, and every wheel autan needs (67 MB on aarch64 today). `AUTAN_OFFLINE=1 AUTAN_FIND_LINKS=<dir> sh <dir>/install.sh` installs with no network. | prototype, tested in CI in a container with networking disabled; published as `autan-wheelhouse-<version>-<target>.tar.gz` with the first PyPI release |
| conda-forge | feedstock with a `__cuda` CUDA variant | after v1 |

## What we refuse

- **Bundling NVIDIA or AMD libraries in our wheels.** Their licences are proprietary; we depend on NVIDIA's wheels and AMD's index instead.
- **sudo, driver installation, system packages.** ollama installs CUDA drivers with sudo; a solver installer has no business touching the kernel. `doctor` says what to install.
- **A silent CPU fallback.** Running without a GPU is reported by the installer and by every run.
- **A silently skipped checksum.** uv's script skips it without `sha256sum`; ours stops.
- **musl (Alpine), glibc < 2.28, Intel Macs, native Windows.** Each is refused with one line naming the alternative (glibc distribution, container image, WSL2).
- **Wheel variants, for now.** PEP 825 is provisional, pip and PyPI do not support them, uv only in preview. We revisit once pip or stable uv installs them from PyPI.
- **A CUDA toolkit, ROCm or MPI as install prerequisites.** Multi-node runs use the cluster's own GPU-aware MPI.
- **Telemetry in the installer.**

## Open decisions

1. **TestPyPI Trusted Publisher** for `autan`: owner `Remenby31`, repository `autan`, workflow `release-testpypi.yml`, environment `testpypi` (the GitHub environment exists, protected branches only). Needs the lead's TestPyPI account; until then step 3 of the pipeline fails at upload.
2. **PyPI organization** for `autan`, raising the limits to 200 MB per file and 20 GB per project: at the first real release.
3. **Hosting `install.sh`** at `remenby31.github.io/autan/install.sh`, copied from the latest release by the docs site build.
4. **Spark runner gate:** the GPU job runs when the repository variable `AUTAN_SPARK_RUNNER` is `true`.

Depending on the proprietary `nvidia-*` wheels, never bundling them, is the licence exception recorded in [AGENTS.md](../../AGENTS.md#design). Keep CUDA runtime calls behind one seam in the core so that a driver-API-only build (`libcuda.so.1`, no runtime wheel) stays possible.

## Coordination

| Lane | Proposal |
|---|---|
| SDK & contrats (`pyproject.toml`, `py/`) | Switch the build backend to scikit-build-core with `wheel.py-api = "cp312"`, the `[tool.cibuildwheel]` table and the `nvidia-cuda-runtime` dependency from `packaging/probe/pyproject.toml`; then `wheels.yml` builds the root project and the probe is deleted. `autan doctor`: absorb the probe's driver, fatbin and path checks; on macOS, report the CPU backend instead of "no NVIDIA GPU"; stop warning about MPI on a single node, since the fix it gives tells users to install MPI. |
| Noyau maillage (`CMakeLists.txt`, `core/`) | A `python/` target built with `nanobind_add_module(STABLE_ABI NB_STATIC)` when scikit-build-core drives CMake; CUDA runtime `Shared` with `INSTALL_RPATH $ORIGIN/../nvidia/cu13/lib`; architectures from `AUTAN_CUDA_ARCHITECTURES`. |
| HPC Spark (`docker/`, runners) | Runner labels `self-hosted, linux, arm64, gpu, gb10` as in issue #4; set `AUTAN_SPARK_RUNNER=true` when they run. The runtime image belongs in `packaging/` unless you prefer `docker/runtime/`. |
| Standards du dépôt | `wheels.yml` and `release-*.yml` follow the existing rules (SHA pins, `permissions: {}`, zizmor). The GPU job uses the runner labels already declared in `.github/actionlint.yaml`. |
| Docs site | Serve `install.sh` from the latest release at the site root. |
