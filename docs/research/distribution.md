# Distribution: how GPU Python packages ship in 2026

Checked 2026-10-31. Sizes and versions come from the PyPI JSON API on that date; three wheels (`jax-cuda13-plugin` 0.12.0.1, `cupy-cuda13x` 14.2.0, `nvidia-cuda-runtime` 13.4.92) were downloaded and their `DT_NEEDED`, `RUNPATH` and fatbin entries read directly. Installer scripts were read in full. "Unverified" marks a claim no primary source confirmed. Facts we measured ourselves on a DGX Spark are marked **measured**. The design that follows from this page is [docs/design/install.md](../design/install.md).

## Decisions

| Question | Decision | Why (section) |
|---|---|---|
| Selecting the GPU build | One `autan` distribution; the Linux wheels carry the CUDA code; AMD through an extra later; no wheel variants yet | Variants are not installable from PyPI with pip or stable uv ([§5](#5-wheel-variants)) |
| CUDA runtime | Link `libcudart.so.13` dynamically, found through `RUNPATH $ORIGIN/../nvidia/cu13/lib` from the `nvidia-cuda-runtime` wheel; only the driver comes from the system | JAX's layout ([§1](#1-how-the-gpu-projects-ship)); NVIDIA libraries carry a proprietary licence, so we depend on them and never bundle them, as AGENTS.md allows ([§2](#2-cuda-13-facts-that-shape-the-wheel)) |
| Architectures | SASS for sm_75, 80, 86, 89, 90, 100, 110, 120 and PTX for compute_120 | CUDA 13 floor is sm_75; sm_120 SASS runs on GB10 (sm_121), **measured** ([§2](#2-cuda-13-facts-that-shape-the-wheel)) |
| Wheel shape | `cp312-abi3` (nanobind stable ABI), `manylinux_2_28` x86_64 and aarch64, `macosx_14_0_arm64` | One wheel per platform instead of one per Python version ([§3](#3-build-tooling)) |
| Size | ≤ 90 MB per file, checked in CI; split the fatbin into a `py3-none` wheel or ask for a limit increase when it grows | PyPI's 100 MB default ([§4](#4-pypi-limits-and-wheel-size)) |
| Build | scikit-build-core + nanobind + cibuildwheel; nvcc installed from NVIDIA's RHEL 8 repository in `before-all`; `auditwheel repair --exclude` the NVIDIA libraries | [§3](#3-build-tooling) |
| `curl \| sh` | Our own POSIX script: pinned uv with embedded SHA-256, then `uv tool install autan` | Python apps with compiled extensions cannot use cargo-dist's installer ([§6](#6-installer-scripts)) |
| Provenance | Trusted Publishing with PEP 740 attestations, GitHub artifact attestations on every asset, immutable releases | [§7](#7-supply-chain-security) |
| AMD | `autan[rocm]` after M1, pulling a HIP plugin; the ROCm runtime comes from AMD's index, not PyPI | [§8](#8-amd-rocm) |
| Other channels | GHCR image without a CUDA base image, Homebrew tap, offline wheelhouse; conda-forge after v1 | [§9](#9-secondary-channels) |

## 1. How the GPU projects ship

| Project | Packages | CUDA libraries | How they are found | Architectures (CUDA 13) |
|---|---|---|---|---|
| JAX 0.12 | `jax` (pure) + `jaxlib` + `jax-cuda13-plugin` (8.2 MB) + `jax-cuda13-pjrt` (`py3-none`, 135 MB); extra `jax[cuda13]` | `nvidia-*` wheels, lower bounds only; `jax[cuda13-local]` uses a system toolkit | `RUNPATH $ORIGIN/../../nvidia/cu13/lib`, libraries opened with `dlopen`, none in `DT_NEEDED` | sm_75, 80, 90, 100, 110 + compute_120 |
| PyTorch 2.15 | `torch` on PyPI is the CUDA 13.2 build, 539 MB x86_64; other builds on `download.pytorch.org/whl/<backend>` | `cuda-toolkit[...]==13.2.2` and `nvidia-*-cu13`, exact pins, `platform_system=="Linux"` | Links them, preloads `nvidia/cu13/lib/*` with `ctypes` when the loader fails | x86_64 75, 80, 86, 90, 100, 120; aarch64 80, 90, 100, 110, 120; no PTX in releases |
| CuPy 14.2 | `cupy-cuda13x` (70 MB); extra `[ctk]` adds the toolkit wheels | Optional | `cuda-pathfinder` (Apache-2.0) | 75, 80, 86, 89, 90, 100f, 120f + PTX 120 |
| RAPIDS 26.10 | thin `cudf-cu13` (`cp311-abi3`, 2.2 MB) over `libcudf-cu13` (`py3-none`, 301 MB), now on pypi.org | `cupy-cuda13x[ctk]`, `cuda-bindings` | Library wheels | Unverified |
| Triton 3.9 | `triton` (`cp312-abi3`, 286 MB) | Bundles ptxas and friends | JIT only | — |

Sources: [jax](https://pypi.org/pypi/jax/json), [jax-cuda13-plugin](https://pypi.org/pypi/jax-cuda13-plugin/json), [jax-cuda13-pjrt](https://pypi.org/pypi/jax-cuda13-pjrt/json), [JAX `.bazelrc`](https://github.com/jax-ml/jax/blob/main/.bazelrc), [JAX install guide](https://docs.jax.dev/en/latest/installation.html); [torch](https://pypi.org/pypi/torch/json), [PyTorch RELEASE.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md), [build_env_setup.py](https://github.com/pytorch/pytorch/blob/adffc04f041945d8bd377145937124343bce24bc/.ci/wheel/linux/build_env_setup.py), [torch/\_\_init\_\_.py](https://github.com/pytorch/pytorch/blob/adffc04f041945d8bd377145937124343bce24bc/torch/__init__.py); [cupy-cuda13x](https://pypi.org/pypi/cupy-cuda13x/json), [CuPy `_compiler.py`](https://github.com/cupy/cupy/blob/f4f7f7e461b647743b13b37775c2121885c844c2/install/cupy_builder/_compiler.py), [CuPy `_environment.py`](https://github.com/cupy/cupy/blob/f4f7f7e461b647743b13b37775c2121885c844c2/cupy/_environment.py); [cudf-cu13](https://pypi.org/pypi/cudf-cu13/json), [libcudf-cu13](https://pypi.org/pypi/libcudf-cu13/json); [triton](https://pypi.org/pypi/triton/json), [Triton toolchain pins](https://github.com/triton-lang/triton/blob/main/cmake/nvidia-toolchain-version.json).

What they share: nobody asks for a CUDA toolkit; the NVIDIA libraries arrive as wheels and are found relative to the extension, so `auditwheel` has nothing to graft; JAX and RAPIDS put the heavy binary in one `py3-none` wheel per platform and keep the Python-facing wheel small.

## 2. CUDA 13 facts that shape the wheel

- CUDA 13.0 dropped Maxwell, Pascal and Volta: the floor is sm_75 ([release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)). nvcc 13.4 accepts sm_75 to sm_121, including sm_103, sm_107, sm_110 and the `f`/`a` suffixes ([nvcc guide](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)).
- Minor version compatibility: any CUDA 13.x application runs on a driver ≥ 580. PTX produced by a newer toolkit than the driver does not JIT ([minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)). **Measured:** our probe built with nvcc 13.0, linked against `nvidia-cuda-runtime` 13.4.92, runs on GB10 with driver 580.173.02 (CUDA 13.0).
- A cubin for X.y loads on X.z for z ≥ y ([programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)). GB10 is 12.1, RTX PRO 6000 Blackwell 12.0, B200 10.0 ([CUDA GPUs](https://developer.nvidia.com/cuda-gpus)). **Measured:** with `CUDA_DISABLE_PTX_JIT=1`, the sm_120 SASS of the probe runs on GB10, so sm_121 needs no SASS of its own.
- The CUDA 13 runtime wheels dropped the `-cuXX` suffix: `nvidia-cuda-runtime` 13.4.92 (2.5 MB), `nvidia-cuda-nvrtc` (53 MB), `nvidia-nvjitlink` (43 MB), `nvidia-cublas` (439–553 MB). The `-cu13` names are 0.0.1 placeholders. They install to `site-packages/nvidia/cu13/lib` ([nvidia-cuda-runtime](https://pypi.org/pypi/nvidia-cuda-runtime/json), [nvidia-cuda-nvrtc](https://pypi.org/pypi/nvidia-cuda-nvrtc/json), [nvidia-cublas](https://pypi.org/pypi/nvidia-cublas/json)).
- Their licence is `LicenseRef-NVIDIA-Proprietary` ([nvidia-cuda-nvcc](https://pypi.org/pypi/nvidia-cuda-nvcc/json)). AGENTS.md lets us ship only permissive or weak-copyleft code, so a static `cudart` inside our wheel is out; declaring these wheels as dependencies is the one exception AGENTS.md allows.
- nvcc `--compress-mode=size` needs a driver from CUDA 12.4 or later, which every CUDA 13 driver is ([nvcc guide](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)); PyTorch uses it with `-Xfatbin -compress-all` on CUDA 13.

## 3. Build tooling

- cibuildwheel 4.3.5 (2026-10-27): default image `manylinux_2_28` (AlmaLinux 8), `before-all` runs once per container, Linux repair is `auditwheel repair` (macOS has its own repair tool), abi3 wheels pass `abi3audit --strict` ([options](https://cibuildwheel.pypa.io/en/stable/options/)). Its FAQ points to prebuilt `quay.io/manylinux_cuda/...` images, documented up to CUDA 13.1 ([FAQ](https://github.com/pypa/cibuildwheel/blob/main/docs/faq.md)). We install `cuda-nvcc-13-0` and `cuda-cudart-devel-13-0` from NVIDIA's `rhel8` repository instead (sbsa on aarch64): about 120 MB, no third-party image. **Measured:** 30 s on the Spark.
- scikit-build-core 1.1.1: `wheel.py-api = "cp312"` gives `cp312-abi3`; `cmake.define` reads environment variables ([configuration](https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html)). It does no wheel repair and notes that the repair tools forbid cross-wheel library links, so relative `RUNPATH` is set by hand ([dynamic linking](https://github.com/scikit-build/scikit-build-core/blob/main/docs/guide/dynamic_link.md)).
- nanobind 3.1.0: `STABLE_ABI` needs CPython ≥ 3.12 ([CMake API](https://github.com/wjakob/nanobind/blob/master/docs/api_cmake.rst)). The SDK already requires 3.12.
- auditwheel 6.8.2 has no CUDA special case: `libcuda` is not in its policy whitelist, so every NVIDIA library is passed to `--exclude` ([policy](https://github.com/pypa/auditwheel/blob/main/src/auditwheel/policy/manylinux-policy.json), [main_repair.py](https://github.com/pypa/auditwheel/blob/main/src/auditwheel/main_repair.py)). **Measured:** with `--exclude libcuda.so.1 --exclude libcudart.so.13`, the repaired wheel keeps `RUNPATH $ORIGIN/../nvidia/cu13/lib` and is tagged `manylinux_2_26.manylinux_2_28`.
- Tags in use: torch `manylinux_2_28`, jaxlib `manylinux_2_27`, cupy `manylinux2014`; nobody ships `manylinux_2_34` ([manylinux README](https://github.com/pypa/manylinux/blob/main/README.rst)). macOS arm64: jaxlib `macosx_11_0`, torch and numpy (Accelerate) `macosx_14_0` ([numpy 2.0 notes](https://github.com/numpy/numpy/blob/main/doc/source/release/2.0.0-notes.rst)).
- GitHub runners: `ubuntu-24.04-arm` and `macos-15` (arm64) are free for public repositories; GPU runners are paid larger runners only ([runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)). GPU tests therefore run on our self-hosted Sparks.

## 4. PyPI limits and wheel size

- 100 MB per file and 10 GB per project by default; 200 MB and 20 GB for a project owned by a PyPI organization. Increases go through a `pypi/support` issue after one release under the current limit. Yanking frees nothing; deleting does ([storage limits](https://docs.pypi.org/project-management/storage-limits/)).
- Over 100 MB today, all with raised limits: torch 539 MB, nvidia-cublas 553 MB, libcudf-cu13 301 MB, triton 286 MB, jax-cuda13-pjrt 135 MB. Under: jaxlib 87 MB, cupy-cuda13x 70 MB.
- What drives size is (architectures × kernels) per wheel, times wheels per release. abi3 removes the Python-version factor; CuPy halved its aarch64 wheel (147 → 73 MB) by dropping pre-Turing architectures with CUDA 13.
- **Measured:** the probe kernel with 8 SASS + 1 PTX weighs 0.1 MB compressed. A real solver will carry hundreds of kernels in up to three precisions; the 90 MB CI budget tells us when to split.

## 5. Wheel variants

- PEP 817 (overview) is Draft; PEP 825 "Wheel Variants: Package Format" was provisionally accepted on 2026-09-05, final only once the whole suite is accepted; PEP 850 "Providers" is Draft since 2026-10-15 ([PEP 817](https://peps.python.org/pep-0817/), [PEP 825](https://peps.python.org/pep-0825/), [acceptance](https://discuss.python.org/t/pep-825-wheel-variants-package-format-split-from-pep-817/106196/232), [PEP 850](https://peps.python.org/pep-0850/)).
- uv: preview feature `wheel-variants` since 0.13.1 (2026-10-14); NVIDIA detection since 0.13.6 (2026-10-26); no ROCm provider; no universal lockfiles or direct URLs ([uv#22039](https://github.com/astral-sh/uv/pull/22039), [uv#22690](https://github.com/astral-sh/uv/pull/22690), [changelog](https://github.com/astral-sh/uv/blob/main/CHANGELOG.md)).
- pip 26.2.2 has no variant support; `packaging` still rejects variant filenames ([packaging#1148](https://github.com/pypa/packaging/issues/1148)). We found no PyPI upload support (unverified by upload).
- PyTorch's variant wheels (2.8–2.10) lived on test indexes in a pre-PEP-825 format ([PyTorch blog](https://pytorch.org/blog/pytorch-wheel-variants/)).
- `uv --torch-backend=auto` detects the driver but applies to a hard-coded list of PyTorch packages ([backend.rs](https://github.com/astral-sh/uv/blob/main/crates/uv-torch/src/backend.rs)); `[tool.uv.sources]` are not in published metadata ([uv dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)).
- Conclusion: variants are a 2027 option at the earliest. What works today with pip, pipx and `uv tool install` is extras (JAX), separate names (CuPy, which conflict when both are installed) or separate indexes (PyTorch), and an installer script that detects the GPU (ollama, [install.sh](https://github.com/ollama/ollama/blob/main/scripts/install.sh)).

## 6. Installer scripts

| Script | Shell | Partial-download guard | Checksum | PATH opt-out |
|---|---|---|---|---|
| uv (cargo-dist 0.32) | POSIX `sh` | yes, single call on the last line | SHA-256 embedded per archive, **skipped silently without `sha256sum`** | `UV_NO_MODIFY_PATH` |
| pixi | POSIX `sh` | yes, `__wrap__` | none | `PIXI_NO_PATH_UPDATE` |
| rustup | POSIX `sh` | yes | none; forces `--proto '=https' --tlsv1.2` | flag |
| Bun | bash | **no** | **none** | none |
| Deno | POSIX `sh` | no | none | `-y` |

Sources: [uv install.sh](https://astral.sh/uv/install.sh), [uv installer reference](https://docs.astral.sh/uv/reference/installer/), [pixi](https://pixi.sh/install.sh), [rustup](https://sh.rustup.rs), [Bun](https://bun.sh/install), [Deno](https://deno.land/install.sh).

- cargo-dist's shell installer keeps only the binary of its archive ([dist shell installer](https://axodotdev.github.io/cargo-dist/book/installers/shell.html)), so it cannot install a Python package with an extension.
- The working pattern for Python applications is "fetch uv, then `uv tool install`": aider's script is cargo-dist's uv installer plus `uv tool install --force --python python3.12 aider-chat@latest` ([aider](https://aider.chat/install.sh)); Hermes Agent pins uv per platform with its SHA-256 ([Hermes](https://hermes-agent.nousresearch.com/install.sh)).
- uv's conventions worth copying: `~/.local/bin` (XDG), no prompt, atomic move into place, `$GITHUB_PATH` in Actions, and an `env` file sourced from shell profiles.

## 7. Supply-chain security

- Risks of `curl | sh`: a truncated stream runs half a line (fixed by the function wrapper, [Sandstorm](https://sandstorm.io/news/2015-09-24-is-curl-bash-insecure-pgp-verified-install)); a server can detect the pipe and serve other content ([idontplaydarts](https://www.idontplaydarts.com/2016/04/detecting-curl-pipe-bash-server-side/)), which is why the script must also be downloadable, readable and verifiable as a file.
- An embedded checksum binds the downloaded binary to the script, not the script to us. That needs a signature checked out of band: GitHub artifact attestations (`actions/attest` v4.2.2, verified with `gh attestation verify FILE -R Remenby31/autan`), which by themselves give SLSA Build L2, L3 when produced inside a reusable workflow ([attestations](https://docs.github.com/en/actions/concepts/security/artifact-attestations), [SLSA levels](https://slsa.dev/spec/v1.0/levels)).
- Immutable releases (GA 2025-10-28): tags cannot move and assets cannot change; GitHub attests the release; `gh release verify-asset TAG FILE` checks it ([changelog](https://github.blog/changelog/2025-10-28-immutable-releases-are-now-generally-available/), [docs](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/immutable-releases)). uv's own releases are immutable and attested.
- PyPI Trusted Publishing exchanges an OIDC token for a 15-minute upload token; a GitHub environment is strongly recommended ([trusted publishers](https://docs.pypi.org/trusted-publishers/)). `pypa/gh-action-pypi-publish` ≥ 1.11 uploads PEP 740 attestations by default; v1.14.3 is current; TestPyPI uses `repository-url: https://test.pypi.org/legacy/` ([action](https://github.com/pypa/gh-action-pypi-publish)). Neither pip nor uv verifies them at install time ([uv#9122](https://github.com/astral-sh/uv/issues/9122)); `pypi-attestations` does by hand ([consuming attestations](https://docs.pypi.org/attestations/consuming-attestations/)).
- Same-repository reusable workflows can be called as `$/.github/workflows/x.yml`, pinned to the running commit (GitHub, 2026-07-30, runner ≥ 2.336.0) ([changelog](https://github.blog/changelog/2026-07-30-reference-same-repository-actions-with-self-repository-syntax/)).

## 8. AMD ROCm

- ROCm is not on PyPI: AMD's TheRock project publishes `rocm[libraries,device-gfxNNN]` on `repo.amd.com` and `stable.repo.amd.com/rocm/whl-next/`; `rocm` and `rocm-sdk` on PyPI are placeholders ([TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md), [pypi rocm](https://pypi.org/pypi/rocm/json)).
- Device packages are per GPU and large: gfx942 (MI300X) 1.6 GB, gfx1201 338 MB, `rocm-sdk-core` 415 MB ([whl-multi-arch](https://repo.amd.com/rocm/whl-multi-arch/)). ROCm code objects have no PTX-like forward compatibility.
- PyTorch's ROCm wheel (1.73 GB) is on its own index and depends on `rocm[device-all,libraries]==7.14.*`; JAX's `jax-rocm7-plugin` is on PyPI but its `with-rocm` dependencies are not, so users add `--extra-index-url` ([JAX install](https://docs.jax.dev/en/latest/installation.html)). Both are x86_64 only.
- PyPI rejects direct-URL dependencies in uploaded metadata (unverified here), so `autan[rocm]` cannot pull AMD's index by itself.

## 9. Secondary channels

- **Homebrew:** homebrew-core forbids dependencies on proprietary software (so no CUDA build) and builds Python resources from sdists only ([Acceptable Formulae](https://docs.brew.sh/Acceptable-Formulae), [Language-Specific Formulae](https://docs.brew.sh/Language-Specific-Formulae)). uv and ruff are in homebrew-core, built from source ([uv.rb](https://github.com/Homebrew/homebrew-core/blob/main/Formula/u/uv.rb)). A macOS source build of Autan is CPU (and later Metal) only, so it fits core once stable; a tap `Remenby31/homebrew-autan` first ([taps](https://docs.brew.sh/How-to-Create-and-Maintain-a-Tap)).
- **GHCR:** packages are private until made public; `org.opencontainers.image.source` links the image to the repository; multi-arch descriptions come from manifest annotations; attest images with `actions/attest` and `push-to-registry: true` ([container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)). With cudart from a wheel, the image needs no `nvidia/cuda` base: the NVIDIA Container Toolkit injects the driver.
- **Offline:** uv has no download command ([uv#3163](https://github.com/astral-sh/uv/issues/3163)); `pip download --only-binary=:all: --platform … --python-version … --abi …` fetches wheels for another machine, and `--require-hashes` with `uv export` hashes pins them ([pip download](https://pip.pypa.io/en/stable/cli/pip_download/), [repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/)). Untested end to end by the source; we test our bundle in CI.
- **conda-forge:** supports CUDA 12 and 13 (`cuda_compiler_version` 12.9 and 13.4, aarch64 as sbsa, which covers the Spark); recipes use `compiler('cuda')`, a `cuda-version` pin and a `__cuda` run requirement on the CUDA variant ([write the recipe](https://github.com/conda-forge/conda-forge.github.io/blob/main/docs/how-to/advanced/cuda/write-the-recipe.mdx), [pinning](https://github.com/conda-forge/conda-forge-pinning-feedstock/blob/main/recipe/conda_build_config.yaml)). staged-recipes has over 1000 open pull requests.
- **MPI:** not researched in depth here. Single-node runs need none; multi-node needs the cluster's GPU-aware MPI, which no wheel can supply portably.
