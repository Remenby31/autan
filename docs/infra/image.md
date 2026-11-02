# Dev image

`docker/dev/Dockerfile` builds the image every build, test and benchmark runs in. CI publishes it as `ghcr.io/remenby31/autan-dev:<commit>` and `:main` for `linux/amd64` and `linux/arm64` (workflow `gpu-image`).

## Contents

| Component | Version | Source | Notes |
|---|---|---|---|
| CUDA toolkit | 13.0.3 | `nvidia/cuda` devel, Ubuntu 24.04 | base image pinned by digest |
| GCC, CMake, Ninja | 13.3, 3.28, 1.11 | Ubuntu 24.04 | `CMAKE_GENERATOR=Ninja` |
| UCX | 1.22.0 | source | CUDA, verbs, rdmacm |
| Open MPI | 5.0.11 | source | CUDA-aware, UCX; CUDA components are plugins, so `libmpi` loads without a GPU driver |
| HDF5 | 2.1.1 | source | parallel (MPI-IO), zlib |
| CGNS | 4.5.2 | source | parallel, 64-bit `cgsize_t`, HDF5 backend |
| METIS | 5.1.0 | Ubuntu 24.04 | 32-bit `idx_t` |
| PT-Scotch | 7.0.17 | source | 64-bit `SCOTCH_Num`; no METIS compatibility library, so METIS symbols come only from METIS |
| OSU micro-benchmarks | 7.5.2 | source | CUDA buffers, on `PATH` |
| Python, uv | 3.12, 0.13.6 | Ubuntu 24.04, `ghcr.io/astral-sh/uv` | `uv` never downloads an interpreter; every `setup-uv` step in `.github/workflows/` pins the same uv: change them together |

Libraries built from source install to `/opt/autan`, which is on `PATH`, `CMAKE_PREFIX_PATH` and the loader path. Every download is checked against a SHA-256 written in the Dockerfile; to upgrade a component, change its version and checksum together. All components carry permissive or weak-copyleft licences (BSD, Apache-2.0, zlib, CeCILL-C); ParMETIS is excluded by its licence.

## Use

On a DGX Spark, from the repository root:

```sh
image=ghcr.io/remenby31/autan-dev:main                 # or: docker build -t autan-dev:local docker/dev
docker pull "$image"
docker run --rm -it --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e AUTAN_IMAGE="$(docker image inspect -f '{{index .RepoDigests 0}}' "$image")" \
  -v /home/admin/autan-gpu:/home/admin/autan-gpu:ro \
  -v /home/admin/autan-gpu/holders.log:/home/admin/autan-gpu/holders.log \
  -v "$PWD:/src" -w /src "$image" bash
```

Mount the GPU lock's directory and its holders log ([bench.md](bench.md#the-gpu-lock)); `docker/runner/deploy.sh` creates the log, and mounting it where it is missing would make docker create a directory there: GPU work in every container then queues on the same lock with `bench/lock/autan-gpu-lock -m "AGENT: what for" -- <cmd>`. `AUTAN_IMAGE` records the image digest in benchmark results (for a local build, use `{{.Id}}`). Use `--gpus all`: one of the two Sparks has no CDI specification, so `--device nvidia.com/gpu=all` fails there. Run as your own user so `git` accepts the mounted checkout. Name your containers with an `autan-` prefix, and never stop or inspect containers you did not start: the Sparks run other teams' stacks.

## Build and test on a Spark from your workstation

Build on a Spark, not on your laptop. From any worktree:

```sh
docker/spark.sh -- 'cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build'
docker/spark.sh --gpu -- 'bench/lock/autan-gpu-lock -m "AGENT: <gpu test>" -- ./build/<gpu test>'
```

`docker/spark.sh` keeps one checkout per worktree in `~/autan-work/<worktree>` on the Spark, checks out your `HEAD` when it is pushed (otherwise `origin/main`, and builds that embed the revision record a dirty tree), copies your working files over it, and runs the command in the dev image as your user, limited to 12 CPUs and 48 GB so several agents can build at once. Ignored files such as `build/` stay on the Spark between runs, so rebuilds are incremental. The default host is `spark`; `--host spark-5359` targets the benchmark machine, which is better left quiet. Builds and CPU tests get no GPU and must not take the GPU lock; pass `--gpu` for GPU work and take the lock for anything timed or GPU-heavy. Benchmarks for a pull request still come from a clean, pushed commit ([bench.md](bench.md)).

## Smoke test

`docker/dev/smoke/run.sh BUILD_DIR` configures a CMake project that finds every library above, then runs it on two MPI ranks. Each check has a result fixed by the library's specification: the MPI sum of ranks, a balanced bisection of a 4×4 grid by METIS and Scotch with the reported edge cut, a parallel HDF5 and CGNS write/read round trip, and an `MPI_Allreduce` on device buffers with `MPIX_Query_cuda_support()`. The CUDA check exits 77 (skipped) without a GPU; set `REQUIRE_GPU=1` to make that a failure. CI runs it without a GPU (`gpu-image`) and on a Spark runner with one (`gpu-bench`).
