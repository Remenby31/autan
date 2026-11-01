#!/usr/bin/env bash
# Builds and tests the current worktree on a DGX Spark, in the dev image, from a workstation.
# Usage: docker/spark.sh [--host HOST] [--gpu] [--image IMAGE] -- COMMAND...
#   docker/spark.sh -- 'cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build'
#   docker/spark.sh --gpu -- 'flock /home/admin/autan-gpu/gpu.lock ./build/my_gpu_test'
# The Spark keeps one git checkout per worktree in ~/autan-work/<name>: it checks out your HEAD
# when that commit is pushed (otherwise origin/main, and the build records a dirty tree), then
# receives your working files by rsync. build/ and other ignored outputs stay there between runs.
# Builds run without the GPU: pass --gpu only for GPU work, and take the GPU lock for anything
# that is timed or uses the GPU heavily. Default host: spark (spark-5359 is kept for benchmarks).
set -euo pipefail

host=spark
gpu=0
image=autan-dev:local
while [[ $# -gt 0 ]]; do
  case $1 in
    --host) host=$2; shift 2 ;;
    --gpu) gpu=1; shift ;;
    --image) image=$2; shift 2 ;;
    --) shift; break ;;
    *) sed -n '2,10p' "$0"; exit 2 ;;
  esac
done
[[ $# -gt 0 ]] || { sed -n '2,10p' "$0"; exit 2; }

top=$(git rev-parse --show-toplevel)
name=$(basename "$top")
head=$(git -C "$top" rev-parse HEAD)
remote_dir="autan-work/$name"
repo=https://github.com/Remenby31/autan.git

ssh "$host" "set -e
  [ -d $remote_dir/.git ] || git clone -q $repo $remote_dir
  cd $remote_dir && git fetch -q origin
  if git cat-file -e $head^{commit} 2>/dev/null; then git checkout -q -f --detach $head
  else echo 'spark.sh: HEAD $head is not pushed; using origin/main (the build records a dirty tree)' >&2
       git checkout -q -f --detach origin/main; fi"

# Ignored files (build trees, venvs) are neither sent nor deleted on the Spark.
git -C "$top" ls-files --cached --others --exclude-standard -z |
  rsync -a --from0 --ignore-missing-args --files-from=- "$top/" "$host:$remote_dir/"
git -C "$top" ls-files --deleted -z | ssh "$host" "cd $remote_dir && xargs -0 -r rm -f --"

gpu_args=""
[[ $gpu -eq 0 ]] || gpu_args="--gpus all"
# The quoting of COMMAND is preserved by sending it as one argument to bash -c.
command=$(printf '%q ' "$*")
ssh "$host" "docker run --rm --name autan-work-$name-\$\$ $gpu_args \
  --cpus 12 --memory 48g --user \"\$(id -u):\$(id -g)\" -e HOME=/tmp \
  -v /home/admin/autan-gpu:/home/admin/autan-gpu:ro \
  -v \"\$HOME/$remote_dir:/src\" -w /src $image bash -c $command"
