#!/usr/bin/env bash
# Manages the GPU runner of one DGX Spark. Run on the Spark, from the repository root.
#   docker/runner/deploy.sh build DEV_IMAGE   build the runner and proxy images
#   docker/runner/deploy.sh register          register (token on stdin, see docs/infra/runners.md)
#   docker/runner/deploy.sh start | stop | status | logs
# Everything it creates is prefixed autan-runner; it never touches other containers.
# A Spark shared with other teams sets its placement in the environment, for every command
# (docs/infra/runners.md): AUTAN_RUNNER_HOME (the lock's directory under it, and no guard cron),
# AUTAN_RUNNER_CPUSET, AUTAN_RUNNER_MEMORY, AUTAN_RUNNER_NICE, AUTAN_RUNNER_BUILD_JOBS and, for
# build, AUTAN_RUNNER_MIN_AVAILABLE_GIB.
set -euo pipefail

# A second runner on the same Spark (e.g. AUTAN_RUNNER_INSTANCE=main, labels bench-main)
# gets its own container and volumes; the proxy is shared.
instance=${AUTAN_RUNNER_INSTANCE:+-$AUTAN_RUNNER_INSTANCE}
name=${AUTAN_RUNNER_NAME:-$(hostname)$instance}
repo_url=https://github.com/Remenby31/autan
labels=${AUTAN_RUNNER_LABELS:-gpu,gb10}
container=autan-runner$instance
image=autan-runner:local
proxy_image=autan-runner-proxy:local
internal_net=autan-runner-internal
egress_net=autan-runner-egress
state=autan-runner-state$instance
work=autan-runner-work$instance
home=${AUTAN_RUNNER_HOME:-}
# Where the lock's directory lives on the host; jobs always see it at the path the workflows name.
lock_dir=${home:+$home/autan-gpu}
lock_dir=${lock_dir:-/home/admin/autan-gpu}
job_lock_dir=/home/admin/autan-gpu
lock=$lock_dir/gpu.lock
cpuset=${AUTAN_RUNNER_CPUSET:-}
memory=${AUTAN_RUNNER_MEMORY:-64g}
nice=${AUTAN_RUNNER_NICE:-0}
build_jobs=${AUTAN_RUNNER_BUILD_JOBS:-}
min_available=${AUTAN_RUNNER_MIN_AVAILABLE_GIB:-0}
here=$(cd "$(dirname "$0")" && pwd)

bad() { echo "error: $1" >&2; exit 2; }
[[ -z $home || $home == /* ]] || bad "AUTAN_RUNNER_HOME must be an absolute path"
[[ -z $cpuset || $cpuset =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] || bad "AUTAN_RUNNER_CPUSET is a cpu list such as 10-18"
[[ $memory =~ ^[1-9][0-9]*g$ ]] || bad "AUTAN_RUNNER_MEMORY is a whole number of GiB such as 16g"
[[ $nice =~ ^([0-9]|1[0-9])$ ]] || bad "AUTAN_RUNNER_NICE is 0 to 19"
[[ -z $build_jobs || $build_jobs =~ ^[1-9][0-9]?$ ]] || bad "AUTAN_RUNNER_BUILD_JOBS is 1 to 99"
[[ $min_available =~ ^[0-9]{1,3}$ ]] || bad "AUTAN_RUNNER_MIN_AVAILABLE_GIB is 0 to 999"
# One GPU lock per host: a relocated lock beside the usual one would let two jobs hold the GPU.
if [[ -n $home && -e $job_lock_dir/gpu.lock ]]; then
  bad "$job_lock_dir/gpu.lock exists: with AUTAN_RUNNER_HOME the runner would take another lock"
fi

prepare() {
  # Not in sticky /tmp: there fs.protected_regular stops other uids from opening a file they
  # do not own with O_CREAT, which flock(1) uses.
  install -d -m 0755 "$lock_dir"
  [[ -e $lock ]] || install -m 0666 /dev/null "$lock"
  local file
  for file in "$lock_dir/holders.log" "$lock_dir/ci.lock" "$lock_dir/main.lock" "$lock_dir/turn.lock"; do
    [[ -f $file ]] || { (set -C; : >"$file") 2>/dev/null && chmod 0666 "$file"; }
    [[ -f $file ]] || { echo "error: $file is not a regular file" >&2; exit 1; }
  done
  # The lock tools, at a path every container that mounts the lock's directory sees.
  install -d -m 0755 "$lock_dir/bin"
  install -m 0755 "$here/../../bench/lock/autan-gpu-lock" "$here/../../bench/lock/autan-gpu-guard" "$lock_dir/bin/"
  install -m 0644 "$here/../../bench/lock/autan_memfloor.py" "$lock_dir/bin/"
  # The guard runs from cron, one instance (flock -n), so it comes back within a minute and after
  # a reboot; not on a host that confines us to AUTAN_RUNNER_HOME, whose crontab is not ours.
  if [[ -z $home ]]; then
    install -d -m 0755 "$HOME/autan-gpu-guard"
    # A running guard keeps its old code: stop it, and cron starts the new one within a minute.
    pkill -f "^python3 $lock_dir/bin/autan-gpu-guard" || true
    local cron="* * * * * flock -n $HOME/autan-gpu-guard/run.lock $lock_dir/bin/autan-gpu-guard >> $HOME/autan-gpu-guard/guard.log 2>&1"
    { crontab -l 2>/dev/null | grep -v autan-gpu-guard || true; echo "$cron"; } | crontab -
  fi
  # Internal: no route out. Isolated gateway: the host gets no address on this bridge, so a job
  # cannot reach host services listening on all interfaces.
  if ! docker network inspect "$internal_net" >/dev/null 2>&1; then
    docker network create --internal -o com.docker.network.bridge.gateway_mode_ipv4=isolated \
      "$internal_net" >/dev/null
  fi
  local shape
  shape=$(docker network inspect -f '{{.Internal}} {{index .Options "com.docker.network.bridge.gateway_mode_ipv4"}}' "$internal_net")
  [[ $shape == "true isolated" ]] || {
    echo "error: $internal_net is not internal with an isolated gateway ($shape); remove it" >&2
    exit 1
  }
  docker network inspect "$egress_net" >/dev/null 2>&1 || docker network create "$egress_net" >/dev/null
}

placement=(${cpuset:+--cpuset-cpus "$cpuset"})
runner_args=(
  --gpus all
  --network "$internal_net"
  --cap-drop ALL
  --security-opt no-new-privileges
  ${placement[@]+"${placement[@]}"}
  --memory "$memory" --memory-swap "$memory" --pids-limit 8192
  # Every job step runs under the runner at this niceness; without CAP_SYS_NICE none lowers it.
  -e AUTAN_RUNNER_NICE="$nice"
  ${build_jobs:+-e CMAKE_BUILD_PARALLEL_LEVEL="$build_jobs"}
  -e HTTP_PROXY=http://autan-runner-proxy:3128 -e HTTPS_PROXY=http://autan-runner-proxy:3128
  -e http_proxy=http://autan-runner-proxy:3128 -e https_proxy=http://autan-runner-proxy:3128
  -e NO_PROXY=localhost,127.0.0.1 -e no_proxy=localhost,127.0.0.1
  -e AUTAN_BENCH_HOST="$name"
  -v "$state":/state
  -v "$work":/work
  -v "$lock_dir":"$job_lock_dir":ro
  --log-opt max-size=20m --log-opt max-file=3
)

start_proxy() {
  # A proxy built from an older image keeps its old policy, and one started with other cores
  # keeps them: replace it.
  local running
  running=$(docker container inspect -f '{{.Image}} {{.HostConfig.CpusetCpus}}' autan-runner-proxy 2>/dev/null || true)
  if [[ -n $running && $running != "$(docker image inspect -f '{{.Id}}' "$proxy_image") $cpuset" ]]; then
    docker rm -f autan-runner-proxy >/dev/null
  fi
  if ! docker container inspect autan-runner-proxy >/dev/null 2>&1; then
    docker run -d --name autan-runner-proxy --restart unless-stopped \
      --network "$egress_net" --cap-drop ALL --security-opt no-new-privileges ${placement[@]+"${placement[@]}"} \
      --memory 1g --pids-limit 256 --log-opt max-size=20m --log-opt max-file=3 \
      "$proxy_image" >/dev/null
  fi
  if ! docker container inspect -f '{{json .NetworkSettings.Networks}}' autan-runner-proxy |
    grep -q "\"$internal_net\""; then
    docker network connect "$internal_net" autan-runner-proxy
  fi
  docker start autan-runner-proxy >/dev/null
}

case ${1:-} in
  build)
    dev_image=${2:?usage: deploy.sh build DEV_IMAGE}
    # Build from, and record, the exact image the reference resolves to now: a tag named after
    # its ID cannot move to another image.
    id=$(docker image inspect -f '{{.Id}}' "$dev_image")
    base=autan-runner-base:${id#sha256:}
    docker tag "$id" "$base"
    docker build -t "$proxy_image" "$here/proxy"
    docker build -t "$image" --build-arg DEV_IMAGE="$base" --build-arg MIN_AVAILABLE_GIB="$min_available" \
      --label org.autan.dev-image="${dev_image%@*}@$id" "$here"
    ;;
  register)
    prepare
    start_proxy
    docker run --rm -i "${runner_args[@]}" "$image" register "$name" "$labels" "$repo_url"
    ;;
  start)
    prepare
    start_proxy
    docker rm -f "$container" >/dev/null 2>&1 || true
    dev_image=$(docker image inspect -f '{{index .Config.Labels "org.autan.dev-image"}}' "$image")
    docker run -d --name "$container" --restart unless-stopped -e AUTAN_IMAGE="$dev_image" "${runner_args[@]}" "$image" run
    ;;
  # The proxy goes with the last runner.
  stop)
    docker rm -f "$container"
    docker ps -a --format '{{.Names}}' | grep -v '^autan-runner-proxy$' | grep -qE '^autan-runner(-[a-z0-9-]+)?$' ||
      docker rm -f autan-runner-proxy
    ;;
  status) docker ps -a --filter name=^autan-runner --format '{{.Names}}\t{{.Status}}\t{{.Image}}' ;;
  logs) docker logs --tail 100 "$container" ;;
  *) sed -n '2,7p' "$0"; exit 2 ;;
esac
