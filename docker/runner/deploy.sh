#!/usr/bin/env bash
# Manages the GPU runner of one DGX Spark. Run on the Spark, from the repository root.
#   docker/runner/deploy.sh build DEV_IMAGE   build the runner and proxy images
#   docker/runner/deploy.sh register          register (token on stdin, see docs/infra/runners.md)
#   docker/runner/deploy.sh start | stop | status | logs
# Everything it creates is prefixed autan-runner; it never touches other containers.
set -euo pipefail

name=${AUTAN_RUNNER_NAME:-$(hostname)}
repo_url=https://github.com/Remenby31/autan
labels=${AUTAN_RUNNER_LABELS:-gpu,gb10}
image=autan-runner:local
proxy_image=autan-runner-proxy:local
internal_net=autan-runner-internal
egress_net=autan-runner-egress
state=autan-runner-state
work=autan-runner-work
lock_dir=/home/admin/autan-gpu
lock=$lock_dir/gpu.lock
here=$(cd "$(dirname "$0")" && pwd)

prepare() {
  # Not in sticky /tmp: there fs.protected_regular stops other uids from opening a file they
  # do not own with O_CREAT, which flock(1) uses.
  install -d -m 0755 "$lock_dir"
  [[ -e $lock ]] || install -m 0666 /dev/null "$lock"
  local log=$lock_dir/holders.log
  [[ -f $log ]] || { (set -C; : >"$log") 2>/dev/null && chmod 0666 "$log"; }
  [[ -f $log ]] || { echo "error: $log is not a regular file" >&2; exit 1; }
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

runner_args=(
  --gpus all
  --network "$internal_net"
  --cap-drop ALL
  --security-opt no-new-privileges
  --memory 64g --pids-limit 8192
  -e HTTP_PROXY=http://autan-runner-proxy:3128 -e HTTPS_PROXY=http://autan-runner-proxy:3128
  -e http_proxy=http://autan-runner-proxy:3128 -e https_proxy=http://autan-runner-proxy:3128
  -e NO_PROXY=localhost,127.0.0.1 -e no_proxy=localhost,127.0.0.1
  -e AUTAN_BENCH_HOST="$name"
  -v "$state":/state
  -v "$work":/work
  -v "$lock_dir":"$lock_dir":ro
  --log-opt max-size=20m --log-opt max-file=3
)

start_proxy() {
  # A proxy built from an older image keeps its old policy: replace it.
  local running
  running=$(docker container inspect -f '{{.Image}}' autan-runner-proxy 2>/dev/null || true)
  if [[ -n $running && $running != "$(docker image inspect -f '{{.Id}}' "$proxy_image")" ]]; then
    docker rm -f autan-runner-proxy >/dev/null
  fi
  if ! docker container inspect autan-runner-proxy >/dev/null 2>&1; then
    docker run -d --name autan-runner-proxy --restart unless-stopped \
      --network "$egress_net" --cap-drop ALL --security-opt no-new-privileges \
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
    docker build -t "$image" --build-arg DEV_IMAGE="$base" \
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
    docker rm -f autan-runner >/dev/null 2>&1 || true
    dev_image=$(docker image inspect -f '{{index .Config.Labels "org.autan.dev-image"}}' "$image")
    docker run -d --name autan-runner --restart unless-stopped -e AUTAN_IMAGE="$dev_image" "${runner_args[@]}" "$image" run
    ;;
  stop) docker rm -f autan-runner autan-runner-proxy ;;
  status) docker ps -a --filter name=^autan-runner --format '{{.Names}}\t{{.Status}}\t{{.Image}}' ;;
  logs) docker logs --tail 100 autan-runner ;;
  *) sed -n '2,7p' "$0"; exit 2 ;;
esac
