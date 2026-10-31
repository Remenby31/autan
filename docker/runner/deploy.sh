#!/usr/bin/env bash
# Manages the GPU runner of one DGX Spark. Run on the Spark, from the repository root.
#   docker/runner/deploy.sh build DEV_IMAGE   build the runner and proxy images
#   docker/runner/deploy.sh register          register (token on stdin, see docs/infra/runners.md)
#   docker/runner/deploy.sh start | stop | status | logs
# Everything it creates is prefixed autan-runner; it never touches other containers.
set -euo pipefail

name=${AUTAN_RUNNER_NAME:-$(hostname)}
repo_url=https://github.com/Remenby31/autan
labels=gpu,gb10
image=autan-runner:local
proxy_image=autan-runner-proxy:local
internal_net=autan-runner-internal
egress_net=autan-runner-egress
state=autan-runner-state
work=autan-runner-work
lock=/tmp/cfd-gpu.lock
here=$(cd "$(dirname "$0")" && pwd)

prepare() {
  [[ -e $lock ]] || install -m 0666 /dev/null "$lock"
  docker network inspect "$internal_net" >/dev/null 2>&1 || docker network create --internal "$internal_net" >/dev/null
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
  -e AUTAN_HOST="$name"
  -v "$state":/state
  -v "$work":/work
  -v "$lock":"$lock"
)

start_proxy() {
  if ! docker container inspect autan-runner-proxy >/dev/null 2>&1; then
    docker run -d --name autan-runner-proxy --restart unless-stopped \
      --network "$egress_net" --cap-drop ALL --security-opt no-new-privileges \
      --memory 1g --pids-limit 256 "$proxy_image" >/dev/null
    docker network connect "$internal_net" autan-runner-proxy
  fi
  docker start autan-runner-proxy >/dev/null
}

case ${1:-} in
  build)
    dev_image=${2:?usage: deploy.sh build DEV_IMAGE}
    # Benchmarks record the dev image by digest, not by a movable tag.
    dev_image="${dev_image%@*}@$(docker image inspect -f '{{.Id}}' "$dev_image")"
    docker build -t "$proxy_image" "$here/proxy"
    docker build -t "$image" --build-arg DEV_IMAGE="${dev_image%@*}" --label org.autan.dev-image="$dev_image" "$here"
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
