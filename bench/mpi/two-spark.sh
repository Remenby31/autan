#!/usr/bin/env bash
# MPI sanity check between the two DGX Spark over Tailscale: correctness only, the link is
# not a cluster interconnect. Run from a workstation that can ssh to both hosts:
#   bench/mpi/two-spark.sh [HOST_A] [HOST_B] [IMAGE]
# Holds the GPU lock of both hosts for the whole run, starts a throwaway sshd container on
# HOST_B (fresh host and client keys, bound to its Tailscale address on a free port, exits
# after an hour whatever happens to this script), runs mpirun from a container on HOST_A with
# strict host-key checking, runs the OSU latency and bandwidth tests with host and device
# buffers, then removes what it created.
set -euo pipefail

a=${1:-spark}
b=${2:-spark-5359}
image=${3:-autan-dev:local}
run=autan-mpi-$(openssl rand -hex 6)
lifetime_s=3600
ip_a=$(ssh "$a" tailscale ip -4)
ip_b=$(ssh "$b" tailscale ip -4)
local_tmp=$(mktemp -d)
remote_dir=""
created_b=""
created_a=""

cleanup() {
  local rc=$? status=0
  [[ -z $created_a ]] || ssh "$a" "docker rm -f $created_a >/dev/null" || status=1
  [[ -z $created_b ]] || ssh "$b" "docker rm -f $created_b >/dev/null" || status=1
  [[ -z $remote_dir ]] || ssh "$a" "rm -rf $remote_dir" || status=1
  rm -rf "$local_tmp"
  if [[ $status -ne 0 ]]; then
    echo "error: cleanup failed; remove $run containers and $remote_dir on $a and $b by hand" >&2
    rc=1
  fi
  exit "$rc"
}
trap cleanup EXIT

# Each lock is held by a remote flock that lives as long as this script keeps its fifo open.
# Hosts are locked in name order so two concurrent runs cannot deadlock.
lock_holders=()
hold_gpu_lock() {
  local host=$1 fifo="$local_tmp/lock-$1"
  mkfifo "$fifo"
  ssh "$host" 'flock /home/admin/autan-gpu/gpu.lock sh -c "echo locked; cat >/dev/null"' <"$fifo" >"$fifo.out" &
  lock_holders+=("$!")
  exec {lock_fd}>"$fifo"
  until grep -q locked "$fifo.out" 2>/dev/null; do
    kill -0 "$!" 2>/dev/null || { echo "error: could not take the GPU lock on $host" >&2; exit 1; }
    sleep 1
  done
}
require_locks() {
  local pid
  for pid in "${lock_holders[@]}"; do
    kill -0 "$pid" 2>/dev/null || { echo "error: a GPU lock holder exited; aborting" >&2; exit 1; }
  done
}
for host in $(printf '%s\n' "$a" "$b" | sort); do hold_gpu_lock "$host"; done

port=$(ssh "$b" 'for p in $(shuf -i 20000-29999 -n 50); do ss -Hltn "sport = :$p" | grep -q . || { echo "$p"; break; }; done')
[[ -n $port ]] || { echo "error: no free port found on $b" >&2; exit 1; }

remote_dir=$(ssh "$a" "mktemp -d /tmp/$run.XXXXXX")
ssh "$a" "ssh-keygen -q -t ed25519 -N '' -C $run -f $remote_dir/id"
client_key=$(ssh "$a" "cat $remote_dir/id.pub")

# The sshd container generates its own host key and prints the public half; nothing is
# written on HOST_B outside the container.
created_b=$(ssh "$b" "docker run -d --rm --name $run --network host --gpus all --user ubuntu \
  -e CLIENT_KEY='$client_key' $image sh -c '
    ssh-keygen -q -t ed25519 -N \"\" -f /tmp/hostkey && cat /tmp/hostkey.pub &&
    echo \"\$CLIENT_KEY\" > /tmp/authorized_keys && chmod 600 /tmp/authorized_keys &&
    exec timeout $lifetime_s /usr/sbin/sshd -D -e -p $port -o ListenAddress=$ip_b -h /tmp/hostkey \
      -o AuthorizedKeysFile=/tmp/authorized_keys -o PidFile=none -o PasswordAuthentication=no \
      -o KbdInteractiveAuthentication=no -o StrictModes=no'")
until ssh "$b" "docker logs $created_b 2>&1" | grep -q "Server listening on $ip_b port $port"; do
  ssh "$b" "docker inspect $created_b >/dev/null 2>&1" || { echo "error: sshd on $b exited" >&2; exit 1; }
  sleep 1
done
host_key=$(ssh "$b" "docker logs $created_b 2>&1" | grep -m1 '^ssh-ed25519 ')
ssh "$a" "echo '[$ip_b]:$port $host_key' > $remote_dir/known_hosts"

mpirun_cmd="mpirun -n 2 --host $ip_a:1,$ip_b:1 --prefix /opt/autan \
  --prtemca plm_ssh_args '-p $port -i /mpi/id -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/mpi/known_hosts -o LogLevel=ERROR' \
  --prtemca oob_tcp_if_include tailscale0 \
  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include tailscale0"

osu() {
  require_locks
  created_a=$run
  ssh "$a" "docker run --rm --name $run --network host --gpus all --user ubuntu -e HOME=/tmp \
    -v $remote_dir:/mpi:ro $image timeout $lifetime_s bash -c \"$mpirun_cmd $*\""
  created_a=""
  require_locks
}

echo "# $a ($ip_a) <-> $b ($ip_b), image $image, TCP over tailscale0"
# Tailscale between the Sparks may run through a DERP relay (milliseconds, a few MB/s): keep
# message sizes and iteration counts small enough to finish in a few minutes.
pt2pt=/opt/autan/osu/libexec/osu-micro-benchmarks/mpi/pt2pt
echo "## osu_latency, host buffers";   osu $pt2pt/osu_latency -m 1:4096 -i 100 -x 10
echo "## osu_latency, device buffers"; osu $pt2pt/osu_latency -d cuda -m 1:4096 -i 100 -x 10 D D
echo "## osu_bw, host buffers";        osu $pt2pt/osu_bw -m 1048576:1048576 -W 8 -i 20 -x 2
echo "## osu_bw, device buffers";      osu $pt2pt/osu_bw -d cuda -m 1048576:1048576 -W 8 -i 20 -x 2 D D
