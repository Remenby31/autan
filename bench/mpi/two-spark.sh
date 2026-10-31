#!/usr/bin/env bash
# MPI sanity check between the two DGX Spark over Tailscale: correctness only, the link is
# not a cluster interconnect. Run from a workstation that can ssh to both hosts:
#   bench/mpi/two-spark.sh [HOST_A] [HOST_B] [IMAGE]
# Starts a throwaway sshd container on HOST_B (key generated for this run, bound to its
# Tailscale address on a free port), launches mpirun from a container on HOST_A, runs the OSU latency and
# bandwidth tests with host and device buffers, then removes both containers and the key.
set -euo pipefail

a=${1:-spark}
b=${2:-spark-5359}
image=${3:-autan-dev:local}
run=autan-mpi-$$
dir=/tmp/$run
ip_a=$(ssh "$a" tailscale ip -4)
ip_b=$(ssh "$b" tailscale ip -4)

cleanup() {
  ssh "$b" "docker rm -f $run >/dev/null 2>&1; rm -rf $dir" || true
  ssh "$a" "docker rm -f $run >/dev/null 2>&1; rm -rf $dir" || true
}
trap cleanup EXIT

# A free port per run, so concurrent runs by different people do not collide.
port=$(ssh "$b" 'for p in $(shuf -i 20000-29999 -n 50); do ss -Hltn "sport = :$p" | grep -q . || { echo "$p"; break; }; done')
[[ -n $port ]] || { echo "error: no free port found on $b" >&2; exit 1; }

for h in "$a" "$b"; do ssh "$h" "install -d -m 0700 $dir"; done
ssh "$a" "ssh-keygen -q -t ed25519 -N '' -C $run -f $dir/id"
ssh "$a" "cat $dir/id.pub" | ssh "$b" "cat > $dir/authorized_keys && chmod 600 $dir/authorized_keys && ssh-keygen -q -t ed25519 -N '' -f $dir/hostkey"

# The image's ubuntu user (uid 1000) owns the key files mounted from the hosts' admin user (uid 1000).
ssh "$b" "docker run -d --name $run --network host --gpus all --user ubuntu -v $dir:/mpi:ro $image \
  /usr/sbin/sshd -D -e -p $port -o ListenAddress=$ip_b -h /mpi/hostkey \
  -o AuthorizedKeysFile=/mpi/authorized_keys -o PidFile=none -o PasswordAuthentication=no \
  -o KbdInteractiveAuthentication=no -o StrictModes=no" >/dev/null

mpirun_cmd="mpirun -n 2 --host $ip_a:1,$ip_b:1 --prefix /opt/autan \
  --prtemca plm_ssh_args '-p $port -i /mpi/id -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR' \
  --prtemca oob_tcp_if_include tailscale0 \
  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include tailscale0"

osu() {
  ssh "$a" "docker run --rm --name $run --network host --gpus all --user ubuntu -e HOME=/tmp -v $dir:/mpi:ro $image \
    bash -c \"$mpirun_cmd $*\""
}

echo "# $a ($ip_a) <-> $b ($ip_b), image $image, TCP over tailscale0"
# Tailscale between the Sparks may run through a DERP relay (milliseconds, a few MB/s): keep
# message sizes and iteration counts small enough to finish in a few minutes.
pt2pt=/opt/autan/osu/libexec/osu-micro-benchmarks/mpi/pt2pt
echo "## osu_latency, host buffers";   osu $pt2pt/osu_latency -m 1:4096 -i 100 -x 10
echo "## osu_latency, device buffers"; osu $pt2pt/osu_latency -d cuda -m 1:4096 -i 100 -x 10 D D
echo "## osu_bw, host buffers";        osu $pt2pt/osu_bw -m 1048576:1048576 -W 8 -i 20 -x 2
echo "## osu_bw, device buffers";      osu $pt2pt/osu_bw -d cuda -m 1048576:1048576 -W 8 -i 20 -x 2 D D
