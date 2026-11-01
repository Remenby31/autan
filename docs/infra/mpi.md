# MPI on the DGX Spark

The image's Open MPI is CUDA-aware (UCX and the `accelerator/cuda` component): pass device pointers to MPI calls.

## One Spark

```sh
docker run --rm --gpus all --user ubuntu -e HOME=/tmp -v /home/admin/autan-gpu:/home/admin/autan-gpu:ro \
  -v "$PWD:/src" -w /src autan-dev:local mpirun -n 4 --mca pml ucx <program>
```

Measured on spark-5359 (2 ranks, one GB10, OSU 7.5.2, `--mca pml ucx`): 0.19 µs latency and 5.6 GB/s at 16 MiB with host buffers; 99 µs latency and 24 GB/s at 16 MiB with device buffers. The small-message device latency is high and not yet investigated.

## Two Sparks: correctness only

The two Sparks have no direct link: MPI runs over TCP on `tailscale0`, and Tailscale currently relays their traffic through a DERP server (`tailscale ping` reports `via DERP(par)`). Use it to check that multi-node code is correct, never to measure it.

`bench/mpi/two-spark.sh [HOST_A] [HOST_B] [IMAGE]`, run from a workstation that can SSH to both, does the whole sequence and cleans up:

1. Takes `/home/admin/autan-gpu/gpu.lock` on both hosts, in host-name order, and holds both until it exits.
2. Starts an unprivileged `sshd` in a `--network host` container on HOST_B, bound to HOST_B's Tailscale address on a free port picked for this run. The container generates its host key, prints the public half, accepts only a client key generated for this run, writes nothing on the host, and exits after an hour whatever happens to the script.
3. Runs `mpirun` from a `--network host` container on HOST_A with `--prefix /opt/autan`, `--prtemca plm_ssh_args '-p <port> -i <key> -o StrictHostKeyChecking=yes …'` against the host key read from the container, `--prtemca oob_tcp_if_include tailscale0`, `--mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include tailscale0`.
4. Removes the containers and keys it created, and says so if it could not.

To run your own program, mount it at the same path on both hosts and replace the OSU commands. Remote ranks get only the environment `--prefix` sets; pass anything else with `-x VAR`.

Last run (spark ↔ spark-5359, 2026-10-31): 3.0–3.1 ms latency up to 1 KiB and 2.85 MB/s at 1 MiB, identical with host and device buffers. Device buffers go through Open MPI's staging over TCP, which is what the check verifies.
