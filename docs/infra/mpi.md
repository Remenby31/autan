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

## Hangs in CI

CI's "MPI ranks" job runs the tests labelled `mpi` under `bench/ci/ctest-watchdog.sh`: an MPI launcher alive past 120 s has the stacks of every process under it dumped with gdb (job log and the `mpi-hang-stacks-*` artifact), then is killed, so the job fails at once with the evidence. One case is not a failure of the test: Open MPI's launcher (PRRTE 3.0.x) sometimes stays alive after every rank has ended, at a rate of about 1 % after an early refusal by the ranks (#304; PRRTE 4.0.0 was tried and rejected there). The proof that the ranks ended is theirs (`run_on_ranks`, #380): N markers `exit-0` … `exit-<N-1>`, in the launcher's `AUTAN_TEST_EXIT_DIR` or else `<parent of --runs>/exits`, each written since the launcher started and holding `<status> <pid>` of a process that is gone (as `run_on_ranks` requires) or a zombie of the launcher itself (N from `-n`, `-np`, `--np`, `-c` or their `=` forms; a relative directory is the launcher's). With that proof and nothing living below the launcher for 5 s (`AUTAN_WATCHDOG_NET_S`), the launcher's stack is dumped (job log and `net-<pid>.txt`) and it gets SIGKILL (a launcher that handled TERM by exiting would end normally, and its test would judge its status instead of the markers), with a warning naming the test, the launcher and the markers found; the watchdog does not fail the test, which judges on the markers instead of the launcher's status, and the proof takes precedence over the 120 s limit. Without it (a rank alive, in the launcher's tree or reparented elsewhere, a marker missing, stale or of a running process), a launcher past 120 s is a hang. #304 stays open as long as the launcher can hang outside CI. To chase an intermittent hang in that same environment, run the `MPI stress` workflow by hand (`gh workflow run mpi-stress.yml -f runs=50 -f tests='^Several ranks give the bits of one$'`): it repeats the selected tests under the watchdog, counts hangs and failures in the job summary and uploads every run's log and stacks.
