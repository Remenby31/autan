# Benchmarks

`bench/` holds the GPU benchmarks and `autan-bench`, the harness that times them, records the conditions of the run and gates regressions. Every performance number quoted in a pull request comes from it (`AGENTS.md`, non-negotiable 2).

## Run

Inside the dev image on a DGX Spark, from the repository root:

```sh
cmake -S bench -B build/bench && cmake --build build/bench
uv sync --project bench --locked
uv run --project bench --locked autan-bench stream --binary build/bench/stream --out results/stream.json
uv run --project bench --locked autan-bench compare results/stream.json bench/baselines/gb10.json
```

## Timing method

`autan-bench` applies the same procedure to every benchmark:

1. Takes an exclusive `flock` on `/tmp/cfd-gpu.lock`, waiting as long as needed. The runner container mounts the host's lock file, so CI and interactive runs queue on the same lock.
2. Refuses to start (`GPU_BUSY`) unless, within 5 s, five consecutive `nvidia-smi` samples over one second show no compute process and at most 5 % utilisation (the utilisation counter is a trailing average, so a GPU released a moment ago needs a second to read idle). Another team's job is not something to wait for: rerun later.
3. Samples SM clock, power, temperature, clock event reasons and the number of compute processes every 200 ms during the run, and fails (`GPU_CONTENDED`) if another process appeared. GB10 reports no memory clock, so none is recorded.
4. Times each repetition with CUDA events around the kernel after warm-up repetitions, and reports the median, quartiles (Hyndman–Fan type 7), IQR and coefficient of variation (CV = standard deviation / mean) of the per-repetition metric.
5. Records provenance with the run-identity names of [results.md](../contracts/results.md) (`solver.commit`, `backend`, `precision`, `devices`), plus a dirty-tree flag, host, image (`AUTAN_IMAGE`, `AUTAN_IMAGE_REVISION`), CUDA runtime and driver, GPU name, UUID and properties.

## STREAM and the GB10 peak

`bench/stream/stream.cu` runs the four STREAM kernels (McCalpin 1995) on three FP64 arrays of 2²⁸ elements (2 GiB each, far beyond the 24 MiB L2), 16-byte loads, 256 threads per block, 10 warm-up and 100 timed samples per kernel, each sample the mean of 5 back-to-back launches (about 85 ms). On the shared Sparks, single launches are disturbed by other tenants' CPU traffic on the same LPDDR5x; batching five halves the CV and moves the median by less than 0.5 %. Bytes moved count each array read and each array written once, as STREAM does. Each kernel repeats on unchanged inputs, so the final arrays have exact values and the binary checks them bitwise.

The **GB10 peak bandwidth** is the highest median of the four kernels in `bench/baselines/gb10.json` (`autan-bench peak bench/baselines/gb10.json`). It is the denominator of "% of peak" for every kernel, and it is a measured, achievable peak, below the 273 GB/s of the LPDDR5x datasheet.

## Results, baselines and the regression gate

Results follow `bench/src/autan_bench/result.schema.json` (`autan.bench-result/1`); `autan-bench validate` checks a file. `autan-bench compare RESULT BASELINE` compares medians benchmark by benchmark and exits 1 when one is more than 5 % worse, 2 when the files cannot be compared (different GPU, missing benchmark, different metric). `autan-bench baseline RESULT --out bench/baselines/<gpu>.json` promotes a result only if it comes from a clean checkout and every benchmark has a CV below 2 %.

To refresh a baseline, run `autan-bench stream` on an idle Spark from a clean checkout of `main`, promote it with `autan-bench baseline`, and commit the file with the reason in the pull request.
