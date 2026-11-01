# Experiments for #178: the side of each incidence in Metal's store-gather gather

Not merged: neither variant is faster. Findings in #178 and #21.

- Commit (a), side from the twin: `left = k < incidence_twin[k]` instead of `owner[f] == c` (exact: interior faces have owner < neighbour).
- Commit (b), one record per face: the store pass writes the owner's bits then the values; the gather reads both in one access.

Both are byte-identical to `main` (`store_gather_hashes.py`, every store-gather face loop of the spike on tet, hex, poly and hybrid meshes in `fp32` and `fp32+acc64`) and pass `pytest tests/test_gpu.py tests/test_metal.py tests/test_jacobian.py --backend metal` and `autan_physics_metal_tests`.

Reproduce, from a build with `-DAUTAN_ENABLE_METAL=ON` of each commit and of `main`:

1. `store_gather_hashes.py` with each build's `libautan_spike_metal.dylib`: the same lines.
2. `autan_metal_loop_bench tet-n096.msh --policy fp32+acc64 --reps 10`, alternating builds: `advection_store_gather` ns/cell.
3. `split_passes.mm` (comment at its top, build line below) times the store and gather passes alone:
   `clang++ -std=c++20 -O3 -fobjc-arc split_passes.mm -I$B/codegen/generated -Icodegen/include -Icodegen/spike -Icore/include -Iio/include -Ipartition/include $B/codegen/libautan_kernels.a $B/codegen/libautan_metal.a $B/io/libautan_io.a $B/core/libautan_core.a -framework Metal -framework Foundation -framework IOKit -lcgns -lhdf5`

Measured on an Apple M4 Pro under a host load of 13–24 (indicative only), tet n=96, `fp32+acc64`, ns/cell:

| Build | store-gather (loop bench) | store pass | gather pass |
|---|---|---|---|
| `main` | 0.79–0.85 | 0.54–0.58 | 0.25–0.28 |
| (a) twin | 0.91–0.98 | — | — |
| (b) record | 0.83–0.86 | 0.59–0.60 | 0.25 |
