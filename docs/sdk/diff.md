# `autan diff`

`autan diff` compares two run directories ([results.md](../contracts/results.md)): their record, their history and their solution.

```text
autan diff RUN_A RUN_B [--fields NAME ...] [--bitwise | --rtol X] [--json]
```

In Python: `autan.diff(a, b, *, bitwise=False, rtol=1e-6, fields=None)` takes two paths and returns the report below.

## What is compared

1. **The record**, from each `run.json`. Compared: the fields of the run's [identity](../contracts/results.md#identity) except `devices.count` and `cell_order` (two runs that differ only by these give the same bits under reproducibility `"ranks"`, results.md), then `status` and `iterations`. Reported, never compared: `run_id`, `devices.count`, `cell_order`, `solver.binary_sha256`, `started`, `finished` and `timing`.
2. **The history**, line by line over the iterations both runs hold: every residual of [History](../contracts/results.md#history) and every quantity of every monitor. The report gives the first iteration that differs and, per series, the largest difference.
3. **The solution**, only when both runs have a `solution.cgns` and the same `mesh.sha256`: every field of `FlowSolution` and of `FlowSolutionMean`, cell by cell in mesh order, and every field of `WallSolution`, face by face (`RestartState` and `SolverState`, which readers ignore, are not compared). A field that only one run has is a difference. With `--fields`, only the fields named are compared; the record and the history always are.

## When two values agree

- With `--bitwise`, two values agree when their bytes are equal.
- Otherwise, values `a` and `b` of one series or field agree when |a − b| ≤ `rtol` · max(|a|, |b|, s), where s is 10⁻⁸ times the largest magnitude of that series or field in either run, so that round-off far below the field's scale is not a difference. The default `rtol` is 10⁻⁶: one unit in the last place agrees in FP32 and FP64, and a value scaled by 1.001 differs. Two NaN agree; a NaN and a number differ.

## Report and exit status

The report lists every compared record entry with both values, the history's first differing iteration, and for each field its largest absolute and relative difference. It names each entry, series and field that differs. With `--json` it is one JSON document; `"agree"` is the verdict.

As `diff` and `cmp`, `autan diff` exits 0 when the runs agree, 1 when they differ and 2 when it cannot compare them: a path that is not a run directory, `E0701`, printed as every `autan` command prints a diagnostic. A difference is never an error.

The acceptance gates are in [acceptance.md](../validation/acceptance.md), M4.
