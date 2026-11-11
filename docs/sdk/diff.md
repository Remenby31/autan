# `autan diff`

`autan diff` compares two run directories ([results.md](../contracts/results.md)): their record, their history and their solution.

```text
autan diff RUN_A RUN_B [--fields NAME ...] [--bitwise | --rtol X] [--json]
```

In Python: `autan.diff(a, b, *, bitwise=False, rtol=1e-6, fields=None)` takes two paths and returns the report below.

## What is compared

1. **The record**, from each `run.json`.
   - Compared: the fields of the run's [identity](../contracts/results.md#identity) except `devices.count` and `cell_order` (two runs that differ only by these give the same bits under reproducibility `"ranks"`, results.md), then `status`, `iterations` and `monitors`, the latter value by value with the rule below.
   - Reported, never compared: `run_id`, `devices.count`, `cell_order`, `solver.binary_sha256`, `started`, `finished` and `timing`.
2. **The history**, line by line: the two runs must hold the same iterations, and each line's every value (every key of [History](../contracts/results.md#history): residuals, `time`, `provisional`, monitor quantities) is compared. The report gives the first iteration that differs and, per series, the largest difference.
3. **The solution**, when the two runs have the same `mesh.sha256`.
   - A `solution.cgns` that only one run has is a difference; when neither has one, the report says so.
   - Compared: every array of `FlowSolution` and of `FlowSolutionMean`, cell by cell in mesh order, and every field of `WallSolution`, whose face ids (`PointList`) must be equal, face by face. These are the solution arrays; `RestartState` and `SolverState`, which readers ignore, are not compared.
   - A field that only one run has is a difference. With `--fields`, only the fields named are compared; the record and the history always are.

## When two values agree

- With `--bitwise`, two values agree when their bytes are equal.
- Otherwise, values `a` and `b` agree when |a − b| ≤ `rtol` · max(|a|, |b|) + m, m being the smallest normal FP64 number, 2.2 × 10⁻³⁰⁸.
  - The default `rtol` is 10⁻⁶. One unit in the last place of a value agrees, a zero included, and a value of magnitude above 10⁻³⁰⁴ scaled by 1.001 differs.
  - Every value is compared relatively, near zero too, so round-off on a field that is nearly zero (a transverse velocity) can differ between runs that are not bit for bit.
  - Non-finite values are refused by the reader where results.md refuses them (`E0701`).

## Report and exit status

The report lists every compared record entry with both values, the history's first differing iteration, and for each field its largest absolute and relative difference. It names each entry, series and field that differs. With `--json` it is one JSON document; `"agree"` is the verdict.

As `diff` and `cmp`, `autan diff` exits 0 when the runs agree, 1 when they differ and 2 when it cannot compare them: a path that is not a run directory, `E0701`, printed as every `autan` command prints a diagnostic. A difference is never an error.

The acceptance gates are in [acceptance.md](../validation/acceptance.md), M4.
