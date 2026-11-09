# Verification and validation

The `vv/` package makes correctness measurable before and alongside the solver. It is its own uv project:

```sh
cd vv
uv sync --locked                       # dev + mesh groups; gmsh is not installable on linux-aarch64
uv run pytest -m "not network"         # offline tests; -m network fetches every reference case
uv run autan-vv --help
```

| Page | What it owns |
|---|---|
| [mms.md](mms.md) | Manufactured solutions, their source-term convention, and the C++ and JSON exports the kernel generator consumes |
| [meshes.md](meshes.md) | Mesh families of the unit cube: kinds, files, zones, perturbation, manifest, reproducibility |
| [order.md](order.md) | Observed order, Richardson extrapolation and GCI; the order-study report; the reference finite-volume solve and its measured orders |
| [oracles.md](oracles.md) | Exact and reference solutions runs are verified against (MMS, Ringleb, vortex, shock tubes, Couette, Poiseuille, Blasius) |
| [harness.md](harness.md) | `autan-vv verify`: runs against an oracle, errors, observed order, gate verdict |
| [cases.md](cases.md) | Reference cases (TMR, ONERA M6, RAE 2822): pinned downloads, checksums, parsers |
| [su2.md](su2.md) | The comparison with SU2: build, fairness protocol, mesh conversion, results |
| [acceptance.md](acceptance.md) | The numbers each milestone gate must meet, with their sources |
| [guard.md](guard.md) | The nightly convergence guard on the TMR flat plate: case, gates, outcomes, timers |

## Typical use by a numerical PR

1. `uv run autan-vv meshes tet --levels 8,16,32,64 --out out/tet` (repeat per kind, with and without `--perturbation 0.3`). Gmsh has no aarch64 wheel: on a Spark, take the families from the `vv-meshes` workflow instead (GitHub-hosted x86-64; `gh workflow run vv-meshes.yml`, by default the four `ns-mms-order` families perturbed with `--perturbation 0.3 --seed 2026` at their gate levels). Each family comes as an artifact `meshes-KIND-pPERTURBATION-sSEED` with its `manifest.json` and a `SHA256SUMS` (`gh run download RUN -n …`, then `sha256sum -c SHA256SUMS`). `onera-m6` is dispatched with explicit levels and `perturbation` 0.
2. Run the solver on each level with the manufactured source from `autan-vv mms <name> --out out/mms`.
3. Write the errors as `{"dim": 3, "levels": [{"n_cells", "volume", "errors": {"L2": ...}}]}` and run `uv run autan-vv order errors.json --out order.json --plot order.png`.
4. Paste the orders and the plot into the PR description, as AGENTS.md requires.

Errors raised by these tools carry a stable code `V####` and the CLI exits with status 2 on them.
