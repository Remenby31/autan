# Verification and validation

The `vv/` package makes correctness measurable before and alongside the solver. It is its own uv project:

```sh
cd vv
uv sync --locked                       # dev + mesh groups; gmsh is not installable on linux-aarch64
uv run pytest -m "not network"         # all tests; add -m slow to run only the order studies
uv run autan-vv --help
```

| Page | What it owns |
|---|---|
| [mms.md](mms.md) | Manufactured solutions, their source-term convention, and the C++ and JSON exports the kernel generator consumes |
| [meshes.md](meshes.md) | Mesh families of the unit cube: kinds, files, zones, perturbation, manifest, reproducibility |
| [order.md](order.md) | Observed order, Richardson extrapolation and GCI; the order-study report; the reference finite-volume solve and its measured orders |

## Typical use by a numerical PR

1. `uv run autan-vv meshes tet --levels 8,16,32,64 --out out/tet` (repeat per kind, with and without `--perturbation 0.3`).
2. Run the solver on each level with the manufactured source from `autan-vv mms <name> --out out/mms`.
3. Write the errors as `{"dim": 3, "levels": [{"n_cells", "volume", "errors": {"L2": ...}}]}` and run `uv run autan-vv order errors.json --out order.json --plot order.png`.
4. Paste the orders and the plot into the PR description, as AGENTS.md requires.

Errors raised by these tools carry a stable code `V####` and the CLI exits with status 2 on them.
