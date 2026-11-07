# 0002: `rans-sa-mms-order` refines a near-isotropic family that resolves the wall layer

**Status:** accepted, 2026-11-07. **Decided by:** the lead's delegate for CFD (Main meta travail), relayed by the orchestrator. **Issues:** #497, #137.

## Context

`rans-sa-mms-order` refined the Cartesian family (n, 4n, n) at n = 24, 32, 48 and 64. Its cells are four times finer across the manufactured wall layer (y) than along it (x, z). With X2p, ρE failed the gate at those levels: fitted order 1.844, pairs 1.792, 1.846, 1.893 (#497). The diagnosis on #497 (H-P1 to H-P6) found the following:
- the deficit is not in the wall layer;
- it is not in the SA terms;
- it is not in the convective flux's dissipation (Roe, HLLC, AUSM+-up);
- it is not in the viscous terms;
- pressure, and with it ρE, converges faster as the cells become isotropic: n³ > (n, 2n, n) > (n, 4n, n).

The family's anisotropy, not the scheme, is what keeps pressure pre-asymptotic at the gate's levels.

## Decision

**This definition comes after results.** The failure at 24–64 on (n, 4n, n) is recorded as it stands. This definition applies only to runs made after it, and no existing run counts towards it: neither the (n, 4n, n) runs nor the (n, 2n, n) runs of the diagnosis.

The gate refines the Cartesian family **(n, 2n, n), n = 24, 32, 48, 64**. The choice rests on two requirements of the physics and of refinement studies, not on which family passes:

1. **The wall layer is resolved.** The manufactured solution's layer is 0.15 thick, and the refinement must resolve it on every level. As before (2026-11-01), the requirement is at least 6 cells across it at the coarsest level. With a spacing of 1/(2n) across the layer, n = 24 puts 7.2 cells there and n = 64 puts 19.2. The cube families n³ put 3.6 cells there at n = 24 and 4.8 at n = 32, fewer than 6, and stay excluded.
2. **The refinement is near-isotropic.** The observed order is measured against h = (V/N)^(1/3), and it reflects the scheme's order only when every direction's spacing follows h within a bounded ratio. Geometrically similar grids are the premise of systematic refinement (Roache 1998; Eça and Hoekstra 2014). The spacing along any direction stays within a factor 2 of the others, so the aspect ratio is at most 2. The former family's ratio of 4 kept the coarse directions' error, here pressure's, pre-asymptotic at every affordable level.

**A mandatory check keeps the stretched case in view.** The same build also runs `rans_sa` on the former family (n, 4n, n), n = 24, 32, 48, 64. There, the L2 pairwise orders of p and ρE must not fall: each pair is at least the one before it minus 0.02, the round-off of a pairwise order at these levels. A pre-asymptotic family rises; an asymptotic one stays at its order, which also passes. A falling order fails the gate, even if the (n, 2n, n) verdict passes.

**The verdict is composite.** `autan-vv verify --gate rans-sa-mms-order` on the (n, 2n, n) runs gives the order verdict; `PASS` there is necessary, not sufficient. The check is computed from the per-level `h` and the `Pressure` and `rhoE` L2 errors in the JSON of `autan-vv verify` on the (n, 4n, n) runs. The PR that reports the gate states both results; the gate passes only if both do. Until `autan-vv` computes the check itself, the PR shows that computation.

## Consequences

- `acceptance.md` states the family, its levels and the check, and links here.
- A finer level of the former family (n = 96, about 75 GiB with X2p) is queued as a measurement only, outside the gate (#497, option (b)).
- Running the gate costs the two families' runs.
