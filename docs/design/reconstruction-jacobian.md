# The reconstruction's gradients in the approximate Jacobian of viscous meshes

Design for #395 (note 1 of 2: the flow's model; note 2, the coupled SA solve, follows #137 and the RAE 2822 grid), written before any code by CFD - Euler and CFD - Multigrille. The steady implicit step solves (V/Δt + J̃) ΔU = −R(U) with J̃ the assembled approximate Jacobian: every face's dF/dW with the reconstruction's gradients and the limiter held fixed ([euler.md](euler.md#time-stepping)). This note adds, in the shared convective assembly and only where `NavierStokes` asks for it, the convective flux's dependence on the cells' reconstruction gradients through their least-squares weights.

## Problem, measured

TMR flat plate 69×49 (#395, `physics/ns-sa` c3966068): once the SER raises the CFL number past ~10³ at second order, the SA plate locks into a period-2 cycle (full steps, solves within tolerance, residual at −3 to −4); the laminar plate wanders without converging; with ν̃ and μ_t frozen inside the cycle, the flow wanders as the laminar one does (#395, isolation diagnostic). At a state of the cycle, ‖(J − J̃) v‖ / ‖J v‖ is 10⁻⁹–10⁻² at first order and, at second order, 0.15–0.25 over the domain and 0.38–0.70 on the wall cells' density and energy rows (central differences of R, three random v, ε 10⁻⁵ and 10⁻⁷). Split into convective and viscous parts at the same state, the gap is convective: ‖g_c‖ / ‖g‖ = 0.995–1.000 in every row group, on the wall ring and over the domain; the viscous blocks, the boundary fit's included, carry at most 0.7 % (#395). The missing part is the convective flux's dependence on the reconstruction's gradients, which J̃ holds fixed and which the wall cells' aspect ratio (10³–10⁴) makes large. With the true Jacobian applied matrix-free (#397), the laminar plate converged in 1655 steps.

## Why

*(CFD - Euler.)*

## Form

*(CFD - Euler: the chain dF/dW_face · dW_face/d(∇W_c) · d(∇W_c)/dW_j through the stored least-squares weights, φ and the reconstruction's fallback held as in today's J̃; the explicit parameter of the shared convective assembly (`Euler::Impl`, `kernels/euler.py`) that only `NavierStokes` sets; the pattern constraint — only face-neighbour entries enter the 5 × 5 block matrix — and the share of least-squares weight it drops, measured on the 69×49 plate mesh as a geometric quantity before any run.)*

## Scope

`NavierStokes` sets the parameter; `EulerPhysics` does not, so every Euler run is bit for bit what `main` gives (the Euler physics' assembly is untouched). The same terms for Euler, where the limited cases are, are #409, with their own design and gate. The multigrid, the line smoother and the steady step are unchanged: the new entries land in the existing face blocks; the hierarchy and the Galerkin coarse operators take them as they take today's.

## Gate, declared before any run

On `physics/ns-sa` with this change, fp64, against the same branch without it (the case files identical):

- **laminar TMR flat plate 69×49** (`physics/tmr/case.sh flatplate`, laminar, 6000 steps): ten orders below each equation's peak (the identically zero z-momentum excepted) in at most 3332 steps, and every such equation at most 10⁻¹⁰ of its peak at the last step (#395's measure);
- **Euler bit for bit**: the Euler MMS (hex, tet, poly, 16–32), Ni's bump 96 and 384 and the NACA 0012 cases of #356 (M 0.5 α 1.25°, M 0.8 α 0°, M 0.8 α 1.25°; 128² and 256²): `history.jsonl` byte-identical to `main`'s (SHA-256), since the Euler physics never sets the parameter;
- **reported, not gated**: per step, the linear solve's drop and cycles and the number of linear failures, before and after, on the laminar plate and SA 35×25 / 69×49 / 137×97 (the new entries can reduce the blocks' diagonal dominance: the multigrid's and line smoother's behaviour is reported); ‖(J − J̃) v‖ / ‖J v‖ at the same cycle state, before and after, by #395's protocol;
- **accuracy** (AGENTS.md): the discretisation does not change (only J̃); the NS MMS observed orders of `ns-mms-order` unchanged.

The SA plates (35×25, 69×49, 137×97) are note 2's gate, with this note's flow model; RAE 2822 (transonic, limited) joins when its grid exists.
