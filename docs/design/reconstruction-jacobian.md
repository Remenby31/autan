# The reconstruction's gradients in the approximate Jacobian of viscous meshes

Design for #395 (note 1 of 2: the flow's model; note 2, the coupled SA solve, follows #137 and the RAE 2822 grid), written before any code by CFD - Euler and CFD - Multigrille. The steady implicit step solves (V/Δt + J̃) ΔU = −R(U) with J̃ the assembled approximate Jacobian: every face's dF/dW with the reconstruction's gradients and the limiter held fixed ([euler.md](euler.md#time-stepping)). This note adds, in the shared convective assembly and only where `NavierStokes` asks for it, the convective flux's dependence on the cells' reconstruction gradients through their least-squares weights.

## Problem, measured

TMR flat plate 69×49 (#395, `physics/ns-sa` c3966068): once the SER raises the CFL number past ~10³ at second order, the SA plate locks into a period-2 cycle (full steps, solves within tolerance, residual at −3 to −4); the laminar plate wanders without converging; with ν̃ and μ_t frozen inside the cycle, the flow wanders as the laminar one does (#395, isolation diagnostic). At a state of the cycle, ‖(J − J̃) v‖ / ‖J v‖ is 10⁻⁹–10⁻² at first order and, at second order, 0.15–0.25 over the domain and 0.38–0.70 on the wall cells' density and energy rows (central differences of R, three random v, ε 10⁻⁵ and 10⁻⁷). Split into convective and viscous parts at the same state, the gap is convective: ‖g_c‖ / ‖g‖ = 0.995–1.000 in every row group, on the wall ring and over the domain; the viscous blocks, the boundary fit's included, carry at most 0.7 % (#395). The missing part is the convective flux's dependence on the reconstruction's gradients, which J̃ holds fixed and which the wall cells' aspect ratio (10³–10⁴) makes large. With the true Jacobian applied matrix-free (#397), the laminar plate converged in 1655 steps.

## Why

The second-order face state is w_f = w_c + φ_c ⊙ (∇w_c · r_cf), with ∇w_c = Σ_j W_cj (w_j − w_c) the least-squares fit over the cell's stencil (`stencil_lsq_weight`, face neighbours and their neighbours) and φ the limiter, componentwise. J̃ differentiates F(w_L,f, w_R,f) with respect to w_L and w_R only through the first term: ∂w_f/∂w_c = I. The fit adds ∂w_f/∂w_j = diag(φ_c) (W_cj · r_cf) for every stencil cell j and −diag(φ_c) Σ_j (W_cj · r_cf) on w_c. On an isotropic cell |W_cj · r_cf| is of order ½ for the cell across the face, so the omission is a fraction of the term kept. In a wall cell stretched 10³–10⁴, the fit's weights along the wall normal fall on the cells above and below a cell of height h, while the faces' r runs along the wall over lengths up to 10³–10⁴ h. Then |W_cj · r_cf| is no longer bounded by ½, and the omitted part can exceed the kept one. That is what #395 measures: 38–70 % of J v missing on the wall cells' continuity and energy rows, which carry no viscous flux.

Two choices keep this from becoming #397:
- **The operator stays the assembled J̃**: the solve's operator and the preconditioner, with these entries added. It is not the matrix-free Jacobian, and nothing depends on a perturbed state.
- **φ and the reconstruction's positivity fallback stay held at U**, exactly as J̃ already holds them. This note adds the gradients' dependence only: no derivative through the limiter's minimum and maximum, or through the fallback's branch.

#397's loss on the limited Euler cases came with a model that changed both the operator (matrix-free) and how much of the true derivative it followed. This one only fills in a smooth part of the linear model, the least-squares fit, which is linear in w. There is still no a priori guarantee that the step stays as robust. The new entries move weight off the diagonal and can reduce the blocks' diagonal dominance, which the multigrid and the line smoother rely on. Hence the gate reports the linear drops, cycles and failures before and after, and the Euler cases stay bit for bit (Scope).

## Form

**The chain.** For an interior face f = (L, R) and each side s ∈ {L, R} whose reconstruction is not the cell value (the fallback held as in J̃), the generated face Jacobian already gives A_s = ∂F/∂w_s,f (5 × 5, the `*_faces_jacobian` loops, gradients and φ held). The added blocks, times dW/dU of the columns' cells as J̃'s blocks are, are:
- for every column cell j of the side's gradient: A_s diag(φ_s) (Ŵ_sj · r_sf), into row L with the face's sign (+) and row R with its sign (−);
- on column s: −A_s diag(φ_s) Σ_j (Ŵ_sj · r_sf).

Boundary faces add the same terms to their owner's row, A from `*_boundaries_jacobian`. These are assembly-time products of stored blocks: no residual evaluation, no generated loop changes.

**Which weights.** The block matrix has a 5 × 5 block for a cell and each of its face neighbours, and nothing else. The stencil's own weights W reach the face neighbours' neighbours. Truncating them to the pattern drops 42 % of Σ |W_sj · r_sf| over the wall ring's faces and 52 % over the domain. It also leaves a gradient that no longer reproduces a linear field (Σ_j W_cj (x_j − x_c) ≠ I once entries are removed). The chain therefore uses its own fit Ŵ, a least-squares gradient over the cell's face neighbours only, with the same determinacy rule as the stencil's (a cell whose face neighbours cannot determine a direction leaves it out). Ŵ is exact for linear fields and lies entirely in row s's pattern. What it still cannot place are the cross terms: row t (the other side) of a face receives column j only when j is t or one of t's face neighbours. Measured on the 69×49 plate mesh, as a geometric quantity before any run (Σ over faces and both rows of |Ŵ_sj · r_sf| with the column outside the pattern, against all terms): 19 % on the wall ring's faces, 25 % over the domain. Ŵ is computed once at setup: 2D-or-3D like the stencil's (the plate is one cell thick: the offsets' components within 64 ulps are zero, euler.md), stored 3 per face incidence (3 × 2 F_interior doubles), freed with the matrix.

**The parameter.** `Scheme::reconstruction_jacobian` (default false): `Euler::assemble` adds the blocks above when set. `NavierStokes` sets it, and `EulerPhysics` does not. The code is written once, in `Euler::Impl`, where the face blocks are assembled; #409 would set it for Euler.

**Cost.** Setup: one 3 × 3 solve per cell and 3 doubles per face incidence. Per assembly: per face side, one 5 × 5 by diagonal-5 product per column, about the face neighbour count, added into existing blocks. Measured in the gate as CPU time per step against the branch without it.

## Scope

`NavierStokes` sets the parameter; `EulerPhysics` does not, so every Euler run is bit for bit what `main` gives (the Euler physics' assembly is untouched). The same terms for Euler, where the limited cases are, are #409, with their own design and gate. The multigrid, the line smoother and the steady step are unchanged: the new entries land in the existing face blocks; the hierarchy and the Galerkin coarse operators take them as they take today's.

## Gate, declared before any run

On `physics/ns-sa` with this change, fp64, against the same branch without it (the case files identical):

- **laminar TMR flat plate 69×49** (`physics/tmr/case.sh flatplate`, laminar, 6000 steps): ten orders below each equation's peak (the identically zero z-momentum excepted) in at most 3332 steps, and every such equation at most 10⁻¹⁰ of its peak at the last step (#395's measure);
- **Euler bit for bit**: the Euler MMS (hex, tet, poly, 16–32), Ni's bump 96 and 384 and the NACA 0012 cases of #356 (M 0.5 α 1.25°, M 0.8 α 0°, M 0.8 α 1.25°; 128² and 256²): `history.jsonl` byte-identical to `main`'s (SHA-256), since the Euler physics never sets the parameter;
- **reported, not gated**: per step, the linear solve's drop and cycles and the number of linear failures, before and after, on the laminar plate and SA 35×25 / 69×49 / 137×97 (the new entries can reduce the blocks' diagonal dominance: the multigrid's and line smoother's behaviour is reported); ‖(J − J̃) v‖ / ‖J v‖ at the same cycle state, before and after, by #395's protocol;
- **accuracy** (AGENTS.md): the discretisation does not change (only J̃); the NS MMS observed orders of `ns-mms-order` unchanged.

The SA plates (35×25, 69×49, 137×97) are note 2's gate, with this note's flow model; RAE 2822 (transonic, limited) joins when its grid exists.
