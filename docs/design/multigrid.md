# Agglomeration multigrid

The linear solver of the implicit step of [PLAN.md](../PLAN.md) M3: each pseudo-transient step of a steady run solves (V/Δt + ∂R/∂U) ΔU = −R ([euler.md](euler.md), [ns-sa.md](ns-sa.md#implicit-solver)) with an agglomeration multigrid whose cost per order of residual drop does not grow with the mesh. Code: `solvers/` (`autan::solvers::Multigrid`), called by the steady physics in `physics/src/euler_physics.cpp`. Issue: #148.

## Why

Block Gauss-Seidel damps the error components a few cells long. The viscous Jacobian is elliptic, so its smooth components need O(n²) sweeps on a mesh n cells wide; four sweeps per step left the linear error large and the number of steps grew with the mesh (laminar MMS: 216 steps on 24³, 706 on 48³). FUN3D, NSU3D and the Fluent coupled solver remove the smooth error on coarser agglomerated levels and solve the anisotropic wall layers along lines ([research §1.7](../research/state-of-the-art-2026-10.md)).

## Choices

| Question | Choice | Reason |
|---|---|---|
| Linear or nonlinear (FAS) multigrid | **Linear**, on the Jacobian of the implicit step | Mavriplis (2002, J. Comput. Phys. 175:302–325): with the Jacobian the outer iteration already uses, linear multigrid converges as fast per cycle as FAS and costs less, since no residual, geometry or boundary condition is evaluated on coarse levels. The outer loop (SER CFL, admissible step, first-order start) and its fixed point R(U) = 0 are unchanged, so the converged solution is the same discrete solution. |
| Coarse operators | **Galerkin** A_c = Pᵀ A P, P constant on each agglomerate | Exact for any agglomerate and any Jacobian (Euler, viscous, boundary blocks, the V/Δt shift) with no coarse geometry: block sums, in a fixed order. Rediscretisation would need the residual's kernels and boundary conditions on agglomerated polyhedra. Its weakness, a coarse diffusion operator about twice too stiff for a 2 × 2 × 2 agglomerate, is corrected by the Krylov acceleration below. |
| Cycle | **K-cycle** (Notay and Vassilevski 2008, Numer. Linear Algebra Appl. 15:123–145): each coarse solve is two steps of GCR preconditioned by the next level's cycle | V and W cycles with piecewise-constant transfers lose convergence as levels are added on diffusion; the Krylov steps restore it at the cost of two products per level. V and W remain options. |
| Outer solve | **GCR** preconditioned by one cycle per iteration, to a relative residual of 0.1 (at most 10 iterations) | Inexact Newton: a tighter linear tolerance does not reduce the number of steps of the defect-correction outer loop. A step whose solve stops above the tolerance is taken and logged (`linear` warning in `log.jsonl`); nothing falls back to Gauss-Seidel. |
| Smoother | **Line-implicit block Gauss-Seidel**, one symmetric sweep before and after each coarse correction | Lines are solved exactly by block-tridiagonal elimination; the other cells are point block Gauss-Seidel, as before. Lines of one colour share no face, so their solves are independent; colours are visited forwards then backwards. |

## Lines

The coupling of a face is w_f = A_f / |x_N − x_O| (that of a boundary face uses the distance to the mirror cell, 2 |x_f − x_c|): the diffusion coefficient a viscous Jacobian has between the two cells. A cell is anisotropic when its strongest coupling is at least 4 times its weakest (aspect ratio ≥ 2). Seeds are the anisotropic cells, boundary cells first, then by decreasing anisotropy. A line grows from its seed, in both directions, through the strongest face whose coupling is at least 4 times the weakest coupling of both cells (Mavriplis 1998, J. Comput. Phys. 145:141–165). On wall layers, each wall cell starts a line normal to the wall that stops where the cells become isotropic.

## Agglomeration

Each level is a graph: nodes (cells, then agglomerates), edges (faces between them) with their summed couplings, the cell zone, the boundary zones touched and the sum of the boundary area vectors. A coarse node:

- never mixes cell zones;
- holds only boundary zones its seed touches, and boundary faces facing the same side (Σ S_b · Σ S_b' > 0), so that an agglomerate keeps its patches and never wraps around a thin wall.

Lines coarsen first, **in the wall plane**: a line is grouped with the neighbouring lines it touches layer by layer, and layer k of the group becomes one coarse node, so a coarse level keeps lines of the same length and the line smoother keeps working on coarse levels. The other cells form compact isotropic agglomerates of at most 8 nodes, grown from a seed by adding the free neighbour with the most faces on the agglomerate, the nearest to the seed among equals (2 × 2 × 2 blocks of hexahedra); seeds follow an advancing front from the boundary corners inwards (Mavriplis and Venkatakrishnan 1996, Comput. Fluids 25:361–378). A node left alone joins the neighbouring agglomerate it is most coupled to. Coarsening stops at 64 nodes, or when a level would keep more than 80 % of the nodes; the coarsest level is factored densely (LU with partial pivoting) when it has at most 1000 unknowns.

The hierarchy is built once from the mesh and its geometry. A mesh with ghost cells is refused (`std::invalid_argument`): coarse levels across partitions are M5.

## Determinism and precision

Every loop runs in a fixed order and every choice breaks ties by index, so a run is bitwise reproducible. Blocks, vectors and reductions are FP64 in both precision policies; the Jacobian is assembled in FP64 by the physics.

## Verification

| Test | Oracle | Where |
|---|---|---|
| Agglomerates connected, one cell zone, coarse edges one per adjacent pair; lines join adjacent nodes and partition the nodes; reference meshes and wall layers | definition | `tests/solvers/test_multigrid.cpp` |
| Wall layers (hexahedra, prisms, first cell 10⁻⁴): one line per wall cell, kept on every coarse level | geometry of the fixture | same |
| One line sweep solves a block-tridiagonal system | exact solution, 1000 ε | same |
| Two-level cycle without smoothing = P (PᵀAP)⁻¹ Pᵀ b | dense product and elimination | same |
| Iterations to a 10⁻⁶ drop on diffusion, isotropic hexahedra and prisms 8 → 32 wide | ≤ 9, growing by at most 2 | same |
| Wall layers of aspect ratio 10⁴, with and without lines | ≤ 9 with lines, at least twice as many without | same |
| Steps to ten orders on Euler and laminar Navier-Stokes MMS, hexahedra 16³ → 96³, against Gauss-Seidel alone | [measurements](#measurements) | `physics/mms/study.sh` |

## Measurements

Filled from the study of the pull request of #148.
