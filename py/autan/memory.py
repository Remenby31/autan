from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from autan.schema import Steady

if TYPE_CHECKING:
    from autan.meshinfo import MeshInfo
    from autan.schema import CaseSpec

DeviceKind = Literal["cpu", "cuda", "hip"]
_CUDA_CONTEXT = 512 * 2**20
# The multigrid factors its coarsest level densely up to this many unknowns (multigrid.cpp).
_DENSE_UNKNOWNS = 1000


class Component(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    bytes_host: int
    bytes_device: int
    transient: bool


class Memory(BaseModel):
    """A run's memory per rank at its peak (model ``autan.memory/2``, docs/sdk/plan.md): what
    each rank allocates on the host and on its device, by component."""

    model_config = ConfigDict(frozen=True)
    model: Literal["autan.memory/2"] = "autan.memory/2"
    device_kind: DeviceKind
    gpus: int
    equations: int
    bytes_host: int
    bytes_device: int
    bytes_per_gpu: int
    bytes_total: int
    components: tuple[Component, ...]


def estimate(
    spec: CaseSpec,
    mesh: MeshInfo,
    *,
    equations: int,
    ranks: int,
    device_kind: DeviceKind,
    vortices: int = 0,
) -> Memory:
    return max(
        (
            _rank(
                spec,
                mesh,
                share,
                equations=equations,
                ranks=ranks,
                device_kind=device_kind,
                vortices=vortices,
            )
            for share in _shares(mesh, ranks)
        ),
        key=lambda memory: memory.bytes_per_gpu,
    )


def _shares(mesh: MeshInfo, ranks: int) -> list[tuple[int, int, int, int]]:
    """Per rank: its cells with ghosts, faces, interior faces and stencil entries."""
    counts, partition = mesh.counts, mesh.partition
    if partition is not None and partition.ranks == ranks:
        return list(
            zip(
                (
                    a + b
                    for a, b in zip(partition.cells_by_rank, partition.ghosts_by_rank, strict=True)
                ),
                partition.faces_by_rank,
                partition.interior_faces_by_rank,
                partition.stencil_entries_by_rank,
                strict=True,
            )
        )
    interior = counts.faces - sum(zone.faces for zone in mesh.boundary_zones)
    if ranks == 1:
        # A single rank's periodic images, each with at most max_faces_per_cell faces.
        extra = counts.periodic_ghosts * mesh.quality.max_faces_per_cell
        cells = counts.cells + counts.periodic_ghosts
        return [(cells, counts.faces + extra, interior + extra, counts.stencil_entries)]
    # No partition for these ranks: the plan refuses the run (E0210), the estimate omits halos.
    return [
        (
            -(-counts.cells // ranks),
            -(-counts.faces // ranks),
            -(-interior // ranks),
            -(-counts.stencil_entries // ranks),
        )
    ]


def _rank(
    spec: CaseSpec,
    mesh: MeshInfo,
    share: tuple[int, int, int, int],
    *,
    equations: int,
    ranks: int,
    device_kind: DeviceKind,
    vortices: int,
) -> Memory:
    counts = mesh.counts
    n = equations
    s = 8 if spec.physics.precision == "fp64" else 4
    cells, faces, nodes = counts.cells, counts.faces, counts.nodes
    interior = faces - sum(zone.faces for zone in mesh.boundary_zones)
    boundary = faces - interior
    c, f, fi, stencil = share
    fb = f - fi
    k = 2 * fi + fb

    parts: list[Component] = []

    def add(name: str, host: float, device: float = 0, *, transient: bool = False) -> None:
        parts.append(
            Component(
                name=name, bytes_host=int(host), bytes_device=int(device), transient=transient
            )
        )

    # Core (the #325 inventory): the global mesh and its FP64 geometry on every rank, the
    # colourings, and what setup leaves resident (the reader's form and the MeshInfo pass,
    # about the mesh and its geometry again).
    mesh_bytes = 24 * nodes + 16 * faces + 4 * counts.face_nodes + 12 * interior + 4 * boundary
    mesh_bytes += 20 * cells
    geometry = 56 * faces + 32 * cells
    add("mesh", mesh_bytes + geometry + 12 * cells)
    add("mesh setup", mesh_bytes + geometry)  # kept resident by the allocator: not transient
    if ranks > 1:
        add("partition", (mesh_bytes + geometry) / ranks + 9 * f)
    # The solution gathered on rank 0 at the end and the writer's copy: the n + 1 primitive
    # fields and their means; the restart's n conserved, n + 1 average sums and their weight, the
    # steady solver's n + 5 values, one circulation per point-vortex condition.
    fields = 2 * (n + 1) + n + (n + 2) + (n + 5) + vortices
    add("solution output", 2 * 8 * cells * fields, transient=True)

    # The physics (Euler's inventory, scaled from its 5 equations to n): geometry, stencil
    # and state on the host, always; the same on the device for a device run.
    geometry_packed = 8 * (16 * f + 4 * c + 3 * stencil)
    host_physics = (
        geometry_packed
        + 8 * c
        + 4 * stencil
        + 8 * f
        + 2 * n * f * s
        + 5 * n * c * s
        + 3 * n * c * s
        + (4 * n + n + 1 + n) * c * 8
        + (n + 1) * c * s
        + n * fb * 8
    )
    add("physics", host_physics)
    # The constructor's packed geometry (25F + 4C + 4K in FP64) and the stencil's twins and
    # FP64 weights, freed when it returns.
    packed = 8 * (25 * f + 4 * c + 4 * k)
    add("physics setup", packed + 32 * stencil, transient=True)
    if device_kind != "cpu":
        # DeviceMesh keeps on the host the packed geometry, its FP64 stencil weights, a copy of
        # the stencil and the face colourings. Its setup passes through three phases in turn:
        # construction (a second packed copy, incidence twins, colouring work), the weights'
        # upload (an FP64 copy, the limiter's unit field) and bind (one field converted, the
        # face moments 72F at most).
        add("device mesh host", packed + 24 * stencil + 8 * (c + 1) + 4 * stencil + 4 * f)
        setup = max(packed + 8 * k + 8 * f, 24 * stencil + n * c * s, 72 * f)
        add("device setup", setup, transient=True)
        device = (
            2 * n * c * 8
            + 2 * c * 8
            + 2 * n * c * s
            + n * f * s
            + 3 * n * c * s
            + 3 * n * c * s
            + c * s
            + n * c * 8
            + n * c * s
            + 8 * f
            + n * f * s
            + c
            + 8 * c
            + (8 * f + 8 * c + 12 * k)
            + (12 * stencil + 8 * c)
            + geometry_packed
            + n * f * 8
        )
        add("physics device", 0, device)
        add("device context", _CUDA_CONTEXT)

    # The implicit steady solve (host, CPU runs only): Jacobians, block matrix and the linear
    # solver (the multigrid inventory: 5x5 blocks, scaled by n²/25).
    if isinstance(spec.time, Steady) and device_kind == "cpu":
        block = 8 * n * n
        add(
            "jacobian",
            3 * n * n * f * s + n * n * c * s + block * c + 2 * block * fi + 8 * c + 4 * k + 8 * c,
        )
        name, persistent, solver_setup = _linear_solver(
            spec.time.linear_solver, c, fi, block, min(mesh.quality.lines.cells, c), mesh
        )
        add(name, persistent)
        add(f"{name} setup", solver_setup, transient=True)

    # The peak holds every persistent component and the largest transient one: transients
    # are freed in turn, and later allocations reuse their memory.
    host = sum(p.bytes_host for p in parts if not p.transient)
    host += max((p.bytes_host for p in parts if p.transient), default=0)
    device = sum(p.bytes_device for p in parts)
    per_rank = host + device
    return Memory(
        device_kind=device_kind,
        gpus=ranks,
        equations=n,
        bytes_host=host,
        bytes_device=device,
        bytes_per_gpu=per_rank,
        bytes_total=per_rank * ranks,
        components=tuple(parts),
    )


def _linear_solver(
    solver: str, cells: int, interior: int, block: int, lines: int, mesh: MeshInfo
) -> tuple[str, float, float]:
    scale = block / 200
    graph = 300 * cells + 60 * interior
    if solver == "gauss-seidel":
        return "gauss-seidel", block * cells + 8 * cells, 0
    if solver == "lines":
        # Persistent 1 700 n0 + 8 E0, and the lines built on the whole mesh at setup.
        return "lines", scale * 1700 * cells + 8 * interior, graph
    # The hierarchy the run builds (MeshInfo quality.multigrid, one rank): per coarse level its
    # nodes and edges; line smoothing on the finest level's cells in lines (quality.lines) and,
    # when there are lines, on every coarse node.
    hierarchy = mesh.quality.multigrid
    coarse = sum(hierarchy.cells_by_level[1:])
    finest = 1200 * cells + 400 * lines + 21 * interior
    levels = (
        1000 * coarse + 400 * (coarse if lines else 0) + 410 * sum(hierarchy.edges_by_level[1:])
    )
    # The coarsest level, factored densely up to 1000 unknowns of 5 equations: LU and pivots.
    unknowns = 5 * hierarchy.cells_by_level[-1]
    dense = 8 * unknowns * unknowns + 8 * unknowns if unknowns <= _DENSE_UNKNOWNS else 0
    return "multigrid", scale * (finest + levels + dense), graph
