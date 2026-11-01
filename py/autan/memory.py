from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from autan.schema import Steady

if TYPE_CHECKING:
    from autan.meshinfo import MeshInfo
    from autan.schema import CaseSpec

DeviceKind = Literal["cpu", "cuda", "hip"]
# Gradient stencil entries per cell: the largest measured on the V&V families (the
# combinatorial bound f² for pyramids, which were not measured).
_STENCIL = {"tetra": 16, "pyramid": 25, "prism": 19, "hexa": 24, "polyhedron": 105}
_CUDA_CONTEXT = 512 * 2**20
# Multigrid levels below the finest, as a fraction of its cells: 0.6 on meshes with boundary
# layers (prisms, pyramids), whose stretched cells coarsen along lines; 0.15 otherwise
# (measured 0.13 to 0.14 in 3D); 7 coupled neighbours per coarse cell.
_COARSE_STRETCHED, _COARSE_ISOTROPIC, _COARSE_DEGREE = 0.6, 0.15, 7


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
    spec: CaseSpec, mesh: MeshInfo, *, equations: int, ranks: int, device_kind: DeviceKind
) -> Memory:
    counts = mesh.counts
    n = equations
    s = 8 if spec.physics.precision == "fp64" else 4
    cells, faces, nodes = counts.cells, counts.faces, counts.nodes
    boundary = sum(zone.faces for zone in mesh.boundary_zones)
    interior = faces - boundary
    # Per rank: the cells and faces a rank solves; the global mesh is kept on every rank.
    c, f, fi = -(-cells // ranks), -(-faces // ranks), -(-interior // ranks)
    fb = f - fi
    k = 2 * fi + fb
    stencil = -(
        -sum(_STENCIL[t] * m for t, m in counts.cells_by_type.model_dump().items()) // ranks
    )

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
    # The solution gathered on rank 0 at the end, fields and restart, and the writer's copy.
    add("solution output", 2 * 8 * cells * (2 * n + 2), transient=True)

    # The physics (Euler's inventory, scaled from its 5 equations to n): geometry, stencil
    # and state on the host, always; the same on the device for a device run.
    geometry_packed = 8 * (16 * f + 4 * c + 3 * stencil)
    host_physics = (
        geometry_packed
        + 8 * c
        + 36 * stencil
        + 8 * f
        + 2 * n * f * s
        + 7 * n * c * s
        + 2 * 3 * n * c * s
        + (4 * n + n + 1 + n) * c * 8
        + (n + 1) * c * s
        + n * fb * 8
    )
    add("physics", host_physics)
    add("physics setup", 8 * (16 * f + 4 * c + 3 * k) + 24 * stencil, transient=True)
    if device_kind != "cpu":
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
        stretched = counts.cells_by_type.prism + counts.cells_by_type.pyramid > 0
        name, persistent, setup = _linear_solver(spec.time.linear_solver, c, fi, block, stretched)
        add(name, persistent)
        add(f"{name} setup", setup, transient=True)

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
    solver: str, cells: int, interior: int, block: int, stretched: bool
) -> tuple[str, float, float]:
    scale = block / 200
    graph = 300 * cells + 60 * interior
    if solver == "gauss-seidel":
        return "gauss-seidel", block * cells + 8 * cells, 0
    if solver == "lines":
        # Persistent 1 700 n0 + 8 E0, and the lines built on the whole mesh at setup.
        return "lines", scale * 1700 * cells + 8 * interior, graph
    coarse = (_COARSE_STRETCHED if stretched else _COARSE_ISOTROPIC) * cells
    finest = 1200 * cells + 400 * cells + 21 * interior
    levels = 1000 * coarse + 400 * coarse + 410 * _COARSE_DEGREE * coarse
    return "multigrid", scale * (finest + levels), graph
