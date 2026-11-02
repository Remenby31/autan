from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from autan.schema import Steady
from autan.texts import TEXT_BYTES

if TYPE_CHECKING:
    from autan.meshinfo import MeshInfo
    from autan.schema import CaseSpec

DeviceKind = Literal["cpu", "cuda", "hip"]
StartMode = Literal["continue", "warm"]


@dataclass(frozen=True)
class StateFileLoad:
    """A file of the case the runtime reads: its bytes, its rows, the faces of the case's
    oracle zones it gives states for and its zones, and its kind."""

    size: int
    rows: int
    faces: int
    zones: int
    kind: Literal["initial", "faces", "manufactured"]


_CUDA_CONTEXT = 512 * 2**20
# The runtime process itself on every rank (binary, libraries, MPI, the hashing buffer): the
# largest VmHWM of autan-run on the 27-cell hex/polyMesh case, 1 and 2 ranks, rounded up to
# the next MiB (#358, round 8 protocol).
_RUNTIME = 26 * 2**20
# The multigrid factors its coarsest level densely up to this many unknowns (multigrid.cpp).
_DENSE_UNKNOWNS = 1000
# CGNS element sections: 64 B per element plus its connectivity block, by shape (glibc).
_CGNS_BLOCK = {"tetra": 48, "pyramid": 48, "prism": 64, "hexa": 80}
_CGNS_FACE_BLOCK = 48  # quad, the larger boundary element
_CGNS_SECTION_BUFFERS = 2**20 + 32 * 2**10
# HDF5's defaults, which the CGNS library keeps (adfh/ADFH.c:64): a 4 MiB metadata cache
# (H5Cprivate.h:85), an 8 MiB chunk cache (H5Pfapl.c:94) and a 64 KiB sieve (H5Pfapl.c:121).
_HDF5_CACHE = 4 * 2**20 + 8 * 2**20 + 64 * 2**10
# A parsed JSON document (nlohmann) per byte of its text, at most: 16 B a value and 24 B an
# array for at least 2 characters, 112 B an object member for at least 6, doubled for a
# vector's growth.
_JSON_DOCUMENT = 40
# DeviceMesh stages each upload through at most this many values (device.cuh, upload_chunk).
_UPLOAD_CHUNK = 2**20


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
    start: StartMode | None = None,
    files: tuple[StateFileLoad, ...] = (),
    input_bytes: int = 0,
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
                start=start,
                files=files,
                input_bytes=input_bytes,
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
    start: StartMode | None,
    files: tuple[StateFileLoad, ...],
    input_bytes: int,
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
    add("runtime", _RUNTIME)
    # The solver input (at most input_bytes): the runtime keeps the case's name and text, and
    # reads and parses the document once.
    add("solver input", input_bytes)
    add("solver input setup", (1 + _JSON_DOCUMENT) * input_bytes, transient=True)
    add("mesh", mesh_bytes + geometry + 12 * cells)
    add("mesh setup", mesh_bytes + geometry)  # kept resident by the allocator: not transient
    add("mesh reading", _reading(mesh, interior, boundary), transient=True)
    # In RCM order (the runtime's default, quality.lines.order) the run keeps the renumbered
    # mesh beside the one it read, with the cell order and the face and point maps.
    if mesh.quality.lines.order == "rcm":
        add("renumbered mesh", mesh_bytes + 4 * cells + 5 * faces + 4 * nodes)
    if ranks > 1:
        add("partition", (mesh_bytes + geometry) / ranks + 9 * f)
    # The solution gathered on rank 0 at the end and the writer's copy: the n + 1 primitive
    # fields and their means; the restart's n conserved, n + 1 average sums and their weight, the
    # steady solver's n + 5 values, one circulation per point-vortex condition.
    fields = 2 * (n + 1) + n + (n + 2) + (n + 5) + vortices
    # The writer also copies the coordinates and writes 64-bit face and cell connectivity.
    k_global = 2 * interior + boundary
    writer = 24 * nodes + 8 * (max(faces, cells) + 1) + 8 * max(counts.face_nodes, k_global)
    add("solution output", 2 * 8 * cells * fields + writer, transient=True)
    # A start reads the previous solution on every rank, every field for a continuation, the
    # n + 1 primitive fields for a warm start, then takes the rank's share and frees the rest;
    # with them the descriptors its writer wrote, unfiltered, which cg_open holds until close
    # (E0707 beyond them, io/src/cgns_solution.cpp): run metadata of at most 64 KiB, also
    # copied and parsed; zone names of at most 25 600 B a zone and one more; a translation of
    # at most 4 096 B per periodic zone; and HDF5's caches.
    if start is not None:
        read = fields if start == "continue" else n + 1
        zones = len(mesh.boundary_zones) + len(mesh.cell_zones)
        metadata = (1 + 1 + _JSON_DOCUMENT) * 64 * 2**10
        descriptors = 25_600 * (zones + 1) + 4096 * 2 * len(mesh.periodic) + _HDF5_CACHE
        add("start", 8 * (cells + c) * read + metadata + descriptors, transient=True)
    # Files of the case, each read once at its size S (runtime.md): an initial state holds its
    # text, the document's strings (at most S), 1 KiB of members, its rows streamed into
    # vectors (64 B a row, 128 B while one grows) and their pairing (138 B a row, 160 B per
    # cell); face states the same with 1 200 B a zone, then 40 B per face of its oracle zones,
    # which a rank keeps; a manufactured solution's text stays for the run with its compiled
    # expressions, at most one 32 B node per byte of text in vectors up to twice that, and the
    # physics parses it once, at most 40 B of document per byte of text, while the node
    # vectors grow (the old buffer, half the new one, is freed after the copy).
    loads = 0
    for x in files:
        # A string (a member name or the oracle, at most TEXT_BYTES) lives in the text, the
        # parser's token buffer (twice its length at most), the key or value it builds and the
        # document's (runtime/src/run.cpp).
        names = 4 * TEXT_BYTES
        if x.kind == "initial":
            loads += 2 * x.size + names + 1024 + 266 * x.rows + 160 * cells
        elif x.kind == "faces":
            loads += 2 * x.size + names + 1024 + 128 * x.rows + 1200 * x.zones + 40 * x.faces
        else:
            loads += (_JSON_DOCUMENT + 32) * x.size
    if loads:
        add("state files", loads, transient=True)
    if any(x.kind == "faces" for x in files):
        add("face states", sum(40 * x.faces for x in files if x.kind == "faces"))
    if any(x.kind == "manufactured" for x in files):
        add("manufactured solution", sum(65 * x.size for x in files if x.kind == "manufactured"))

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
        + (n + 1 + n) * c * 8
        + (n + 3 * n * (device_kind == "cpu")) * c * 8
        + (n + 1) * c * s
        + n * fb * 8
    )
    add("physics", host_physics)
    # The constructor's packed geometry (25F + 4C + 4K in FP64) and the stencil's twins and
    # FP64 weights, freed when it returns.
    packed = 8 * (25 * f + 4 * c + 4 * k)
    add("physics setup", packed + 32 * stencil, transient=True)
    if device_kind != "cpu":
        # DeviceMesh keeps the face colourings on the host; it holds the packed geometry moved
        # into it until release_host after bind. Its setup passes through three phases in
        # turn, each with that geometry: construction (incidence twins, colouring work), the
        # weights' upload (its FP64 weights and the physics' copy, the limiter's unit field)
        # and bind (its weights, one staged chunk of 2^20 values).
        add("device mesh host", 4 * f)
        phases = (8 * k + 8 * f, 48 * stencil + n * c * s, 24 * stencil + 8 * _UPLOAD_CHUNK)
        add("device setup", packed + max(phases), transient=True)
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


def _reading(mesh: MeshInfo, interior: int, boundary: int) -> int:
    """What a mesh reader holds beyond the mesh it returns, at its peak (docs/sdk/plan.md,
    the readers' inventory: io/src/read_mesh.cpp, gmsh.cpp, fluent.cpp, cgns_read.cpp,
    polymesh.cpp and core/src/cellwise.cpp, mesh.cpp). N_s counts the nodes the file holds,
    unreferenced ones included; each text is read once at its size S."""
    counts = mesh.counts
    n, c, f, p, s = counts.nodes, counts.cells, counts.faces, counts.face_nodes, mesh.source.bytes
    ns = counts.source_nodes
    k = 2 * interior + boundary
    if mesh.source.format == "gmsh":
        # to_face_based, with the text, tags, points and order alive; face_nodes may double.
        reading = s + 60 * ns - 24 * n - 8 * c + 52 * f + 16 * p + 4 * p
    elif mesh.source.format == "fluent":
        # The end of assemble, with the reader's members alive; face_nodes_ may double.
        reading = s + 53 * ns + 20 * c + 12 * p + 45 * f + 4 * p
    elif mesh.source.format == "cgns":
        by_type = counts.cells_by_type.model_dump()
        elements = sum((64 + _CGNS_BLOCK[t]) * m for t, m in by_type.items() if t in _CGNS_BLOCK)
        # Labelled cells (64 B each) when the sections carry a family: bounded by every cell.
        labels = 64 * boundary + 64 * c
        # Sections are read through cgio in chunks (4 096 offsets and 2^17 entries, 1 MiB and
        # 32 KiB of buffers); the points and edges a section skips are kept as runs, at most
        # one more than its kept elements, 24 B each (io/src/cgns_read.cpp).
        runs = 24 * (c + boundary + 1) + _CGNS_SECTION_BUFFERS
        alive = 24 * ns + elements + (64 + _CGNS_FACE_BLOCK) * boundary + labels + runs
        standard = alive + 24 * ns - 24 * n + 4 * c + 52 * f + 16 * p
        # NGON_n/NFACE_n: a block per face and per cell, at most 32 B plus 4 B an entry.
        blocks = 24 * ns + (96 * f + 4 * p) + (96 * c + 4 * k) + labels + runs
        ngon = blocks + 76 * f - 4 * interior - 4 * k - 8 * c
        # cg_open loads every descriptor and keeps it until close; one filtered dataset at a
        # time is inflated beside its destination (filtered_bytes, 2L + C, docs/contracts/
        # mesh.md#memory-of-a-cgns-read); the reader parses ZoneNames, at most 40 B a byte.
        source = mesh.source
        texts = source.descriptor_bytes + source.filtered_bytes
        reading = (
            max(standard, ngon) + texts + _JSON_DOCUMENT * source.zone_names_bytes + _HDF5_CACHE
        )
    else:
        # polyMesh: the end of assemble, with the owner, neighbour and boundary texts (at most
        # the files' bytes) and the 64-bit owner and neighbour arrays alive.
        reading = s + 52 * ns - 24 * n + 33 * f + 8 * interior + 4 * p + 8 * c
    if mesh.periodic and mesh.source.format in ("gmsh", "cgns"):
        # Periodic matching at the end of assemble, every node bounding the boundary ones.
        paired = sum(pair.faces for pair in mesh.periodic) * 2
        reading += 21 * ns + 128 * ns + 113 * paired
    return reading
