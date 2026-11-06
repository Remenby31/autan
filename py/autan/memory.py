from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from autan.manufactured import MMS_BYTES
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
    oracle zones it gives states for and its zones, its kind, and the bytes of its member
    names."""

    size: int
    rows: int
    faces: int
    zones: int
    kind: Literal["initial", "faces", "manufactured"]
    names: int = 0


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
# What cg_open and the reader hold per node of a CGNS tree, the upper envelope measured by
# « CFD - Noyau maillage » (mesh.md#memory-of-a-cgns-read): 1 MiB and 16 KiB a node.
_CGNS_TREE = 2**20
_CGNS_NODE = 16 * 2**10
# The most nodes a CGNS file may hold (E0133 for a mesh, E0707 for a start file): four times the
# largest shipped grid's 4 032 (CRM-HL), rounded to a power of two.
_CGNS_NODES_MAX = 2**14
# The data cg_open reads at once beyond descriptors, point sets and the arrays it reads lazily
# (cgns_internals.c cgi_read_array): at most 1 MiB a file (E0133, E0707 for a start).
_CGNS_EAGER = 2**20
# A Fluent file's fluid, solid and interior zone records, a 64 B map node each (id, offset), at
# most 16 384 (E0133).
_FLUENT_ZONES = 64 * 16_384
# Its sections of interior faces, at most 16 384 (E0133), each a zone named by its id: 299 B
# measured per zone (autan-mesh-info on a 16 385-cell column), charged 512 B.
_FLUENT_INTERIOR_ZONES = 512 * 16_384
# Gmsh's batch of element-block entities looked up in $Entities: 65 536 pairs of 8 B.
_GMSH_BATCH = 65_536 * 8
# The CGNS reader holds 8 B per entry its point lists give, in storage sized once, and reads them
# through a 1 MiB chunk (io/src/cgns_read.cpp).
_CGNS_POINT_LIST_CHUNK = 2**20
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


def _shares(mesh: MeshInfo, ranks: int) -> list[tuple[int, int, int, int, int | None]]:
    """Per rank: its cells with ghosts, faces, interior faces and stencil entries, and its index
    in the MeshInfo's partition (``None`` without one)."""
    counts, partition = mesh.counts, mesh.partition
    if partition is not None and partition.ranks == ranks:
        return [
            (
                partition.cells_by_rank[r] + partition.ghosts_by_rank[r],
                partition.faces_by_rank[r],
                partition.interior_faces_by_rank[r],
                partition.stencil_entries_by_rank[r],
                r,
            )
            for r in range(ranks)
        ]
    interior = counts.faces - sum(zone.faces for zone in mesh.boundary_zones)
    if ranks == 1:
        # A single rank's periodic images, each with at most max_faces_per_cell faces; the
        # MeshInfo's one-part partition, when it gives one, for the part the runtime extracts.
        extra = counts.periodic_ghosts * mesh.quality.max_faces_per_cell
        cells = counts.cells + counts.periodic_ghosts
        one = 0 if partition is not None and partition.ranks == 1 else None
        return [(cells, counts.faces + extra, interior + extra, counts.stencil_entries, one)]
    # No partition for these ranks: the plan refuses the run (E0210), the estimate omits halos.
    return [
        (
            -(-counts.cells // ranks),
            -(-counts.faces // ranks),
            -(-interior // ranks),
            -(-counts.stencil_entries // ranks),
            None,
        )
    ]


def extracts(spec: CaseSpec, mesh: MeshInfo, ranks: int) -> bool:
    """Whether the runtime runs on an extracted partition: on several ranks, or on one with
    periodic pairs or reproducibility "ranks" (runtime/src/run.cpp, make_part)."""
    return ranks > 1 or bool(mesh.periodic) or spec.physics.reproducibility == "ranks"


def _partition(mesh: MeshInfo, r: int) -> tuple[int, int, int]:
    """Rank r's local mesh and halo, kept for the run (host, then the device copy of the halo
    buffers), and what extracting it holds at its peak (docs/design/setup-memory.md, partition
    and halos, « CFD - Noyau maillage »)."""
    p, counts = mesh.partition, mesh.counts
    assert p is not None
    n, fn, sent = p.nodes_by_rank[r], p.face_nodes_by_rank[r], p.sent_cells_by_rank[r]
    c, g, f, fi = (
        p.cells_by_rank[r],
        p.ghosts_by_rank[r],
        p.faces_by_rank[r],
        p.interior_faces_by_rank[r],
    )
    local = c + g
    # The local mesh (points, face nodes, faces, cells, ghost translations), its geometry and
    # face ids, the halo pattern and exchange (176 B a sent cell) and the ghosts' slots.
    kept = (
        24 * n
        + 4 * fn
        + 20 * f
        + 8 * fi
        + 20 * local
        + 24 * g
        + 56 * f
        + 32 * local
        + 9 * f
        + 176 * sent
        + 128 * g
    )
    halo_device = 128 * sent + 128 * g
    # Extraction: MeshInput, the maps and sets (64 B a node), assemble's temporaries, one
    # neighbour's recomputed ghosts, and the covering and cell lists of the global mesh beside
    # them; the partitioner's weights, freed before, are the alternative.
    neighbour = max(a + b for a, b in zip(p.cells_by_rank, p.ghosts_by_rank, strict=True))
    extraction = (
        124 * n
        + 8 * fn
        + 122 * f
        + 84 * local
        + 120 * g
        + 8 * sent
        + 128 * neighbour
        + 6 * counts.faces
        + 16 * counts.nodes
        + 8 * counts.cells
        + 16 * fn  # the local mesh's closure check, at most 16 B a face node
    )
    return kept, halo_device, max(extraction, 8 * counts.cells)


def _rank(
    spec: CaseSpec,
    mesh: MeshInfo,
    share: tuple[int, int, int, int, int | None],
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
    c, f, fi, stencil, rank = share
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
    # Rank 0 reads its executable in 1 MiB chunks to hash it (results.md, solver.binary_sha256).
    add("binary hash", 2**20, transient=True)
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
    if rank is not None and mesh.partition is not None and extracts(spec, mesh, ranks):
        kept, halo_device, extraction = _partition(mesh, rank)
        add("partition", kept, halo_device if device_kind != "cpu" else 0)
        add("partition setup", extraction, transient=True)
    elif ranks > 1:
        # No partition for these ranks: the plan refuses the run (E0210).
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
        # cg_open's tree (mesh.md#memory-of-a-cgns-read) at the node count the runtime accepts
        # in a start file (E0707 beyond, before cg_open).
        descriptors += _CGNS_TREE + _CGNS_NODE * _CGNS_NODES_MAX + _CGNS_EAGER
        add("start", 8 * (cells + c) * read + metadata + descriptors, transient=True)
    # Files of the case, each read once at its size S (runtime.md): an initial state holds its
    # text, the document's strings (at most S), 1 KiB of members, its rows streamed into
    # vectors (64 B a row, 128 B while one grows) and their pairing (138 B a row, 160 B per
    # cell); face states the same with 1 200 B a zone, then 40 B per face of its oracle zones,
    # which a rank keeps; a manufactured solution's text stays for the run, read into a buffer of
    # the contract's bound, with its compiled
    # expressions, at most one 32 B node per byte of text in vectors up to twice that, and the
    # physics parses it once, at most 40 B of document per byte of text, while the node
    # vectors grow (the old buffer, half the new one, is freed after the copy).
    loads = 0
    for x in files:
        # A string (a member name or the oracle, at most TEXT_BYTES) lives in the text, the
        # parser's token buffer (twice its length at most), the key or value it builds and the
        # document's (runtime/src/run.cpp); an object that grows copies all its member names
        # beside the old ones (nlohmann/ordered_map.hpp), at most every name of the file once.
        names = 4 * TEXT_BYTES + x.names
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
        # The text in a buffer of MMS_BYTES + 1, which the runtime reads into and keeps
        # (runtime/src/run.cpp), and its compiled expressions.
        texts = sum(MMS_BYTES + 1 + 64 * x.size for x in files if x.kind == "manufactured")
        add("manufactured solution", texts)

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
    # Under reproducibility "ranks", the global order the physics keeps: 4 B an incidence, 8 its
    # twin, 1 a face's orientation (codegen/cpp/geometry.cpp, global_order).
    ranks_order = 12 * k + f if spec.physics.reproducibility == "ranks" else 0
    add("physics", host_physics + ranks_order)
    # The constructor's packed geometry (25F + 4C + 4K in FP64), the stencil's twins and FP64
    # weights, the stencil's marks (4C + 4F_b), a vertex stencil's node lists (16N + 8P) and,
    # under "ranks", a first global order built and freed in pack_geometry: freed when it
    # returns (docs/design/setup-memory.md).
    packed = 8 * (25 * f + 4 * c + 4 * k)
    marks = 4 * c + 4 * fb + 16 * nodes + 8 * counts.face_nodes
    first_order = 12 * k if spec.physics.reproducibility == "ranks" else 0
    add("physics setup", packed + 32 * stencil + marks + first_order, transient=True)
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
        # The kernels' face scratch, 5 values of 8 B a face (euler_device face_scratch_width).
        add("device face scratch", 0, 40 * f)
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
    return "multigrid", scale * (finest + levels + dense), _agglomeration(mesh)


def _agglomeration(mesh: MeshInfo) -> int:
    """What building the hierarchy holds at its peak beyond what it keeps (docs/design/
    setup-memory.md, « CFD - Multigrille »): the line grouping with its votes (which bounds the
    isotropic loop after it), the coarse-edge build, or the colouring of the finest level's
    lines, whichever is larger; every face of a cell pair counts,
    and the lines are at most the cells."""
    hierarchy = mesh.quality.multigrid
    n, f = hierarchy.cells_by_level[0], hierarchy.edges_by_level[0]
    n1 = hierarchy.cells_by_level[1] if len(hierarchy.cells_by_level) > 1 else 0
    f1 = hierarchy.edges_by_level[1] if len(hierarchy.edges_by_level) > 1 else 0
    b = mesh.counts.faces - f
    grouping = 160 * f + 284 * n + 72 * b
    coarse_edges = 69 * f + 48 * f1 + 296 * n + 156 * n1 + 144 * b
    colouring = 32 * n + 32 * f
    return max(grouping, coarse_edges, colouring)


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
        reading = s + 60 * ns - 24 * n - 8 * c + 52 * f + 16 * p + 4 * p + _GMSH_BATCH
    elif mesh.source.format == "fluent":
        # The end of assemble, with the reader's members alive; face_nodes_ may double.
        reading = (
            s + 53 * ns + 20 * c + 12 * p + 45 * f + 4 * p + _FLUENT_ZONES + _FLUENT_INTERIOR_ZONES
        )
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
        # cg_open and the reader hold up to 16 KiB per node of the tree, which the MeshInfo
        # counts (at most _CGNS_NODES_MAX, E0133 beyond).
        texts += _CGNS_TREE + _CGNS_NODE * source.cgns_nodes + _CGNS_EAGER
        # The BC phase's labels: 16 B a distinct labelled face or cell (at most F_b + C), read
        # in chunks of 2^17 ids (3 MiB); added to the prescan, whose release no test can see.
        texts += 8 * source.point_list_ids + _CGNS_POINT_LIST_CHUNK
        texts += 3 * 2**20 + 16 * (boundary + c)
        reading = (
            max(standard, ngon) + texts + _JSON_DOCUMENT * source.zone_names_bytes + _HDF5_CACHE
        )
    else:
        # polyMesh: the end of assemble, with the owner, neighbour and boundary texts (at most
        # the files' bytes) and the 64-bit owner and neighbour arrays alive.
        reading = s + 52 * ns - 24 * n + 33 * f + 8 * interior + 4 * p + 8 * c
    # Assembly checks every cell closed with 16 B an edge, reserved for the cell with the most
    # face nodes (core/src/mesh.cpp, check_cells_closed): at most 16 B a face node.
    reading += 16 * p
    if mesh.periodic and mesh.source.format in ("gmsh", "cgns"):
        # Periodic matching at the end of assemble, every node bounding the boundary ones.
        paired = sum(pair.faces for pair in mesh.periodic) * 2
        reading += 21 * ns + 128 * ns + 113 * paired
    return reading
