import json
import struct
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import numpy.typing as npt

SURFACE_SCHEMA = "autan-ui.surface/1"

F32 = npt.NDArray[np.float32]
U32 = npt.NDArray[np.uint32]


@dataclass
class Zone:
    name: str
    positions: F32
    triangles: U32
    fields: dict[str, F32] = field(default_factory=dict)


def encode(zones: list[Zone], *, origin: tuple[float, float, float]) -> bytes:
    """Pack zones into one surface message: u32 header length, JSON header, 4-aligned buffers.

    Zones are concatenated in order, so zone k owns a contiguous triangle range; every zone
    carries the same node fields.
    """
    positions = np.concatenate([zone.positions.reshape(-1, 3) for zone in zones])
    node_base = np.cumsum([0] + [len(zone.positions) for zone in zones[:-1]], dtype=np.uint32)
    indices = np.concatenate(
        [zone.triangles.reshape(-1, 3) + base for zone, base in zip(zones, node_base, strict=True)]
    )
    names = list(zones[0].fields)
    fields = {name: np.concatenate([zone.fields[name] for zone in zones]) for name in names}

    buffers: list[tuple[str, str, np.ndarray]] = [
        ("positions", "f32", positions.astype(np.float32, copy=False)),
        ("indices", "u32", indices.astype(np.uint32, copy=False)),
        *((f"field:{name}", "f32", values) for name, values in fields.items()),
    ]
    offset = 0
    layout = []
    for name, dtype, array in buffers:
        layout.append({"name": name, "dtype": dtype, "offset": offset, "length": array.size})
        offset += array.nbytes

    first = 0
    zone_ranges = []
    for zone in zones:
        count = len(zone.triangles.reshape(-1, 3))
        zone_ranges.append({"name": zone.name, "first_triangle": first, "triangles": count})
        first += count

    header = {
        "schema": SURFACE_SCHEMA,
        "origin": list(origin),
        "nodes": len(positions),
        "triangles": len(indices),
        "zones": zone_ranges,
        "fields": [
            {"name": name, "location": "node", "min": float(v.min()), "max": float(v.max())}
            for name, v in fields.items()
        ],
        "buffers": layout,
    }
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    return b"".join(
        [struct.pack("<I", len(text)), text, *(array.tobytes() for _, _, array in buffers)]
    )


def _grid(nu: int, nv: int) -> U32:
    """Two triangles per quad of an nu x nv node grid, row-major nodes."""
    i, j = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1), indexing="ij")
    a = (i * nv + j).ravel()
    b, c, d = a + nv, a + nv + 1, a + 1
    return np.stack([a, b, c, a, c, d], axis=1).reshape(-1, 3).astype(np.uint32)


def _naca0012(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 0.6 * (0.2969 * np.sqrt(u) - 0.1260 * u - 0.3516 * u**2 + 0.2843 * u**3 - 0.1036 * u**4)


@lru_cache(maxsize=2)
def synthetic_wing(triangles: int) -> bytes:
    """A swept, tapered NACA 0012 wing with a made-up pressure field, for the viewer only."""
    span, root_chord, tip_chord, sweep = 6.0, 2.0, 0.8, 0.55
    per_side = max(triangles // 2, 64)
    nu = max(int(np.sqrt(per_side / 2 / 4)), 4)
    nv = max(per_side // (2 * nu), 4) + 1

    u = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, nu))
    s = np.linspace(0.0, 1.0, nv)
    uu, ss = np.meshgrid(u, s, indexing="ij")
    chord = root_chord + (tip_chord - root_chord) * ss
    x = ss * span * sweep + uu * chord
    y = ss * span
    thickness = _naca0012(uu) * chord

    zones = []
    for name, side in (("wing.upper", 1.0), ("wing.lower", -1.0)):
        z = side * thickness
        if side > 0:
            cp = -(0.4 + 2.6 * np.exp(-uu / 0.03)) * (1 - 0.6 * ss**2) + 0.7 * uu
        else:
            cp = 1.0 * np.exp(-uu / 0.02) + 0.15 - 0.1 * uu
        positions = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
        tri = _grid(nu, nv)
        if side < 0:
            tri = tri[:, ::-1].copy()
        zones.append(Zone(name, positions, tri, {"Cp": cp.ravel().astype(np.float32)}))

    v = np.linspace(-1.0, 1.0, 9)
    uu_tip, vv = np.meshgrid(u, v, indexing="ij")
    tip = np.stack(
        [
            span * sweep + uu_tip * tip_chord,
            np.full_like(uu_tip, span),
            vv * _naca0012(uu_tip) * tip_chord,
        ],
        axis=-1,
    )
    cp_tip = -0.3 + 0.5 * uu_tip
    zones.append(
        Zone(
            "wing.tip",
            tip.reshape(-1, 3).astype(np.float32),
            _grid(nu, len(v)),
            {"Cp": cp_tip.ravel().astype(np.float32)},
        )
    )
    centre = (span * sweep / 2 + root_chord / 2, span / 2, 0.0)
    for zone in zones:
        zone.positions -= np.asarray(centre, dtype=np.float32)
    return encode(zones, origin=centre)
