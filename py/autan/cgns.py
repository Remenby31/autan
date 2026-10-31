"""The CGNS/HDF5 subset of ``solution.cgns`` (docs/contracts/results.md#solutioncgns).

Node layout follows the CGNS/HDF5 file mapping: each node is an HDF5 group with
``name``, ``label`` (33-byte strings), ``type`` (3-byte string) and ``flags``
attributes, and its data in the dataset ``" data"``. Arrays of rank > 1 are
stored with their dimensions reversed (Fortran order).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

LIBRARY_VERSION = 4.4

# Exponents of kg, m, s, K, rad for each field we write.
EXPONENTS: dict[str, tuple[float, float, float, float, float]] = {
    "CoordinateX": (0, 1, 0, 0, 0),
    "CoordinateY": (0, 1, 0, 0, 0),
    "CoordinateZ": (0, 1, 0, 0, 0),
    "Density": (1, -3, 0, 0, 0),
    "VelocityX": (0, 1, -1, 0, 0),
    "VelocityY": (0, 1, -1, 0, 0),
    "VelocityZ": (0, 1, -1, 0, 0),
    "Pressure": (1, -1, -2, 0, 0),
    "Temperature": (0, 0, 0, 1, 0),
    "TurbulentSANuTilde": (0, 2, -1, 0, 0),
    "TurbulentEnergyKinetic": (0, 2, -2, 0, 0),
    "TurbulentDissipationRate": (0, 0, -1, 0, 0),
    "WallShearStressX": (1, -1, -2, 0, 0),
    "WallShearStressY": (1, -1, -2, 0, 0),
    "WallShearStressZ": (1, -1, -2, 0, 0),
}

_TYPES = {
    np.dtype(np.int32): "I4",
    np.dtype(np.int64): "I8",
    np.dtype(np.float32): "R4",
    np.dtype(np.float64): "R8",
    np.dtype(np.int8): "C1",
}


def _string(value: str, size: int) -> np.bytes_:
    return np.bytes_(value.encode("ascii")[: size - 1])


def node(parent: h5py.Group, name: str, label: str, data: Any = None) -> h5py.Group:
    """Create a CGNS node; ``data`` is a str (C1), or an array of a supported type."""
    group = parent.create_group(name)
    kind = "MT"
    if data is not None:
        array = (
            np.frombuffer(data.encode("ascii"), dtype=np.int8)
            if isinstance(data, str)
            else np.asarray(data)
        )
        kind = _TYPES[array.dtype]
        group.create_dataset(" data", data=array.T if array.ndim > 1 else array)
    for key, value, size in (("name", name, 33), ("label", label, 33), ("type", kind, 3)):
        group.attrs.create(key, _string(value, size), dtype=h5py.string_dtype("ascii", size))
    group.attrs.create("flags", np.array([1], dtype=np.int32))
    return group


def data_array(parent: h5py.Group, name: str, values: NDArray[Any]) -> h5py.Group:
    array = node(parent, name, "DataArray_t", values)
    if name in EXPONENTS:
        node(
            array,
            "DimensionalExponents",
            "DimensionalExponents_t",
            np.array(EXPONENTS[name], dtype=np.float32),
        )
    return array


def create(path: Path) -> h5py.File:
    """A new CGNS/HDF5 file with its root node, library version and nothing else."""
    handle = h5py.File(path, "w", libver=("earliest", "v110"))
    for key, value in (("label", "Root Node of HDF5 File"), ("name", "HDF5 MotherNode")):
        handle.attrs.create(key, _string(value, 33), dtype=h5py.string_dtype("ascii", 33))
    handle.attrs.create("type", _string("MT", 3), dtype=h5py.string_dtype("ascii", 3))
    handle.create_dataset(" format", data=np.frombuffer(b"IEEE_LITTLE_32\0", dtype=np.int8))
    version = f"HDF5 Version {'.'.join(map(str, h5py.h5.get_libversion()))}".ljust(32, "\0")
    handle.create_dataset(
        " hdf5version", data=np.frombuffer(version.encode() + b"\0", dtype=np.int8)
    )
    node(
        handle,
        "CGNSLibraryVersion",
        "CGNSLibraryVersion_t",
        np.array([LIBRARY_VERSION], dtype=np.float32),
    )
    return handle


@dataclass
class Node:
    """A node read back: label, data and children by name."""

    name: str
    label: str
    data: Any
    children: dict[str, Node] = field(default_factory=dict)

    def text(self) -> str:
        return bytes(np.asarray(self.data, dtype=np.int8).astype(np.uint8)).decode("ascii")

    def child(self, name: str) -> Node:
        return self.children[name]

    def with_label(self, label: str) -> list[Node]:
        return [c for c in self.children.values() if c.label == label]


def read(path: Path) -> Node:
    """Read a whole CGNS/HDF5 tree into memory."""

    def load(group: h5py.Group, name: str) -> Node:
        raw = group.get(" data")
        data = None
        if isinstance(raw, h5py.Dataset):
            array = raw[()]
            data = array.T if array.ndim > 1 else array
        label = group.attrs.get("label", b"")
        result = Node(name, label.decode() if isinstance(label, bytes) else str(label), data)
        for key, child in group.items():
            if isinstance(child, h5py.Group):
                result.children[key] = load(child, key)
        return result

    with h5py.File(path, "r") as handle:
        return load(handle, "/")
