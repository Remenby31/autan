"""Reference validation cases: checksummed downloads and parsers for their grids and data.

The catalogue (`catalog.json`, schema `autan.case-library/1`) pins every file by URL, size and
SHA-256; files inside an archive also pin the member path and its SHA-256. Nothing is fetched at
import or test-collection time, and a checksum mismatch is an error, never a warning.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from autan_vv.errors import VVError

CATALOG_SCHEMA = "autan.case-library/1"
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CaseFile:
    """One pinned file of a case."""

    name: str
    role: str
    url: str
    sha256: str
    size: int
    parser: str
    description: str
    member: str | None = None
    member_sha256: str | None = None
    corrupt_sections: tuple[int, ...] = ()


@dataclass(frozen=True)
class Case:
    """A validation case: conditions, reference values with their source, and pinned files."""

    name: str
    title: str
    conditions: dict[str, Any]
    reference_values: dict[str, Any]
    references: tuple[str, ...]
    licence: str
    files: tuple[CaseFile, ...]

    def file(self, name: str) -> CaseFile:
        for f in self.files:
            if f.name == name:
                return f
        msg = f"case {self.name} has no file {name}"
        raise VVError(msg, code="V0604")


def _catalog() -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(
        resources.files(__package__).joinpath("catalog.json").read_text()
    )
    if doc.get("schema") != CATALOG_SCHEMA:
        msg = f"case catalogue schema is {doc.get('schema')!r}, expected {CATALOG_SCHEMA}"
        raise VVError(msg, code="V0605")
    return doc


def list_cases() -> list[Case]:
    """All cases of the catalogue, in catalogue order."""
    out = []
    for name, c in _catalog()["cases"].items():
        files = tuple(
            CaseFile(**{**f, "corrupt_sections": tuple(f.get("corrupt_sections", ()))})
            for f in c["files"]
        )
        out.append(
            Case(
                name,
                c["title"],
                c["conditions"],
                c["reference_values"],
                tuple(c["references"]),
                c["licence"],
                files,
            )
        )
    return out


def get_case(name: str) -> Case:
    """The case called `name` (V0604 if unknown)."""
    for case in list_cases():
        if case.name == name:
            return case
    msg = f"unknown case {name!r}"
    raise VVError(msg, code="V0604")


def pointers() -> dict[str, Any]:
    """Landing pages and sizes of datasets too large to fetch (HLPW-5, DPW)."""
    result: dict[str, Any] = _catalog()["pointers"]
    return result


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _publish_verified(write: Any, dest: Path, sha256: str, code: str, what: str) -> None:
    """Write through `write(fh)` to a unique temporary file, verify it, then rename it into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=dest.name + ".", delete=False) as fh:
        tmp = Path(fh.name)
        write(fh)
    got = _sha256(tmp)
    if got != sha256:
        tmp.unlink()
        msg = f"{what}: SHA-256 {got} differs from the catalogue ({sha256})"
        raise VVError(msg, code=code)
    tmp.replace(dest)


def _download(url: str, sha256: str, size: int, blobs: Path) -> Path:
    blob = blobs / sha256
    if blob.exists() and _sha256(blob) == sha256:
        return blob
    request = urllib.request.Request(url, headers={"User-Agent": "autan-vv"})

    def write(fh: Any) -> None:
        with urllib.request.urlopen(request, timeout=120) as response:
            shutil.copyfileobj(response, fh, 1 << 20)
        if fh.tell() != size:
            fh.truncate(0)

    _publish_verified(write, blob, sha256, "V0601", url)
    return blob


def fetch(
    case: Case, cache: Path, roles: tuple[str, ...] = ("grid", "reference", "experiment")
) -> list[Path]:
    """Download and verify the files of `case` with the given roles into `cache/<case>/`.

    Archives are downloaded once and kept by checksum in `cache/blobs/`; archive members are
    extracted and verified against their own SHA-256. Every file is written to a unique temporary
    name, verified, then renamed, so concurrent fetches never expose unverified bytes. Raises V0601
    (download) or V0602 (file or member) on a mismatch. Returns the local paths in catalogue order.
    """
    paths = []
    for f in case.files:
        if f.role not in roles:
            continue
        dest = cache / case.name / f.name
        expected = f.member_sha256 or f.sha256
        if not (dest.exists() and _sha256(dest) == expected):
            blob = _download(f.url, f.sha256, f.size, cache / "blobs")

            def write(fh: Any, f: CaseFile = f, blob: Path = blob) -> None:
                if f.member is None:
                    with blob.open("rb") as src:
                        shutil.copyfileobj(src, fh)
                else:
                    with zipfile.ZipFile(blob) as zf, zf.open(f.member) as src:
                        shutil.copyfileobj(src, fh)

            _publish_verified(write, dest, expected, "V0602", f"{f.name} of {case.name}")
        paths.append(dest)
    return paths


# --- parsers ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Table:
    """A titled table of named columns (one Tecplot zone, or one headerless block)."""

    title: str
    variables: tuple[str, ...]
    data: FloatArray

    def column(self, name: str) -> FloatArray:
        if name not in self.variables:
            msg = f"table {self.title!r} has no column {name!r}; columns are {self.variables}"
            raise VVError(msg, code="V0603")
        return self.data[:, self.variables.index(name)]


def _open_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("latin-1")


_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?$")


def _floats(tokens: list[str]) -> list[float] | None:
    if not tokens or not all(_NUMBER.match(t) for t in tokens):
        return None
    return [float(t.replace("D", "E").replace("d", "e")) for t in tokens]


def read_tecplot(path: Path) -> list[Table]:
    """Tecplot ASCII in POINT format: `variables` with quoted names, `zone` lines, `#` comments.

    Data before the first zone form an untitled table. Values of a zone are read in order and
    regrouped by the number of variables, so wrapped records are accepted; a remainder is V0603.
    """
    variables: tuple[str, ...] = ()
    tables: list[Table] = []
    title = ""
    values: list[float] = []

    def close() -> None:
        if not values:
            return
        if not variables or len(values) % len(variables):
            msg = f"{path.name}: zone {title!r} has {len(values)} values for {len(variables)} variables"
            raise VVError(msg, code="V0603")
        tables.append(Table(title, variables, np.array(values).reshape(-1, len(variables))))

    for line in _open_text(path).splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        lower = text.lower()
        if lower.startswith("variables"):
            variables = tuple(re.findall(r'"([^"]*)"', text))
            continue
        if lower.startswith("zone"):
            close()
            values = []
            match = re.search(r'\bt\s*=\s*"([^"]*)"', text, re.IGNORECASE)
            title = match.group(1).strip() if match else ""
            continue
        nums = _floats(re.split(r"[\s,]+", text))
        if nums is None:
            if lower.startswith(("title", "text", "geometry")):
                continue
            msg = f"{path.name}: unexpected line {text[:60]!r}"
            raise VVError(msg, code="V0603")
        values.extend(nums)
    close()
    return tables


def read_columns(path: Path) -> Table:
    """Whitespace-separated numeric columns; lines that are not all numbers (headers) are skipped.

    All numeric lines must have the same number of columns (V0603 otherwise).
    """
    rows = [r for r in (_floats(line.split()) for line in _open_text(path).splitlines()) if r]
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        msg = f"{path.name}: numeric lines have {sorted(widths)} columns"
        raise VVError(msg, code="V0603")
    return Table(path.name, tuple(f"c{i}" for i in range(widths.pop())), np.array(rows))


def read_plot3d_2d(path: Path) -> list[tuple[FloatArray, FloatArray]]:
    """Formatted multi-block 2D PLOT3D grid (TMR layout): returns (x, y) arrays of shape (jdim, idim)."""
    tokens = _open_text(path).split()
    nbl = int(tokens[0])
    dims = [(int(tokens[1 + 2 * b]), int(tokens[2 + 2 * b])) for b in range(nbl)]
    values = np.array([float(t.replace("D", "E")) for t in tokens[1 + 2 * nbl :]])
    if values.size != sum(2 * i * j for i, j in dims):
        msg = f"{path.name}: {values.size} coordinates for blocks {dims}"
        raise VVError(msg, code="V0603")
    blocks, start = [], 0
    for i, j in dims:
        n = i * j
        blocks.append(
            (
                values[start : start + n].reshape(j, i),
                values[start + n : start + 2 * n].reshape(j, i),
            )
        )
        start += 2 * n
    return blocks


_TAPE_REAL = re.compile(r"^(?:[ -]\d\.\d{6}E[ +-]\d{2})+$")
_TAPE_INT = re.compile(r"^(?:[ -]{0,5}\d{1,6})+$")


def read_tape_1981(path: Path, corrupt: tuple[int, ...] = ()) -> dict[int, FloatArray]:
    """Coded data files of the 1980-81 AFOSR-HTTM-Stanford conference tape (Bradshaw's f8621.txt).

    A coded file holds one record of maxima and one of minima (E13.6, a blank exponent sign meaning
    +), then integer records (I6) I in 0..10000 per variable: value = min + (max - min) I / 10000;
    I = 20000 means no value (NaN). Text files of the tape (descriptions) are skipped. A coded file
    with a malformed record raises V0603 unless its number is in `corrupt`, the sections the
    catalogue declares damaged. Returns {file number: array (rows, variables)}.
    """
    out: dict[int, FloatArray] = {}
    pattern = r"FILE NUMBER\s+(\d+)-+\s*\n(.*?)\n-+ END OF FILE"
    for number, body in re.findall(pattern, _open_text(path), re.DOTALL):
        lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
        n_real = 0
        while n_real < len(lines) and _TAPE_REAL.match(lines[n_real]):
            n_real += 1
        if n_real == 0:
            continue
        if not all(_TAPE_INT.match(ln) for ln in lines[n_real:]):
            if int(number) in corrupt:
                continue
            msg = f"{path.name}: coded file {number} has a malformed record"
            raise VVError(msg, code="V0603")
        reals = [
            float(ln[k : k + 13].replace("E ", "E+"))
            for ln in lines[:n_real]
            for k in range(0, len(ln), 13)
        ]
        ints = [int(ln[k : k + 6]) for ln in lines[n_real:] for k in range(0, len(ln), 6)]
        nv = len(reals) // 2
        if len(reals) != 2 * nv or len(ints) % nv:
            msg = f"{path.name}: file {number} has {len(reals)} bounds and {len(ints)} integers"
            raise VVError(msg, code="V0603")
        hi, lo = np.array(reals[:nv]), np.array(reals[nv:])
        coded = np.array(ints, dtype=np.float64).reshape(-1, nv)
        values = lo + (hi - lo) * coded / 10000.0
        values[coded == 20000] = np.nan
        out[int(number)] = values
    if not out:
        msg = f"{path.name}: no coded data file"
        raise VVError(msg, code="V0603")
    return out


PARSERS = {
    "tecplot": read_tecplot,
    "columns": read_columns,
    "plot3d2d": read_plot3d_2d,
    "tape1981": read_tape_1981,
}


def parse(case_file: CaseFile, path: Path) -> Any:
    """Parse a fetched file with the parser the catalogue names for it."""
    if case_file.parser == "tape1981":
        return read_tape_1981(path, case_file.corrupt_sections)
    return PARSERS[case_file.parser](path)
