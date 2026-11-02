"""``autan.initial-state/1`` documents (docs/contracts/initial-state.md): the state of every
fluid cell of one mesh at the start of a run, cells identified by their centroid."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    model_validator,
)

from autan.texts import bounded

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SCHEMA = "autan.initial-state/1"
VARIABLES = ("density", "velocity-x", "velocity-y", "velocity-z", "pressure")

Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Positive = Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0)]
Text = Annotated[str, AfterValidator(bounded)]


class InitialState(BaseModel):
    """A validated ``autan.initial-state/1`` document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: Literal["autan.initial-state/1"] = Field(alias="schema")
    oracle: Text
    mesh_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    variables: tuple[
        Literal["density"],
        Literal["velocity-x"],
        Literal["velocity-y"],
        Literal["velocity-z"],
        Literal["pressure"],
    ]
    cells: Annotated[StrictInt, Field(ge=1)]
    centroids: list[tuple[Finite, Finite, Finite]]
    values: list[tuple[Positive, Finite, Finite, Finite, Positive]]

    @model_validator(mode="after")
    def _one_row_per_cell(self) -> Self:
        if not len(self.centroids) == len(self.values) == self.cells:
            message = "centroids and values need one row per cell"
            raise ValueError(message)
        return self


def unique_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """A JSON object whose members all have distinct names of at most 4 096 bytes: the
    runtime's streaming reader refuses a repeated or longer one, so the SDK does too."""
    names = [bounded(name) for name, _ in pairs]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        message = f"member {repeated[0]!r} appears twice"
        raise ValueError(message)
    return dict(pairs)


def parse(data: bytes) -> InitialState | str:
    """The document in ``data``, or why it is not a valid one."""
    try:
        document = json.loads(data, object_pairs_hook=unique_members)
    except ValueError as error:
        return str(error)
    try:
        return InitialState.model_validate(document)
    except ValidationError as error:
        first = error.errors()[0]
        return f"{'.'.join(map(str, first['loc']))}: {first['msg']}"


def write(
    path: Path,
    *,
    oracle: str,
    mesh_sha256: str,
    centroids: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
) -> str:
    """Write the initial state of every fluid cell of the mesh whose hash is ``mesh_sha256``:
    one centroid (m) and one row of density, velocity (mesh axes) and pressure (SI) per cell.
    Returns the file's SHA-256, which the case gives with its path. Raises ``ValueError`` for
    a document the contract refuses."""
    document = InitialState.model_validate(
        {
            "schema": SCHEMA,
            "oracle": oracle,
            "mesh_sha256": mesh_sha256,
            "variables": VARIABLES,
            "cells": len(centroids),
            "centroids": [tuple(float(x) for x in c) for c in centroids],
            "values": [tuple(float(x) for x in v) for v in values],
        }
    )
    data = json.dumps(document.model_dump(mode="json", by_alias=True)).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
