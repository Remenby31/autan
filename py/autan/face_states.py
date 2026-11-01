"""``autan.face-states/1`` documents (docs/contracts/face-states.md)."""

from __future__ import annotations

import json
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, model_validator

SCHEMA = "autan.face-states/1"
VARIABLES = ("density", "velocity-x", "velocity-y", "velocity-z", "pressure")

Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Positive = Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ZoneStates(_Strict):
    faces: Annotated[StrictInt, Field(ge=0)]
    centroids: list[tuple[Finite, Finite, Finite]]
    values: list[tuple[Positive, Finite, Finite, Finite, Positive]]

    @model_validator(mode="after")
    def _one_row_per_face(self) -> Self:
        if not len(self.centroids) == len(self.values) == self.faces:
            message = "centroids and values need one row per face"
            raise ValueError(message)
        return self


class FaceStates(_Strict):
    schema_: Literal["autan.face-states/1"] = Field(alias="schema")
    oracle: str
    mesh_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    variables: tuple[
        Literal["density"],
        Literal["velocity-x"],
        Literal["velocity-y"],
        Literal["velocity-z"],
        Literal["pressure"],
    ]
    zones: dict[str, ZoneStates]


def parse(data: bytes) -> FaceStates | str:
    """The document in ``data``, or why it is not a valid one."""
    try:
        document = json.loads(data)
    except ValueError as error:
        return str(error)
    try:
        return FaceStates.model_validate(document)
    except ValidationError as error:
        first = error.errors()[0]
        return f"{'.'.join(map(str, first['loc']))}: {first['msg']}"
