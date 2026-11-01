"""Named oracles, as `autan-vv verify --against <name>` refers to them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from autan_vv.errors import VVError
from autan_vv.mms import euler, navier_stokes, rans_sa
from autan_vv.oracles import Oracle, from_mms
from autan_vv.oracles.inviscid import isentropic_vortex, ringleb, shock_tube
from autan_vv.oracles.viscous import blasius, couette, poiseuille

if TYPE_CHECKING:
    from collections.abc import Callable

ORACLES: dict[str, Callable[[], Oracle]] = {
    "mms-euler": lambda: from_mms(euler()),
    "mms-navier-stokes": lambda: from_mms(navier_stokes()),
    "mms-rans-sa": lambda: from_mms(rans_sa()),
    "mms-rans-sa-noft2": lambda: from_mms(rans_sa(ft2=False)),
    "ringleb": ringleb,
    "isentropic-vortex": isentropic_vortex,
    "sod": lambda: shock_tube("sod"),
    "lax": lambda: shock_tube("lax"),
    "couette": couette,
    "poiseuille": poiseuille,
    "blasius": blasius,
}


def get_oracle(name: str) -> Oracle:
    """The oracle called `name` with its default parameters (V0704 if unknown)."""
    if name not in ORACLES:
        msg = f"unknown oracle {name!r}; known: {', '.join(sorted(ORACLES))}"
        raise VVError(msg, code="V0704")
    return ORACLES[name]()
