"""CRM-HL zone map: the HLPW-5 FUN3D .mapbc of h5c23_rans_c01 as a versioned table."""

import json
from importlib import resources

from autan_vv.cases import pointers

# FUN3D boundary codes (FUN3D manual, "Boundary conditions"): viscous wall, farfield, y-symmetry.
FUN3D_KINDS = {4000: "wall", 5050: "farfield", 6662: "symmetry"}


def test_zone_map_covers_every_patch_of_the_grid_it_names() -> None:
    doc = json.loads(resources.files("autan_vv.cases").joinpath("crm_hl_zones.json").read_text())
    patches = doc["patches"]
    assert [p["patch"] for p in patches] == list(range(1, 669))
    assert all(FUN3D_KINDS[p["bc"]] == p["kind"] for p in patches)
    assert {p["component"] for p in patches if p["kind"] == "wall"} >= {
        "wing.slat",
        "wing.main",
        "wing.flap",
        "fuselage",
        "empennage",
    }
    assert {p["component"] for p in patches if p["kind"] != "wall"} == {"farfield", "symmetry"}
    # The map and the catalogue name the same grid file.
    grid = pointers()["HLPW5"]["grid"]
    assert doc["grid"]["sha256"] == grid["sha256"]
    assert doc["grid"]["member_sha256"] == grid["member_sha256"]
