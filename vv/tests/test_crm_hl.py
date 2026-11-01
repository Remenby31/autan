"""CRM-HL zone groups: the HLPW-5 FUN3D .mapbc of h5c23_rans_c01 as an autan.zone-groups/1 sidecar."""

import json
import re
from importlib import resources

from autan_vv.cases import pointers

COMPONENTS = (
    "wing.slat",
    "wing.slat-brackets",
    "wing.main",
    "wing.flap",
    "wing.flap-fairings",
    "fuselage",
    "empennage",
    "farfield",
    "symmetry",
)


def test_sidecar_groups_every_patch_of_the_grid_it_names() -> None:
    text = resources.files("autan_vv.cases").joinpath("h5c23_rans_c01.cgns.zones.json").read_text()
    doc = json.loads(text)
    assert doc["schema"] == "autan.zone-groups/1"
    grid = pointers()["HLPW5"]["grid"]
    assert doc["mesh_sha256"] == grid["member_sha256"]
    groups = doc["groups"]
    assert all(re.fullmatch(r"[A-Za-z0-9_.][A-Za-z0-9_.-]*", name) for name in groups)
    # The workshop's 668 patches, the symmetry patch split into a triangle and a quad section.
    patches = {int(z.split()[1]) for zones in groups.values() for z in zones}
    assert patches == set(range(1, 669))
    # Components partition the zones; families partition them too.
    components = [z for name in COMPONENTS for z in groups[name]]
    families = [
        z for name, zones in groups.items() if name.isupper() or name == "Box" for z in zones
    ]
    assert len(components) == len(set(components)) == 669
    assert sorted(families) == sorted(components)
    assert sorted(groups["symmetry"]) == ["QUAD_bdy  668", "TRI_bdy  668"]
