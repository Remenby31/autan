# Test meshes

Small meshes read by `tests/io`. Each directory holds one mesh in every format that can represent it, all written by independent tools so that the readers are checked against producers other than our own code.

| Mesh | Cells | Gmsh `.ascii.msh`, `.binary.msh` | `polyMesh/` | Fluent `.fluent.msh`, `.fluent-binary.msh` |
|---|---|---|---|---|
| `hex` | 27 hexahedra | yes | yes | yes |
| `prism` | 54 prisms | yes | yes | yes |
| `pyramid` | 48 pyramids on cube centres | yes | yes | yes |
| `tet` | 48 Kuhn tetrahedra | yes | yes | yes |
| `hybrid` | hexahedra, pyramids, prisms in two orientations, tetrahedra; cell zones `lower`, `upper` | yes | yes, also `polyMesh-binary/` | yes |
| `tet-gmsh` | unstructured tetrahedra meshed by Gmsh | yes | yes | yes |
| `poly` | polyhedral dual of `tet-gmsh` | — | yes, also `polyMesh-binary/` | yes |
| `periodic-box` | unstructured tetrahedra meshed by Gmsh, periodic in x and y (`$Periodic`) | yes | — | — |
| `ringleb` | one layer of 32 hexahedra on the Ringleb domain (HiOCFD1 C1.2), with `ringleb-n004.states.json`, the exact states of its four in-plane sides | yes | — | — |
| `hanging` | one 105-face polyhedron with hanging nodes facing 100 hexahedra; cell zones `coarse`, `fine` | — | yes | yes |

Every mesh except `hanging` and `ringleb` fills the unit cube with sides `xmin` … `zmax`; `ymin` is a wall and `zmax` a symmetry plane in the polyMesh and Fluent files. `hanging` fills [0,2]×[0,1]×[0,1] with patches `inlet`, `outlet` and `walls` (wall).

## Provenance

`generate/make.sh` regenerates everything (needs uv and Docker):

1. `generate.py` builds the meshes through the Gmsh API (gmsh 4.15.2 from `generate/uv.lock`); Gmsh writes the MSH 4.1 ASCII and binary files. `hanging` is written as a polyMesh directly.
2. `convert.sh` runs in `opencfd/openfoam-default:2506`: `gmshToFoam`, then `polyDualMesh` for `poly`, `checkMesh` (must report `Mesh OK`), `foamMeshToFluent` for the Fluent files and `foamFormatConvert` for the binary polyMesh.
3. `fluent_binary.py` re-encodes the ASCII Fluent files with binary node, cell and face sections, copying values unchanged.

`ringleb/` is not from `make.sh`: `autan-vv meshes ringleb --levels 4` wrote both files ([meshes.md](../../../docs/validation/meshes.md)).

`foamMeshToFluent` writes coordinates with 10 significant digits, a single cell zone `fluid-1`, and types every non-wall, non-symmetry patch as `pressure-outlet`; the tests account for this. Gmsh and OpenFOAM are GPL tools used only to produce these data files; nothing of them is linked or shipped.
