# Meshes

Autan reads any mesh of polyhedral cells and stores it by its faces. A cell is known only through the faces that bound it, so a tetrahedron, a hexahedron and a polyhedron with 200 faces go through the same code. The exact arrays, invariants and error codes are in the [mesh contract](../contracts/mesh.md).

## How faces connect cells

![Owner and neighbour of a face](../assets/diagrams/mesh-faces.svg)

Each face stores the cell on each side. The **owner** is the cell its area vector points away from; the **neighbour** is the cell it points into. A boundary face has an owner and no neighbour, and its vector points out of the domain.

A flux kernel loops over faces: it reads the owner and the neighbour, computes one flux, and adds it to one cell and subtracts it from the other. That is why the solver needs no cell type.

## Zones

Every face belongs to one **face zone** and every cell to one **cell zone**. Zones come from the mesh file: Fluent zone names, Gmsh physical groups, CGNS boundary conditions and families, OpenFOAM patches. A zone keeps its name and, when the file gives one, its type, such as `wall` or `farfield`.

Your case refers to zones by name. The [MeshInfo](../contracts/mesh-info.md) summary lists them, with their types and sizes, so the SDK can check a case without loading the mesh.

## Why ghost cells exist

![Owned and ghost cells after partitioning](../assets/diagrams/partition-halos.svg)

To run on several GPUs, Autan cuts the mesh into partitions, one per GPU. Each partition stores the cells it **owns**, then **ghost** copies of the cells across its cut, owned by another partition.

A face on the cut has an owned cell on one side and a ghost on the other, so the face loop runs unchanged. Before the loop, a halo exchange copies owned values into the ghosts. After it, the contributions added to ghosts are partial sums: a reverse exchange adds them to the owning partition. The adjoint needs this reverse exchange; see [Architecture](architecture.md#how-autan-differentiates-its-kernels). [Partitioning and halo exchange](../design/partitioning.md) describes the partitioner and the exchange.

## Geometry

Areas, centroids and volumes are computed once, in FP64, whatever precision the solver computes in. [Mesh contract: geometry](../contracts/mesh.md#geometry) defines them.
