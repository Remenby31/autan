# Boundary conditions

Every boundary zone of the mesh needs a condition before a run can start. Autan takes it from your case when you set one, and otherwise from the type the mesh file gives the zone. `autan plan` shows the result for every zone, with where it came from: its **provenance**.

## How a zone gets its condition

![How a boundary zone gets its condition](../assets/diagrams/bc-resolution.svg)

| Provenance | The condition comes from |
|---|---|
| `code` | A rule of your case, such as `case.boundary("wing.*").wall()`. |
| `mesh` | The zone type in the mesh file, such as a Fluent `wall` or a CGNS `BCFarfield`. |
| `default` | Cell zones only: a cell zone the mesh leaves untyped is a fluid with the case's gas. |

A rule may override the mesh type; `autan plan` then reports `W0415` so the override is visible. A mesh type that needs values only you can give, such as an inlet's total pressure, stops the plan with `E0413`.

The rules, including glob matching and every error code, are in [Case schema: boundary resolution](../contracts/case-schema.md#boundary-resolution). The normalised mesh types are in [MeshInfo](../contracts/mesh-info.md#normalized-boundary-types).

## Why the mesh comes first

Meshing tools already name and type the boundaries. Reading those types means a case states only what the mesh cannot know: values, and the zones you want to treat differently. Showing the provenance means no condition is applied without you seeing where it came from.

## Next steps

- [Check a case before you run it](../guides/check-a-case.md)
- [Diagnostic codes](../sdk/errors.md)
