# Mesh and Exodus export

Pass `output_path` to `run_mesh_case` or call `MeshResult.export_mesh` after a
successful volume run:

```python
result = run_mesh_case(case)
result.export_mesh("outputs/model.exo")
```

The recognized Exodus/NetCDF suffixes are `.exo`, `.e`, `.nc`, `.nc4`, and
`.cdf`. Other suffixes use PyVista's mesh writer. Exodus export requires the
TetGen generator metadata retained by a complete `run_mesh_case` call.

## Exodus entities

| PyMeshIt data | Exodus representation | Identifier |
|---|---|---|
| Formation material | Tetrahedral element block | `MaterialSpec.attribute` |
| Border or unit surface `i` | Sideset | `i + 1` |
| Fault surface `i` | Sideset | `1000 + i` |
| Well | Node set by default | `WellSpec.marker` |
| Well with `Element Blocks` mode | BAR2 element block | marker-derived block ID |

Well node sets share nodes with the tetrahedral mesh and are suitable for
point-source/sink workflows. BAR2 blocks preserve line elements when the
downstream application requires explicit well segments.

## Custom names

Custom-name dictionaries are keyed by the numeric IDs above:

```python
result = run_mesh_case(
    case,
    output_path="outputs/model.exo",
    custom_block_names={1: "lower_unit", 2: "upper_unit"},
    custom_sideset_names={1: "bottom", 2: "top", 1002: "main_fault"},
    custom_well_names={2001: "injection_well"},
    well_export_type="Node Sets",
)
```

Names are sanitized for Exodus compatibility: spaces, hyphens, and periods
become underscores; other non-alphanumeric characters are removed; names that
start with a digit receive a prefix; and names are limited to 64 characters.

## Verify an export

Always inspect the return value or `MeshResult.failures`. For critical models,
open the file in ParaView and verify:

- the number and names of element blocks;
- `MaterialID` assignments;
- sideset names and marker IDs;
- well node sets or BAR2 blocks; and
- the expected number of nodes and elements.

