# Understanding results

`run_mesh_case` returns a `MeshResult`. Always inspect both `ok` and `failures`:

```python
result = run_mesh_case(case)

if not result.ok or result.failures:
    print(result.failures)
```

For volume cases, `ok` means that a tetrahedral mesh exists. A later export
failure can therefore coexist with `ok=True` and will appear in `failures`.

## Final volume mesh

`tetra_mesh` is the PyVista `UnstructuredGrid` returned by TetGen. When material
regions are available, tetrahedral cells contain a `MaterialID` array.

```python
mesh = result.tetra_mesh
if mesh is not None:
    print(mesh.n_points, mesh.n_cells)
    print(mesh.cell_data.get("MaterialID"))
```

## Conforming surface data

`conforming_surface_data` is keyed by zero-based surface index. Each entry has:

| Key | Meaning |
|---|---|
| `name` | Original surface name. |
| `vertices` | Conforming `(N, 3)` coordinates. |
| `triangles` | Triangle connectivity. |
| `facet_markers` | Marker for every triangle. |
| `original_dataset_index` | Index in the original surface sequence. |
| `conforming_mesh_source` | Indicates a conforming workflow mesh. |
| `constraint_source` | `selected` or `hull_fallback`. |
| `mesh_metadata` | Duplicate/supporting triangulation metadata. |

These entries are useful for inspecting the PLC before TetGen:

```python
for index, surface in result.conforming_surface_data.items():
    print(index, surface["name"], surface["constraint_source"])
```

## Intersections and triple points

Intersection records contain `dataset_id1`, `dataset_id2`,
`is_polyline_mesh`, and ordered `points`. They are grouped under the lower
participating dataset index.

Triple-point records contain:

```python
{
    "point": [x, y, z],
    "intersection_ids": [0, 2, 3],
}
```

## Intermediate datasets

With `keep_intermediate=True`, `datasets` contains normalized surfaces followed
by wells. Internal processing adds fields such as `hull_points`, `segments`,
`triangulation_result`, `intersection_points`, `refined_well_points`, and
`conforming_mesh`. These dictionaries are diagnostic data, not a stable public
serialization format. Prefer the documented result fields for long-lived code.

With `keep_intermediate=False`, `datasets` and `conforming_surface_data` are
cleared in the returned result; the final volume mesh and failure information
remain available.

