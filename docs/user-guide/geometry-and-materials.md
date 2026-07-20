# Surfaces, wells, and materials

## Surface roles

Surface order and role determine how the PLC and exported marker IDs are
constructed.

| Role | Purpose | Marker for surface index `i` |
|---|---|---:|
| `border` | External domain boundary | `i + 1` |
| `unit` | Internal stratigraphic interface | `i + 1` |
| `fault` | Internal fault constraint | `1000 + i` |

An index is zero-based and follows the order of `MeshCase.surfaces`. Keep this
order stable if downstream boundary-condition files use numeric markers.

```python
surfaces = [
    SurfaceSpec("bottom", bottom, role="border"),  # marker 1
    SurfaceSpec("top", top, role="border"),        # marker 2
    SurfaceSpec("fault", fault, role="fault"),     # marker 1002
]
```

## Material seeds

A formation material requires at least one seed strictly inside every volume
region that should receive its `attribute` value:

```python
materials = [
    MaterialSpec("lower", [0.0, 0.0, -500.0], attribute=1),
    MaterialSpec("upper", [0.0, 0.0, 500.0], attribute=2),
]
```

The attribute becomes the `MaterialID` cell value and the Exodus element-block
ID. A material with `type="FAULT"` is surface-only metadata and does not create
a volumetric TetGen region.

If no materials are supplied, the TetGen wrapper creates a default material
region near the PLC centre with attribute zero. Explicit seeds are strongly
recommended for layered models.

## Wells

Well coordinates are ordered polylines. Intersections with surfaces are
inserted into the refined trajectory before volume meshing.

```python
well = WellSpec(
    "injection_well",
    well_points,
    target_size=10.0,
    marker=2001,
)
```

Set `marker` explicitly when Exodus node-set or block IDs must remain stable.
Automatic well markers depend on the number and order of datasets.

## Watertight volume requirements

TetGen needs a closed, consistent piecewise-linear complex. Before generating
a volume mesh, check that:

- border surfaces enclose the complete domain;
- intersections meet without gaps larger than `intersection_tolerance`;
- internal surfaces do not extend inconsistently outside the domain;
- material seeds lie inside closed regions; and
- conforming surface meshes have no missing pieces.

