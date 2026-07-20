# Input data

PyMeshIt accepts coordinates as in-memory arrays or file paths. All geometry
coordinates, target sizes, and tolerances must use one consistent unit system.

## Arrays

Pass any array-like object with shape `(N, 2)` or `(N, 3+)` to `SurfaceSpec`
or `WellSpec`:

```python
surface = SurfaceSpec(
    "top",
    [[0.0, 0.0, 10.0], [100.0, 0.0, 10.0], [0.0, 100.0, 10.0]],
)
```

Two-column inputs receive `z=0`. Columns after `x`, `y`, and `z` are ignored.

## Text and CSV files

Text files may use whitespace, commas, semicolons, or tabs. Blank lines and
lines beginning with `#` are ignored. Two or three numeric columns are
accepted:

```text
# x, y, z in metres
0.0,   0.0,  -100.0
100.0, 0.0,  -95.0
100.0, 100.0, -90.0
```

Lines that cannot be parsed are skipped. Validate important input files before
large batch runs instead of relying on skipped-line behavior.

```python
from Pymeshit import read_points

points = read_points("fault_01.csv")
assert points.shape[1] == 3
```

## VTK-family files

`.vtu`, `.vtk`, and `.vtp` files are read through PyVista. The mesh point array
is used; existing cell topology is not treated as the headless input topology.

## Surface point ordering

Surface inputs are treated as point clouds. PyMeshIt projects them to a local
surface plane, finds a hull, interpolates elevation, and triangulates the
constraints. Well inputs differ: their point order defines consecutive
polyline segments and must follow the well trajectory.

## Validation recommendations

- Remove duplicate or nearly duplicate samples.
- Use at least three non-collinear points per surface.
- Keep all case coordinates in the same units and coordinate reference system.
- Confirm well trajectories are ordered from one end to the other.
- Keep material seeds strictly inside their intended regions.

