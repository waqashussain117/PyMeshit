# Meshing options

All lengths and tolerances use the same units as the geometry coordinates.

| Option | Default | Meaning |
|---|---:|---|
| `target_size` | `1.0` | Global target edge length; must be positive. |
| `min_angle` | `20.0` | Requested minimum surface-triangle angle in degrees. |
| `gradient` | `2.0` | Transition factor between fine and coarse surface regions. |
| `uniform` | `True` | Use approximately uniform refinement along constraint lines. |
| `interpolation` | `Thin Plate Spline (TPS)` | Surface elevation interpolation; TPS and IDW are supported. |
| `smoothing` | `0.0` | TPS smoothing parameter. |
| `hull_method` | `delaunay` | Projected boundary method: `delaunay` or `alpha`. |
| `hull_alpha_factor` | `1.0` | Alpha-shape scale used with `hull_method="alpha"`. |
| `constraint_mode` | `intersections` | Select hull and/or intersection constraints. |
| `preserve_boundary_hulls` | `True` | Keep outer hulls in intersection mode. |
| `tetgen_switches` | `pq1.414aAY` | Switch string passed to TetGen. |
| `generate_volume` | `True` | Run TetGen after surface meshing. |
| `include_hull_fallback_faults_in_volume` | `False` | Force fallback fault hulls into the PLC. |
| `intersection_tolerance` | `1e-5` | Intersection/triple-point matching tolerance. |
| `keep_intermediate` | `True` | Retain intermediate datasets and surface meshes. |

## Constraint modes

| Mode | Hull constraints | Intersection constraints |
|---|---|---|
| `all` | Yes | Yes |
| `intersections` | Boundary hulls when `preserve_boundary_hulls=True` | Yes |
| `hulls` | Yes | No |

`intersections` with preserved boundary hulls corresponds to the GUI workflow
where intersections are selected as internal constraints while each surface
hull remains the outer triangulation boundary.

If selected intersection constraints cannot form a surface PLC, PyMeshIt may
retry that surface using its hull. The resulting surface has
`constraint_source="hull_fallback"`. Fault fallback surfaces are excluded from
the volume PLC by default because forcing them into TetGen can make the model
fragile.

## Choosing target sizes

Start with a size appropriate for the full domain, then use
`SurfaceSpec.target_size` and `WellSpec.target_size` only where local refinement
is scientifically required. Reducing target size increases memory and runtime
and can expose very small geometry inconsistencies.

## TetGen switches

The default `pq1.414aAY` requests PLC tetrahedralization, quality control,
volume constraints, material attributes, and preservation of boundary markers.
Treat a custom switch string as part of the reproducibility record. Test it on
a representative case before starting a large batch.

