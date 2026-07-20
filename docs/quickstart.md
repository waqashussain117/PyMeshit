# Headless quickstart

This example creates a conforming mesh for one planar surface without opening
the GUI. Volume generation is disabled so the example is small and
self-contained.

```python
import numpy as np

from Pymeshit import MeshCase, MeshOptions, SurfaceSpec, run_mesh_case

points = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ]
)

case = MeshCase(
    surfaces=[
        SurfaceSpec(
            name="surface",
            points=points,
            role="border",
            target_size=0.25,
        )
    ],
    options=MeshOptions(
        target_size=0.25,
        generate_volume=False,
    ),
)

result = run_mesh_case(case)

print("Successful:", result.ok)
print("Failures:", result.failures)
print("Surface meshes:", len(result.conforming_surface_data))
```

Expected output includes:

```text
Successful: True
Failures: []
Surface meshes: 1
```

## Loading points from a file

`SurfaceSpec.points` and `WellSpec.points` accept an array or a file path.
Text inputs should contain three numeric coordinate columns. PyVista-readable
VTK formats are also supported.

```python
from Pymeshit import SurfaceSpec, read_points

points = read_points("top.dat")
surface = SurfaceSpec("top", points, role="border", target_size=50.0)
```

## Moving to a volume case

A volume case normally contains enough boundary surfaces to form a closed,
watertight domain. Add internal unit or fault surfaces, define material seed
points with `MaterialSpec`, and leave `generate_volume=True`.

See the [complete batch-workflow notebook](examples/headless_batch_workflow.ipynb)
for surfaces, wells, material seeds, TetGen options, and Exodus export.
