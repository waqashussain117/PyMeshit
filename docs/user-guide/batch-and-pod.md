# Batch processing and POD studies

The headless API can process multiple geometry cases without opening the GUI.
Create one `MeshCase` per geometry configuration and keep the meshing options
explicit so the study can be reproduced.

```python
from pathlib import Path

from Pymeshit import run_mesh_case

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

for case_index, case in enumerate(cases):
    result = run_mesh_case(
        case,
        output_path=output_dir / f"case_{case_index:03d}.exo",
    )
    if not result.ok or result.failures:
        raise RuntimeError(f"Case {case_index} failed: {result.failures}")
```

## Node correspondence

Independently remeshing different geometries does **not** guarantee:

- the same number of nodes;
- the same element topology;
- the same node ordering; or
- that node `i` represents the same physical location in every case.

Fixing only the total node count would not establish the correspondence needed
for direct snapshot stacking in proper orthogonal decomposition (POD).

For varying fault dip or other geometry parameters, the robust workflow is:

1. Generate and simulate every watertight geometry independently.
2. Choose a common reference grid or reference mesh.
3. Interpolate each simulation field onto that reference representation.
4. Apply one consistent inside/outside-domain mask.
5. Stack the interpolated field vectors for POD.
6. Store interpolation settings and the reference-grid definition with the
   surrogate-model metadata.

A fixed-topology deformation workflow may be possible for small, smooth
geometry changes, but it is a separate meshing strategy and must be checked for
inverted or poor-quality elements. PyMeshIt currently prioritizes watertightness
and mesh quality over forced node correspondence.

## Reproducibility checklist

- Pin the PyMeshIt version.
- Record all `MeshOptions` values and TetGen switches.
- Keep input geometry files for every case.
- Use deterministic case and surface names.
- Check `MeshResult.ok` and `MeshResult.failures`; an export can fail after a
  volume mesh was created successfully.
- Record the common-grid coordinates, mask, and interpolation method.
