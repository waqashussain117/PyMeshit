# MeshIt

A Python library for mesh generation and manipulation with C++ backend.

## Installation

```bash
pip install meshit
```

## Features

- Generate meshes from polylines and boundaries
- Support for different mesh algorithms (Delaunay, advancing front)
- Custom triangle refinement with gradient control
- Export to VTU format for visualization
- Vector operations and geometry utilities

## Quick Start

```python
import meshit

# Create a model
model = meshit.MeshItModel()

# Add a simple triangle
points = [
    [0, 0, 0],
    [1, 0, 0],
    [0.5, 1, 0],
    [0, 0, 0]  # Close the loop
]
model.add_polyline(points)

# Generate mesh
model.set_mesh_algorithm("delaunay")
model.set_mesh_quality(1.2)
model.mesh()

# Export result
model.export_vtu("triangle_mesh.vtu")
```

## Advanced Features

### Custom Triangle Refinement

MeshIt includes a custom triangle refinement algorithm that mimics the behavior of the C++ `triunsuitable` function. This allows for precise control over mesh density based on features in the model.

```python
from meshit import extensions
from meshit.core import Surface, Vector3D

# Create a surface
surface = Surface()
for point in points:
    v = Vector3D(point[0], point[1], point[2])
    surface.add_vertex(v)

# Triangulate with gradient control
# Higher gradient values allow more variation in triangle size
# Lower values create more uniform meshes
gradient = 2.0
vertices, triangles = extensions.triangulate_with_triangle(surface, gradient=gradient)

# The result will have a higher density of triangles near important features
# and fewer triangles in areas of less detail
```

### GradientControl

The `GradientControl` class lets you specify feature points and their associated sizes to control mesh refinement:

```python
from meshit.core import GradientControl

# Get the gradient control instance
gc = GradientControl.get_instance()

# Update with gradient parameters
# gradient: How quickly triangle size increases with distance (default: 1.0)
# base_size: Base size for the triangulation
# feature_points: Points where finer triangulation is needed
# feature_sizes: Size parameters for each feature point
gc.update(gradient, base_size, num_feature_points, feature_point, feature_size)
```

## Troubleshooting

If you encounter issues with the installation, make sure you have the required dependencies:

```bash
pip install numpy scipy triangle matplotlib
```

For more complex builds with C++ components, you'll need:

- C++ compiler (Visual Studio on Windows, GCC on Linux)
- CMake
- pybind11

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the GNU Affero General Public License v3.0.