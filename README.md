# MeshIt

A Python library for mesh generation and manipulation with C++ backend.

## Installation

```bash
pip install meshit

Features
Generate meshes from polylines and boundaries
Support for different mesh algorithms (Delaunay, advancing front)
Export to VTU format for visualization
Vector operations and geometry utilities

# Quick Start
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