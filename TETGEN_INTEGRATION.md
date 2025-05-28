# Tetrahedral Mesh Generation with PyVista TetGen

The MeshIt application now includes a comprehensive tetrahedral mesh generation tab that uses the Python version of TetGen via the PyVista library.

## Installation Requirements

To use the tetrahedral mesh generation features, you need to install the following Python packages:

```bash
pip install tetgen pyvista
```

### About TetGen

TetGen is a quality tetrahedral mesh generator created by Hang Si. The PyVista wrapper provides a Python interface to this powerful C++ library, allowing for:

- Exact constrained Delaunay tetrahedralization
- Boundary conforming Delaunay meshes
- Voronoi partitions
- Quality mesh generation suitable for finite element methods

## Features

### Surface Selection
- Interactive table showing all available constrained surfaces from the pre-tetramesh tab
- Checkbox selection for each surface
- Select All/Deselect All functionality
- Displays surface name, type, and triangle count

### Material Management
- Define multiple material regions with unique properties
- Each material can have multiple location points
- 3D coordinate controls (X, Y, Z spinboxes) for precise positioning
- Automatic default location calculation based on surface centers
- Add/remove materials and locations dynamically

### TetGen Options
- Command line options input (default: "pq1.2AY")
- Quality ratio control (1.0-3.0, default: 1.2)
- Support for common TetGen switches:
  - `p`: Tetrahedralize a piecewise linear complex (PLC)
  - `q`: Quality mesh generation with ratio control
  - `A`: Apply a maximum area constraint
  - **Note**: `Y` option is not supported by PyVista TetGen wrapper

### Mesh Generation
- Real-time status updates during generation
- Comprehensive error handling and validation
- Mesh quality analysis (min/mean/max quality metrics)
- Volume calculations
- Detailed mesh statistics

### 3D Visualization
- Embedded PyVista 3D plotter
- Toggle controls for:
  - Surface visualization
  - Wireframe display
  - Material point visualization
- Interactive 3D navigation
- Color-coded material regions

### Export Capabilities
- Multiple format support: VTK, VTU, PLY, STL
- PyVista-based export for maximum compatibility
- Fallback manual export methods

## Workflow

1. **Prepare Surfaces**: Complete the pre-tetramesh tab to generate constrained triangulated surfaces
2. **Select Surfaces**: In the tetra mesh tab, select which surfaces to include in the tetrahedralization
3. **Define Materials**: Add material regions with location points
4. **Configure Options**: Set TetGen options and quality parameters
5. **Generate Mesh**: Click "Generate Tetrahedral Mesh" to create the volume mesh
6. **Visualize**: Use the 3D plotter to examine the resulting mesh
7. **Export**: Save the mesh in your preferred format

## TetGen Command Line Options

The most commonly used TetGen options in the command line format:

- `p`: Tetrahedralize a piecewise linear complex (PLC)
- `q[ratio]`: Quality mesh generation (e.g., `q1.5` for ratio of 1.5)
- `a[area]`: Maximum area constraint for tetrahedra
- `A`: Apply area constraints from input
- `Y`: Preserve the input surface mesh
- `V`: Verbose output

Example: `pq1.2A` generates a quality mesh with minimum radius-to-edge ratio of 1.2. 

**Note**: The `Y` flag is not supported by the PyVista TetGen wrapper, so use `pq1.2A` instead of `pq1.2AY`.

## Technical Details

### Data Structures
- **Surface Data**: Collected from constrained triangulation results
- **Material Regions**: Defined by 3D point locations with material attributes
- **Mesh Result**: PyVista UnstructuredGrid with tetrahedra, vertices, and quality metrics

### Quality Metrics
- **Minimum Scaled Jacobian**: Measures element quality (0-1, higher is better)
- **Aspect Ratio**: Ratio of circumradius to shortest edge
- **Volume**: Total mesh volume

### Error Handling
- Non-manifold surface detection
- Missing dependency warnings
- Surface data validation
- TetGen execution error reporting

## Integration with C++ Workflow

This Python implementation closely mirrors the C++ MeshIt tetgen workflow:

- Similar surface selection interface
- Material location management matching C++ patterns
- Compatible tetgen command line options
- Equivalent 3D visualization capabilities
- Advanced quality analysis features

## License Note

TetGen is licensed under AGPL v3, which applies to this integration. Please review the license terms before using in commercial applications.

## References

- [PyVista TetGen Documentation](https://tetgen.pyvista.org/)
- [Original TetGen by Hang Si](https://wias-berlin.de/software/tetgen/)
- [TetGen Academic Paper](http://doi.acm.org/10.1145/2629697) 