# PyMeshIt

PyMeshIt is a complete Python package for mesh generation and manipulation with a full-featured Qt-based GUI. It provides a comprehensive workflow to process point clouds and polylines into conforming surface meshes and tetrahedral meshes.

**Note:** This version runs entirely in Python without C++ dependencies, making it easier to install and deploy.

## Highlights (GUI-driven workflow)

The included GUI (main.py) implements a full MeshIt workflow with the following main tabs:

- 1. Load Data — load points, wells (polylines) or VTU/Poly formats; manage multiple datasets and colors.
- 2. Convex Hull — compute dataset boundaries (convex or rim for/quasi-planar sheets) with corner detection.
- 3. Segmentation — refine hulls by target feature size and per-surface length tables (RefineByLength).
- 4. Triangulation — generate surface triangulations with gradient, min-angle, interpolation and uniform options.
- 5. Intersections — compute & visualize global surface–surface and polyline–surface intersections; triple point detection.
- 6. Refine & Mesh — refine intersection/hull lines, generate conforming surface meshes, constraint selection UI, per-surface mesh density table.
- 7. Pre‑Tetramesh — select conforming surfaces, validate for TetGen, manage selection tree for tetrahedralization.
- 8. Tetra Mesh — generate and visualize tetrahedral meshes, assign materials, export results.


## Installation

### From PyPI (Recommended)

```bash
pip install triangle
pip install pymeshit # Needs to be fixed
```

### From Source

If you want to install from source:

```bash
git clone https://github.com/waqashussain/meshit.git
cd meshit
pip install -e .
```

### Requirements

The package will automatically install all required dependencies:
- numpy
- scipy
- matplotlib
- PySide6
- pyvista
- tetgen
- triangle (optional)


## Quick start (GUI)

After installation, run the GUI:

```bash
meshit-gui
```

Or from Python:

```python
import Pymeshit
Pymeshit.main_wrapper()
```

Typical workflow:
1. Load one or more point or VTU files (File → Load).
2. Compute hulls (Convex Hull tab).
3. Compute segmentation (Segmentation tab) — set "Target Feature Size" or per-surface values.
4. Run triangulation (Triangulation tab), choose interpolation and quality settings.
5. Compute intersections (Intersections tab) to extract shared constraints and triple points.
6. Refine intersection lines and generate conforming meshes (Refine & Mesh tab).
7. Select conforming surfaces and validate for TetGen (Pre‑Tetramesh tab).
8. Generate and visualize tetrahedral mesh (Tetra Mesh tab) and export.

## Programmatic Usage

```python
import pymeshit
from pymeshit.intersection_utils import align_intersections_to_convex_hull

# Use pymeshit functions programmatically
# ...
```

## Contributing

Contributions are welcome. Please open an issue for discussion and submit PRs for fixes and features. Keep GUI behavior consistent with the tab-based workflow.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.