# PyMeshIt

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
   <img src="images/pymeshit_logo.png"/>
</p>

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

### From Release (Recommended)
        
PyMeshIt provides standalone executables for **Windows 10/11** and **Ubuntu (22.04+)**. These do not require Python to be installed.
        
1. Go to the [Releases page](https://github.com/waqashussain117/PyMeshit/releases).
2. Download the appropriate file for your OS:
   - **Windows**: `PyMeshit-vX.X.X-win64.zip`
   - **Ubuntu**: `PyMeshit-vX.X.X-linux.zip`
3. Extract the archive.
4. On Linux, you may need to make the file executable:
   ```bash
   chmod +x PyMeshIt
   ```
5. Run the executable (`./PyMeshIt` on Linux, double-click `PyMeshIt.exe` on Windows).

> **Note**: macOS support is planned for future releases.

### From PyPI (Cross-Platform)

If you prefer to run from source or use the Python API, you can install via pip:

```bash
pip install triangle
pip install pymeshit
```

### From Source

If you want to install from source:

```bash
git clone https://github.com/waqashussain117/PyMeshit
cd PyMeshit
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

**Linux Users**: You may need to install system libraries for Qt. On Ubuntu:
```bash
sudo apt-get install libxcb-cursor0 libxkbcommon-x11-0 libegl1 libopengl0 libgl1
```


## Quick start (GUI)



For installation either install the Requirements and then open through Python.
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



## Troubleshooting

### Linux / Virtual Machine Issues

If you encounter an error like `X Error of failed request: BadWindow` or the application crashes on startup, it is likely due to graphics driver compatibility, especially in Virtual Machines (VirtualBox, VMware).

Try running the application with software rendering forced:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
./PyMeshIt
```

Or force the Qt platform to X11:

```bash
export QT_QPA_PLATFORM=xcb
./PyMeshIt
```

## Contributing

Contributions are welcome. Please open an issue for discussion and submit PRs for fixes and features. Keep GUI behavior consistent with the tab-based workflow.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.