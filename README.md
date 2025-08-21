# MeshIt

MeshIt is a Python library and Qt-based GUI for mesh generation and manipulation with a C++ backend. The repository contains both library code and a full featured workflow GUI to process point clouds / polylines into conforming surface meshes and tetrahedral meshes.

## Highlights (GUI-driven workflow)

The included GUI (meshit_workflow_gui.py) implements a full MeshIt workflow with the following main tabs:

- 1. Load Data — load points, wells (polylines) or VTU/Poly formats; manage multiple datasets and colors.
- 2. Convex Hull — compute dataset boundaries (convex or rim for/quasi-planar sheets) with corner detection.
- 3. Segmentation — refine hulls by target feature size and per-surface length tables (RefineByLength).
- 4. Triangulation — generate surface triangulations with gradient, min-angle, interpolation and uniform options.
- 5. Intersections — compute & visualize global surface–surface and polyline–surface intersections; triple point detection.
- 6. Refine & Mesh — refine intersection/hull lines, generate conforming surface meshes, constraint selection UI, per-surface mesh density table.
- 7. Pre‑Tetramesh — select conforming surfaces, validate for TetGen, manage selection tree for tetrahedralization.
- 8. Tetra Mesh — generate and visualize tetrahedral meshes, assign materials, export results.

Other GUI features:
- Optional 3D interactive rendering via PyVista / pyvistaqt (gracefully disabled if PyVista is missing).
- Background worker thread support for batch hull/segment/triangulation/intersection processing with progress dialog and cancel.
- Per-surface refinement tables and automatic propagation of shared intersection constraints.
- Material seeds editor for tetrahedral material assignment with coordinate editors and auto-placement.
- Export options: OBJ/PLY/STL/CSV for triangulations and related export helpers included.

## Installation

Recommended minimal Python dependencies:

```bash
pip install numpy scipy matplotlib pyqt5 pyvista pyvistaqt
# optional: tetgen, pybind11, cython, triangle wrappers, scikit-learn, pandas
```

Notes:
- PyVista is optional — if not present the GUI will run but 3D visualization features are disabled.
- Some C++ acceleration (triangle/tetgen) is optional and requires a C++ toolchain (Visual Studio on Windows) and pybind11/CMake to build native wrappers.

## Quick start (GUI)

Run the GUI from the repository root:

```bash
python meshit_workflow_gui.py
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

## CLI / Scripting usage

The MeshIt core functions are exposed in the python package (see meshit.*). The GUI is a convenient workflow wrapper but core utilities (triangulation, intersection utilities, tetra mesh helpers) can be used programmatically for automated pipelines.

## Export & Interchange

- Surface triangulations can be exported to OBJ / PLY / STL / CSV.
- Conforming meshes produced in the refiner are stored in-memory and may be exported for TetGen input.
- VTU boundary segments are parsed and may be re-built if triangulation is re-run.

## Troubleshooting

- If 3D view buttons are disabled, install PyVista and pyvistaqt:
  pip install pyvista pyvistaqt
- For native speedups (triangle / tetgen wrappers), ensure a working C++ toolchain and build the wrappers using CMake and pybind11.
- Long batch operations use a background thread; cancel via the progress dialog if necessary.

## Contributing

Contributions are welcome. Please open an issue for discussion and submit PRs for fixes and features. Keep GUI behavior consistent with the tab-based workflow.

## License

This project is licensed under the GNU Affero General Public