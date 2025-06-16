"""
MeshIt Workflow GUI

This application provides a graphical interface for the complete MeshIt workflow,
including file loading, convex hull computation, segmentation, triangulation, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import logging
import os
import sys
import time
from matplotlib.colors import ListedColormap, to_rgba
import re
# import QAbsractItemView
from PyQt5.QtWidgets import QAbstractItemView
from meshit.intersection_utils import align_intersections_to_convex_hull, Vector3D, Intersection, refine_intersection_line_by_length, insert_triple_points
from meshit.intersection_utils import prepare_plc_for_surface_triangulation, run_constrained_triangulation_py, calculate_triple_points, TriplePoint
# Import PyQt5
# import QMessageBox
import tetgen
from meshit.tetra_mesh_utils import TetrahedralMeshGenerator, create_tetrahedral_mesh
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                            QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QTabWidget,
                            QFileDialog, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
                            QCheckBox, QGroupBox, QRadioButton, QSlider, QLineEdit,
                            QSplitter, QDialog, QFormLayout, QButtonGroup, QMenu, QAction,
                            QListWidget, QColorDialog, QListWidgetItem, QProgressDialog,
                            QActionGroup, QSpacerItem, QTableWidget, QTableWidgetItem,
                            QTreeWidget, QTreeWidgetItem, QScrollArea)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QObject, QThread, QTimer, QSettings # Add QTimer and QSettings
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette, QPixmap
# Add these imports at the top of meshit_workflow_gui.py
from meshit.intersection_utils import run_constrained_triangulation_py
from scipy.spatial.distance import pdist, squareform
import gc
import atexit
# Import PyVista for 3D visualization
try:
    import pyvista as pv
    from pyvista import examples
    HAVE_PYVISTA = True
    logging.info("Successfully imported PyVista for 3D visualization")
except ImportError:
    HAVE_PYVISTA = False
    logging.warning("PyVista not available. 3D visualization disabled.")

# Configure logging to show ALL messages at INFO level
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MeshIt-Workflow")

# Also ensure meshit module logs are visible
logging.getLogger("meshit").setLevel(logging.INFO)

logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the DirectTriangleWrapper
try:
    from meshit.triangle_direct import DirectTriangleWrapper
    HAVE_DIRECT_WRAPPER = True
    logger.info("Successfully imported DirectTriangleWrapper")
except ImportError as e:
    logger.error(f"Failed to import DirectTriangleWrapper: {e}")
    HAVE_DIRECT_WRAPPER = False

# Add Scipy import for interpolation
from scipy.interpolate import griddata

# Worker class for background computations
class ComputationWorker(QObject):
    dataset_finished = pyqtSignal(int, str, bool) # index, name, success
    batch_finished = pyqtSignal(int, int, float) # success_count, total_eligible, elapsed_time
    error_occurred = pyqtSignal(str)

    def __init__(self, gui_instance):
        super().__init__()
        self.gui = gui_instance # Keep a reference to the GUI instance to call its methods
        self._is_running = True

    def stop(self):
        logger.debug("ComputationWorker stop requested.")
        self._is_running = False

    def compute_hulls_batch(self):
        """Worker method to compute hulls for all datasets."""
        logger.info("Worker: Starting hull computation batch.")
        if not self.gui.datasets:
            self.error_occurred.emit("No datasets loaded.")
            self.batch_finished.emit(0, 0, 0)
            logger.info("Worker: Hull batch finished (no datasets).")
            return

        success_count = 0
        total_datasets = len(self.gui.datasets)
        start_time = time.time()

        for i in range(total_datasets):
            if not self._is_running:
                logger.info(f"Worker: Hull computation loop canceled at index {i}.")
                break
            dataset_name = self.gui.datasets[i].get('name', f"Dataset {i}")
            logger.debug(f"Worker: Computing hull for '{dataset_name}' (index {i}).")
            # Moved the check inside the loop iteration, just before the potentially long call
            if not self._is_running:
                 logger.info(f"Worker: Hull computation canceled just before processing index {i}.")
                 break
            success = self.gui._compute_hull_for_dataset(i) # Call GUI's compute method
            logger.debug(f"Worker: Hull computation for index {i} finished (Success: {success}).")
            self.dataset_finished.emit(i, dataset_name, success)
            if success:
                success_count += 1

        elapsed = time.time() - start_time
        logger.info(f"Worker: Hull batch finished. Success: {success_count}/{total_datasets}. Elapsed: {elapsed:.2f}s.")
        self.batch_finished.emit(success_count, total_datasets, elapsed)
        # self._is_running = False # Resetting here might be redundant if worker is deleted

    def compute_segments_batch(self):
        """Worker method to compute segments for all eligible datasets."""
        logger.info("Worker: Starting segment computation batch.")
        datasets_with_hulls_indices = [i for i, d in enumerate(self.gui.datasets) if d.get('hull_points') is not None]

        if not datasets_with_hulls_indices:
            self.error_occurred.emit("No datasets have computed hulls.")
            self.batch_finished.emit(0, 0, 0)
            logger.info("Worker: Segment batch finished (no eligible datasets).")
            return

        success_count = 0
        total_eligible = len(datasets_with_hulls_indices)
        start_time = time.time()

        for i in datasets_with_hulls_indices:
            if not self._is_running:
                 logger.info(f"Worker: Segment computation loop canceled at index {i}.")
                 break
            dataset_name = self.gui.datasets[i].get('name', f"Dataset {i}")
            logger.debug(f"Worker: Computing segments for '{dataset_name}' (index {i}).")
             # Moved the check inside the loop iteration
            if not self._is_running:
                 logger.info(f"Worker: Segment computation canceled just before processing index {i}.")
                 break
            success = self.gui._compute_segments_for_dataset(i) # Call GUI's compute method
            logger.debug(f"Worker: Segment computation for index {i} finished (Success: {success}).")
            self.dataset_finished.emit(i, dataset_name, success)
            if success:
                success_count += 1

        elapsed = time.time() - start_time
        logger.info(f"Worker: Segment batch finished. Success: {success_count}/{total_eligible}. Elapsed: {elapsed:.2f}s.")
        self.batch_finished.emit(success_count, total_eligible, elapsed)
        # self._is_running = False

    def compute_triangulations_batch(self):
        """Worker method to compute triangulations for all eligible datasets."""
        logger.info("Worker: Starting triangulation computation batch.")
        datasets_with_segments_indices = [i for i, d in enumerate(self.gui.datasets) if d.get('segments') is not None]

        if not datasets_with_segments_indices:
            self.error_occurred.emit("No datasets have computed segments.")
            self.batch_finished.emit(0, 0, 0)
            logger.info("Worker: Triangulation batch finished (no eligible datasets).")
            return

        success_count = 0
        total_eligible = len(datasets_with_segments_indices)
        start_time = time.time()

        for i in datasets_with_segments_indices:
            if not self._is_running:
                 logger.info(f"Worker: Triangulation computation loop canceled at index {i}.")
                 break
            dataset_name = self.gui.datasets[i].get('name', f"Dataset {i}")
            logger.debug(f"Worker: Computing triangulation for '{dataset_name}' (index {i}).")
             # Moved the check inside the loop iteration
            if not self._is_running:
                 logger.info(f"Worker: Triangulation computation canceled just before processing index {i}.")
                 break
            success = self.gui._run_triangulation_for_dataset(i) # Call GUI's compute method
            logger.debug(f"Worker: Triangulation computation for index {i} finished (Success: {success}).")
            self.dataset_finished.emit(i, dataset_name, success)
            if success:
                success_count += 1

        elapsed = time.time() - start_time
        logger.info(f"Worker: Triangulation batch finished. Success: {success_count}/{total_eligible}. Elapsed: {elapsed:.2f}s.")
        self.batch_finished.emit(success_count, total_eligible, elapsed)
        # self._is_running = False

    def compute_global_intersections_task(self):
        """Worker task to trigger global intersection computation on the GUI instance."""
        logger.info("Worker: Starting global intersection computation task.")
        if not hasattr(self, 'gui') or self.gui is None:
            self.error_occurred.emit("GUI instance not available for intersection computation.")
            self.batch_finished.emit(0, 0, 0) # Indicate failure
            return

        start_time = time.time()
        success = False
        try:
            # Call the GUI's method to perform the actual computation
            # We expect this method to handle its own data preparation and error reporting
            # and return True/False for overall success.
            success = self.gui._compute_global_intersections()
            logger.info(f"Worker: GUI's global intersection computation returned: {success}")
        except Exception as e:
            error_msg = f"Unhandled error during global intersection computation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            success = False # Ensure failure is recorded

        elapsed_time = time.time() - start_time
        # Emit batch finished: 1 success if calculation succeeded, 0 otherwise.
        # Total eligible is conceptually 1 (the single global task).
        self.batch_finished.emit(1 if success else 0, 1, elapsed_time)
        logger.info(f"Worker: Global intersection task finished. Success: {success}. Elapsed: {elapsed_time:.2f}s.")


class SurfaceConstraintManager:
    """
    C++-faithful constraint manager that still matches the Python GUI.

    Key points
    ----------
    1. Triple-points are stored per-intersection (lookup table).
    2. Each intersection line is segmented only with its own triple-points
       → always produces 3 segments for a line with two internal triple-points.
    3. API expected by the GUI (tetra-mesh tab) is preserved:
         • generate_constraints_from_pre_tetra_data(datasets)
         • surface_constraints       (dict)
         • constraint_states         (dict)
         • constraint_rgb_map        (dict)
         • surface_visibility        (dict)
         • get_selected_surfaces() / get_constraint_summary()
    """

    DEFAULT_RGB_START = [1, 0, 0]

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self, parent=None):
        self.parent = parent
        self.surface_constraints = {}
        self.constraint_rgb_map = {}
        self.constraint_states = {}
        self.surface_visibility = {}        # NEW – used by the GUI
        self.next_rgb_color = self.DEFAULT_RGB_START.copy()

        # internal stores
        self._triple_lookup = {}            # intersection_id → [points]
        self._triple_points_global = []     # unique list for hull

    # ------------------------------------------------------------------ #
    # public entry point (wrapper kept for GUI compatibility)            #
    # ------------------------------------------------------------------ #
    def generate_constraints_from_pre_tetra_data(self, datasets):
        """
        Wrapper kept so the existing call
             constraint_manager.generate_constraints_from_pre_tetra_data(...)
        in the tetra-mesh tab continues to work.
        """
        self.build_constraints_from_datasets(datasets)

        # mark every surface “visible” by default (GUI uses this flag)
        self.surface_visibility = {idx: True for idx in range(len(datasets))}

        # ------------------------------------------------------------------ #
    # build GUI-constraints from the datasets (hull + intersections)     #
    # ------------------------------------------------------------------ #
    def build_constraints_from_datasets(self, datasets):
        """
        Creates the per-surface constraint lists.

        1)   adds the surface’s own hull boundary             (closed   loop)
        2)   adds every intersection poly-line exactly once   (open     line)

        Resulting order inside each surface’s list:
            – all hull segments   (ctype = "hull", "hull_1", …)
            – all intersection segments                       ("intersection_*")
        """
        # ---------- reset internal state --------------------------------
        self.surface_constraints.clear()
        self.constraint_rgb_map.clear()
        self.constraint_states.clear()
        self.next_rgb_color = self.DEFAULT_RGB_START.copy()

        # ---------- map triple-points first -----------------------------
        self._build_triple_point_lookup(datasets)

        # ---------- global intersection list (with provenance) ----------
        global_lines = self._reconstruct_global_intersection_network(datasets)

        # ---------- build constraints surface by surface ----------------
        for surf_idx, ds in enumerate(datasets):
            constraints = []

            # ---- 1) HULL boundary  -------------------------------------
            hull_lines = self._extract_hull_boundary_from_constrained_mesh(
                ds, surf_idx
            )
            for k, hull in enumerate(hull_lines):
                ctype = "hull" if k == 0 else f"hull_{k}"
                constraints.extend(
                    self._segment_open_or_closed_line(
                        hull,                       # poly-line
                        extras=[],                  # no triple points on hull
                        surf_idx=surf_idx,
                        ctype=ctype,
                        is_closed=True              # closed loop
                    )
                )

            # ---- 2) INTERSECTIONS  -------------------------------------
            for g in self._filter_lines_for_surface(global_lines,
                                                    ds, surf_idx):
                key = (g['dataset_idx'], g['intersection_id'])
                tp_for_line = self._triple_lookup.get(key, [])
                constraints.extend(
                    self._segment_open_or_closed_line(
                        g['line'],                 # poly-line
                        tp_for_line,               # triple-points mapped earlier
                        surf_idx=surf_idx,
                        ctype=f"intersection_{len(constraints)}",
                        is_closed=False            # open line
                    )
                )

            # store the finished list for this surface
            self.surface_constraints[surf_idx] = constraints
    # ------------------------------------------------------------------ #
    def _build_triple_point_lookup(self, datasets):
        """
        Build  self._triple_lookup : (dataset_idx, intersection_id) → [tp]
               self._triple_points_global : list[[x,y,z], …]
        Works whether triple-points are stored per dataset or only on the
        parent GUI, and whether intersection IDs already match or not.
        """
        import numpy as np

        self._triple_lookup.clear()
        self._triple_points_global.clear()

        # -------------------------------------------------------------- #
        # helper: return numeric xyz from any vertex representation
        # -------------------------------------------------------------- #
        def _xyz(pt):
            if hasattr(pt, "x"):
                return [float(pt.x), float(pt.y), float(pt.z)]
            elif isinstance(pt, (list, tuple)) and len(pt) >= 3:
                return [float(pt[0]), float(pt[1]), float(pt[2])]
            raise ValueError("Cannot extract xyz from vertex")

        # -------------------------------------------------------------- #
        # 1) collect every triple-point we can find
        # -------------------------------------------------------------- #
        raw_tp_items = []               # [(coord, ids), …]

        def _add_raw(tp_obj):
            if tp_obj is None:
                return
            if isinstance(tp_obj, dict):
                coord = tp_obj.get("point")
                ids   = tp_obj.get("intersection_ids") or tp_obj.get("intersections") or []
            else:
                coord = getattr(tp_obj, "point", None)
                ids   = getattr(tp_obj, "intersection_ids", [])
            if coord is None:
                return
            coord = _xyz(coord)
            raw_tp_items.append((coord, list(ids)))

        # a) stored inside each dataset
        for ds in datasets:
            for tp in ds.get("triple_points", []):
                _add_raw(tp)
        # b) global list on the parent GUI
        if hasattr(self, "_parent_gui"):
            for tp in getattr(self._parent_gui, "triple_points", []):
                _add_raw(tp)

        # deduplicate (1 e-8 tolerance)
        def _norm(p): return tuple(round(float(c), 8) for c in p)
        seen, uniq_tp_items = set(), []
        for c, ids in raw_tp_items:
            k = _norm(c)
            if k not in seen:
                seen.add(k)
                uniq_tp_items.append((c, ids))

        # -------------------------------------------------------------- #
        # 2) fast, id-based assignment  (works if ids already match)
        # -------------------------------------------------------------- #
                # 2) fast, id-based assignment – two passes
        #    a) TP stored inside each dataset           (old behaviour)
        #    b) TP stored only on the parent GUI list   (NEW)
        # -------------------------------------------------------------- #
        id_hit = False

        # a) local copies inside each dataset
        for ds_idx, ds in enumerate(datasets):
            for tp in ds.get("triple_points", []):
                coord = _xyz(tp["point"])
                ids   = tp.get("intersection_ids", [])
                for iid in ids:
                    self._triple_lookup.setdefault((ds_idx, iid), []).append(coord)
                if _norm(coord) not in {_norm(p) for p in self._triple_points_global}:
                    self._triple_points_global.append(coord)
                id_hit = True

        # b) global list on the parent GUI
        if hasattr(self, "_parent_gui"):
            for tp in getattr(self._parent_gui, "triple_points", []):
                coord = _xyz(tp["point"])
                ids   = tp.get("intersection_ids", [])
                for ds_idx, ds in enumerate(datasets):
                    max_iid = len(ds.get("intersection_constraints", [])) - 1
                    for iid in ids:
                        if 0 <= iid <= max_iid:      # id exists in this dataset
                            self._triple_lookup.setdefault((ds_idx, iid), []).append(coord)
                            id_hit = True
                if _norm(coord) not in {_norm(p) for p in self._triple_points_global}:
                    self._triple_points_global.append(coord)

        if id_hit:
            return                                  # done                            # done

        # -------------------------------------------------------------- #
        # 3) geometric fall-back – map every triple-point to the
        #    intersection polylines it truly lies on
        # -------------------------------------------------------------- #
        # pre-extract NUMERIC versions of all intersection lines
        intersection_lines = {
            ds_idx: [[_xyz(v) for v in line]
                     for line in ds.get("intersection_constraints", [])]
            for ds_idx, ds in enumerate(datasets)
        }

        # fast helpers
        def _pt_seg_distance(p, a, b):
            pa, ba = p - a, b - a
            l2 = np.dot(ba, ba)
            if l2 == 0.0:
                return np.linalg.norm(pa)
            t = np.clip(np.dot(pa, ba) / l2, 0.0, 1.0)
            proj = a + t * ba
            return np.linalg.norm(p - proj)

        tol = 1e-3       # absolute lower bound (model units ≈ mm)

        for coord, _unused_ids in uniq_tp_items:
            p_np = np.asarray(coord, dtype=float)

            for ds_idx, lines in intersection_lines.items():
                for int_id, line in enumerate(lines):
                    if len(line) < 2:
                        continue
                    l_np = np.asarray(line, dtype=float)

                    # quick bounding-box reject
                    if not ((l_np.min(axis=0) - tol <= p_np).all() and
                            (p_np <= l_np.max(axis=0) + tol).all()):
                        continue

                    # adaptive tolerance: 1 % of total poly-line length
                    line_len = np.sum(
                        np.linalg.norm(l_np[1:] - l_np[:-1], axis=1)
                    )
                    adaptive_tol = max(tol, 0.01 * line_len)

                    # (a) distance to any segment
                    on_seg = any(
                        _pt_seg_distance(p_np, l_np[i], l_np[i + 1]) < adaptive_tol
                        for i in range(len(l_np) - 1)
                    )

                    # (b) OR distance to any existing vertex
                    to_vertex = np.min(
                        np.linalg.norm(l_np - p_np, axis=1)
                    ) < adaptive_tol

                    if on_seg or to_vertex:
                        self._triple_lookup.setdefault(
                            (ds_idx, int_id), []
                        ).append(coord)

            # keep a global copy for hull handling
            if _norm(coord) not in {_norm(p) for p in self._triple_points_global}:
                self._triple_points_global.append(coord)
        # -------------------------------------------------------------- #
        # DEBUG summary – how many TP per intersection?
        # -------------------------------------------------------------- #
        log = logging.getLogger("MeshIt-Workflow")
        for (ds_i, int_i), pts in sorted(self._triple_lookup.items()):
            log.info(f"TP lookup  ds={ds_i}  int={int_i}  count={len(pts)}")
        # -------------------------------------------------------------- #
        # INFO summary – how many TP per intersection?
        # -------------------------------------------------------------- #
        log = logging.getLogger("MeshIt-Workflow")
        for (ds_i, int_i), pts in sorted(self._triple_lookup.items()):
            log.info(f"TP lookup  ds={ds_i}  int={int_i}  count={len(pts)}")
    def _extract_hull_boundary_from_constrained_mesh(self, dataset, surf_id):
        """
        Re-implements the original Python logic:

        • If constrained triangulation exists, find edges that belong to only
          one triangle (boundary edges) and walk them to build closed lines.
        • Fallback: if only ‘hull_points’ exist, return that as a single line.
        Returns a list of poly-lines (each a list[[x,y,z], …]).
        """
        import numpy as np
        verts  = dataset.get("constrained_vertices")
        tris   = dataset.get("constrained_triangles")

        # fallback – simple stored hull
        if (verts is None or tris is None or
                len(verts) == 0 or len(tris) == 0):
            hp = dataset.get("hull_points", [])
            return [hp] if len(hp) >= 3 else []

        verts = np.asarray(verts)
        tris  = np.asarray(tris)

        # 1) build edge→count map
        edge_count = {}
        for tri in tris:
            if len(tri) < 3:
                continue
            e0 = tuple(sorted((tri[0], tri[1])))
            e1 = tuple(sorted((tri[1], tri[2])))
            e2 = tuple(sorted((tri[2], tri[0])))
            for e in (e0, e1, e2):
                edge_count[e] = edge_count.get(e, 0) + 1

        # 2) keep boundary edges (appear only once)
        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if not boundary_edges:
            return []

        # 3) trace edges into poly-lines
        #    build adjacency map vertex → connected vertices
        adj = {}
        for a, b in boundary_edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        lines = []
        visited = set()
        for start in adj:
            if start in visited:
                continue
            line = [start]
            visited.add(start)
            cur = start
            while True:
                nxts = [n for n in adj[cur] if n not in visited]
                if not nxts:
                    break
                nxt = nxts[0]
                line.append(nxt)
                visited.add(nxt)
                cur = nxt
            # convert indices to coords
            lines.append([verts[idx][:3].tolist() for idx in line])
        return lines

    # ------------------------------------------------------------------ #
    # global intersection network                                        #
    # ------------------------------------------------------------------ #
    def _reconstruct_global_intersection_network(self, datasets):
        """
        Returns a list of dictionaries:
            { 'line'          : poly-line [[x,y,z], …],
              'dataset_idx'   : int   (where it came from),
              'intersection_id': int   (its id inside that dataset) }
        The position in this list (enumeration) is still used only
        for naming/colouring, but the real lookup uses the pair
        (dataset_idx, intersection_id).
        """
        global_lines = []
        for ds_idx, ds in enumerate(datasets):
            for int_id, line in enumerate(ds.get("intersection_constraints", [])):
                if len(line) >= 2:
                    global_lines.append({
                        'line'          : line,
                        'dataset_idx'   : ds_idx,
                        'intersection_id': int_id      # original id
                    })
        return global_lines

    # ------------------------------------------------------------------ #
        # ------------------------------------------------------------------ #
    # decide which global lines belong to this surface                   #
    # ------------------------------------------------------------------ #
    def _filter_lines_for_surface(self, global_lines, dataset, surf_idx):
        """
        Return global intersection lines that

        (a) originate from *this* dataset (g['dataset_idx'] == surf_idx)
        (b) are actually on / very near this surface (KD-tree test)
        (c) are not geometric duplicates of another accepted line
            (same points, maybe reversed order)

        Result: each real intersection appears only once in the GUI.
        """
        import numpy as np
        from scipy.spatial import cKDTree as KD

        # ---- build a KD-tree of this surface’s own points -------------
        own_xyz = [
            [_pt[:3] if isinstance(_pt, (list, tuple))
             else [float(_pt.x), float(_pt.y), float(_pt.z)]
             for _pt in line]
            for line in dataset.get("intersection_constraints", [])
        ]
        if not own_xyz:
            return []

        all_pts = np.asarray([p for l in own_xyz for p in l], dtype=float)
        tree = KD(all_pts)

        # ---- helper: geometric equality test -------------------------
        def _same_poly(a, b, eps=1e-5):
            """
            True if the two polylines have the same vertices (within eps),
            either in identical order or completely reversed order.
            """
            if len(a) != len(b):
                return False
            a_np = np.asarray(a)
            b_np = np.asarray(b)
            if np.allclose(a_np, b_np, atol=eps):
                return True
            if np.allclose(a_np, b_np[::-1], atol=eps):
                return True
            return False

        eps = 1e-2
        accepted = []           # will hold dicts from global_lines

        for g in global_lines:
            # keep only the copy that belongs to THIS dataset
            if g['dataset_idx'] != surf_idx:
                continue

            pts_xyz = [p[:3] if isinstance(p, (list, tuple))
                       else [p.x, p.y, p.z] for p in g['line']]
            # (b) – proximity test
            hits = sum(tree.query(p, k=1)[0] < eps for p in pts_xyz)
            if hits / len(pts_xyz) < 0.5:
                continue

            # (c) – duplicate-geometry test
            if any(_same_poly(pts_xyz, a['xyz']) for a in accepted):
                continue

            g_copy = g.copy()
            g_copy['xyz'] = pts_xyz        # cache for next comparisons
            accepted.append(g_copy)

        # strip helper key before returning
        for d in accepted:
            d.pop('xyz', None)
        return accepted
        # ------------------------------------------------------------------ #
    # line segmentation  (keeps geometry, honours triple-points)         #
    # ------------------------------------------------------------------ #
    def _segment_open_or_closed_line(self, pts, extras,
                                     surf_idx, ctype, is_closed):
        """
        Split a poly-line into selectable GUI segments while preserving the
        original geometry.

        • pts      original vertex list (never modified)
        • extras   triple-point coordinates that must act as splitters
        • ctype    “hull”, “intersection_*”, etc.
        • is_closed   True  → wrap-around (hull)   |  False → open line
        """
        import numpy as np
        from scipy.spatial.distance import cdist
        log = logging.getLogger("MeshIt-Workflow")

        if len(pts) < 2:
            return []

        # helper: np.array([x,y,z]) regardless of representation
        def _np_xyz(v):
            if hasattr(v, "x"):
                return np.asarray([v.x, v.y, v.z], dtype=float)
            elif isinstance(v, (list, tuple)):
                return np.asarray(v[:3], dtype=float)
            raise ValueError("Bad vertex type")

        log.info(f"SEGMENT  surf={surf_idx}  ctype={ctype}  TP={len(extras or [])}")

        # ------------------------------------------------------------------
        # 1) working copy – insert projected triple-points
        # ------------------------------------------------------------------
        line = pts.copy()

        # adaptive tolerance: 1 % of total length, minimum 1 mm
        seg_lens = [np.linalg.norm(_np_xyz(line[i + 1]) - _np_xyz(line[i]))
                    for i in range(len(line) - 1)]
        line_len = sum(seg_lens) if seg_lens else 0.0
        tol = max(1e-3, 0.01 * line_len)

        inserted_split = set()          # indices created by TP insertion

        def _project(p, a, b):
            p, a, b = map(_np_xyz, (p, a, b))
            ab = b - a
            l2 = np.dot(ab, ab)
            if l2 == 0.0:
                return a.copy(), np.linalg.norm(p - a)
            t = np.clip(np.dot(p - a, ab) / l2, 0.0, 1.0)
            proj = a + t * ab
            return proj, np.linalg.norm(p - proj)

        for tp in extras or []:
            tp_coord = [tp.x, tp.y, tp.z] if hasattr(tp, "x") else tp[:3]

            best_i, best_d, best_proj = None, float("inf"), None
            for i in range(len(line) - 1):
                proj, dist = _project(tp_coord, line[i], line[i + 1])
                if dist < best_d:
                    best_i, best_d, best_proj = i, dist, proj

            if best_i is None:
                continue

            if best_d < tol:
                # insert only if truly between the two vertices
                if (not np.allclose(best_proj, _np_xyz(line[best_i]),     atol=1e-9) and
                    not np.allclose(best_proj, _np_xyz(line[best_i + 1]), atol=1e-9)):
                    line.insert(best_i + 1, best_proj.tolist())
                    inserted_split.add(best_i + 1)
            else:
                # logical split at nearest existing vertex
                dists = cdist([tp_coord], [_np_xyz(v) for v in line])[0]
                inserted_split.add(int(np.argmin(dists)))

        log.info(f"  inserted TP indices {sorted(inserted_split)}")

        # ------------------------------------------------------------------
        # 2) extra splits at refined vertices (type != DEFAULT)
        # ------------------------------------------------------------------
        refined_split = {
            idx for idx, v in enumerate(line)
            if ((isinstance(v, (list, tuple)) and len(v) > 3 and
                 str(v[3]).upper() != 'DEFAULT') or
                (hasattr(v, 'point_type') and str(v.point_type).upper() != 'DEFAULT') or
                (hasattr(v, 'type') and str(v.type).upper() != 'DEFAULT'))
        }
        log.info(f"  refined-split indices {sorted(refined_split)}")

        # ------------------------------------------------------------------
        # 3) decide where to break the poly-line
        # ------------------------------------------------------------------
        if is_closed and str(ctype).lower().startswith("hull"):
            # Hull boundary – expose every edge, one segment per edge
            break_ids = set(range(len(line)))
        else:
            break_ids = {0, len(line) - 1}.union(inserted_split, refined_split)

        split = sorted(break_ids)
        if is_closed:
            split.append(split[0])        # wrap-around for closed loop

        # ------------------------------------------------------------------
        # 4) build the segments
        # ------------------------------------------------------------------
        segments = []
        for k in range(len(split) - 1):
            a, b = split[k], split[k + 1]
            part = (line[a:b + 1] if a < b else
                    line[a:] + line[:b + 1])
            if len(part) < 2:
                continue

            rgb = self._next_rgb()
            seg_info = dict(
                points=part,
                type=ctype,
                surface_id=surf_idx,
                segment_id=len(segments),
                rgb=rgb,
                constraint_type="UNDEFINED"
            )
            segments.append(seg_info)
            self.constraint_rgb_map[tuple(rgb)] = (surf_idx, len(segments) - 1)

        log.info(f"  → {len(segments)} GUI-segments produced")
        return segments
    def _insert_points_preserving_order(self, line, to_insert):
        import numpy as np

        def dist(p, q):
            return np.linalg.norm(np.asarray(p) - np.asarray(q))

        new_line = line.copy()
        for p in to_insert:
            best_seg, best_d = None, float("inf")
            for i in range(len(new_line) - 1):
                a, b = np.asarray(new_line[i]), np.asarray(new_line[i + 1])
                ab = b - a
                L2 = np.dot(ab, ab)
                if L2 == 0:
                    continue
                t = np.dot(np.asarray(p) - a, ab) / L2
                if 0.0 <= t <= 1.0:
                    proj = a + t * ab
                    d = np.linalg.norm(proj - p)
                    if d < best_d:
                        best_seg, best_d = i, d
            if best_seg is None:
                if all(dist(p, q) > 1e-8 for q in new_line):
                    new_line.append(p)
            else:
                new_line.insert(best_seg + 1, p)
        return new_line

    # ------------------------------------------------------------------ #
    # RGB generator                                                      #
    # ------------------------------------------------------------------ #
    def _next_rgb(self):
        rgb = self.next_rgb_color.copy()
        self.next_rgb_color[0] += 1
        if self.next_rgb_color[0] > 255:
            self.next_rgb_color[0] = 0
            self.next_rgb_color[1] += 1
            if self.next_rgb_color[1] > 255:
                self.next_rgb_color[1] = 0
                self.next_rgb_color[2] += 1
                if self.next_rgb_color[2] > 255:
                    self.next_rgb_color = self.DEFAULT_RGB_START.copy()
        return rgb

    # ------------------------------------------------------------------ #
    # quick helpers queried by the GUI                                   #
    # ------------------------------------------------------------------ #
    def get_selected_surfaces(self):
        out = []
        for s_idx, cons in self.surface_constraints.items():
            if any(self.constraint_states.get((s_idx, i)) == "SEGMENTS"
                   for i in range(len(cons))):
                out.append(s_idx)
        return out

    def get_constraint_summary(self):
        total = len(self.constraint_states)
        selected = sum(1 for v in self.constraint_states.values()
                       if v == "SEGMENTS")
        holes = sum(1 for v in self.constraint_states.values()
                    if v == "HOLES")
        return dict(total=total, selected=selected,
                    holes=holes, undefined=total - selected - holes)

class MeshItWorkflowGUI(QMainWindow):
    # Define a color cycle for datasets
    DEFAULT_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    def __init__(self):
        """Initialize the GUI"""
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("MeshIt Workflow GUI")
        self.setGeometry(100, 100, 1400, 900) # Increased default size
        # Add color cycle for surface visualization
        self.color_cycle = [
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta',
            'brown', 'pink', 'gray', 'olive', 'navy', 'teal', 'lime', 'indigo'
        ]
        # Initialize data structures for multiple datasets
        self.datasets = []  # List to hold dataset dictionaries
        self.current_dataset_index = -1  # Index of the currently active dataset
        self._color_index = 0 # For assigning colors to datasets
        
        self.plotters = {}
        # Visualization placeholders and state
        self.points = None
        self.hull_points = None
        self.segments = None
        self.triangulation_result = None
        self.active_view = "points" # To track which visualization is active
        
        # Initialize optional variables that could be used for 3D visualization
        self.view_3d_enabled = HAVE_PYVISTA
        self.height_factor = 1.0
        self.current_plotter = None
        self.pv_plotter = None
        
        # Initialize legend widgets
        self.file_legend_widget = None
        self.hull_legend_widget = None
        self.segment_legend_widget = None
        self.tri_legend_widget = None
        
        # Create layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create notebook with multiple tabs
        self.notebook = QTabWidget()
        self.main_layout.addWidget(self.notebook)
        
        # Setup file tab
        self.file_tab = QWidget()
        self.notebook.addTab(self.file_tab, "1. Load Data")
        self._setup_file_tab()
        
        # Setup hull tab
        self.hull_tab = QWidget()
        self.notebook.addTab(self.hull_tab, "2. Convex Hull")
        self._setup_hull_tab()
        
        # Setup segment tab
        self.segment_tab = QWidget()
        self.notebook.addTab(self.segment_tab, "3. Segmentation")
        self._setup_segment_tab()
        
        # Setup triangulation tab
        self.triangulation_tab = QWidget()
        self.notebook.addTab(self.triangulation_tab, "4. Triangulation")
        self._setup_triangulation_tab()
        
        # Add to __init__ after other tabs are created
        self._setup_intersection_tab()
        
        # Setup Refine & Mesh Settings tab
        self.refine_mesh_tab = QWidget() # Add this line
        self.notebook.addTab(self.refine_mesh_tab, "6. Refine & Mesh") # Add this line
        self._setup_refine_mesh_tab() # Add this line

        # Setup pre-tetramesh tab
        self.pre_tetramesh_tab = QWidget()
        self.notebook.addTab(self.pre_tetramesh_tab, "7. Pre-Tetramesh")
        self._setup_pre_tetramesh_tab()

        # Setup tetra mesh tab
        self.tetra_mesh_tab = QWidget()
        self.notebook.addTab(self.tetra_mesh_tab, "8. Tetra Mesh")
        self._setup_tetra_mesh_tab()

        # Placeholder for the refine_mesh_tab plotter
        self.refine_mesh_plotter = None
        self.refine_mesh_viz_frame = None # Add this line for consistency
        

        # Create menu bar
        self._create_menu_bar()
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Connect signals
        self.notebook.currentChanged.connect(self._on_tab_changed)
        
        # Show ready message
        self.statusBar().showMessage("MeshIt GUI Ready")
        
        # Initialize plot axes (will be properly set up in _setup_..._tab methods)
        self.hull_ax = None
        self.segment_ax = None
        self.tri_ax = None
        self.hull_canvas = None
        self.segment_canvas = None
        self.tri_canvas = None
        
        # For 3D view management
        self.plotter = None
        self.vtk_widget = None
        
        # Register VTK cleanup on exit
        self._ensure_vtk_cleanup_on_exit()
        
        self._create_main_layout()
        self.show()
        self._update_statistics() # Initial update
    
    def _create_menu_bar(self):
        """Create the menu bar"""
        menu_bar = self.menuBar()
        
        # --- File Menu ---
        file_menu = menu_bar.addMenu("&File")

        load_action = QAction("&Load File...", self)
        load_action.setStatusTip("Load a single point data file")
        load_action.triggered.connect(self.load_file)
        file_menu.addAction(load_action)

        load_multiple_action = QAction("Load &Multiple Files...", self)
        load_multiple_action.setStatusTip("Load multiple point data files as separate datasets")
        load_multiple_action.triggered.connect(self.load_multiple_files)
        file_menu.addAction(load_multiple_action)

        generate_action = QAction("&Generate Test Data...", self)
        generate_action.setStatusTip("Generate sample point data for testing")
        generate_action.triggered.connect(self.generate_test_data)
        file_menu.addAction(generate_action)
        
        file_menu.addSeparator()

        export_action = QAction("&Export Active Result...", self)
        export_action.setStatusTip("Export the triangulation result of the active dataset")
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- Dataset Menu ---
        dataset_menu = menu_bar.addMenu("&Datasets")

        process_all_action = dataset_menu.addAction("&Process All Datasets")
        process_all_action.setStatusTip("Run all steps (hull, segments, triangulation) for all datasets")
        process_all_action.triggered.connect(self.process_all_datasets)
        
        dataset_menu.addSeparator()

        remove_active_action = dataset_menu.addAction("&Remove Active Dataset")
        remove_active_action.setStatusTip("Remove the currently selected dataset from the list")
        remove_active_action.triggered.connect(self.remove_active_dataset)

        clear_all_action = dataset_menu.addAction("Clear &All Datasets")
        clear_all_action.setStatusTip("Remove all loaded datasets")
        clear_all_action.triggered.connect(self.clear_all_datasets)


        # --- Visualization Menu ---
        viz_menu = menu_bar.addMenu("&Visualization")

        view_3d_action = viz_menu.addAction("Show &3D View")
        view_3d_action.setStatusTip("Open a separate interactive 3D view (requires PyVista)")
        view_3d_action.triggered.connect(self.show_3d_view)
        view_3d_action.setEnabled(HAVE_PYVISTA) # Disable if PyVista not available
    
        viz_menu.addSeparator()
        
        # --- Height Factor Submenu ---
        height_menu = viz_menu.addMenu("Height Factor (3D)")
        height_group = QActionGroup(self) # Use QActionGroup for radio-button like behavior

        height_factors = [0.1, 0.5, 1.0, 2.0, 5.0]
        for factor in height_factors:
            height_action = QAction(f"{factor:.1f}x", self, checkable=True)
            height_action.setActionGroup(height_group)
            height_action.triggered.connect(lambda checked, f=factor: self._set_height_factor_and_update(f) if checked else None)
            if abs(factor - self.height_factor) < 1e-6: # Check if it's the current factor
                 height_action.setChecked(True)
            height_menu.addAction(height_action)


        # --- Help Menu ---
        help_menu = menu_bar.addMenu("&Help")
        about_action = help_menu.addAction("&About")
        about_action.setStatusTip("Show information about the application")
        about_action.triggered.connect(self._show_about)
    
    def _setup_file_tab(self):
        """Sets up the file loading tab with controls and visualization area"""
        # Main layout for the tab
        tab_layout = QHBoxLayout(self.file_tab)
        
        # --- Control panel (left side) ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(350) # Limit width
        control_layout = QVBoxLayout(control_panel)
        
        # -- File Loading Controls --
        file_group = QGroupBox("Load Data")
        file_layout = QVBoxLayout(file_group)
        
        # Buttons for loading files
        load_btn = QPushButton("Load Single File...")
        load_btn.setToolTip("Load points from a single file (.txt, .csv, .pts)")
        load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(load_btn)
        
        # Button for loading multiple files
        load_multiple_btn = QPushButton("Load Multiple Files...")
        load_multiple_btn.setToolTip("Load points from multiple files as separate datasets")
        load_multiple_btn.clicked.connect(self.load_multiple_files)
        file_layout.addWidget(load_multiple_btn)
        
        # Test data generation
        test_btn = QPushButton("Generate Test Data")
        test_btn.setToolTip("Generate sample point data for testing")
        test_btn.clicked.connect(self.generate_test_data)
        file_layout.addWidget(test_btn)
        
        control_layout.addWidget(file_group)
        
        # -- Dataset List --
        datasets_group = QGroupBox("Datasets")
        datasets_layout = QVBoxLayout(datasets_group)
        
        self.dataset_list_widget = QListWidget()
        self.dataset_list_widget.setToolTip("List of loaded datasets. Select one to view/process.")
        # Enable extended selection later if needed: self.dataset_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.dataset_list_widget.itemSelectionChanged.connect(self._on_dataset_selection_changed)
        self.dataset_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.dataset_list_widget.customContextMenuRequested.connect(self._show_dataset_context_menu)
        datasets_layout.addWidget(self.dataset_list_widget)
        
        control_layout.addWidget(datasets_group)
        
        # -- Statistics --
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        # Number of datasets
        self.num_datasets_label = QLabel("Datasets: 0")
        stats_layout.addWidget(self.num_datasets_label)
        
        # Number of points
        self.points_label = QLabel("Points: 0")
        stats_layout.addWidget(self.points_label)
        
        # Bounding box info
        self.bounds_label = QLabel("Bounds: N/A")
        stats_layout.addWidget(self.bounds_label)
        
        control_layout.addWidget(stats_group)
        control_layout.addStretch()
        
        tab_layout.addWidget(control_panel)
        
        # --- Visualization Area (right side) ---
        viz_group = QGroupBox("Point Cloud Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Create container for visualization
        self.file_viz_frame = QWidget()
        self.file_viz_layout = QVBoxLayout(self.file_viz_frame)
        
        # Initial visualization panel will show placeholder message
        if HAVE_PYVISTA:
            placeholder_text = "Load data to visualize points in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nLoad data to visualize points in 2D."
            
        self.file_viz_placeholder = QLabel(placeholder_text)
        self.file_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.file_viz_layout.addWidget(self.file_viz_placeholder)
        
        viz_layout.addWidget(self.file_viz_frame)
        tab_layout.addWidget(viz_group, 1)  # 1 = stretch factor
    
    def _setup_hull_tab(self):
        """Sets up the convex hull tab with controls and visualization area"""
        # Main layout for the tab
        tab_layout = QHBoxLayout(self.hull_tab)

        # --- Control panel (left side) ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_panel)

        # -- Hull Controls --
        hull_group = QGroupBox("Hull Controls")
        hull_layout = QVBoxLayout(hull_group)
        
        # Compute hull button
        compute_btn = QPushButton("Compute Convex Hull (All Datasets)")
        compute_btn.setObjectName("compute_btn") # Set the object name
        compute_btn.setToolTip("Compute convex hull for all loaded datasets")
        compute_btn.clicked.connect(self.compute_all_hulls)
        hull_layout.addWidget(compute_btn)
        
        control_layout.addWidget(hull_group)
        
        # -- Statistics --
        stats_group = QGroupBox("Hull Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        # Hull points count
        self.hull_points_label = QLabel("Hull vertices: 0")
        stats_layout.addWidget(self.hull_points_label)
        
        # Hull area
        self.hull_area_label = QLabel("Hull area: 0.0")
        stats_layout.addWidget(self.hull_area_label)
        
        control_layout.addWidget(stats_group)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(lambda: self.notebook.setCurrentIndex(0))
        nav_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(lambda: self.notebook.setCurrentIndex(2))
        nav_layout.addWidget(next_btn)
        
        control_layout.addLayout(nav_layout)
        control_layout.addStretch()
        
        tab_layout.addWidget(control_panel)
        
        # --- Visualization Area (right side) ---
        viz_group = QGroupBox("Convex Hull Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Create container for visualization
        self.hull_viz_frame = QWidget()
        self.hull_viz_layout = QVBoxLayout(self.hull_viz_frame)
        
        # Initial visualization panel will show placeholder message
        if HAVE_PYVISTA:
            placeholder_text = "Compute convex hull to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nCompute convex hull to visualize in 2D."
            
        self.hull_viz_placeholder = QLabel(placeholder_text)
        self.hull_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.hull_viz_layout.addWidget(self.hull_viz_placeholder)
        
        viz_layout.addWidget(self.hull_viz_frame)
        tab_layout.addWidget(viz_group, 1)  # 1 = stretch factor
    
    def _setup_segment_tab(self):
        """Sets up the segmentation tab with controls and visualization area"""
        # Main layout for the tab
        tab_layout = QHBoxLayout(self.segment_tab)

        # --- Control panel (left side) ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_panel)

        # -- Segmentation Controls --
        segment_group = QGroupBox("Segmentation Controls")
        segment_layout = QVBoxLayout(segment_group)

        # --- START EDIT: Replace old controls with Target Feature Size ---
        # Remove old segment length and density controls
        # length_layout = QHBoxLayout()
        # length_layout.addWidget(QLabel("Segment Length:"))
        # self.segment_length_input = QLineEdit("1.0")
        # length_layout.addWidget(self.segment_length_input)
        # segment_layout.addLayout(length_layout)
        #
        # density_layout = QHBoxLayout()
        # density_layout.addWidget(QLabel("Density:"))
        # self.segment_density_slider = QSlider(Qt.Horizontal)
        # self.segment_density_slider.setMinimum(50)
        # self.segment_density_slider.setMaximum(200)
        # self.segment_density_slider.setValue(100)
        # density_layout.addWidget(self.segment_density_slider)
        # segment_layout.addLayout(density_layout)

        # Add new Target Feature Size control
        target_size_layout = QFormLayout()
        self.target_feature_size_input = QDoubleSpinBox()
        self.target_feature_size_input.setRange(0.1, 500.0) # Adjust range as needed
        self.target_feature_size_input.setValue(20.0)    # Example default size
        self.target_feature_size_input.setSingleStep(0.5)
        self.target_feature_size_input.setToolTip(
            "Specify the desired approximate size for features (e.g., segment length).\n"
            "This value also influences the base size for triangulation.\n"
            "Smaller values result in denser segmentation and meshes."
        )
        target_size_layout.addRow("Target Feature Size:", self.target_feature_size_input)
        segment_layout.addLayout(target_size_layout)
        # --- END EDIT ---

        # Compute segments button
        compute_btn = QPushButton("Compute Segmentation (All Datasets)") # Update button text
        compute_btn.setObjectName("compute_btn") # Set the object name
        compute_btn.setToolTip("Compute segmentation for all datasets with computed hulls") # Update tooltip
        compute_btn.clicked.connect(self.compute_all_segments) # Connect to the new batch method
        segment_layout.addWidget(compute_btn)

        control_layout.addWidget(segment_group)

        # -- Statistics --
        stats_group = QGroupBox("Segment Statistics")
        stats_layout = QVBoxLayout(stats_group)

        # Number of segments
        self.num_segments_label = QLabel("Segments: 0")
        stats_layout.addWidget(self.num_segments_label)

        # Average segment length
        self.avg_segment_length_label = QLabel("Avg length: 0.0")
        stats_layout.addWidget(self.avg_segment_length_label)

        control_layout.addWidget(stats_group)

        # Navigation buttons
        nav_layout = QHBoxLayout()

        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(lambda: self.notebook.setCurrentIndex(1))
        nav_layout.addWidget(prev_btn)

        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(lambda: self.notebook.setCurrentIndex(3))
        nav_layout.addWidget(next_btn)

        control_layout.addLayout(nav_layout)
        control_layout.addStretch()

        tab_layout.addWidget(control_panel)

        # --- Visualization Area (right side) ---
        viz_group = QGroupBox("Segmentation Visualization")
        viz_layout = QVBoxLayout(viz_group)

        # Create container for visualization
        self.segment_viz_frame = QWidget()
        self.segment_viz_layout = QVBoxLayout(self.segment_viz_frame)

        # Initial visualization panel will show placeholder message
        if HAVE_PYVISTA:
            placeholder_text = "Compute segmentation to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\\nCompute segmentation to visualize in 2D."

        self.segment_viz_placeholder = QLabel(placeholder_text)
        self.segment_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.segment_viz_layout.addWidget(self.segment_viz_placeholder)

        viz_layout.addWidget(self.segment_viz_frame)
        tab_layout.addWidget(viz_group, 1)  # 1 = stretch factor
    
    def _setup_triangulation_tab(self):
        """Sets up the triangulation tab with controls and visualization area"""
        # Main layout for the tab
        tab_layout = QHBoxLayout(self.triangulation_tab)

        # --- Control panel (left side) ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_panel)

        # -- Triangulation Controls --
        tri_group = QGroupBox("Triangulation Controls")
        tri_layout = QVBoxLayout(tri_group)

        # Mesh quality controls (remain the same)
        quality_group = QGroupBox("Quality Settings")
        quality_layout = QFormLayout(quality_group)

        # Gradient
        self.gradient_input = QDoubleSpinBox()
        self.gradient_input.setRange(1.0, 3.0)
        self.gradient_input.setValue(2.0)
        self.gradient_input.setSingleStep(0.1)
        quality_layout.addRow("Gradient:", self.gradient_input)

        # Min angle
        self.min_angle_input = QDoubleSpinBox()
        self.min_angle_input.setRange(10.0, 30.0)
        self.min_angle_input.setValue(20.0)
        self.min_angle_input.setSingleStep(1.0)
        quality_layout.addRow("Min Angle:", self.min_angle_input)

        # Uniform triangulation
        self.uniform_checkbox = QCheckBox()
        self.uniform_checkbox.setChecked(True) # Default to uniform for consistent sizing
        quality_layout.addRow("Uniform:", self.uniform_checkbox)

        tri_layout.addWidget(quality_group)

        # Feature points controls (remain the same)
        # ... (feature_group setup remains the same) ...
        feature_group = QGroupBox("Feature Points") # Added for completeness
        feature_layout = QFormLayout(feature_group) # Added for completeness
        self.use_feature_points_checkbox = QCheckBox() # Added for completeness
        feature_layout.addRow("Use Features:", self.use_feature_points_checkbox) # Added for completeness
        self.num_features_input = QSpinBox() # Added for completeness
        self.num_features_input.setRange(1, 10) # Added for completeness
        self.num_features_input.setValue(3) # Added for completeness
        feature_layout.addRow("Count:", self.num_features_input) # Added for completeness
        self.feature_size_input = QDoubleSpinBox() # Added for completeness
        self.feature_size_input.setRange(0.1, 3.0) # Added for completeness
        self.feature_size_input.setValue(1.0) # Added for completeness
        self.feature_size_input.setSingleStep(0.1) # Added for completeness
        feature_layout.addRow("Size:", self.feature_size_input) # Added for completeness
        tri_layout.addWidget(feature_group) # Added for completeness


        # 3D visualization settings (remain the same)
        # ... (viz3d_group setup remains the same) ...
        viz3d_group = QGroupBox("3D Settings") # Added for completeness
        viz3d_layout = QFormLayout(viz3d_group) # Added for completeness
        self.height_factor_slider = QSlider(Qt.Horizontal) # Added for completeness
        self.height_factor_slider.setMinimum(0) # Added for completeness
        self.height_factor_slider.setMaximum(100) # Added for completeness
        self.height_factor_slider.setValue(20)  # Added for completeness
        viz3d_layout.addRow("Height Scale:", self.height_factor_slider) # Added for completeness
        tri_layout.addWidget(viz3d_group) # Added for completeness

        # Run triangulation button
        run_btn = QPushButton("Run Triangulation (All Datasets)") # Update button text
        run_btn.setObjectName("run_btn") # Set the object name
        run_btn.setToolTip("Run triangulation for all datasets with computed segments") # Update tooltip
        run_btn.clicked.connect(self.run_all_triangulations) # Connect to the new batch method
        tri_layout.addWidget(run_btn)

        control_layout.addWidget(tri_group)

        # -- Statistics -- (remain the same)
        # ... (stats_group setup remains the same) ...
        stats_group = QGroupBox("Triangulation Statistics") # Added for completeness
        stats_layout = QVBoxLayout(stats_group) # Added for completeness
        self.num_triangles_label = QLabel("Triangles: 0") # Added for completeness
        stats_layout.addWidget(self.num_triangles_label) # Added for completeness
        self.num_vertices_label = QLabel("Vertices: 0") # Added for completeness
        stats_layout.addWidget(self.num_vertices_label) # Added for completeness
        self.mean_edge_label = QLabel("Mean edge: 0.0") # Added for completeness
        stats_layout.addWidget(self.mean_edge_label) # Added for completeness
        self.uniformity_label = QLabel("Uniformity: 0.0") # Added for completeness
        stats_layout.addWidget(self.uniformity_label) # Added for completeness
        control_layout.addWidget(stats_group) # Added for completeness


        # Export button (remain the same)
        # ... (export_btn setup remains the same) ...
        export_btn = QPushButton("Export Results...") # Added for completeness
        export_btn.clicked.connect(self.export_results) # Added for completeness
        control_layout.addWidget(export_btn) # Added for completeness


        # Navigation buttons (remain the same)
        # ... (nav_layout setup remains the same) ...
        nav_layout = QHBoxLayout() # Added for completeness
        prev_btn = QPushButton("← Previous") # Added for completeness
        prev_btn.clicked.connect(lambda: self.notebook.setCurrentIndex(2)) # Added for completeness
        nav_layout.addWidget(prev_btn) # Added for completeness
        control_layout.addLayout(nav_layout) # Added for completeness
        control_layout.addStretch() # Added for completeness

        tab_layout.addWidget(control_panel)

        # --- Visualization Area (right side) --- (remain the same)
        # ... (viz_group setup remains the same) ...
        viz_group = QGroupBox("Triangulation Visualization") # Added for completeness
        viz_layout = QVBoxLayout(viz_group) # Added for completeness
        self.tri_viz_frame = QWidget() # Added for completeness
        self.tri_viz_layout = QVBoxLayout(self.tri_viz_frame) # Added for completeness
        if HAVE_PYVISTA: # Added for completeness
            placeholder_text = "Run triangulation to visualize in 3D" # Added for completeness
        else: # Added for completeness
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nRun triangulation to visualize in 2D." # Added for completeness
        self.tri_viz_placeholder = QLabel(placeholder_text) # Added for completeness
        self.tri_viz_placeholder.setAlignment(Qt.AlignCenter) # Added for completeness
        self.tri_viz_layout.addWidget(self.tri_viz_placeholder) # Added for completeness
        viz_layout.addWidget(self.tri_viz_frame) # Added for completeness
        tab_layout.addWidget(viz_group, 1)  # 1 = stretch factor # Added for completeness

    def _setup_intersection_tab(self):
        """Create the intersection tab and its components with embedded PyVista view."""
        self.intersection_tab = QWidget()
        self.notebook.addTab(self.intersection_tab, "5. Intersections")

        # Set up the layout
        layout = QVBoxLayout(self.intersection_tab)
        
        # Add information label
        info_label = QLabel("Compute intersections between surfaces and polylines:")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 12px; color: #555;")
        layout.addWidget(info_label)
        
        # Horizontal layout for controls
        controls_layout = QHBoxLayout()
        
        # Create compute buttons
        self.compute_intersections_btn = QPushButton("Compute Intersections")
        self.compute_intersections_btn.setObjectName("compute_intersections_btn")
        self.compute_intersections_btn.setIcon(QIcon.fromTheme("system-run", QIcon()))
        self.compute_intersections_btn.clicked.connect(self.compute_intersections)
        controls_layout.addWidget(self.compute_intersections_btn)
        
        self.compute_all_intersections_btn = QPushButton("Compute All")
        self.compute_all_intersections_btn.setIcon(QIcon.fromTheme("system-run", QIcon()))
        self.compute_all_intersections_btn.clicked.connect(self.compute_all_intersections)
        controls_layout.addWidget(self.compute_all_intersections_btn)
        
        self.clear_intersections_btn = QPushButton("Clear Results")
        self.clear_intersections_btn.setIcon(QIcon.fromTheme("edit-clear", QIcon()))
        self.clear_intersections_btn.clicked.connect(self._clear_intersection_results)
        controls_layout.addWidget(self.clear_intersections_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Create a splitter for the main interface
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)  # Add the splitter with stretch
        
        # Left panel for intersection list and details
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Statistics group
        stats_group = QGroupBox("Intersection Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.surface_intersection_count_label = QLabel("Surface-Surface: 0")
        stats_layout.addWidget(self.surface_intersection_count_label)
        
        self.polyline_intersection_count_label = QLabel("Polyline-Surface: 0")
        stats_layout.addWidget(self.polyline_intersection_count_label)
        
        self.triple_point_count_label = QLabel("Triple Points: 0")
        stats_layout.addWidget(self.triple_point_count_label)
        
        left_layout.addWidget(stats_group)
        
        # List of intersections
        intersection_group = QGroupBox("Intersections")
        intersection_layout = QVBoxLayout(intersection_group)
        
        self.intersection_list = QListWidget()
        self.intersection_list.setSelectionMode(QAbstractItemView.SingleSelection)
        # Defer connecting selection change until plotter is ready
        # self.intersection_list.itemSelectionChanged.connect(self._on_intersection_selection_changed)
        intersection_layout.addWidget(self.intersection_list)
        
        left_layout.addWidget(intersection_group, 1)  # Add stretch
        
        # Right panel for visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Embed PyVista Plotter --- 
        self.intersection_view_frame = QFrame()
        self.intersection_view_frame.setFrameShape(QFrame.StyledPanel)
        self.intersection_view_frame.setMinimumSize(400, 300)
        self.intersection_plot_layout = QVBoxLayout(self.intersection_view_frame)
        self.intersection_plot_layout.setContentsMargins(0, 0, 0, 0)
        
        if HAVE_PYVISTA:
            from pyvistaqt import QtInteractor
            self.intersection_plotter = QtInteractor(self.intersection_view_frame)
            self.intersection_plot_layout.addWidget(self.intersection_plotter.interactor)
            self.intersection_plotter.set_background('white')
            # Connect selection change now that plotter exists
            self.intersection_list.itemSelectionChanged.connect(self._on_intersection_selection_changed)
        else:
            # Placeholder if PyVista is not available
            placeholder = QLabel("PyVista is required for 3D intersection visualization.")
            placeholder.setAlignment(Qt.AlignCenter)
            self.intersection_plot_layout.addWidget(placeholder)
            self.intersection_plotter = None # Ensure plotter is None
        
        right_layout.addWidget(self.intersection_view_frame, 1)  # Add frame with stretch
        # --- End PyVista Embedding ---
        
        # Remove the separate 3D view button
        # self.intersection_3d_view_btn = QPushButton("Show 3D View")
        # self.intersection_3d_view_btn.clicked.connect(self.show_intersections_3d_view)
        # right_layout.addWidget(self.intersection_3d_view_btn)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 550])  # Adjust initial sizes
        
        # Store references to intersection data (remains the same)
        if not hasattr(self, 'datasets_intersections'):
            self.datasets_intersections = {}
        
        if not hasattr(self, 'triple_points'):
            self.triple_points = []
    def _setup_refine_mesh_tab(self):
        """Sets up the Refine & Mesh Settings tab."""
        tab_layout = QHBoxLayout(self.refine_mesh_tab)

        # --- Control panel (left side) ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(350)  # Limit width
        control_layout = QVBoxLayout(control_panel)

        # -- Refinement Controls --
        refinement_group = QGroupBox("Intersection Refinement")
        refinement_layout = QVBoxLayout(refinement_group)

        self.refine_intersections_btn = QPushButton("Refine Intersection Lines")
        self.refine_intersections_btn.setToolTip(
            "Align intersection line endpoints to the convex hulls of involved surfaces."
        )
        self.refine_intersections_btn.clicked.connect(self._refine_intersection_lines_action)
        refinement_layout.addWidget(self.refine_intersections_btn)
        self.show_original_lines_checkbox = QCheckBox("Show Original Lines")
        self.show_original_lines_checkbox.setChecked(True)
        self.show_original_lines_checkbox.toggled.connect(self._update_refined_visualization)
        refinement_layout.addWidget(self.show_original_lines_checkbox)
        control_layout.addWidget(refinement_group)

        # -- Global Mesh Settings --
        mesh_settings_group = QGroupBox("Global Mesh Settings")
        mesh_settings_layout = QFormLayout(mesh_settings_group) # Use QFormLayout for label-input pairs

        # Target Feature Size (copied from segment tab's target_feature_size_input)
        self.mesh_target_feature_size_input = QDoubleSpinBox()
        self.mesh_target_feature_size_input.setRange(0.1, 500.0)
        self.mesh_target_feature_size_input.setValue(20.0) # Default
        self.mesh_target_feature_size_input.setSingleStep(0.5)
        self.mesh_target_feature_size_input.setToolTip(
            "Global target size for features/elements in the final mesh."
        )
        mesh_settings_layout.addRow("Target Feature Size:", self.mesh_target_feature_size_input)

        # Gradient (copied from triangulation tab's gradient_input)
        self.mesh_gradient_input = QDoubleSpinBox()
        self.mesh_gradient_input.setRange(1.0, 3.0)
        self.mesh_gradient_input.setValue(2.0) # Default
        self.mesh_gradient_input.setSingleStep(0.1)
        mesh_settings_layout.addRow("Gradient:", self.mesh_gradient_input)

        # Min Angle (copied from triangulation tab's min_angle_input)
        self.mesh_min_angle_input = QDoubleSpinBox()
        self.mesh_min_angle_input.setRange(10.0, 30.0)
        self.mesh_min_angle_input.setValue(20.0) # Default
        self.mesh_min_angle_input.setSingleStep(1.0)
        mesh_settings_layout.addRow("Min Angle:", self.mesh_min_angle_input)

        # Uniform Meshing (copied from triangulation tab's uniform_checkbox)
        self.mesh_uniform_checkbox = QCheckBox()
        self.mesh_uniform_checkbox.setChecked(True) # Default
        mesh_settings_layout.addRow("Uniform Meshing:", self.mesh_uniform_checkbox)

        # Add information labels for constraints
        self.constraints_label = QLabel("<b>Mesh Constraints</b>")
        self.constraints_label.setAlignment(Qt.AlignCenter)
        mesh_settings_layout.addRow("", self.constraints_label)

        # Add a separator line
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Sunken)
        mesh_settings_layout.addRow("", self.separator)

        # Add explanatory text
        self.target_size_info = QLabel("Controls the general element size in the final mesh")
        self.target_size_info.setWordWrap(True)
        mesh_settings_layout.addRow("", self.target_size_info)

        self.gradient_info = QLabel("Controls the sizing transition rate from smaller to larger elements")
        self.gradient_info.setWordWrap(True)
        mesh_settings_layout.addRow("", self.gradient_info)

        self.min_angle_info = QLabel("Ensures triangle quality by setting minimum internal angles")
        self.min_angle_info.setWordWrap(True)
        mesh_settings_layout.addRow("", self.min_angle_info)

        # Visual indicator for constraints impact (simple color code)
        self.constraint_indicator = QLabel("◼ Strict constraints may result in more elements/longer processing")
        self.constraint_indicator.setStyleSheet("color: orange;")
        mesh_settings_layout.addRow("", self.constraint_indicator)

        control_layout.addWidget(mesh_settings_group)
        # Placeholder for future "Generate Mesh" button
        # generate_mesh_btn = QPushButton("Generate Final Mesh (Placeholder)")
        # generate_mesh_btn.setEnabled(False) # Disabled for now
        # control_layout.addWidget(generate_mesh_btn)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        prev_btn = QPushButton("← Previous (Intersections)")
        prev_btn.clicked.connect(lambda: self.notebook.setCurrentIndex(self.notebook.indexOf(self.intersection_tab)))
        nav_layout.addWidget(prev_btn)
        # No "Next" button for the last tab for now
        control_layout.addLayout(nav_layout)
        control_layout.addStretch()
        tab_layout.addWidget(control_panel)

        # --- Visualization Area (right side) ---
        viz_group = QGroupBox("Refined Intersections & Mesh Preview")
        viz_layout = QVBoxLayout(viz_group)

        self.refine_mesh_viz_frame = QFrame() # Use the class attribute
        self.refine_mesh_viz_frame.setFrameShape(QFrame.StyledPanel)
        self.refine_mesh_viz_frame.setMinimumSize(400, 300)
        self.refine_mesh_plot_layout = QVBoxLayout(self.refine_mesh_viz_frame)
        self.refine_mesh_plot_layout.setContentsMargins(0, 0, 0, 0)

        if HAVE_PYVISTA:
            try:
                from pyvistaqt import QtInteractor
                plotter = QtInteractor(self.refine_mesh_viz_frame)
                self.refine_mesh_plot_layout.addWidget(plotter.interactor)
                plotter.set_background([0.318, 0.341, 0.431])
                plotter.add_text("Refine intersections to visualize or load data.", position='upper_edge', color='white')
                # Store in both attribute and dictionary
                self.refine_mesh_plotter = plotter
                self.plotters['refine_mesh'] = plotter
                logger.info("Successfully created refine_mesh_plotter")
            except Exception as e:
                logger.error(f"Error initializing Refine/Mesh plotter: {e}", exc_info=True)
                placeholder = QLabel(f"Error initializing PyVista plotter: {e}")
                placeholder.setAlignment(Qt.AlignCenter)
                placeholder.setWordWrap(True)
                self.refine_mesh_plot_layout.addWidget(placeholder)
                self.refine_mesh_plotter = None
        else:
            placeholder = QLabel("PyVista is required for 3D visualization.")
            placeholder.setAlignment(Qt.AlignCenter)
            self.refine_mesh_plot_layout.addWidget(placeholder)
            self.refine_mesh_plotter = None

        viz_layout.addWidget(self.refine_mesh_viz_frame, 1)
        # Add to _setup_refine_mesh_tab in viz_layout
        self.refinement_summary_label = QLabel("Run refinement to see summary")
        self.refinement_summary_label.setWordWrap(True)
        self.refinement_summary_label.setTextFormat(Qt.RichText)
        viz_layout.addWidget(self.refinement_summary_label)
        tab_layout.addWidget(viz_group, 1)
    # Event handlers - placeholder implementations
    def _setup_pre_tetramesh_tab(self):
            """Sets up the Constrained Surface Meshing tab."""
            # Initialize the main layout for this specific tab
            tab_layout = QHBoxLayout(self.pre_tetramesh_tab)

            # --- Control panel (left side) ---
            control_panel = QWidget()
            control_panel.setMaximumWidth(350)
            control_layout = QVBoxLayout(control_panel) # This is the layout for the control_panel's content
            
            # -- Computation Controls --
            compute_group = QGroupBox("Constrained Triangulation")
            compute_layout = QVBoxLayout(compute_group)

            self.compute_constrained_mesh_btn = QPushButton("Generate Constrained Surface Meshes")
            self.compute_constrained_mesh_btn.setToolTip(
                "Re-triangulate surfaces using refined hulls and intersections as constraints."
            )
            self.compute_constrained_mesh_btn.clicked.connect(self._compute_constrained_meshes_action)
            compute_layout.addWidget(self.compute_constrained_mesh_btn)
            
            # Add tetgen validation button
            self.validate_for_tetgen_btn = QPushButton("Validate Surfaces for Tetgen")
            self.validate_for_tetgen_btn.setToolTip(
                "Check if constrained surfaces are ready for tetgen tetrahedralization.\n"
                "Validates mesh quality, topology, and constraint processing."
            )
            self.validate_for_tetgen_btn.clicked.connect(self._validate_surfaces_for_tetgen)
            self.validate_for_tetgen_btn.setEnabled(False)
            self.validate_for_tetgen_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    padding: 8px;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """)
            compute_layout.addWidget(self.validate_for_tetgen_btn)
            
            self.constrained_mesh_target_feature_size_input = QDoubleSpinBox()
            self.constrained_mesh_target_feature_size_input.setRange(0.1, 500.0)
            self.constrained_mesh_target_feature_size_input.setValue(20.0)
            self.constrained_mesh_target_feature_size_input.setToolTip("Target feature size for constrained triangulation.")
            
            self.constrained_mesh_min_angle_input = QDoubleSpinBox()
            self.constrained_mesh_min_angle_input.setRange(10.0, 30.0)
            self.constrained_mesh_min_angle_input.setValue(20.0)
            
            form_layout = QFormLayout()
            form_layout.addRow("Target Feature Size:", self.constrained_mesh_target_feature_size_input)
            form_layout.addRow("Min Angle:", self.constrained_mesh_min_angle_input)
            compute_layout.addLayout(form_layout)

            # -- Constraint Processing Controls --
            constraint_group = QGroupBox("Constraint Processing (C++ MeshIt Logic)")
            constraint_layout = QVBoxLayout(constraint_group)
            
            # Enable constraint processing checkbox
            self.use_constraint_processing_checkbox = QCheckBox("Enable Constraint Processing")
            self.use_constraint_processing_checkbox.setChecked(True)
            self.use_constraint_processing_checkbox.setToolTip(
                "Enable advanced constraint processing that replicates C++ MeshIt logic"
            )
            constraint_layout.addWidget(self.use_constraint_processing_checkbox)
            
            # Type-based sizing checkbox
            self.type_based_sizing_checkbox = QCheckBox("Type-Based Sizing")
            self.type_based_sizing_checkbox.setChecked(True)
            self.type_based_sizing_checkbox.setToolTip(
                "Assign different mesh sizes based on point types (TRIPLE_POINT, CORNER, etc.)"
            )
            constraint_layout.addWidget(self.type_based_sizing_checkbox)
            
            # Hierarchical constraints checkbox
            self.hierarchical_constraints_checkbox = QCheckBox("Hierarchical Constraints")
            self.hierarchical_constraints_checkbox.setChecked(True)
            self.hierarchical_constraints_checkbox.setToolTip(
                "Process constraints hierarchically as in C++ MeshIt"
            )
            constraint_layout.addWidget(self.hierarchical_constraints_checkbox)
            
            # Gradient control
            self.constraint_gradient_input = QDoubleSpinBox()
            self.constraint_gradient_input.setRange(1.0, 5.0)
            self.constraint_gradient_input.setValue(2.0)
            self.constraint_gradient_input.setSingleStep(0.1)
            self.constraint_gradient_input.setToolTip("Gradient control for smooth size transitions")
            
            constraint_form_layout = QFormLayout()
            constraint_form_layout.addRow("Gradient:", self.constraint_gradient_input)
            constraint_layout.addLayout(constraint_form_layout)

            control_layout.addWidget(compute_group) # Add compute_group to control_panel's layout
            control_layout.addWidget(constraint_group) # Add constraint_group to control_panel's layout
            control_layout.addStretch()
            
            # Add the control_panel to the main tab_layout
            tab_layout.addWidget(control_panel)

            # --- Visualization Area (right side) ---
            viz_group = QGroupBox("Constrained Surface Mesh Visualization")
            viz_layout = QVBoxLayout(viz_group) # This is the layout for the viz_group's content

            self.pre_tetramesh_viz_frame = QFrame()
            self.pre_tetramesh_viz_frame.setFrameShape(QFrame.StyledPanel)
            self.pre_tetramesh_plot_layout = QVBoxLayout(self.pre_tetramesh_viz_frame)
            self.pre_tetramesh_plot_layout.setContentsMargins(0, 0, 0, 0)

            if HAVE_PYVISTA:
                from pyvistaqt import QtInteractor
                plotter = QtInteractor(self.pre_tetramesh_viz_frame)
                self.pre_tetramesh_plot_layout.addWidget(plotter.interactor)
                plotter.set_background([0.2, 0.2, 0.25]) # Darker background
                plotter.add_text("Compute constrained meshes to visualize.", position='upper_edge', color='white')
                self.plotters['pre_tetramesh'] = plotter 
                self.pre_tetramesh_plotter = plotter
                logger.info("Successfully created pre_tetramesh_plotter")
            else:
                placeholder = QLabel("PyVista is required for 3D visualization.")
                placeholder.setAlignment(Qt.AlignCenter)
                self.pre_tetramesh_plot_layout.addWidget(placeholder)
                self.pre_tetramesh_plotter = None

            viz_layout.addWidget(self.pre_tetramesh_viz_frame, 1) # Add the frame to viz_group's layout
            
            # Add the viz_group to the main tab_layout
            tab_layout.addWidget(viz_group, 1)

    # Add a new clear method for this tab's plotter
    def _clear_pre_tetramesh_plot(self):
        plotter = self.plotters.get('pre_tetramesh')
        if plotter:
            plotter.clear()
            plotter.add_text("Compute constrained meshes to visualize.", position='upper_edge', color='white')
            plotter.reset_camera()
        elif hasattr(self, 'pre_tetramesh_plot_layout'): # Fallback
            # ... (similar clearing logic as _clear_refine_mesh_plot if plotter is None)
            pass
            
    # Update _on_tab_changed
   

    # Action method for the button
    def _compute_constrained_meshes_action(self):
        """Compute constrained meshes for surfaces using stored constraints from refinement step (C++ approach)"""
        if not hasattr(self, 'datasets') or not self.datasets:
            self.statusBar().showMessage("Constrained meshing skipped: No datasets.")
            return

                # Check if we have stored constraints from refinement step
        constraint_found = False
        for dataset in self.datasets:
            if 'stored_constraints' in dataset and dataset['stored_constraints']:
                constraint_found = True
                break

        if not constraint_found:
            logger.warning("No stored constraint lines found. Please run 'Refine Intersection Lines' first.")
            self.statusBar().showMessage("Please run 'Refine Intersection Lines' first to generate constraints.")
            return

        # Convert stored constraints to expected format
        self.stored_constraint_lines = {}
        for dataset_idx, dataset in enumerate(self.datasets):
            stored_constraints = dataset.get('stored_constraints', [])
            if not stored_constraints:
                continue
                
            constraint_lines = []
            
            # Add hull boundary as first constraint line
            hull_points = dataset.get('hull_points', [])
            if hull_points is not None and hasattr(hull_points, '__len__') and len(hull_points) >= 3:
                hull_coords = []
                for hp in hull_points:
                    if len(hp) >= 3:
                        hull_coords.append([float(hp[0]), float(hp[1]), float(hp[2])])
                if hull_coords:
                    constraint_lines.append(hull_coords)
            
            # Add intersection lines as additional constraint lines
            for constraint in stored_constraints:
                if constraint['type'] == 'intersection_line':
                    constraint_lines.append(constraint['points'])
            
            if constraint_lines:
                self.stored_constraint_lines[dataset_idx] = constraint_lines
                logger.info(f"Converted {len(constraint_lines)} constraint lines for dataset {dataset_idx}")

                # Get triangulation parameters from UI (with fallbacks if controls don't exist)
        try:
            target_feature_size = self.constrained_mesh_target_feature_size_input.value()
        except AttributeError:
            target_feature_size = 20.0  # Default target feature size
            logger.warning("Using default target feature size (20.0) - UI control not found")
        
        try:
            min_angle_deg = self.constrained_mesh_min_angle_input.value()
        except AttributeError:
            min_angle_deg = 20.0  # Default min angle
            logger.warning("Using default min angle (20°) - UI control not found")
        
        # Calculate reasonable max area based on target feature size
        # Use a much larger area to prevent over-refinement
        max_area = (target_feature_size ** 2) * 2.0  # Larger area to reduce over-refinement

        logger.info("Starting constrained surface meshing using stored constraints (C++ approach)...")
        
        success_count = 0
        total_count = 0
        
        # Clear previous results
        if not hasattr(self, 'constrained_meshes'):
            self.constrained_meshes = {}
        
        # Process each dataset using stored constraints
        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_name = dataset.get('name', f'Dataset_{dataset_idx}')
            
            # Skip if no stored constraints for this dataset
            if dataset_idx not in self.stored_constraint_lines:
                logger.warning(f"No stored constraints for {dataset_name}")
                continue
                
            total_count += 1
            
            try:
                logger.info(f"Processing C++ CONSTRAINED triangulation for: {dataset_name}")
                
                # Get stored constraint lines for this dataset (C++ approach)
                constraint_lines = self.stored_constraint_lines[dataset_idx]
                
                # 1. Build simplified constraint approach - just use hull + intersection lines
                all_points_3d = []
                all_segments = []
                
                # A. First, add ALL original triangulation vertices to preserve internal structure
                original_triangulation = self.datasets[dataset_idx].get('triangulation_result', {})
                original_vertices = original_triangulation.get('vertices', [])
                
                # Also check alternative storage locations if not found
                if original_vertices is None or len(original_vertices) == 0:
                    # Try legacy triangulation storage
                    legacy_triangulation = self.datasets[dataset_idx].get('triangulation', {})
                    original_vertices = legacy_triangulation.get('vertices', [])
                    
                    # Try to get vertices from the triangles data structure
                    if (original_vertices is None or len(original_vertices) == 0) and 'triangles' in self.datasets[dataset_idx]:
                        triangles_data = self.datasets[dataset_idx]['triangles']
                        if isinstance(triangles_data, dict) and 'vertices' in triangles_data:
                            original_vertices = triangles_data['vertices']
                        elif hasattr(triangles_data, 'vertices'):
                            original_vertices = triangles_data.vertices
                    
                    # Try alternative storage locations
                    if (original_vertices is None or len(original_vertices) == 0) and 'mesh' in self.datasets[dataset_idx]:
                        mesh_data = self.datasets[dataset_idx]['mesh']
                        if isinstance(mesh_data, dict) and 'vertices' in mesh_data:
                            original_vertices = mesh_data['vertices']
                
                # C++ approach: Only use constraint boundaries, let Triangle generate internal points
                # The C++ version relies on Triangle's internal point generation with proper area constraints
                logger.info(f"Using C++ constraint-only approach for {dataset_name} - Triangle will generate internal points")
                
                # B. Add hull boundary
                hull_line = constraint_lines[0]  # First line is always hull
                logger.info(f"Adding hull boundary: {len(hull_line)} points")
                
                hull_indices = []
                for pt in hull_line:
                    xyz = pt[:3]            # ignore the type string
                    all_points_3d.append(xyz)
                    hull_indices.append(len(all_points_3d) - 1)
                
                # Create hull segments (closed loop)
                for j in range(len(hull_indices)):
                    next_j = (j + 1) % len(hull_indices)
                    all_segments.append([hull_indices[j], hull_indices[next_j]])
                
                # C. Add intersection lines as constraints (with basic duplicate checking)
                for line_idx in range(1, len(constraint_lines)):
                    intersection_line = constraint_lines[line_idx]
                    logger.info(f"Adding intersection constraint {line_idx}: {len(intersection_line)} points")
                    
                    line_indices = []
                    for pt in intersection_line:
                        xyz = pt[:3]                        # numeric part only

                        existing_idx = None
                        for idx, existing_pt in enumerate(all_points_3d):
                            if np.linalg.norm(np.array(xyz, dtype=float) -
                                            np.array(existing_pt, dtype=float)) < 1e-6:
                                existing_idx = idx
                                break

                        if existing_idx is None:
                            all_points_3d.append(xyz)
                            line_indices.append(len(all_points_3d) - 1)
                        else:
                            line_indices.append(existing_idx)
                    
                    # Create intersection segments (open line)
                    for j in range(len(line_indices) - 1):
                        all_segments.append([line_indices[j], line_indices[j + 1]])
                
                # D. CRITICAL: Add triple points as individual point constraints
                # This is what was missing - triple points need to be added as single-point constraints
                triple_points_added = 0
                if hasattr(self, 'triple_points') and self.triple_points:
                    logger.info(f"Adding {len(self.triple_points)} triple points as point constraints")
                    for tp in self.triple_points:
                        try:
                            tp_coords = tp.get('point')
                            if tp_coords:
                                if hasattr(tp_coords, '__len__') and len(tp_coords) >= 3:
                                    triple_point = [float(tp_coords[0]), float(tp_coords[1]), float(tp_coords[2])]
                                elif hasattr(tp_coords, 'x') and hasattr(tp_coords, 'y') and hasattr(tp_coords, 'z'):
                                    triple_point = [float(tp_coords.x), float(tp_coords.y), float(tp_coords.z)]
                                else:
                                    continue
                                
                                # Check if this triple point is already in our points (avoid duplicates)
                                existing_idx = None
                                for existing_pt_idx, existing_pt in enumerate(all_points_3d):
                                    if np.linalg.norm(np.array(triple_point) - np.array(existing_pt)) < 1e-6:
                                        existing_idx = existing_pt_idx
                                        break
                                
                                if existing_idx is None:
                                    all_points_3d.append(triple_point)
                                    triple_points_added += 1
                                    logger.debug(f"Added triple point: {triple_point}")
                                else:
                                    logger.debug(f"Triple point already exists at index {existing_idx}")
                                    
                        except Exception as e:
                            logger.warning(f"Error processing triple point: {e}")
                    
                    logger.info(f"Added {triple_points_added} new triple points as point constraints")
                else:
                    logger.warning("No triple points found - missing critical junction points for triangulation!")

                logger.info(f"C++ APPROACH: Using {len(constraint_lines)} constraint lines "
                        f"(1 hull + {len(constraint_lines)-1} intersections) + {triple_points_added} triple points for {dataset_name}")
                logger.info(f"TOTAL POINTS: {len(all_points_3d)} constraint points ({len(all_points_3d) - triple_points_added} boundary + {triple_points_added} triple points)")
                logger.info(f"C++ CONSTRAINT PROCESSING: {len(all_points_3d)} total points, {len(all_segments)} segments")
                
                # Safety checks
                if len(all_segments) > 200:
                    logger.warning(f"Too many constraint segments ({len(all_segments)}) for {dataset_name}. May cause hanging.")
                    continue
                
                if len(all_points_3d) > 500:
                    logger.warning(f"Too many constraint points ({len(all_points_3d)}) for {dataset_name}. Using simplified approach.")
                    # Keep only constraint boundaries, not all original vertices
                    all_points_3d = []
                    all_segments = []
                    
                    # Re-add just the constraint boundaries
                    for line_idx, constraint_line in enumerate(constraint_lines):
                        line_indices = []
                        for pt in constraint_line:
                            all_points_3d.append(pt)
                            line_indices.append(len(all_points_3d) - 1)
                        
                        if line_idx == 0:  # Hull - create closed loop
                            for j in range(len(line_indices)):
                                next_j = (j + 1) % len(line_indices)
                                all_segments.append([line_indices[j], line_indices[next_j]])
                        else:  # Intersection - create open line
                            for j in range(len(line_indices) - 1):
                                all_segments.append([line_indices[j], line_indices[j + 1]])
                    
                    logger.info(f"SIMPLIFIED: Using {len(all_points_3d)} constraint boundary points, {len(all_segments)} segments")
                
                # 2. Project to 2D with C++ duplicate removal approach  
                projection_params = dataset.get('projection_params')
                if not projection_params:
                    logger.warning(f"No projection parameters for {dataset_name}. Skipping.")
                    continue

                centroid = np.array(projection_params['centroid'])
                basis = np.array(projection_params['basis'])

                all_points_3d_np = np.array(all_points_3d)
                centered_points = all_points_3d_np - centroid
                
                # Fix dimension mismatch: ensure basis is 3x2 for projection
                if basis.shape[0] == 2:
                    # If basis is 2x2 or 2x3, we need to transpose it to 3x2 or create proper 3x2
                    if basis.shape[1] == 2:
                        # Create a 3x2 basis matrix
                        basis_3x2 = np.zeros((3, 2))
                        basis_3x2[:2, :] = basis
                        basis_3x2[2, :] = [0, 0]  # Z maps to 0 in both 2D axes
                    else:  # basis.shape[1] == 3, so transpose
                        basis_3x2 = basis.T[:, :2]  # Take first 2 columns after transpose
                elif basis.shape[0] >= 3:
                    basis_3x2 = basis[:, :2]  # Take first 2 columns
                else:
                    logger.error(f"Invalid basis shape: {basis.shape}")
                    continue
                
                points_2d = centered_points @ basis_3x2
                
                # NO DEDUPLICATION - Keep ALL points to preserve triple points and geometry
                deduplicated_points_2d = points_2d  # Keep all 2D points
                deduplicated_segments = all_segments  # Keep all segments
                
                logger.info(f"NO DEDUPLICATION: Preserving ALL {len(points_2d)} points and {len(all_segments)} segments")
                logger.info(f"TRIPLE POINT PRESERVATION: All points including {triple_points_added} triple points are preserved")
                
                if len(deduplicated_segments) < 3 or len(deduplicated_points_2d) < 3:
                    logger.warning(f"Too few points/segments after deduplication for {dataset_name}")
                    continue

                # Apply scaling to avoid precision issues (like in logs)
                points_2d_array = np.array(deduplicated_points_2d)
                if points_2d_array.size > 0:
                    scale_factor = 50.0 / max(np.ptp(points_2d_array[:, 0]), np.ptp(points_2d_array[:, 1]), 1e-10)
                    points_2d_array *= scale_factor
                    logger.info(f"SCALED 2D coordinates by factor {scale_factor:.6f} to avoid precision issues")
                    logger.info(f"2D coordinate range: [{points_2d_array.min():.3f}, {points_2d_array.max():.3f}]")

                logger.info(f"Projecting {len(deduplicated_points_2d)} constraint points to 2D for {dataset_name}")
                logger.info(f"Running C++ triangulation with {len(deduplicated_segments)} constraint segments for {dataset_name}")

                # 3. NEW: Use the constraint processing approach instead of manual constraint building
                import math
                try:
                    from meshit.intersection_utils import prepare_plc_for_surface_triangulation, run_constrained_triangulation_py
                    from meshit.intersection_utils import Vector3D
                    
                    # Convert constraint lines to proper format for new constraint processing
                    surface_data = {
                        'hull_points': [],
                        'size': target_feature_size,
                        'projection_params': projection_params
                    }
                    
                    intersections_on_surface_data = []
                    
                    # Convert constraint lines to Vector3D format
                    for line_idx, constraint_line in enumerate(constraint_lines):
                        if line_idx == 0:  # Hull boundary
                            hull_points_vector3d = []
                            for pt in constraint_line:
                                v3d = Vector3D(pt[0], pt[1], pt[2])
                                v3d.type = "DEFAULT"  # Hull points are DEFAULT type
                                hull_points_vector3d.append(v3d)
                            surface_data['hull_points'] = hull_points_vector3d
                        else:  # Intersection lines
                            intersection_points_vector3d = []
                            for pt in constraint_line:
                                v3d = Vector3D(pt[0], pt[1], pt[2])
                                v3d.type = "INTERSECTION_POINT"  # Intersection points
                                intersection_points_vector3d.append(v3d)
                            
                            intersections_on_surface_data.append({
                                'points': intersection_points_vector3d,
                                'size': target_feature_size * 0.7,  # Smaller size for intersections
                                'type': 'INTERSECTION'
                            })
                    
                    logger.info(f"NEW CONSTRAINT PROCESSING: Using {len(surface_data['hull_points'])} hull points, {len(intersections_on_surface_data)} intersection lines")
                    
                    # Prepare triple points for protection
                    triple_points_vector3d = []
                    if hasattr(self, 'triple_points') and self.triple_points:
                        for tp in self.triple_points:
                            try:
                                tp_coords = tp.get('point')
                                if tp_coords:
                                    if hasattr(tp_coords, '__len__') and len(tp_coords) >= 3:
                                        v3d = Vector3D(float(tp_coords[0]), float(tp_coords[1]), float(tp_coords[2]))
                                        v3d.type = "TRIPLE_POINT"
                                        triple_points_vector3d.append(v3d)
                                    elif hasattr(tp_coords, 'x') and hasattr(tp_coords, 'y') and hasattr(tp_coords, 'z'):
                                        v3d = Vector3D(float(tp_coords.x), float(tp_coords.y), float(tp_coords.z))
                                        v3d.type = "TRIPLE_POINT"
                                        triple_points_vector3d.append(v3d)
                            except Exception as e:
                                logger.warning(f"Error converting triple point to Vector3D: {e}")
                    
                    logger.info(f"PROTECTED TRIPLE POINTS: {len(triple_points_vector3d)} triple points prepared for protection")
                    
                    # Use the new constraint processing approach
                    config = {
                        'target_feature_size': target_feature_size,
                        'max_area': max_area,
                        'min_angle': min_angle_deg,
                        'gradient': 2.0,
                        'uniform_meshing': True,
                        'use_constraint_processing': True,  # FORCE enable constraint processing
                        'type_based_sizing': self.type_based_sizing_checkbox.isChecked(),
                        'hierarchical_constraints': self.hierarchical_constraints_checkbox.isChecked(),
                        'preserve_constraints': True,
                        'constraint_enforcement': 'strict',
                        'triple_points': triple_points_vector3d,  # PROTECTED TRIPLE POINTS
                        'use_cpp_exact_logic': True
                    }
                    
                    # Use the new PLC preparation method
                    plc_result = prepare_plc_for_surface_triangulation(surface_data, intersections_on_surface_data, config)
                    
                    if plc_result and len(plc_result) == 4:
                        points_2d_new, segments_new, holes_2d_new, points_3d_new = plc_result
                        
                        if points_2d_new is not None and len(points_2d_new) > 0:
                            logger.info(f"NEW CONSTRAINT PROCESSING SUCCESS: {len(points_2d_new)} points, {len(segments_new)} segments")
                            
                            # Use the new constraint processing result
                            result = run_constrained_triangulation_py(
                                points_2d_new,
                                segments_new,
                                holes_2d_new,
                                projection_params,
                                points_3d_new,
                                config
                            )
                        else:
                            logger.warning("New constraint processing returned empty result, falling back to manual approach")
                            # Fall back to manual approach
                            edge_length = math.sqrt(max_area * 4 / math.sqrt(3))
                            deduplicated_points_3d = []
                            for i in range(len(deduplicated_points_2d)):
                                for orig_idx, mapped_idx in point_index_map.items():
                                    if mapped_idx == i:
                                        deduplicated_points_3d.append(all_points_3d[orig_idx])
                                        break
                            
                            result = run_constrained_triangulation_py(
                                points_2d_array,
                                np.array(deduplicated_segments),
                                [],  # holes
                                projection_params,
                                np.array(deduplicated_points_3d),
                                config
                            )
                    else:
                        logger.warning("New constraint processing failed, falling back to manual approach")
                        # Fall back to manual approach
                        edge_length = math.sqrt(max_area * 4 / math.sqrt(3))
                        deduplicated_points_3d = []
                        for i in range(len(deduplicated_points_2d)):
                            for orig_idx, mapped_idx in point_index_map.items():
                                if mapped_idx == i:
                                    deduplicated_points_3d.append(all_points_3d[orig_idx])
                                    break
                        
                        result = run_constrained_triangulation_py(
                            points_2d_array,
                            np.array(deduplicated_segments),
                            [],  # holes
                            projection_params,
                            np.array(deduplicated_points_3d),
                            config
                        )
                    
                    if result is not None and len(result) == 3:
                        vertices_3d, triangles, triple_point_indices = result
                        
                        # Additional safety checks
                        if (vertices_3d is not None and triangles is not None and 
                            len(vertices_3d) > 0 and len(triangles) > 0):
                            
                            # Store result in constrained_meshes
                            self.constrained_meshes[dataset_idx] = {
                                'vertices': vertices_3d,
                                'triangles': triangles,
                                'constraint_lines': constraint_lines,
                                'num_constraints': len(constraint_lines),
                                'num_segments': len(deduplicated_segments)
                            }
                            
                            # Also store in dataset for validation
                            self.datasets[dataset_idx]['constrained_vertices'] = vertices_3d
                            self.datasets[dataset_idx]['constrained_triangles'] = triangles
                            self.datasets[dataset_idx]['constrained_triple_point_indices'] = triple_point_indices  # CRITICAL FIX
                            self.datasets[dataset_idx]['constraint_processing_used'] = True
                            self.datasets[dataset_idx]['intersection_constraints'] = constraint_lines[1:]  # Exclude hull boundary
                            
                            logger.info(f"SUCCESS: C++ CONSTRAINED mesh for {dataset_name}: "
                                    f"{len(vertices_3d)} vertices, {len(triangles)} triangles "
                                    f"using {len(constraint_lines)} constraint lines, {len(deduplicated_segments)} segments")
                            success_count += 1
                    else:
                        logger.error(f"Failed to generate constrained mesh for {dataset_name}")
                        
                except Exception as e:
                    logger.error(f"Error during constrained triangulation for {dataset_name}: {str(e)}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing constrained mesh for {dataset_name}: {str(e)}")
                continue

        # Update visualization
        self._visualize_constrained_meshes()
        
        if success_count > 0:
            self.statusBar().showMessage(f"Generated {success_count}/{total_count} constrained meshes using C++ approach.")
            logger.info(f"Constrained meshing completed: {success_count}/{total_count} surfaces processed successfully.")
            
            # Enable the tetgen validation button after successful computation
            if hasattr(self, 'validate_for_tetgen_btn'):
                self.validate_for_tetgen_btn.setEnabled(True)
                logger.info("Tetgen validation button enabled")
        else:
            self.statusBar().showMessage("No constrained meshes generated. Check constraints and parameters.")

    def _visualize_constrained_meshes(self):
        """Visualize constrained meshes with highlighted constraint boundaries"""
        if not hasattr(self, 'pre_tetramesh_plotter') or not self.pre_tetramesh_plotter:
            return
            
        if not hasattr(self, 'constrained_meshes') or not self.constrained_meshes:
            logger.info("No constrained meshes to visualize")
            return

        plotter = self.pre_tetramesh_plotter
        plotter.clear()
        
        # Create visualization for each constrained mesh
        for dataset_idx, mesh_data in self.constrained_meshes.items():
            if dataset_idx >= len(self.datasets):
                continue
                
            dataset = self.datasets[dataset_idx]
            dataset_name = dataset.get('name', f'Dataset_{dataset_idx}')
            dataset_color = dataset.get('color', [0.7, 0.7, 0.7])
            
            vertices = mesh_data['vertices']
            triangles = mesh_data['triangles']
            constraint_lines = mesh_data.get('constraint_lines', [])
            
            if len(vertices) == 0 or len(triangles) == 0:
                continue
                
            # Convert to numpy arrays
            vertices_np = np.array(vertices)
            triangles_np = np.array(triangles)
            
            # Create PyVista mesh
            import pyvista as pv
            faces = []
            for tri in triangles_np:
                faces.extend([3, tri[0], tri[1], tri[2]])
            
            mesh = pv.PolyData(vertices_np, faces)
            
            # Add main mesh
            plotter.add_mesh(
                mesh,
                color=dataset_color,
                style='surface',
                opacity=0.8,
                name=f'constrained_mesh_{dataset_idx}',
                show_edges=True,
                edge_color='black',
                line_width=1
            )
            
            # Highlight constraint boundaries
            for line_idx, constraint_line in enumerate(constraint_lines):
                if len(constraint_line) < 2:
                    continue

                # keep only numeric coordinates
                numeric_coords = [[pt[0], pt[1], pt[2]] for pt in constraint_line]

                constraint_points = np.array(numeric_coords, dtype=float)
                
                if line_idx == 0:  # Hull boundary
                    # Close the loop for hull
                    constraint_points_closed = np.vstack([constraint_points, constraint_points[0:1]])
                    line_color = 'red'
                    line_width = 4
                    label = f'{dataset_name}_hull_boundary'
                else:  # Intersection lines
                    constraint_points_closed = constraint_points
                    line_color = 'blue'
                    line_width = 3
                    label = f'{dataset_name}_intersection_{line_idx}'
                
                # Create line mesh
                line_mesh = pv.PolyData(constraint_points_closed)
                line_mesh.lines = np.hstack([[2, i, i+1] for i in range(len(constraint_points_closed)-1)])
                
                plotter.add_mesh(
                    line_mesh,
                    color=line_color,
                    line_width=line_width,
                    name=label,
                    style='wireframe'
                )
                
                # Add small spheres to show constraint line points (to see if triple points are embedded)
                constraint_point_cloud = pv.PolyData(constraint_points)
                constraint_spheres = constraint_point_cloud.glyph(
                    geom=pv.Sphere(radius=0.5, phi_resolution=6, theta_resolution=6),
                    scale=False
                )
                plotter.add_mesh(
                    constraint_spheres,
                    color='white' if line_idx == 0 else 'lightblue',
                    style='surface',
                    opacity=0.8,
                    name=f'{label}_points',
                    show_edges=False
                )
        
        # Add HIGHLY VISIBLE triple points visualization
        self._add_triple_points_visualization(plotter)
        
        # Add legend
        legend_entries = []
        for dataset_idx in self.constrained_meshes.keys():
            if dataset_idx < len(self.datasets):
                dataset_name = self.datasets[dataset_idx].get('name', f'Dataset_{dataset_idx}')
                mesh_data = self.constrained_meshes[dataset_idx]
                legend_entries.append([f"{dataset_name}: {len(mesh_data['vertices'])} vertices", 
                                    self.datasets[dataset_idx].get('color', [0.7, 0.7, 0.7])])
        
        # Add triple points to legend
        if hasattr(self, 'triple_points') and self.triple_points:
            legend_entries.append([f"Triple Points: {len(self.triple_points)}", 'yellow'])
        
        if legend_entries:
            plotter.add_legend(legend_entries, bcolor='white', face='rectangle')
        
        # Set view
        plotter.show_axes()
        plotter.reset_camera()
        
        logger.info(f"Visualized {len(self.constrained_meshes)} constrained meshes with constraint boundaries and {len(self.triple_points) if hasattr(self, 'triple_points') else 0} triple points")

    def _add_triple_points_visualization(self, plotter):
        """Add highly visible triple points to the pre-tetra mesh visualization"""
        if not hasattr(self, 'triple_points') or not self.triple_points:
            logger.info("No triple points found to visualize")
            return
        
        import pyvista as pv
        
        # DEBUG: Show triple point structure
        logger.info(f"DEBUG: Found {len(self.triple_points)} triple points")
        for i, tp in enumerate(self.triple_points[:3]):  # Show first 3 for debugging
            logger.info(f"  Triple point {i}: type={type(tp)}, value={tp}")
        
        # Extract triple point coordinates with better debugging
        triple_coords = []
        for i, tp in enumerate(self.triple_points):
            coord = None
            
            # Handle dictionary format: {'point': Vector3D(...), 'intersections': [...]}
            if isinstance(tp, dict) and 'point' in tp:
                point = tp['point']
                if hasattr(point, 'x') and hasattr(point, 'y') and hasattr(point, 'z'):
                    # Vector3D object
                    coord = [float(point.x), float(point.y), float(point.z)]
                elif isinstance(point, (list, tuple)) and len(point) >= 3:
                    # List/tuple coordinates
                    coord = [float(point[0]), float(point[1]), float(point[2])]
            # Handle direct object format
            elif hasattr(tp, 'point'):
                # Triple point has a point attribute
                point = tp.point
                if hasattr(point, 'x') and hasattr(point, 'y') and hasattr(point, 'z'):
                    # Vector3D object
                    coord = [float(point.x), float(point.y), float(point.z)]
                elif isinstance(point, (list, tuple)) and len(point) >= 3:
                    # List/tuple coordinates
                    coord = [float(point[0]), float(point[1]), float(point[2])]
            # Handle direct Vector3D object
            elif hasattr(tp, 'x') and hasattr(tp, 'y') and hasattr(tp, 'z'):
                # Direct Vector3D object
                coord = [float(tp.x), float(tp.y), float(tp.z)]
            # Handle direct list/tuple
            elif isinstance(tp, (list, tuple)) and len(tp) >= 3:
                # Direct list/tuple coordinates
                coord = [float(tp[0]), float(tp[1]), float(tp[2])]
            
            if coord:
                triple_coords.append(coord)
                logger.info(f"  Extracted triple point {i}: {coord}")
            else:
                logger.warning(f"  Could not extract coordinates from triple point {i}: {tp}")
        
        if not triple_coords:
            logger.warning("No valid triple point coordinates found")
            # Instead, highlight triple points in constraint lines if they exist
            self._highlight_triple_points_in_constraints(plotter)
            return
        
        # Convert to numpy array
        triple_coords_np = np.array(triple_coords)
        logger.info(f"Adding {len(triple_coords_np)} highly visible triple points to visualization")
        
        # Create point cloud for triple points
        triple_point_cloud = pv.PolyData(triple_coords_np)
        
        # Add LARGE, BRIGHT spheres for triple points
        sphere_radius = 2.0  # Large radius for visibility
        spheres = triple_point_cloud.glyph(
            geom=pv.Sphere(radius=sphere_radius, phi_resolution=8, theta_resolution=8),
            scale=False
        )
        
        # Add to plotter with very visible styling
        plotter.add_mesh(
            spheres,
            color='yellow',           # Bright yellow color
            style='surface',
            opacity=1.0,             # Fully opaque
            name='triple_points_spheres',
            show_edges=False,
            lighting=True,
            ambient=0.6,             # High ambient lighting
            diffuse=0.8,             # High diffuse reflection
            specular=0.2             # Some specular reflection for shine
        )
        
        # Also add text labels for each triple point
        for i, coord in enumerate(triple_coords_np):
            plotter.add_point_labels(
                coord.reshape(1, -1),
                [f'TP{i+1}'],
                point_size=0,  # No point, just label
                font_size=12,
                text_color='black',
                name=f'triple_point_label_{i}'
            )
        
        logger.info(f"Added {len(triple_coords_np)} triple points as bright yellow spheres with labels")

    def _highlight_triple_points_in_constraints(self, plotter):
        """Highlight triple points by finding them in constraint line vertices"""
        if not hasattr(self, 'constrained_meshes') or not self.constrained_meshes:
            return
        
        import pyvista as pv
        
        # Get original triple point coordinates for comparison
        original_triple_coords = []
        if hasattr(self, 'triple_points') and self.triple_points:
            for tp in self.triple_points:
                coord = None
                
                # Handle dictionary format: {'point': Vector3D(...), 'intersections': [...]}
                if isinstance(tp, dict) and 'point' in tp:
                    point = tp['point']
                    if hasattr(point, 'x') and hasattr(point, 'y') and hasattr(point, 'z'):
                        coord = [float(point.x), float(point.y), float(point.z)]
                # Handle direct object format
                elif hasattr(tp, 'point'):
                    point = tp.point
                    if hasattr(point, 'x') and hasattr(point, 'y') and hasattr(point, 'z'):
                        coord = [float(point.x), float(point.y), float(point.z)]
                # Handle direct Vector3D object
                elif hasattr(tp, 'x') and hasattr(tp, 'y') and hasattr(tp, 'z'):
                    coord = [float(tp.x), float(tp.y), float(tp.z)]
                
                if coord:
                    original_triple_coords.append(coord)
        
        if not original_triple_coords:
            logger.info("No original triple point coordinates for comparison")
            return
        
        original_triple_coords_np = np.array(original_triple_coords)
        logger.info(f"Looking for {len(original_triple_coords)} triple points in constraint mesh vertices")
        
        # Find triple points in mesh vertices
        found_triple_points = []
        tolerance = 1e-3  # Tolerance for coordinate matching
        
        for dataset_idx, mesh_data in self.constrained_meshes.items():
            vertices = mesh_data.get('vertices', [])
            if len(vertices) == 0:
                continue
                
            vertices_np = np.array(vertices)
            
            # Check each original triple point against mesh vertices
            for i, orig_tp in enumerate(original_triple_coords_np):
                distances = np.linalg.norm(vertices_np - orig_tp, axis=1)
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                
                if min_dist < tolerance:
                    found_vertex = vertices_np[min_dist_idx]
                    found_triple_points.append(found_vertex)
                    logger.info(f"  Found triple point {i} in dataset {dataset_idx} at {found_vertex} (dist: {min_dist:.6f})")
        
        if found_triple_points:
            # Create visualization for found triple points
            found_triple_points_np = np.array(found_triple_points)
            triple_point_cloud = pv.PolyData(found_triple_points_np)
            
            # Add EXTRA LARGE spheres for found triple points
            sphere_radius = 3.0  # Even larger radius
            spheres = triple_point_cloud.glyph(
                geom=pv.Sphere(radius=sphere_radius, phi_resolution=8, theta_resolution=8),
                scale=False
            )
            
            plotter.add_mesh(
                spheres,
                color='red',           # Bright red color for found triple points
                style='surface',
                opacity=1.0,
                name='found_triple_points_spheres',
                show_edges=False,
                lighting=True,
                ambient=0.8,
                diffuse=0.9,
                specular=0.3
            )
            
            logger.info(f"Added {len(found_triple_points)} found triple points as red spheres")
        else:
            logger.warning("No triple points found in constrained mesh vertices")

    def _analyze_refinement_results(self):
        """Analyze refinement results and return a summary string focusing on special points and hull status."""
        if not hasattr(self, 'original_intersections_backup') or not self.original_intersections_backup or not self.datasets_intersections:
            return "No data to analyze."
        
        # Compare original vs refined intersections
        modified_count = 0
        original_total_points = 0
        refined_total_points = 0
        
        # Special point counters
        triple_points_count = 0
        special_points_count = 0
        triple_points_on_hulls = 0
        
        # Count triple points from the dedicated attribute
        stored_triple_points_count = len(self.triple_points) if hasattr(self, 'triple_points') else 0
        
        # Intersection type counters
        polyline_surf_intersections = 0
        surf_surf_intersections = 0
        
        # Count intersection lines by type
        for ds_key in self.datasets_intersections.keys():
            if ds_key in self.original_intersections_backup:
                for i, refined_intersection in enumerate(self.datasets_intersections[ds_key]):
                    if i < len(self.original_intersections_backup[ds_key]):
                        original_intersection = self.original_intersections_backup[ds_key][i]
                        
                        # Count by intersection type
                        if refined_intersection.get('is_polyline_mesh', False):
                            polyline_surf_intersections += 1
                        else:
                            surf_surf_intersections += 1
                        
                        # Get points
                        original_points = original_intersection.get('points', [])
                        refined_points = refined_intersection.get('points', [])
                        
                        original_total_points += len(original_points)
                        refined_total_points += len(refined_points)
                        
                        # Count modifications
                        if len(original_points) != len(refined_points):
                            modified_count += 1
                        
                        # Count different types of special points in the refined data
                        for point in refined_points:
                            point_type = None
                            # Check point format - could be a list with type at index 3 or an object with point_type
                            if isinstance(point, list) and len(point) > 3:
                                point_type = point[3]
                            elif hasattr(point, 'point_type'):
                                point_type = point.point_type
                            
                            if point_type:
                                type_str = str(point_type).upper()
                                if "TRIPLE_POINT" in type_str:
                                    triple_points_count += 1
                                elif "HIGH_CORNER_POINT" in type_str:
                                    special_points_count += 1
                                elif "SPECIAL_POINT" in type_str:
                                    special_points_count += 1
        
        # Check for special points on convex hulls
        hull_special_points_count = 0
        total_hull_points = 0
        for dataset in self.datasets:
            hull_points = dataset.get('hull_points', [])
            total_hull_points += len(hull_points)
            for hp in hull_points:
                if len(hp) > 3 and isinstance(hp[3], str):
                    type_str = hp[3].upper()
                    if "HIGH_CORNER_POINT" in type_str or "SPECIAL_POINT" in type_str:
                        hull_special_points_count += 1
                    elif "TRIPLE_POINT" in type_str:
                        triple_points_on_hulls += 1
        
        # Create detailed summary
        summary = "<b>Refinement Summary:</b><br>"
        
        # Intersection information
        summary += "<u>Intersection Lines:</u><br>"
        if surf_surf_intersections > 0:
            summary += f"• {surf_surf_intersections} surface-surface intersections<br>"
        if polyline_surf_intersections > 0:
            summary += f"• {polyline_surf_intersections} polyline-surface intersections<br>"
        
        # Convex hull information
        summary += "<br><u>Convex Hulls:</u><br>"
        summary += f"• {total_hull_points} total points on all convex hulls<br>"
        if hull_special_points_count > 0:
            summary += f"• <span style='color:#2ecc71;'>{hull_special_points_count} special points</span> (angle > 135°) on convex hulls<br>"
        if triple_points_on_hulls > 0:
            summary += f"• <span style='color:#e74c3c;'>{triple_points_on_hulls} triple points</span> aligned to convex hulls<br>"
        
        # Special point information
        summary += "<br><u>Special Points:</u><br>"
        if stored_triple_points_count > 0:
            summary += f"• <span style='color:#e74c3c;'>{stored_triple_points_count} triple points</span> identified (saved for constraints)<br>"
        if triple_points_count > 0:
            summary += f"• <span style='color:#e74c3c;'>{triple_points_count} triple points</span> in intersection lines<br>"
        if special_points_count > 0:
            summary += f"• <span style='color:#3498db;'>{special_points_count} special points</span> on intersection lines<br>"
        
        # Length refinement information
        summary += "<br><u>Length Refinement:</u><br>"
        if original_total_points > 0 and refined_total_points > 0:
            if refined_total_points > original_total_points:
                summary += f"• {original_total_points} → {refined_total_points} points ({((refined_total_points - original_total_points) / max(1, original_total_points) * 100):.1f}% increase)<br>"
            else:
                summary += f"• {original_total_points} → {refined_total_points} points<br>"
        
        return summary
    def _refine_intersection_lines_action(self):
        """
        Action to refine intersection lines by:
        1. Identifying triple points at intersection line crossings
        2. Identifying special points (angles > 135 degrees) on hulls and intersections 
        3. Refining lines by dividing into segments of target length
        
        Focus on keeping only convex hull and intersection lines without triangulated surfaces.
        """
        logger.info("Starting refinement of intersection lines...")
        self.statusBar().showMessage("Refining intersection lines...")

        if not hasattr(self, 'datasets_intersections') or not self.datasets_intersections:
            QMessageBox.information(self, "No Intersections", "No intersections found to refine.")
            self.statusBar().showMessage("Refinement skipped: No intersections found.", 5000)
            return

        # --- Store original for summary ---
        import copy
        self.original_intersections_backup = copy.deepcopy(self.datasets_intersections)

        # --- Get UI Parameters ---
        try:
            target_feature_size = float(self.mesh_target_feature_size_input.text())
            gradient = float(self.mesh_gradient_input.text())
            min_angle_deg = float(self.mesh_min_angle_input.text())
            uniform_meshing = self.mesh_uniform_checkbox.isChecked()
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid numeric input for refinement parameters.")
            self.statusBar().showMessage("Refinement failed: Invalid input.", 5000)
            return
        
        # --- Temporary Model Setup ---
        class TempModelWrapper:
            def __init__(self):
                self.surfaces = [] # List of TempDataWrapper for surfaces only (no triangulation needed)
                self.polylines = [] # List of TempDataWrapper for polylines
                self.intersections = [] # List of Intersection objects
                self.triple_points = [] # List to store the triple points explicitly
                
                # Mappings
                self.original_indices_map = {} 
                self.is_polyline = {}
                self.surface_original_to_temp_idx_map = {}
                self.polyline_original_to_temp_idx_map = {}


        temp_model = TempModelWrapper()

        involved_original_indices = set()
        for _, intersections_list in self.datasets_intersections.items():
            for intersection_data in intersections_list:
                involved_original_indices.add(intersection_data['dataset_id1'])
                involved_original_indices.add(intersection_data['dataset_id2'])

        temp_data_idx_counter = 0 # This will be the key for original_indices_map and is_polyline

        for original_idx, dataset_content in enumerate(self.datasets):
            if original_idx not in involved_original_indices:
                continue

            class TempDataWrapper:
                def __init__(self):
                    self.convex_hull = [] # List of Vector3D
                    self.size = 0.1 # Default size
                    self.name = f"Dataset_{original_idx}"
                    # self.vertices = [] # Could add if needed by some refinement step

            data_wrapper = TempDataWrapper()
            data_wrapper.name = dataset_content.get('name', f"Dataset_{original_idx}")
            
            current_geom_points = [] # For calculating size if hull_points is missing
            hull_points_np = dataset_content.get('hull_points')
            if hull_points_np is not None and len(hull_points_np) > 0:
                data_wrapper.convex_hull = []
                for hp in hull_points_np:
                    point_type = hp[3] if len(hp) > 3 and isinstance(hp[3], str) else "DEFAULT"
                    data_wrapper.convex_hull.append(Vector3D(hp[0], hp[1], hp[2] if len(hp) > 2 else 0.0, point_type=point_type))
                current_geom_points = hull_points_np
            else: # Fallback: use 'points' if 'hull_points' is missing
                points_np = dataset_content.get('points')
                if points_np is not None and len(points_np) > 0:
                    data_wrapper.convex_hull = [Vector3D(p[0], p[1], p[2] if len(p) > 2 else 0.0) for p in points_np]
                    current_geom_points = points_np
                else:
                    logger.warning(f"Dataset {original_idx} ({data_wrapper.name}) has no hull_points or points. Skipping for temp_model.")
                    continue
            
            if len(current_geom_points) > 1:
                min_pt, max_pt = np.min(current_geom_points, axis=0), np.max(current_geom_points, axis=0)
                diag_length = np.linalg.norm(max_pt - min_pt)
                data_wrapper.size = diag_length / 10.0 if diag_length > 0 else 0.1
            
            # Determine if it's a polyline (heuristic, improve if type is stored in dataset_content)
            if original_idx == 0:  # Force first dataset to be a surface
                is_p = False
                logger.info(f"Dataset {original_idx} ({data_wrapper.name}) FORCED as surface for testing")
            else:
                # Original logic with triangulation check
                is_p = (dataset_content.get('type') == 'polyline' or
                        ('segments' in dataset_content and 'triangles' not in dataset_content and 'hull_points' not in dataset_content))
                logger.info(f"Dataset {original_idx} ({data_wrapper.name}) classified as {'polyline' if is_p else 'surface'}")
            if not is_p: # Check intersection involvement if not clearly a polyline
                for _, intersection_list_check in self.datasets_intersections.items():
                    for intersection_d_check in intersection_list_check:
                        if intersection_d_check['is_polyline_mesh'] and intersection_d_check['dataset_id1'] == original_idx:
                            is_p = True; break
                    if is_p: break
            
            temp_model.original_indices_map[temp_data_idx_counter] = original_idx
            temp_model.is_polyline[temp_data_idx_counter] = is_p

            if is_p:
                temp_model.polyline_original_to_temp_idx_map[original_idx] = len(temp_model.polylines)
                temp_model.polylines.append(data_wrapper)
            else:
                temp_model.surface_original_to_temp_idx_map[original_idx] = len(temp_model.surfaces)
                temp_model.surfaces.append(data_wrapper)
            
            temp_data_idx_counter += 1

        if not temp_model.surfaces and not temp_model.polylines:
            QMessageBox.warning(self, "No Valid Datasets", "No datasets with geometry were prepared for refinement.")
            self.statusBar().showMessage("Refinement failed: No valid datasets.", 5000)
            return
        logger.info(f"Temp model created: {len(temp_model.surfaces)} surfaces, {len(temp_model.polylines)} polylines.")

        # Map original self.datasets_intersections to temp_model.intersections using temp_data_idx_counter keys
        original_to_temp_combined_idx_map = {v: k for k, v in temp_model.original_indices_map.items()}

        for _, intersections_list in self.datasets_intersections.items():
            for intersection_data in intersections_list:
                temp_combined_id1 = original_to_temp_combined_idx_map.get(intersection_data['dataset_id1'])
                temp_combined_id2 = original_to_temp_combined_idx_map.get(intersection_data['dataset_id2'])

                if temp_combined_id1 is None or temp_combined_id2 is None:
                    logger.warning(f"Skipping intersection: Original IDs {intersection_data['dataset_id1']}/{intersection_data['dataset_id2']} not in temp_model map.")
                    continue
                
                new_int = Intersection(temp_combined_id1, temp_combined_id2, intersection_data['is_polyline_mesh'])
                for pt_coords in intersection_data['points']:
                    new_int.add_point(Vector3D(pt_coords[0], pt_coords[1], pt_coords[2] if len(pt_coords) > 2 else 0.0))
                temp_model.intersections.append(new_int)
        
        if not temp_model.intersections:
            QMessageBox.information(self, "No Intersections", "No intersections populated in temp_model.")
            self.statusBar().showMessage("Refinement skipped: No intersections in temp model.", 5000)
            return
        logger.info(f"Temp model has {len(temp_model.intersections)} intersections to process.")

        # --- Step 1: Identify Triple Points ---
        try:
            # Triple points are crucial for refinement - these are points where 3 or more surfaces intersect
            # First find the triple points where multiple intersections meet
            for i in range(len(temp_model.intersections) - 1):
                for j in range(i + 1, len(temp_model.intersections)):
                    # Calculate triple points between pairs of intersections
                    triple_points = calculate_triple_points(i, j, temp_model, tolerance=1e-5)
                    for tp in triple_points:
                        # Create a TriplePoint object
                        triple_point_obj = TriplePoint(tp)
                        triple_point_obj.add_intersection(i)
                        triple_point_obj.add_intersection(j)
                        # Set point type explicitly to TRIPLE_POINT
                        tp.point_type = "TRIPLE_POINT"
                        temp_model.triple_points.append(triple_point_obj)
            
            # Insert the triple points into the intersections
            insert_triple_points(temp_model)
            
            # Store triple points for constraint references in the pre-tetrameshtab
            self.triple_points = []
            for tp in temp_model.triple_points:
                self.triple_points.append({
                    'point': tp.point,
                    'intersections': tp.intersection_ids
                })
                
            logger.info(f"Found and inserted {len(temp_model.triple_points)} triple points.")
        except Exception as e:
            logger.error(f"Error during triple points calculation: {e}", exc_info=True)
            QMessageBox.warning(self, "Refinement Error", f"Error during triple point identification: {str(e)}")
            return
        
        # --- Step 2: Identify corner points on convex hulls (C++ MakeCornersSpecial) ---
        try:
            from meshit.intersection_utils import make_corners_special
            
            # Process each surface's convex hull to identify corner points first
            # This mirrors the C++ workflow: calculate_convex_hull -> MakeCornersSpecial -> alignIntersectionsToConvexHull
            corner_points_count = 0
            for temp_surface_idx, temp_surface in enumerate(temp_model.surfaces):
                if not hasattr(temp_surface, 'convex_hull') or len(temp_surface.convex_hull) < 3:
                    continue
                
                original_dataset_idx = temp_model.original_indices_map.get(temp_surface_idx)
                logger.info(f"Identifying corner points on convex hull for surface {temp_surface_idx} (original dataset {original_dataset_idx})")
                
                # Apply corner detection using the same angle threshold as C++ MeshIt (135°)
                temp_surface.convex_hull = make_corners_special(temp_surface.convex_hull, angle_threshold_deg=135.0)
                
                # Count corner points
                for pt in temp_surface.convex_hull:
                    pt_type = getattr(pt, 'point_type', getattr(pt, 'type', "DEFAULT"))
                    if pt_type == "CORNER":
                        corner_points_count += 1
            
            logger.info(f"Identified {corner_points_count} corner points across all convex hulls")
        except Exception as e:
            logger.error(f"Error during corner point identification: {e}", exc_info=True)
            QMessageBox.warning(self, "Refinement Error", f"Error during corner point identification: {str(e)}")
            return
        
        # --- Step 3: Align intersections to convex hulls (C++ alignIntersectionsToConvexHull) ---
        try:
            # This step aligns intersection points with the convex hulls of surfaces
            # Creates new special junction points where intersection lines meet convex hull boundaries
            for temp_surface_list_idx in range(len(temp_model.surfaces)):
                align_intersections_to_convex_hull(temp_surface_list_idx, temp_model)
            logger.info("Intersection alignment to convex hulls complete.")
        except Exception as e:
            logger.error(f"Error during convex hull alignment: {e}", exc_info=True)
            QMessageBox.warning(self, "Refinement Error", f"Error during convex hull alignment: {str(e)}")
            return
        
        # --- Step 4: Refine intersection lines by length ---
            
                # --- Step 4: Refine intersection lines by length ---
        try:
            # Custom angle thresholds for point classification in refine_intersection_line_by_length
            # HC_ANGLE_THRESHOLD = 135.0  # Only keep special points with angles > 135 degrees
            
            for intersection in temp_model.intersections:
                # Determine effective target length based on involved objects
                id1 = intersection.id1
                id2 = intersection.id2
                is_poly1 = temp_model.is_polyline.get(id1, False)
                is_poly2 = temp_model.is_polyline.get(id2, False)
                
                # Get the data wrappers for the objects
                obj1_data = None
                if is_poly1:
                    if temp_model.original_indices_map[id1] in temp_model.polyline_original_to_temp_idx_map:
                        obj1_data = temp_model.polylines[temp_model.polyline_original_to_temp_idx_map[temp_model.original_indices_map[id1]]]
                else:
                    if temp_model.original_indices_map[id1] in temp_model.surface_original_to_temp_idx_map:
                        obj1_data = temp_model.surfaces[temp_model.surface_original_to_temp_idx_map[temp_model.original_indices_map[id1]]]
                
                obj2_data = None
                if is_poly2:
                    if temp_model.original_indices_map[id2] in temp_model.polyline_original_to_temp_idx_map:
                        obj2_data = temp_model.polylines[temp_model.polyline_original_to_temp_idx_map[temp_model.original_indices_map[id2]]]
                else:
                    if temp_model.original_indices_map[id2] in temp_model.surface_original_to_temp_idx_map:
                        obj2_data = temp_model.surfaces[temp_model.surface_original_to_temp_idx_map[temp_model.original_indices_map[id2]]]
                
                # Get sizes
                obj1_size = obj1_data.size if obj1_data and hasattr(obj1_data, 'size') else 0.1
                obj2_size = obj2_data.size if obj2_data and hasattr(obj2_data, 'size') else 0.1
                
                # Determine effective target length
                eff_target_length = target_feature_size
                if not uniform_meshing:
                    valid_sizes = [s for s in [obj1_size, obj2_size] if s > 1e-6]
                    if valid_sizes:
                        eff_target_length = min(valid_sizes)
                    if target_feature_size > 1e-6 and target_feature_size < eff_target_length:
                        eff_target_length = target_feature_size
                
                if eff_target_length <= 1e-6:
                    eff_target_length = 0.1
                
                # Refine the intersection line
                refined_points = refine_intersection_line_by_length(
                    intersection, 
                    target_length=eff_target_length,
                    min_angle_deg=min_angle_deg,
                    uniform_meshing=uniform_meshing
                )
                
                # Update the intersection with refined points
                intersection.points = refined_points
                
            logger.info("Length-based refinement complete.")
        except Exception as e:
            logger.error(f"Error during length-based refinement: {e}", exc_info=True)
            QMessageBox.warning(self, "Refinement Error", f"Error during length-based refinement: {str(e)}")
            return
                # --- STEP: Store intersection lines as constraints (C++ approach) ---
        logger.info("Storing intersection lines as constraints for each surface...")
        
        # Clear existing constraints
        for dataset_idx in range(len(self.datasets)):
            self.datasets[dataset_idx]['stored_constraints'] = []
        
        # Store intersection constraints for each surface
        for intersection in temp_model.intersections:
            original_id1 = temp_model.original_indices_map.get(intersection.id1)
            original_id2 = temp_model.original_indices_map.get(intersection.id2)
            
            if original_id1 is not None and original_id1 < len(self.datasets):
                # Store this intersection as constraint for surface 1
                constraint_line = []
                for pt in intersection.points:
                    # keep the vertex “type” so the constraint manager can split later
                    p_type = getattr(pt, "point_type", getattr(pt, "type", "DEFAULT"))
                    constraint_line.append([pt.x, pt.y, pt.z, p_type])
                
                if len(constraint_line) >= 2:  # Only store if it's a valid line
                    self.datasets[original_id1]['stored_constraints'].append({
                        'type': 'intersection_line',
                        'points': constraint_line,
                        'other_surface_id': original_id2
                    })
                    logger.info(f"Stored intersection constraint for surface {original_id1}: {len(constraint_line)} points")
            
            if original_id2 is not None and original_id2 < len(self.datasets) and original_id1 != original_id2:
                # Store this intersection as constraint for surface 2
                constraint_line = []
                for pt in intersection.points:
                    p_type = getattr(pt, "point_type", getattr(pt, "type", "DEFAULT"))
                    constraint_line.append([pt.x, pt.y, pt.z, p_type])
                
                if len(constraint_line) >= 2:  # Only store if it's a valid line
                    self.datasets[original_id2]['stored_constraints'].append({
                        'type': 'intersection_line', 
                        'points': constraint_line,
                        'other_surface_id': original_id1
                    })
                    logger.info(f"Stored intersection constraint for surface {original_id2}: {len(constraint_line)} points")
        
        logger.info("Constraint storage complete.")
        # --- Update main data structures and visualization ---
        self.datasets_intersections.clear() 
        self.refined_intersections_for_visualization = {}

        for temp_s_list_idx, temp_surface_data in enumerate(temp_model.surfaces):
    # Find the original dataset index for this temp surface
            original_dataset_idx = None
            for orig_idx, temp_idx in temp_model.surface_original_to_temp_idx_map.items():
                if temp_idx == temp_s_list_idx:
                    original_dataset_idx = orig_idx
                    break
            
            if original_dataset_idx is not None:
                if hasattr(temp_surface_data, 'convex_hull') and temp_surface_data.convex_hull:
                    # Create hull points array with type information
                    hull_points_with_type = []
                    for p in temp_surface_data.convex_hull:
                        point_type = "DEFAULT"
                        if hasattr(p, 'type') and p.type:
                            point_type = p.type
                        elif hasattr(p, 'point_type') and p.point_type:
                            point_type = p.point_type
                        
                        # Store points with type as 4th element [x, y, z, type]
                        hull_points_with_type.append([p.x, p.y, p.z, point_type])
                        logger.info(f"Copying hull point: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) type={point_type}")

                    # Store hull points with their types
                    self.datasets[original_dataset_idx]['hull_points'] = np.array(hull_points_with_type, dtype=object)
                    logger.info(f"Updated convex hull for dataset {original_dataset_idx} with {len(hull_points_with_type)} points.")
                else:
                    logger.warning(f"No convex hull data in temp_surface_data for original_dataset_idx {original_dataset_idx} to copy back.")
            else:
                logger.warning(f"Could not map temp_model.surfaces index {temp_s_list_idx} to an original dataset index for hull update.")
        for intersection in temp_model.intersections:
            original_id1 = temp_model.original_indices_map.get(intersection.id1)
            original_id2 = temp_model.original_indices_map.get(intersection.id2)

            if original_id1 is None or original_id2 is None:
                logger.warning(f"Post-refinement: Skipping intersection due to missing original ID map for temp IDs {intersection.id1}/{intersection.id2}.")
                continue
            
                      # Prepare points with embedded types for visualization compatibility
            points_for_entry = []
            for p_obj in intersection.points: # Iterate over the point objects from refinement
                p_type_str = "DEFAULT" # Default type if none found
                if hasattr(p_obj, 'point_type') and p_obj.point_type:
                    p_type_str = p_obj.point_type
                
                points_for_entry.append([p_obj.x, p_obj.y, p_obj.z, p_type_str])

            intersection_entry = {
                'dataset_id1': original_id1,
                'dataset_id2': original_id2,
                'is_polyline_mesh': intersection.is_polyline_mesh,
                'points': points_for_entry, # Now contains [x,y,z,type_string]
                # The separate 'point_types' list that was here before is no longer needed
                # as the type is embedded directly with the coordinates for visualization.
            }
            
            # Store in self.datasets_intersections (main data)
            # Key by the dataset that is involved (can be either id1 or id2)
            # To avoid duplicates if an intersection is processed from "both sides", use a combined key or primary key
            primary_key_ds = min(original_id1, original_id2) # Consistent keying
            if primary_key_ds not in self.datasets_intersections:
                 self.datasets_intersections[primary_key_ds] = []
            
            # Check if this specific intersection is already added under this key to avoid exact duplicates
            already_added = False
            for existing_entry in self.datasets_intersections[primary_key_ds]:
                if (existing_entry['dataset_id1'] == original_id1 and \
                    existing_entry['dataset_id2'] == original_id2 and \
                    existing_entry['is_polyline_mesh'] == intersection.is_polyline_mesh) or \
                   (existing_entry['dataset_id1'] == original_id2 and \
                    existing_entry['dataset_id2'] == original_id1 and \
                    existing_entry['is_polyline_mesh'] == intersection.is_polyline_mesh) :
                    # A more robust check might compare point lists if order can vary
                    if len(existing_entry['points']) == len(intersection_entry['points']):
                         already_added = True # Simple check, assume it's the same
                         break
            if not already_added:
                self.datasets_intersections[primary_key_ds].append(intersection_entry)

            # Store for self.refined_intersections_for_visualization (visualization helper)
            # This is keyed by *each* dataset involved.
            for vis_key_id in [original_id1, original_id2]:
                if vis_key_id not in self.refined_intersections_for_visualization:
                    self.refined_intersections_for_visualization[vis_key_id] = []
                # Add if not already present for this specific vis_key_id
                # (to prevent adding the same intersection twice if original_id1 == original_id2, or if processing from other partner)
                vis_already_added = any(
                    ie['dataset_id1'] == intersection_entry['dataset_id1'] and
                    ie['dataset_id2'] == intersection_entry['dataset_id2'] and
                    ie['is_polyline_mesh'] == intersection_entry['is_polyline_mesh'] and
                    len(ie['points']) == len(intersection_entry['points']) # simple identity check
                    for ie in self.refined_intersections_for_visualization[vis_key_id]
                )
                if not vis_already_added:
                     self.refined_intersections_for_visualization[vis_key_id].append(intersection_entry.copy())

        # --- NEW: Sync special convex hull points from intersections to hulls ---
        for temp_surface_idx, temp_surface in enumerate(temp_model.surfaces):
            if not hasattr(temp_surface, 'convex_hull') or not temp_surface.convex_hull:
                continue
            # Find all intersection points for this surface that are special convex hull points
            for intersection in temp_model.intersections:
                # Only consider intersections involving this surface
                if intersection.id1 != temp_surface_idx and intersection.id2 != temp_surface_idx:
                    continue
                for pt in intersection.points:
                    # Only insert if it's a convex hull special point
                    pt_type = getattr(pt, 'type', getattr(pt, 'point_type', None))
                    if pt_type == "COMMON_INTERSECTION_CONVEXHULL_POINT":
                        # Check if this point is already in the hull (within tolerance)
                        already_in_hull = any(
                            abs(pt.x - hp.x) < 1e-10 and abs(pt.y - hp.y) < 1e-10 and abs(pt.z - hp.z) < 1e-10
                            for hp in temp_surface.convex_hull
                        )
                        if not already_in_hull:
                            # Find the closest segment and insert
                            min_dist = float('inf')
                            insert_idx = 0
                            for i in range(len(temp_surface.convex_hull) - 1):
                                seg_start = temp_surface.convex_hull[i]
                                seg_end = temp_surface.convex_hull[i+1]
                                # Project pt onto segment
                                seg_vec = Vector3D(seg_end.x - seg_start.x, seg_end.y - seg_start.y, seg_end.z - seg_start.z)
                                seg_len2 = seg_vec.x**2 + seg_vec.y**2 + seg_vec.z**2
                                if seg_len2 == 0:
                                    continue
                                t = ((pt.x - seg_start.x) * seg_vec.x + (pt.y - seg_start.y) * seg_vec.y + (pt.z - seg_start.z) * seg_vec.z) / seg_len2
                                t = max(0, min(1, t))
                                proj = Vector3D(
                                    seg_start.x + (seg_end.x - seg_start.x) * t,
                                    seg_start.y + (seg_end.y - seg_start.y) * t,
                                    seg_start.z + (seg_end.z - seg_start.z) * t,
                                    point_type=pt_type
                                )
                                dist = (Vector3D(pt.x, pt.y, pt.z) - proj).length()
                                if dist < min_dist:
                                    min_dist = dist
                                    insert_idx = i + 1
                            temp_surface.convex_hull.insert(insert_idx, pt)

        for temp_surface_idx, temp_surface in enumerate(temp_model.surfaces):
            if hasattr(temp_surface, 'convex_hull') and temp_surface.convex_hull:
                logger.info(f"After insertion, convex hull for surface {temp_surface_idx} has {len(temp_surface.convex_hull)} points.")
                # Count special points by type
                type_counts = {}
                for hp in temp_surface.convex_hull:
                    pt_type = getattr(hp, 'type', getattr(hp, 'point_type', 'NONE'))
                    type_counts[pt_type] = type_counts.get(pt_type, 0) + 1
                    if pt_type == "COMMON_INTERSECTION_CONVEXHULL_POINT":
                        logger.info(f"  SPECIAL Hull pt: ({hp.x:.3f}, {hp.y:.3f}, {hp.z:.3f}) type={pt_type}")
        
                logger.info(f"  Hull point types: {type_counts}")
        # Copy back modified convex hulls from temp_model.surfaces to self.datasets
        for temp_s_list_idx, temp_surface_data in enumerate(temp_model.surfaces):
            # Find the original_dataset_idx for temp_model.surfaces[temp_s_list_idx]
            original_dataset_idx = -1
            for orig_idx_key, temp_surf_map_idx_val in temp_model.surface_original_to_temp_idx_map.items():
                if temp_surf_map_idx_val == temp_s_list_idx:
                    original_dataset_idx = orig_idx_key
                    break
            
            if original_dataset_idx != -1 and original_dataset_idx < len(self.datasets):
                if hasattr(temp_surface_data, 'convex_hull') and temp_surface_data.convex_hull:
                    # Create hull points array with type information
                    hull_points_with_type = []
                    for p in temp_surface_data.convex_hull:
                        point_type = "DEFAULT"
                        if hasattr(p, 'type') and p.type:
                            point_type = p.type
                        elif hasattr(p, 'point_type') and p.point_type:
                            point_type = p.point_type
                        
                        # Store points with type as 4th element [x, y, z, type]
                        hull_points_with_type.append([p.x, p.y, p.z, point_type])
                        logger.info(f"Copying hull point: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) type={point_type}")

                    # Store hull points with their types
                    self.datasets[original_dataset_idx]['hull_points'] = np.array(hull_points_with_type, dtype=object)
                    logger.info(f"Updated convex hull for dataset {original_dataset_idx} with {len(hull_points_with_type)} points.")
                    logger.info(f"Updated convex hull for dataset {original_dataset_idx} ('{self.datasets[original_dataset_idx].get('name')}') with {len(hull_points_with_type)} points.")
                else:
                    logger.warning(f"No convex hull data in temp_surface_data for original_dataset_idx {original_dataset_idx} to copy back.")
            else:
                logger.warning(f"Could not map temp_model.surfaces index {temp_s_list_idx} to an original dataset index for hull update.")

        summary = self._analyze_refinement_results() 
        self.refinement_summary_label.setText(summary)

        self.statusBar().showMessage("Intersection lines refined successfully.", 5000)
        logger.info("Intersection lines refined successfully.")

        # At the very end of _refine_intersection_lines_action(), before the final success message:
        logger.info("Consolidating all refined points for triangulation...")
        self.consolidate_points_for_triangulation()
        logger.info("Consolidation complete!")

        logger.info("Intersection lines refined successfully.")
        self._update_refined_visualization()
    def load_file(self):
        """Load data from a file"""
        self.statusBar().showMessage("Loading file...")
        
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a point data file",
            "",
            "Text files (*.txt);;Data files (*.dat);;CSV files (*.csv);;All files (*.*)"
        )
        
        if not file_path:
            self.statusBar().showMessage("File loading canceled")
            return
            
        # Update status with file path
        self.statusBar().showMessage(f"Loading file: {os.path.basename(file_path)}...")
        
        try:
            # Try to read the file
            points = self._read_point_file(file_path)
            
            if points is not None and len(points) > 0:
                self.points = points
                
                # Visualize the loaded points
                self._plot_points(points)
                
                self.statusBar().showMessage(f"Successfully loaded {len(points)} points")
                
                # Clear any previous results from later steps
                self.hull_points = None
                self.segments = None
                self.triangulation_result = None
                
                # Update other views
                self._clear_hull_plot()
                self._clear_segment_plot()
                self._clear_triangulation_plot()
            else:
                self.statusBar().showMessage("Error: No valid points found in file")
                QMessageBox.critical(self, "Error", "No valid points found in file")
        
        except Exception as e:
            self.statusBar().showMessage(f"Error loading file: {str(e)}")
            logger.error(f"Error loading file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
    
    def _read_point_file(self, file_path):
        """Read points from a file, handling various delimiters and formats."""
        points = []
        try:
            # --- Primary Method: Line-by-line parsing for robustness --- 
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue # Skip comments and empty lines

                    # Check for Bounds format first
                    if line.startswith("Bounds:"):
                        bounds_points = self._parse_bounds_format(line)
                        if bounds_points:
                            points.extend(bounds_points)
                        continue
                        
                    # Clean the line: replace delimiters, handle signs
                    cleaned_line = line.replace('\t', ' ').replace(',', ' ').replace(';', ' ')
                    # Handle potential + signs before numbers
                    cleaned_line = re.sub(r'(?<![eE\d.-])\+', '', cleaned_line) # Remove + if not part of scientific notation or preceded by digit/.
                    # Ensure space around minus signs not part of scientific notation
                    cleaned_line = re.sub(r'(?<![eE\d.-])-', ' -', cleaned_line) 
                    # Remove extra whitespace
                    cleaned_line = ' '.join(cleaned_line.split())

                    parts = cleaned_line.split()
                    try:
                        if len(parts) >= 2:
                            x = float(parts[0])
                            y = float(parts[1])
                            z = float(parts[2]) if len(parts) >= 3 else 0.0
                            points.append([x, y, z])
                        else:
                            logger.warning(f"Skipping line {line_num+1}: Not enough values found in '{line}'")
                    except ValueError as ve:
                        logger.warning(f"Skipping line {line_num+1} due to value error: {ve} in '{line}'")
                        continue # Skip lines with non-numeric data after cleaning
            
            if points:
                logger.info(f"Read {len(points)} points using line-by-line parsing.")
                points_arr = np.array(points)
                # Ensure 3D format even if Z was missing
                if points_arr.shape[1] == 2:
                     points_3d = np.zeros((len(points_arr), 3))
                     points_3d[:, 0:2] = points_arr
                     return points_3d
                return points_arr[:, :3] # Return first 3 columns if more exist
            else:
                 logger.info("Line-by-line parsing yielded no points, trying np.loadtxt.")

            # --- Fallback Method: np.loadtxt --- 
            points = None # Reset points if primary method failed
            ext = os.path.splitext(file_path)[1].lower()
            delimiters_to_try = [None, '\t', ',', ' '] # None tries whitespace
            if ext == '.csv': # Prioritize comma for csv
                delimiters_to_try = [',', '\t', ' ', None]
                
            for delim in delimiters_to_try:
                try:
                    # Try loading, skip comment lines
                    loaded_points = np.loadtxt(file_path, delimiter=delim, comments='#')
                    # Check shape after loading
                    if loaded_points.ndim == 1: # Handle case where only one line is read
                        if loaded_points.shape[0] >= 2:
                           points = loaded_points.reshape(1, -1)
                    elif loaded_points.ndim == 2 and loaded_points.shape[1] >= 2:
                         points = loaded_points
                         
                    if points is not None:
                         logger.info(f"Successfully read data using np.loadtxt with delimiter: '{delim if delim else 'whitespace'}'")
                         break # Stop trying delimiters if successful
                except Exception as np_err:
                    logger.debug(f"np.loadtxt failed with delimiter '{delim if delim else 'whitespace'}': {np_err}")
                    continue
            
            if points is None:
                 logger.error("Failed to read valid points using all methods.")
                 raise ValueError("Could not read valid point data from file.")

            # --- Final Processing (applies to np.loadtxt result) --- 
            if points.shape[1] < 2:
                raise ValueError("File must contain at least 2D points (x, y)")
            
            if points.shape[1] == 2:
                points_3d = np.zeros((len(points), 3))
                points_3d[:, 0:2] = points
                logger.info(f"Converted {len(points)} 2D points to 3D.")
                return points_3d
            
            logger.info(f"Read {len(points)} points using np.loadtxt.")
            return points[:, 0:3]
            
        except Exception as e:
            logger.error(f"Error reading file '{file_path}': {str(e)}")
            QMessageBox.critical(self, "Error Reading File", f"Failed to read points from file:\n{os.path.basename(file_path)}\n\nError: {str(e)}")
            return None
            
    def _parse_bounds_format(self, line):
        """
        Parse bounds format line: "Bounds: X[min,max] Y[min,max] Z[min,max]"
        
        Args:
            line: String line in bounds format
            
        Returns:
            List of points representing the bounds corners (4 points, Z=0)
        """
        try:
            # Extract X, Y, Z ranges
            x_range = re.search(r'X\[([-+]?\d+\.?\d*),([-+]?\d+\.?\d*)\]', line)
            y_range = re.search(r'Y\[([-+]?\d+\.?\d*),([-+]?\d+\.?\d*)\]', line)
            z_range = re.search(r'Z\[([-+]?\d+\.?\d*),([-+]?\d+\.?\d*)\]', line)
            
            if not x_range or not y_range:
                logger.debug("Could not find X or Y ranges in bounds line")
                return None
                
            # Extract min/max values for each dimension
            x_min, x_max = float(x_range.group(1)), float(x_range.group(2))
            y_min, y_max = float(y_range.group(1)), float(y_range.group(2))
            
            # Use Z from bounds if present for 3D bounds
            if z_range:
                 z_min, z_max = float(z_range.group(1)), float(z_range.group(2))
            else:
                 z_min, z_max = 0.0, 0.0 # Default Z to 0 if not found
                 
            # Return 8 corners of the 3D bounding box
            bounds_points = [
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max]
            ]
            
            logger.info(f"Parsed bounds line into 8 corner points.")
            return bounds_points
        except Exception as e:
            logger.debug(f"Error parsing bounds format: {str(e)}")
            return None
            
    def generate_test_data(self):
        """Generate test data"""
        self.statusBar().showMessage("Generating test data...")
        
        # Create a dialog to select the type of test data
        dialog = QDialog(self)
        dialog.setWindowTitle("Generate Test Data")
        dialog.setFixedSize(300, 250)
        dialog_layout = QVBoxLayout(dialog)
        
        # Data type selection
        dialog_layout.addWidget(QLabel("Select data type:"))
        
        data_type_group = QButtonGroup(dialog)
        data_types = ["Random Points", "Circle", "Square", "Complex Shape"]
        data_type_buttons = {}
        
        for i, data_type in enumerate(data_types):
            radio = QRadioButton(data_type)
            dialog_layout.addWidget(radio)
            data_type_group.addButton(radio, i)
            data_type_buttons[i] = radio
            if i == 0:  # Select the first option by default
                radio.setChecked(True)
        
        dialog_layout.addWidget(QLabel("Number of points:"))
        
        num_points_input = QSpinBox()
        num_points_input.setRange(10, 1000)
        num_points_input.setValue(100)
        dialog_layout.addWidget(num_points_input)
        
        button_layout = QHBoxLayout()
        generate_btn = QPushButton("Generate")
        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(generate_btn)
        
        dialog_layout.addLayout(button_layout)
        
        # Connect signals
        cancel_btn.clicked.connect(dialog.reject)
        generate_btn.clicked.connect(dialog.accept)
        
        # Show dialog and get result
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            # Get the selected data type
            selected_id = data_type_group.checkedId()
            if selected_id >= 0:
                data_type = data_types[selected_id]
                num_points = num_points_input.value()
                
                # Generate the selected data type
                if data_type == "Random Points":
                    points = self._generate_random_points(num_points)
                elif data_type == "Circle":
                    points = self._generate_circle_points(num_points)
                elif data_type == "Square":
                    points = self._generate_square_points(num_points)
                elif data_type == "Complex Shape":
                    points = self._generate_complex_shape(num_points)
                else:
                    points = None
                
                if points is not None:
                    self.points = points
                    
                    # Visualize the generated points
                    self._plot_points(points)
                    
                    self.statusBar().showMessage(f"Generated {len(points)} {data_type} points")
                    
                    # Clear any previous results from later steps
                    self.hull_points = None
                    self.segments = None
                    self.triangulation_result = None
                    
                    # Update other views
                    self._clear_hull_plot()
                    self._clear_segment_plot()
                    self._clear_triangulation_plot()
        else:
            self.statusBar().showMessage("Test data generation canceled")
    
    def _generate_random_points(self, num_points=100):
        """Generate random points"""
        # Generate random points in a square
        points = np.random.rand(num_points, 2) * 20 - 10  # Range: -10 to 10
        return points
    
    def _generate_circle_points(self, num_points=100):
        """Generate points in a circle"""
        # Generate random points in a circle
        r = 10.0 * np.sqrt(np.random.rand(num_points))  # Radius: 0 to 10
        theta = np.random.rand(num_points) * 2 * np.pi  # Angle: 0 to 2pi
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        return np.column_stack((x, y))
    
    def _generate_square_points(self, num_points=100):
        """Generate points in a square"""
        # Generate random points in a square with some clustering
        points = []
        
        # Add corner points to ensure a square shape
        corners = [[-10, -10], [10, -10], [10, 10], [-10, 10]]
        points.extend(corners)
        
        # Add more random points
        remaining = num_points - len(corners)
        if remaining > 0:
            random_points = np.random.rand(remaining, 2) * 20 - 10  # Range: -10 to 10
            points.extend(random_points)
        
        return np.array(points)
    
    def _generate_complex_shape(self, num_points=100):
        """Generate complex shape with internal features"""
        points = []
        
        # Generate outer boundary (polygon)
        num_vertices = 8
        radius = 10.0
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        
        # Add some randomness to the radius
        radii = radius * (0.8 + 0.4 * np.random.rand(num_vertices))
        
        # Create polygon vertices
        for i in range(num_vertices):
            x = radii[i] * np.cos(angles[i])
            y = radii[i] * np.sin(angles[i])
            points.append([x, y])
        
        # Add interior points with some clustering
        remaining = num_points - num_vertices
        
        if remaining > 0:
            # Create clusters
            num_clusters = 3
            cluster_centers = []
            
            for _ in range(num_clusters):
                # Random position within the shape (closer to center)
                cx = (np.random.rand() * 10 - 5)
                cy = (np.random.rand() * 10 - 5)
                cluster_centers.append([cx, cy])
            
            # Distribute points among clusters
            for i in range(remaining):
                # Pick a random cluster
                cluster = np.random.randint(0, num_clusters)
                center = cluster_centers[cluster]
                
                # Generate point with noise around the cluster center
                noise_scale = 2.0 + np.random.rand() * 2.0
                x = center[0] + np.random.randn() * noise_scale
                y = center[1] + np.random.randn() * noise_scale
                
                # Limit to within the approximate boundary
                dist = np.sqrt(x*x + y*y)
                if dist > radius:
                    scale = (radius * 0.9) / dist
                    x *= scale
                    y *= scale
                
                points.append([x, y])
        
        return np.array(points)

    # Visualization methods
    def _clear_hull_plot(self):
        """Clear the hull plot"""
        # Clear existing visualization
        while self.hull_viz_layout.count():
            item = self.hull_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            
        # Reset placeholder
        if HAVE_PYVISTA:
            placeholder_text = "Compute convex hull to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nCompute convex hull to visualize in 2D."
            
        self.hull_viz_placeholder = QLabel(placeholder_text)
        self.hull_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.hull_viz_layout.addWidget(self.hull_viz_placeholder)
        
        # Reset hull info
        self.hull_points_label.setText("Hull vertices: 0")
        self.hull_area_label.setText("Hull area: 0.0")
    
    def _clear_segment_plot(self):
        """Clear the segmentation plot"""
        # Clear existing visualization
        while self.segment_viz_layout.count():
            item = self.segment_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            
        # Reset placeholder
        if HAVE_PYVISTA:
            placeholder_text = "Compute segmentation to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nCompute segmentation to visualize in 2D."
            
        self.segment_viz_placeholder = QLabel(placeholder_text)
        self.segment_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.segment_viz_layout.addWidget(self.segment_viz_placeholder)
        
        # Reset segment info
        self.num_segments_label.setText("Segments: 0")
        self.avg_segment_length_label.setText("Avg length: 0.0")
    
    def _clear_triangulation_plot(self):
        """Clear the triangulation plot"""
        # Clear existing visualization
        while self.tri_viz_layout.count():
            item = self.tri_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            
        # Reset placeholder
        if HAVE_PYVISTA:
            placeholder_text = "Run triangulation to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nRun triangulation to visualize in 2D."
            
        self.tri_viz_placeholder = QLabel(placeholder_text)
        self.tri_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.tri_viz_layout.addWidget(self.tri_viz_placeholder)
        
        # Reset triangulation info
        self.num_triangles_label.setText("Triangles: 0")
        self.num_vertices_label.setText("Vertices: 0")
        self.mean_edge_label.setText("Mean edge: 0.0")
        self.uniformity_label.setText("Uniformity: 0.0")
    
    
    
    def _compute_hull_for_dataset(self, dataset_index):
        """Compute the convex hull for a specific dataset index. Returns True on success, False on error."""
        # Check index validity
        if not (0 <= dataset_index < len(self.datasets)):
            logger.error(f"Invalid dataset index {dataset_index} for hull computation.")
            return False

        dataset = self.datasets[dataset_index]
        dataset_name = dataset.get('name', f"Dataset {dataset_index}")
        # self.statusBar().showMessage(f"Computing hull for {dataset_name}...") # Remove status update
        
        if dataset.get('points') is None or len(dataset['points']) < 3:
            self.statusBar().showMessage(f"Error: Need at least 3 points for hull in {dataset_name}")
            logger.warning(f"Skipping hull for {dataset_name}: needs >= 3 points.")
            return False # Indicate error or skip
        
        try:
            # Get points
            points = dataset['points']
            
            # For 3D computations, we need to handle the hull differently
            if points.shape[1] >= 3:
                # Check if points are roughly planar (most real-world datasets are)
                # 1. Center points
                centroid = np.mean(points, axis=0)
                centered = points - centroid
                
                # 2. Get dominant plane using PCA/SVD
                u, s, vh = np.linalg.svd(centered, full_matrices=False)
                # The smallest singular value/component tells us how planar the data is
                normal = vh[2]  # This is the approximate normal to the best-fitting plane
                
                # For highly planar data (ratio of smallest to second smallest singular value is small)
                if s[2] / s[1] < 0.2:  # Threshold can be adjusted
                    logger.debug(f"Points in {dataset_name} are mostly planar, using projected hull")
                    # Project points onto a plane defined by first two principal components
                    projected = np.dot(centered, vh[:2].T)
                    
                    # Compute 2D convex hull on projected points
                    from scipy.spatial import ConvexHull
                    hull_2d = ConvexHull(projected)
                    
                    # Map hull vertices back to 3D
                    hull_indices = hull_2d.vertices
                    hull_points = points[hull_indices]
                    
                    # Ensure the hull is closed
                    if not np.array_equal(hull_points[0], hull_points[-1]):
                        hull_points = np.vstack([hull_points, hull_points[0]])
                else:
                    # For truly 3D point clouds, use 3D convex hull
                    logger.debug(f"Points in {dataset_name} are 3D, using 3D convex hull")
                    from scipy.spatial import ConvexHull
                    hull_3d = ConvexHull(points)
                    
                    # Extract unique vertices from the hull
                    # 3D hulls use triangular faces, so we need to get unique vertices
                    hull_indices = []
                    for simplex in hull_3d.simplices:
                        for idx in simplex:
                            if idx not in hull_indices:
                                hull_indices.append(idx)
                    
                    # Extract hull points in original order
                    hull_points = points[hull_indices]
                    
                    # For visualization purposes, we'll close the hull by adding the first point again
                    hull_points = np.vstack([hull_points, hull_points[0]])
            else:
                # Standard 2D hull computation
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points)
                
                # Extract hull points
                hull_points = points[hull.vertices]
                
                # Close the hull
                hull_points = np.append(hull_points, [hull_points[0]], axis=0)
            
            # Store in dataset
            dataset['hull_points'] = hull_points
            
            # Clear any previous results from later steps for this dataset
            dataset.pop('segments', None)
            dataset.pop('triangulation_result', None)
            
            return True # Indicate success
            
        except Exception as e:
            # self.statusBar().showMessage(f"Error computing hull for {dataset_name}: {str(e)}") # Remove status update
            logger.error(f"Error computing hull for {dataset_name}: {str(e)}")
            # Optionally show message box for individual errors during batch?
            # QMessageBox.critical(self, "Hull Error", f"Error computing hull for {dataset_name}: {str(e)}")
            return False # Indicate error

    def compute_hull(self):
        """Compute the convex hull of the loaded points for the *active* dataset (primarily for context menu)"""
        # Check if we have an active dataset
        if self.current_dataset_index < 0 or self.current_dataset_index >= len(self.datasets):
            self.statusBar().showMessage("Error: No active dataset selected")
            QMessageBox.critical(self, "Error", "No active dataset selected")
            return

        self.statusBar().showMessage(f"Computing hull for {self.datasets[self.current_dataset_index]['name']}...") # Add status message here for single run
        success = self._compute_hull_for_dataset(self.current_dataset_index)

        if success:
            # Update statistics and visualization after computing for the active one
            self._update_statistics()
            self._visualize_all_hulls()
            self._update_visualization() # Ensure other dependent views are updated/cleared
            self.notebook.setCurrentIndex(1)  # Switch to hull tab
            self.statusBar().showMessage(f"Computed hull for {self.datasets[self.current_dataset_index]['name']}") # Add success message here

    def compute_all_hulls(self):
        """Compute the convex hull for all loaded datasets using a worker thread."""
        if not self.datasets:
            QMessageBox.information(self, "No Data", "No datasets loaded to compute hulls.")
            return
        self._run_batch_computation("hulls", len(self.datasets))
    
        
    def _update_height_in_plotter(self, height_factor):
        """Update the height factor in the current plotter"""
        # Store the new height factor
        self.height_factor = height_factor
        
        # The height factor is used to exaggerate the Z dimension for better visualization
        # If we have a current plotter, update it
        if hasattr(self, 'current_plotter') and self.current_plotter is not None:
            try:
                # Scale the Z dimension using plotter's scale feature
                # This adjusts how the 3D scene appears without modifying the actual coordinates
                self.current_plotter.set_scale(zscale=height_factor)
                self.current_plotter.reset_camera()
            except Exception as e:
                logger.error(f"Error updating 3D Z exaggeration: {str(e)}")
    
    
    def show_3d_view(self):
        """Show 3D view of all triangulations in a separate window"""
        if not self.view_3d_enabled:
            QMessageBox.information(self, "PyVista Not Available", 
                                 "PyVista is not available. Please install PyVista for 3D visualization.")
            return
            
        # Check if we have any triangulation results
        visible_datasets = [d for d in self.datasets if d.get('visible', True) and d.get('triangulation_result') is not None]
        
        if not visible_datasets:
            QMessageBox.information(self, "No Triangulation", 
                                 "No triangulation results available. Please run triangulation first.")
            return
            
        # Close previous plotter if it exists
        if hasattr(self, 'pv_plotter') and self.pv_plotter is not None:
            try:
                self.pv_plotter.close()
            except:
                pass
        
        # Create a new plotter
        self.pv_plotter = pv.Plotter(window_size=[800, 600], title="MeshIt 3D Visualization")
        self.pv_plotter.set_background("#383F51")  # Dark blue background
        
        # Show each dataset
        for dataset in visible_datasets:
            color = dataset.get('color', '#000000')
            name = dataset.get('name', 'Unnamed')
            
            triangulation_result = dataset['triangulation_result']
            vertices = triangulation_result['vertices']
            triangles = triangulation_result['triangles']
            
            # Ensure vertices are 3D
            vertices_3d = np.zeros((len(vertices), 3))
            if vertices.shape[1] == 2:
                # If 2D vertices, try to get Z from original points
                vertices_3d[:, 0:2] = vertices
                
                # If the dataset has 3D points, use them to set Z values based on XY proximity
                original_points = dataset.get('points')
                if original_points is not None and original_points.shape[1] >= 3:
                    for i in range(len(vertices)):
                        vertex_2d = vertices[i]
                        # Find the closest original point
                        distances = np.sum((original_points[:, 0:2] - vertex_2d)**2, axis=1)
                        closest_idx = np.argmin(distances)
                        vertices_3d[i, 2] = original_points[closest_idx, 2]
            else:
                # If vertices already have Z, use them directly
                vertices_3d = vertices.copy()
            
            # Create a mesh
            cells = np.hstack([np.full((len(triangles), 1), 3), triangles])
            mesh = pv.PolyData(vertices_3d, cells)
            
            # Add mesh to plotter
            self.pv_plotter.add_mesh(mesh, color=color, opacity=0.8, 
                                  show_edges=True, edge_color=color, 
                                  specular=0.5, smooth_shading=True,
                                  name=name)  # Add name to identify in plotter
        
        # Add text with dataset information
        text_lines = []
        for i, dataset in enumerate(visible_datasets):
            name = dataset.get('name', 'Unnamed')
            num_vertices = len(dataset['triangulation_result']['vertices'])
            num_triangles = len(dataset['triangulation_result']['triangles'])
            text_lines.append(f"{i+1}. {name}: {num_vertices} vertices, {num_triangles} triangles")
        
        if text_lines:
            info_text = '\n'.join(text_lines)
            self.pv_plotter.add_text(info_text, font_size=10, position='upper_left')
        
        # Add UI controls for adjusting display
        self.pv_plotter.add_checkbox_button_widget(
            lambda state: [self.pv_plotter.add_axes() if state else self.pv_plotter.remove_axes()],
            value=True,
            position=(10, 180),
            size=30,
            border_size=1,
            color_on='white',
            color_off='grey',
            background_color='darkblue'
        )
        self.pv_plotter.add_text("Axes", position=(50, 180), font_size=10, color='white')
        
        # Add scale slider for height
        def update_height(value):
            self._set_height_factor(value)
            self.show_3d_view()  # Refresh the view
            
        # Add axes for reference
        self.pv_plotter.add_axes()
        
        # Show the plotter in a separate window
        self.pv_plotter.show()
    
    def _set_height_factor(self, height):
        """Set the height factor for 3D visualization"""
        self.height_factor = height
        self.height_factor_slider.setValue(int(height * 100))
        
        # Update existing visualization if any
        if hasattr(self, 'triangulation_result') and self.triangulation_result is not None:
            vertices = self.triangulation_result['vertices']
            triangles = self.triangulation_result['triangles']
            self._plot_triangulation(vertices, triangles, self.hull_points)
    
    def _show_about(self):
        """Show about dialog"""
        about_text = """
MeshIt Workflow GUI

A graphical interface for the complete MeshIt workflow,
including file loading, convex hull computation, 
segmentation, triangulation, and visualization.

3D Visualization: {status}
"""
        
        status = "Enabled" if self.view_3d_enabled else "Disabled (Install PyVista)"
        about_text = about_text.format(status=status)
        
        QMessageBox.information(self, "About MeshIt Workflow GUI", about_text)

    def _compute_segments_for_dataset(self, dataset_index):
        """Compute the segmentation for a specific dataset index. Returns True on success, False on error."""
        # Check index validity
        if not (0 <= dataset_index < len(self.datasets)):
            logger.error(f"Invalid dataset index {dataset_index} for segmentation.")
            return False

        dataset = self.datasets[dataset_index]
        dataset_name = dataset.get('name', f"Dataset {dataset_index}")
        
        if dataset.get('hull_points') is None or len(dataset['hull_points']) < 4:  # 3 vertices + 1 closing point
            self.statusBar().showMessage(f"Skipping segments for {dataset_name}: Compute convex hull first")
            logger.warning(f"Skipping segments for {dataset_name}: hull not computed.")
            return False # Indicate error or skip

        # --- START EDIT: Get target size from new control ---
        # Get the segmentation parameter from the UI
        try:
            effective_segment_length = float(self.target_feature_size_input.value())
            if effective_segment_length <= 1e-6: # Use a small threshold instead of zero
                logger.warning("Target Feature Size is too small, using default value (1.0).")
                effective_segment_length = 1.0
        except ValueError:
            logger.warning("Invalid Target Feature Size, using default value (1.0).")
            effective_segment_length = 1.0
        # Remove old parameter calculations
        # try:
        #     segment_length = float(self.segment_length_input.text())
        #     if segment_length <= 0:
        #         segment_length = 1.0
        # except ValueError:
        #     segment_length = 1.0
        #
        # density_factor = self.segment_density_slider.value() / 100.0
        # effective_segment_length = segment_length / density_factor
        # --- END EDIT ---

        try:
            # Extract the hull boundary (excluding the closing point)
            hull_boundary = dataset['hull_points'][:-1]
            hull_size = len(hull_boundary)
            
            # Performance optimization: Limit the number of segments per edge to prevent excessive calculations
            MAX_SEGMENTS_PER_EDGE = 20
            
            # Estimate average edge length to determine a reasonable minimum segment length
            total_perimeter = 0
            for i in range(hull_size):
                p1 = hull_boundary[i]
                p2 = hull_boundary[(i + 1) % hull_size]
                total_perimeter += np.linalg.norm(p2 - p1)
            
            avg_edge_length = total_perimeter / hull_size
            min_segment_length = avg_edge_length / MAX_SEGMENTS_PER_EDGE
            
            # Use the larger of calculated min_segment_length and effective_segment_length
            effective_segment_length = max(effective_segment_length, min_segment_length)
            
            # Pre-allocate segments list with reasonable capacity
            estimated_segments = int(total_perimeter / effective_segment_length) + hull_size
            segments = []
            
            # Generate segments along the hull boundary with uniform distribution
            for i in range(hull_size):
                p1 = hull_boundary[i]
                p2 = hull_boundary[(i + 1) % hull_size]
                
                # Compute edge length
                dist = np.linalg.norm(p2 - p1)
                
                # Limit number of segments per edge
                num_segments = min(max(1, int(np.ceil(dist / effective_segment_length))), MAX_SEGMENTS_PER_EDGE)
                
                # Vectorized segment creation for this edge
                t_values = np.linspace(0, 1, num_segments + 1)
                edge_segments = []
                
                for j in range(num_segments):
                    t1, t2 = t_values[j], t_values[j+1]
                    segment_start = p1 + t1 * (p2 - p1)
                    segment_end = p1 + t2 * (p2 - p1)
                    segments.append([segment_start, segment_end])
            
            # Store the segments in the dataset - using a normal list for better performance
            dataset['segments'] = segments
            
            # Clear any previous results from later steps for this dataset
            dataset.pop('triangulation_result', None)

            return True # Indicate success

        except Exception as e:
            logger.error(f"Error computing segments for {dataset_name}: {str(e)}")
            return False # Indicate error

    def compute_segments(self):
        """Compute the segmentation of the convex hull for the *active* dataset (primarily for context menu)"""
        # Check if we have an active dataset
        if self.current_dataset_index < 0 or self.current_dataset_index >= len(self.datasets):
            self.statusBar().showMessage("Error: No active dataset selected")
            QMessageBox.critical(self, "Error", "No active dataset selected")
            return

        self.statusBar().showMessage(f"Computing segments for {self.datasets[self.current_dataset_index]['name']}...") # Add status message here for single run
        success = self._compute_segments_for_dataset(self.current_dataset_index)

        if success:
            # Update statistics and visualization after computing for the active one
            self._update_statistics()
            self._visualize_all_segments()
            self._update_visualization() # Ensure other dependent views are updated/cleared
            self.notebook.setCurrentIndex(2)  # Switch to segment tab
            self.statusBar().showMessage(f"Computed segments for {self.datasets[self.current_dataset_index]['name']}") # Add success message here

    def compute_all_segments(self):
        """Compute segmentation for all datasets that have hulls using a worker thread."""
        datasets_with_hulls_indices = [i for i, d in enumerate(self.datasets) if d.get('hull_points') is not None]
        if not datasets_with_hulls_indices:
            QMessageBox.information(self, "No Hulls", "No datasets have computed hulls. Please compute hulls first.")
            return
        self._run_batch_computation("segments", len(datasets_with_hulls_indices))
    
    
    def _run_triangulation_for_dataset(self, dataset_index):
        """Run triangulation for a specific dataset index. Returns True on success, False on error."""
        # Check index validity
        if not (0 <= dataset_index < len(self.datasets)):
            logger.error(f"Invalid dataset index {dataset_index} for triangulation.")
            return False

        dataset = self.datasets[dataset_index]
        dataset_name = dataset.get('name', f"Dataset {dataset_index}")

        segments_data = dataset.get('segments')
        if segments_data is None or len(segments_data) < 3:
            logger.warning(f"Skipping triangulation for {dataset_name}: segments not computed.")
            return False

        # Get triangulation parameters from GUI
        gradient = self.gradient_input.value()
        min_angle = self.min_angle_input.value()
        uniform = self.uniform_checkbox.isChecked()

        # --- START EDIT: Get base_size from the unified control ---
        try:
            base_size = float(self.target_feature_size_input.value())
            if base_size <= 1e-6:
                logger.warning("Target Feature Size (used as base_size) is too small, using default (1.0).")
                base_size = 1.0
        except ValueError:
            logger.warning("Invalid Target Feature Size (used as base_size), using default (1.0).")
            base_size = 1.0
        # Remove old target_edge_length reading
        # target_edge_length = self.target_edge_length_input.value()
        # base_size = max(1e-9, target_edge_length)
        # --- END EDIT ---

        try:
            start_time = time.time()

            # Extract unique points and segment indices from segments
            segment_points = []
            for segment in segments_data:
                segment_points.append(segment[0])
                segment_points.append(segment[1])

            unique_points_list = []
            point_to_index = {}
            segment_indices = []

            current_index = 0
            for segment in segments_data:
                start_pt, end_pt = segment[0], segment[1]
                start_tuple = tuple(start_pt)
                end_tuple = tuple(end_pt)

                if start_tuple not in point_to_index:
                    point_to_index[start_tuple] = current_index
                    unique_points_list.append(start_pt)
                    start_idx = current_index
                    current_index += 1
                else:
                    start_idx = point_to_index[start_tuple]

                if end_tuple not in point_to_index:
                    point_to_index[end_tuple] = current_index
                    unique_points_list.append(end_pt)
                    end_idx = current_index
                    current_index += 1
                else:
                    end_idx = point_to_index[end_tuple]

                segment_indices.append([start_idx, end_idx])

            all_boundary_points = np.array(unique_points_list)
            boundary_segments_indices = np.array(segment_indices)
            logger.info(f"Using {len(all_boundary_points)} unique points and {len(boundary_segments_indices)} segments from segmentation step for triangulation.")

            # Projection and reconstruction logic
            all_dataset_points = np.array(dataset['points'])
            plane_normal = None
            projected_boundary_points = all_boundary_points.copy()

            if all_dataset_points.shape[1] > 2:
                logger.info(f"Projecting ALL 3D points to best-fit plane for triangulation for {dataset_name}")
                
                # Find best-fitting plane using ALL points
                centroid = np.mean(all_dataset_points, axis=0)
                centered_all_points = all_dataset_points - centroid
                u, s, vh = np.linalg.svd(centered_all_points, full_matrices=False)
                projection_basis = vh[:2]
                plane_normal = vh[2]
                
                # Project ALL points to 2D
                all_points_2d = np.dot(centered_all_points, projection_basis.T)
                
                # Project boundary points using the same projection
                centered_boundary = all_boundary_points - centroid
                boundary_points_2d = np.dot(centered_boundary, projection_basis.T)
                
                # Store projection data
                dataset['original_points_3d'] = all_dataset_points.copy()
                dataset['projected_points_2d'] = all_points_2d
                projected_boundary_points = boundary_points_2d
                
                dataset['projection_params'] = {
                    'centroid': centroid,
                    'basis': projection_basis,
                    'normal': plane_normal,
                    'original_points': all_boundary_points.copy(),
                    'all_original_points': all_dataset_points.copy()
                }
            else:
                projected_boundary_points = all_boundary_points
                dataset['projection_params'] = None

            # Run triangulation
            if not HAVE_DIRECT_WRAPPER:
                logger.error("DirectTriangleWrapper not available!")
                raise ImportError("DirectTriangleWrapper failed to import.")

            logger.info(f"Triangulating {dataset_name} with Target Edge Length = {base_size:.4f}")

            # Initialize triangulator with mesh quality parameters
            triangulator = DirectTriangleWrapper(
                gradient=gradient,
                min_angle=min_angle,
                base_size=base_size
            )

            # Set up triangulation with enhanced transition feature points
            triangulation_result = triangulator.triangulate(
                points=projected_boundary_points,
                segments=boundary_segments_indices,
                uniform=uniform,
                create_transition=True  # Enable enhanced transition point generation
            )

            if triangulation_result is None or 'vertices' not in triangulation_result or 'triangles' not in triangulation_result:
                raise ValueError("Triangulation failed to produce valid output.")

            vertices_2d = triangulation_result['vertices']
            triangles = triangulation_result['triangles']

            # Reconstruction logic
            if dataset['projection_params'] is not None:
                projection_params = dataset['projection_params']
                centroid = projection_params['centroid']
                basis = projection_params['basis']
                normal = projection_params.get('normal')
                original_boundary_points = projection_params['original_points']
                all_original_points = projection_params.get('all_original_points')
                all_projected_points = dataset.get('projected_points_2d')

                can_calculate_planar_z = normal is not None and abs(normal[2]) > 1e-9
                final_vertices_3d = np.zeros((len(vertices_2d), 3))
                
                for i, vertex_2d in enumerate(vertices_2d):
                    is_matched_point = False
                    
                    # Try to match with all points first
                    if all_original_points is not None and all_projected_points is not None:
                        for j, proj_pt in enumerate(all_projected_points):
                            if np.allclose(vertex_2d, proj_pt, atol=1e-10):
                                if j < len(all_original_points):
                                    final_vertices_3d[i] = all_original_points[j]
                                    is_matched_point = True
                                    break
                    
                    # Try boundary points if no match found
                    if not is_matched_point:
                        for j, bp_2d in enumerate(projected_boundary_points):
                            if np.allclose(vertex_2d, bp_2d, atol=1e-10):
                                if j < len(original_boundary_points):
                                    final_vertices_3d[i] = original_boundary_points[j]
                                    is_matched_point = True
                                    break
                    
                    # Calculate z using plane equation if still no match
                    if not is_matched_point:
                        # --- START EDIT: Replace planar Z with interpolation ---
                        # Get original 3D points and their 2D projections from stored params
                        original_3d = projection_params.get('all_original_points')
                        projected_2d = dataset.get('projected_points_2d')

                        if original_3d is not None and projected_2d is not None and len(original_3d) > 0:
                            # Extract original Z values
                            original_z = original_3d[:, 2]
                            
                            # Interpolate Z value for the new 2D point (vertex_2d)
                            # Use linear interpolation, fallback to nearest if linear fails 
                            # (e.g., outside convex hull of original projected points)
                            interpolated_z = griddata(projected_2d, original_z, vertex_2d, method='linear')
                            if np.isnan(interpolated_z):
                                interpolated_z = griddata(projected_2d, original_z, vertex_2d, method='nearest')

                            # Check if interpolation was successful (might still be NaN if projected_2d is degenerate)
                            if np.isnan(interpolated_z):
                                logger.warning(f"Interpolation failed for point {vertex_2d}. Falling back to centroid Z.")
                                interpolated_z = centroid[2] # Fallback Z

                            # --- START EDIT: Ensure scalar float ---
                            interpolated_z = float(interpolated_z)
                            # --- END EDIT ---
                            
                            # Reconstruct the 3D point using the interpolated Z
                            vertex_3d_reconstructed = centroid.copy()
                            vertex_3d_reconstructed += vertex_2d[0] * basis[0]
                            vertex_3d_reconstructed += vertex_2d[1] * basis[1]
                            vertex_3d_reconstructed[2] = interpolated_z # Use interpolated Z
                            
                            final_vertices_3d[i] = vertex_3d_reconstructed
                        else:
                            # Fallback if original points or projections are missing
                            logger.warning(f"Original 3D points or 2D projections missing for interpolation. Falling back.")
                            # Fallback to original planar Z calculation (or just centroid Z)
                            vertex_3d_on_plane = centroid.copy()
                            vertex_3d_on_plane += vertex_2d[0] * basis[0]
                            vertex_3d_on_plane += vertex_2d[1] * basis[1]
                            if can_calculate_planar_z:
                                z_planar = centroid[2] - (normal[0]*(vertex_3d_on_plane[0] - centroid[0]) + 
                                                        normal[1]*(vertex_3d_on_plane[1] - centroid[1])) / normal[2]
                                vertex_3d_on_plane[2] = z_planar
                            else:
                                vertex_3d_on_plane[2] = centroid[2] # Use centroid Z if planar calc fails
                            final_vertices_3d[i] = vertex_3d_on_plane
                        # --- END EDIT ---
                
                final_vertices = final_vertices_3d
            else:
                if vertices_2d.shape[1] == 2:
                    final_vertices = np.zeros((len(vertices_2d), 3))
                    final_vertices[:, :2] = vertices_2d
                else:
                    final_vertices = vertices_2d

            # Store results
            dataset['triangulation_result'] = {
                'vertices': final_vertices,
                'triangles': triangles,
                'uniform': uniform,
                'gradient': gradient,
                'min_angle': min_angle,
                'target_edge_length': base_size,
            }
            logger.info(f"Triangulation for {dataset_name} completed. Vertices: {len(final_vertices)}, Triangles: {len(triangles)}")
            return True

        except Exception as e:
            logger.error(f"Error triangulating {dataset_name}: {str(e)}")
            import traceback
            logger.debug(f"Triangulation error traceback: {traceback.format_exc()}")
            return False

    def run_triangulation(self):
        """Run triangulation on the segments and points of the *active* dataset (primarily for context menu)"""
        # Check if we have an active dataset
        if self.current_dataset_index < 0 or self.current_dataset_index >= len(self.datasets):
            self.statusBar().showMessage("Error: No active dataset selected")
            QMessageBox.critical(self, "Error", "No active dataset selected")
            return

        self.statusBar().showMessage(f"Running triangulation for {self.datasets[self.current_dataset_index]['name']}...") # Add status message here for single run
        success = self._run_triangulation_for_dataset(self.current_dataset_index)

        if success:
            # Update statistics and visualization after computing for the active one
            self._update_statistics()
            self._visualize_all_triangulations()
            self._update_visualization() # Ensure other dependent views are updated/cleared
            self.notebook.setCurrentIndex(3)  # Switch to triangulation tab
            self.statusBar().showMessage(f"Completed triangulation for {self.datasets[self.current_dataset_index]['name']}") # Add success message here

    def run_all_triangulations(self):
        """Run triangulation for all datasets that have segments using a worker thread."""
        datasets_with_segments_indices = [i for i, d in enumerate(self.datasets) if d.get('segments') is not None]
        if not datasets_with_segments_indices:
            QMessageBox.information(self, "No Segments", "No datasets have computed segments. Please compute segments first.")
            return
        self._run_batch_computation("triangulations", len(datasets_with_segments_indices))
    
    def export_results(self):
        """Export the triangulation results to a file for the active dataset"""
        self.statusBar().showMessage("Exporting results...")
        
        # Check if we have an active dataset
        if self.current_dataset_index < 0 or self.current_dataset_index >= len(self.datasets):
            self.statusBar().showMessage("Error: No active dataset selected")
            QMessageBox.critical(self, "Error", "No active dataset selected")
            return
        
        # Get active dataset
        dataset = self.datasets[self.current_dataset_index]
        
        if dataset.get('triangulation_result') is None:
            self.statusBar().showMessage("Error: No triangulation results to export")
            QMessageBox.critical(self, "Error", "No triangulation results to export")
            return
        
        # Create export options dialog
        export_dialog = QDialog(self)
        export_dialog.setWindowTitle("Export Options")
        export_dialog.setMinimumWidth(400)
        dialog_layout = QVBoxLayout(export_dialog)
        
        # Export format frame
        format_group = QGroupBox("Export Format")
        format_layout = QVBoxLayout(format_group)
        
        format_radio_group = QButtonGroup(export_dialog)
        
        obj_radio = QRadioButton("OBJ Format (3D objects)")
        ply_radio = QRadioButton("PLY Format (3D mesh)")
        stl_radio = QRadioButton("STL Format (3D printing)")
        csv_radio = QRadioButton("CSV Format (raw data)")
        
        format_radio_group.addButton(obj_radio, 0)
        format_radio_group.addButton(ply_radio, 1)
        format_radio_group.addButton(stl_radio, 2)
        format_radio_group.addButton(csv_radio, 3)
        
        obj_radio.setChecked(True)  # Default selection
        
        format_layout.addWidget(obj_radio)
        format_layout.addWidget(ply_radio)
        format_layout.addWidget(stl_radio)
        format_layout.addWidget(csv_radio)
        
        dialog_layout.addWidget(format_group)
        
        # 3D export options (enabled only if PyVista is available)
        options_group = QGroupBox("3D Export Options")
        options_layout = QGridLayout(options_group)
        
        export_3d_checkbox = QCheckBox("Export as 3D Surface")
        export_3d_checkbox.setChecked(HAVE_PYVISTA)
        export_3d_checkbox.setEnabled(HAVE_PYVISTA)
        options_layout.addWidget(export_3d_checkbox, 0, 0, 1, 2)
        
        options_layout.addWidget(QLabel("Height Scale:"), 1, 0)
        height_scale_slider = QSlider(Qt.Horizontal)
        height_scale_slider.setMinimum(0)
        height_scale_slider.setMaximum(100)
        height_scale_slider.setValue(int(self.height_factor * 100))
        options_layout.addWidget(height_scale_slider, 1, 1)
        
        dialog_layout.addWidget(options_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        export_btn = QPushButton("Export")
        
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(export_btn)
        
        dialog_layout.addLayout(button_layout)
        
        # Connect signals
        cancel_btn.clicked.connect(export_dialog.reject)
        export_btn.clicked.connect(export_dialog.accept)
        
        # Show dialog and get result
        result = export_dialog.exec_()
        
        if result == QDialog.Accepted:
            # Get selected format
            format_id = format_radio_group.checkedId()
            formats = ["obj", "ply", "stl", "csv"]
            if format_id >= 0 and format_id < len(formats):
                export_format = formats[format_id]
            else:
                export_format = "obj"  # Default
                
            # Get 3D options
            export_3d = export_3d_checkbox.isChecked() and HAVE_PYVISTA
            height_scale = height_scale_slider.value() / 100.0
            
            # Create a file save dialog
            file_filter = ""
            if export_format == "obj":
                file_filter = "OBJ files (*.obj);;All files (*.*)"
            elif export_format == "ply":
                file_filter = "PLY files (*.ply);;All files (*.*)"
            elif export_format == "stl":
                file_filter = "STL files (*.stl);;All files (*.*)"
            elif export_format == "csv":
                file_filter = "CSV files (*.csv);;All files (*.*)"
                
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save triangulation results",
                "",
                file_filter,
                options=QFileDialog.Options()
            )
            
            if not file_path:
                self.statusBar().showMessage("Export canceled")
                return
            
            try:
                vertices = dataset['triangulation_result']['vertices']
                triangles = dataset['triangulation_result']['triangles']
                
                # If 3D export is selected and PyVista is available
                if export_3d and HAVE_PYVISTA:
                    try:
                        # Create 3D vertices with Z-values
                        points_3d = np.zeros((len(vertices), 3))
                        points_3d[:, 0] = vertices[:, 0]  # X remains the same
                        points_3d[:, 1] = vertices[:, 1]  # Y remains the same
                        
                        # Calculate Z values (same as in show_3d_view)
                        x_scale = 0.05
                        y_scale = 0.05
                        z_scale = 20.0 * height_scale
                        
                        for i in range(len(vertices)):
                            x, y = vertices[i]
                            z = np.sin(x * x_scale) * np.cos(y * y_scale) * z_scale
                            points_3d[i, 2] = z
                        
                        # Create PyVista mesh
                        cells = np.hstack([np.full((len(triangles), 1), 3), triangles])
                        mesh = pv.PolyData(points_3d, cells)
                        
                        # Save based on format
                        if export_format == "obj":
                            mesh.save(file_path)
                            self.statusBar().showMessage(f"Exported 3D OBJ file to {file_path}")
                        elif export_format == "ply":
                            mesh.save(file_path)
                            self.statusBar().showMessage(f"Exported 3D PLY file to {file_path}")
                        elif export_format == "stl":
                            mesh.save(file_path)
                            self.statusBar().showMessage(f"Exported 3D STL file to {file_path}")
                        elif export_format == "csv":
                            # For CSV, fall back to standard CSV export
                            self._export_csv(file_path, points_3d, triangles)
                            self.statusBar().showMessage(f"Exported 3D CSV file to {file_path}")
                    except Exception as e:
                        self.statusBar().showMessage(f"Error creating 3D export: {str(e)}")
                        logger.error(f"Error creating 3D export: {str(e)}")
                        QMessageBox.critical(self, "Export Error", f"Error creating 3D export: {str(e)}")
                else:
                    # Standard 2D format exports
                    if export_format == "obj":
                        self._export_obj(file_path, vertices, triangles)
                        self.statusBar().showMessage(f"Exported 2D OBJ file to {file_path}")
                    elif export_format == "ply":
                        self._export_ply(file_path, vertices, triangles)
                        self.statusBar().showMessage(f"Exported 2D PLY file to {file_path}")
                    elif export_format == "stl":
                        self._export_stl(file_path, vertices, triangles)
                        self.statusBar().showMessage(f"Exported 2D STL file to {file_path}")
                    elif export_format == "csv":
                        self._export_csv(file_path, vertices, triangles)
                        self.statusBar().showMessage(f"Exported 2D CSV file to {file_path}")
                
            except Exception as e:
                self.statusBar().showMessage(f"Error exporting results: {str(e)}")
                logger.error(f"Error exporting results: {str(e)}")
                QMessageBox.critical(self, "Error", f"Error exporting results: {str(e)}")
        else:
            self.statusBar().showMessage("Export canceled")
    
    def _export_obj(self, file_path, vertices, triangles):
        """Export to OBJ format"""
        with open(file_path, 'w') as f:
            # Write header
            f.write("# OBJ file created by MeshIt Workflow GUI\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Triangles: {len(triangles)}\n\n")
            
            # Write vertices
            for vertex in vertices:
                # Check if vertex has Z coordinate
                if vertex.shape[0] >= 3:
                    f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                else:
                    # If only 2D, use 0 for Z
                    f.write(f"v {vertex[0]} {vertex[1]} 0.0\n")
            
            # Write triangles
            for triangle in triangles:
                # OBJ indices start from 1, not 0
                f.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")
    
    def _export_ply(self, file_path, vertices, triangles):
        """Export to PLY format"""
        with open(file_path, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(triangles)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices
            for vertex in vertices:
                # Check if vertex has Z coordinate
                if vertex.shape[0] >= 3:
                    f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
                else:
                    # If only 2D, use 0 for Z
                    f.write(f"{vertex[0]} {vertex[1]} 0.0\n")
            
            # Write triangles
            for triangle in triangles:
                f.write(f"3 {triangle[0]} {triangle[1]} {triangle[2]}\n")
    
    def _export_stl(self, file_path, vertices, triangles):
        """Export to STL format"""
        with open(file_path, 'w') as f:
            # Write header
            f.write("solid meshit_triangulation\n")
            
            # Write triangles
            for triangle in triangles:
                # Get vertices of this triangle
                v1 = vertices[triangle[0]]
                v2 = vertices[triangle[1]]
                v3 = vertices[triangle[2]]
                
                # Get Z coordinates or default to 0
                z1 = v1[2] if v1.shape[0] >= 3 else 0.0
                z2 = v2[2] if v2.shape[0] >= 3 else 0.0
                z3 = v3[2] if v3.shape[0] >= 3 else 0.0
                
                # Calculate normal (cross product of two sides)
                # For a non-flat mesh, calculate actual normal
                if v1.shape[0] >= 3 and v2.shape[0] >= 3 and v3.shape[0] >= 3:
                    # Simple normal calculation
                    side1 = np.array([v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]])
                    side2 = np.array([v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]])
                    normal = np.cross(side1, side2)
                    # Normalize
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal = normal / norm
                    f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                else:
                    # Default normal for flat mesh
                    f.write(f"  facet normal 0.0 0.0 1.0\n")
                
                f.write(f"    outer loop\n")
                f.write(f"      vertex {v1[0]} {v1[1]} {z1}\n")
                f.write(f"      vertex {v2[0]} {v2[1]} {z2}\n")
                f.write(f"      vertex {v3[0]} {v3[1]} {z3}\n")
                f.write(f"    endloop\n")
                f.write(f"  endfacet\n")
            
            # Write footer
            f.write("endsolid meshit_triangulation\n")
    
    def _export_csv(self, file_path, vertices, triangles):
        """Export to CSV format"""
        with open(file_path, 'w') as f:
            # Write header
            f.write("# MeshIt Workflow GUI triangulation results\n")
            f.write("# VERTICES\n")
            f.write("x,y,z\n")
            
            # Write vertices
            for vertex in vertices:
                if vertex.shape[0] == 3:  # 3D vertex
                    f.write(f"{vertex[0]},{vertex[1]},{vertex[2]}\n")
                else:  # 2D vertex
                    f.write(f"{vertex[0]},{vertex[1]},0.0\n")
            
            # Write separator
            f.write("\n# TRIANGLES\n")
            f.write("v1,v2,v3\n")
            
            # Write triangles
            for triangle in triangles:
                f.write(f"{triangle[0]},{triangle[1]},{triangle[2]}\n")

    def _show_dataset_context_menu(self, position):
        """Show context menu for dataset list items"""
        if self.dataset_list_widget.count() == 0:
            return
            
        # Get selected item
        selected_items = self.dataset_list_widget.selectedItems()
        if not selected_items:
            return
            
        # Create context menu
        menu = QMenu()
        rename_action = menu.addAction("Rename")
        toggle_action = menu.addAction("Toggle Visibility")
        change_color_action = menu.addAction("Change Color")
        menu.addSeparator()
        
        # Add processing actions
        process_menu = menu.addMenu("Process")
        compute_hull_action = process_menu.addAction("Compute Hull")
        compute_segments_action = process_menu.addAction("Compute Segments")
        compute_triangulation_action = process_menu.addAction("Compute Triangulation")
        process_all_action = process_menu.addAction("Process All Steps")
        
        menu.addSeparator()
        remove_action = menu.addAction("Remove")
        
        # Show menu at position
        action = menu.exec_(self.dataset_list_widget.mapToGlobal(position))
        
        # Handle action
        if not action:
            return
            
        # Get selected dataset index
        selected_index = self.dataset_list_widget.row(selected_items[0])
        
        if action == rename_action:
            self._rename_dataset(selected_index)
        elif action == toggle_action:
            self._toggle_dataset_visibility(selected_index)
        elif action == change_color_action:
            self._change_dataset_color(selected_index)
        elif action == remove_action:
            self._remove_dataset(selected_index)
        elif action == compute_hull_action:
            self.current_dataset_index = selected_index
            self.compute_hull()
        elif action == compute_segments_action:
            self.current_dataset_index = selected_index
            self.compute_segments()
        elif action == compute_triangulation_action:
            self.current_dataset_index = selected_index
            self.run_triangulation()
        elif action == process_all_action:
            self.current_dataset_index = selected_index
            self._process_all_steps(selected_index)
    
    def _process_all_steps(self, dataset_index):
        """Process all steps (hull, segments, triangulation) for a dataset"""
        if not (0 <= dataset_index < len(self.datasets)):
            return
            
        # Set as active dataset
        self.current_dataset_index = dataset_index
        
        # Display processing message
        dataset_name = self.datasets[dataset_index]['name']
        self.statusBar().showMessage(f"Processing all steps for {dataset_name}...")
        
        try:
            # Step 1: Compute hull
            self.compute_hull()
            
            # Step 2: Compute segments
            self.compute_segments()
            
            # Step 3: Run triangulation
            self.run_triangulation()
            
            self.statusBar().showMessage(f"Completed all processing steps for {dataset_name}")
        except Exception as e:
            self.statusBar().showMessage(f"Error processing dataset: {str(e)}")
            logger.error(f"Error during dataset processing: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error processing dataset: {str(e)}")
    
    def process_all_datasets(self):
        """Process all steps for all datasets"""
        if not self.datasets:
            QMessageBox.information(self, "No Datasets", "No datasets available to process.")
            return
            
        num_processed = 0
        total_datasets = len(self.datasets)
        
        self.statusBar().showMessage(f"Processing all steps for {total_datasets} datasets...")
        
        try:
            # Process each dataset
            for i in range(total_datasets):
                dataset_name = self.datasets[i]['name']
                self.statusBar().showMessage(f"Processing {dataset_name} ({i+1}/{total_datasets})...")
                
                # Set as active
                self.current_dataset_index = i
                
                # Compute hull if needed
                if self.datasets[i].get('hull_points') is None:
                    # Call compute hull but catch any exceptions to continue with other datasets
                    try:
                        self.compute_hull()
                    except Exception as e:
                        logger.error(f"Error computing hull for {dataset_name}: {str(e)}")
                        continue
                
                # Compute segments if needed
                if self.datasets[i].get('segments') is None:
                    try:
                        self.compute_segments()
                    except Exception as e:
                        logger.error(f"Error computing segments for {dataset_name}: {str(e)}")
                        continue
                
                # Run triangulation if needed
                if self.datasets[i].get('triangulation_result') is None:
                    try:
                        self.run_triangulation()
                    except Exception as e:
                        logger.error(f"Error triangulating {dataset_name}: {str(e)}")
                        continue
                
                num_processed += 1
            
            # Show completion message
            self.statusBar().showMessage(f"Successfully processed {num_processed} out of {total_datasets} datasets")
            
            # Update visualizations for all
            self._update_visualization()
            
        except Exception as e:
            self.statusBar().showMessage(f"Error in batch processing: {str(e)}")
            logger.error(f"Error during batch processing: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error in batch processing: {str(e)}")
    
    def _on_dataset_selection_changed(self):
        """Handle dataset selection change"""
        selected_items = self.dataset_list_widget.selectedItems()
        if not selected_items:
            return
            
        # Get selected dataset index
        selected_index = self.dataset_list_widget.row(selected_items[0])
        
        # Set as active dataset
        self.current_dataset_index = selected_index
        
        # Update statistics and visualization for the selected dataset
        self._update_statistics()
        self._update_visualization()
        
    def _update_statistics(self):
        """Update statistics labels based on loaded datasets"""
        num_datasets = len(self.datasets)
        self.num_datasets_label.setText(f"Datasets: {num_datasets}")
        
        # Count datasets with each processing step completed
        datasets_with_hull = sum(1 for d in self.datasets if d.get('hull_points') is not None)
        datasets_with_segments = sum(1 for d in self.datasets if d.get('segments') is not None)
        datasets_with_triangulation = sum(1 for d in self.datasets if d.get('triangulation_result') is not None)
        
        # Count intersections
        surface_intersections = 0
        polyline_intersections = 0
        
        if hasattr(self, 'datasets_intersections'):
            for intersections in self.datasets_intersections.values():
                for intersection in intersections:
                    if intersection['is_polyline_mesh']:
                        polyline_intersections += 1
                    else:
                        surface_intersections += 1
        
        # Due to double-counting (each intersection is stored for both datasets),
        # we divide surface-surface intersections count by 2
        unique_surface_intersections = surface_intersections // 2
        triple_point_count = len(self.triple_points) if hasattr(self, 'triple_points') else 0
        
        # Update the status bar with processing statistics
        if hasattr(self, 'status_hulls_label'):
            self.status_hulls_label.setText(f"Hulls: {datasets_with_hull}")
        else:
            self.status_hulls_label = QLabel(f"Hulls: {datasets_with_hull}")
            self.statusBar().addPermanentWidget(self.status_hulls_label)
            
        if hasattr(self, 'status_segments_label'):
            self.status_segments_label.setText(f"Segments: {datasets_with_segments}")
        else:
            self.status_segments_label = QLabel(f"Segments: {datasets_with_segments}")
            self.statusBar().addPermanentWidget(self.status_segments_label)
            
        if hasattr(self, 'status_triangulations_label'):
            self.status_triangulations_label.setText(f"Triangulations: {datasets_with_triangulation}")
        else:
            self.status_triangulations_label = QLabel(f"Triangulations: {datasets_with_triangulation}")
            self.statusBar().addPermanentWidget(self.status_triangulations_label)
            
        if hasattr(self, 'status_intersections_label'):
            self.status_intersections_label.setText(f"Intersections: {unique_surface_intersections + polyline_intersections}")
        else:
            self.status_intersections_label = QLabel(f"Intersections: {unique_surface_intersections + polyline_intersections}")
            self.statusBar().addPermanentWidget(self.status_intersections_label)
        
        if num_datasets == 0:
            self.points_label.setText("Points: 0")
            self.bounds_label.setText("Bounds: N/A")
            self.hull_points_label.setText("Hull vertices: 0")
            self.hull_area_label.setText("Hull area: 0.0")
            self.num_segments_label.setText("Segments: 0")
            self.avg_segment_length_label.setText("Avg length: 0.0")
            self.num_triangles_label.setText("Triangles: 0")
            self.num_vertices_label.setText("Vertices: 0")
            self.mean_edge_label.setText("Mean edge: 0.0")
            self.uniformity_label.setText("Uniformity: 0.0")
            
            # Update intersection statistics if they exist
            if hasattr(self, 'surface_intersection_count_label'):
                self.surface_intersection_count_label.setText("Surface-Surface: 0")
            if hasattr(self, 'polyline_intersection_count_label'):
                self.polyline_intersection_count_label.setText("Polyline-Surface: 0")
            if hasattr(self, 'triple_point_count_label'):
                self.triple_point_count_label.setText("Triple Points: 0")
            return
        
        # Update statistics for the active dataset
        if 0 <= self.current_dataset_index < num_datasets:
            dataset = self.datasets[self.current_dataset_index]
            
            # Points statistics
            if dataset['points'] is not None:
                self.points_label.setText(f"Points: {len(dataset['points'])}")
                
                # Calculate bounds - only use X and Y for display
                min_coords = np.min(dataset['points'], axis=0)
                max_coords = np.max(dataset['points'], axis=0)
                min_x, min_y = min_coords[0], min_coords[1]
                max_x, max_y = max_coords[0], max_coords[1]
                
                # If 3D data is available, also show Z range
                if dataset['points'].shape[1] >= 3:
                    min_z, max_z = min_coords[2], max_coords[2]
                    self.bounds_label.setText(f"Bounds: X[{min_x:.2f},{max_x:.2f}] Y[{min_y:.2f},{max_y:.2f}] Z[{min_z:.2f},{max_z:.2f}]")
                else:
                    self.bounds_label.setText(f"Bounds: X[{min_x:.2f},{max_x:.2f}] Y[{min_y:.2f},{max_y:.2f}]")
            else:
                self.points_label.setText("Points: 0")
                self.bounds_label.setText("Bounds: N/A")
            
            # Hull statistics
            if dataset.get('hull_points') is not None:
                hull_points = dataset['hull_points']
                self.hull_points_label.setText(f"Hull vertices: {len(hull_points)-1}")  # -1 for the closing point
                
                # Calculate hull area (approximate)
                area = 0.0
                for i in range(len(hull_points)-1):
                    # Extract x,y coordinates only
                    x1, y1 = hull_points[i][0], hull_points[i][1]
                    x2, y2 = hull_points[i+1][0], hull_points[i+1][1]
                    area += x1*y2 - x2*y1
                area = abs(area) / 2.0
                self.hull_area_label.setText(f"Hull area: {area:.2f}")
            else:
                self.hull_points_label.setText("Hull vertices: 0")
                self.hull_area_label.setText("Hull area: 0.0")
            
            # Segment statistics
            if dataset.get('segments') is not None:
                segments = dataset['segments']
                self.num_segments_label.setText(f"Segments: {len(segments)}")
                
                # Calculate average segment length
                segment_lengths = [np.linalg.norm(segment[1] - segment[0]) for segment in segments]
                avg_length = np.mean(segment_lengths)
                self.avg_segment_length_label.setText(f"Avg length: {avg_length:.2f}")
            else:
                self.num_segments_label.setText("Segments: 0")
                self.avg_segment_length_label.setText("Avg length: 0.0")
            
            # Triangulation statistics
            if dataset.get('triangulation_result') is not None:
                triangulation = dataset['triangulation_result']
                vertices = triangulation['vertices']
                triangles = triangulation['triangles']
                
                self.num_triangles_label.setText(f"Triangles: {len(triangles)}")
                self.num_vertices_label.setText(f"Vertices: {len(vertices)}")
                
                # Calculate edge lengths and statistics
                edge_lengths = []
                for tri in triangles:
                    v1, v2, v3 = tri
                    p1, p2, p3 = vertices[v1], vertices[v2], vertices[v3]
                    
                    edge_lengths.extend([
                        np.linalg.norm(p2 - p1),
                        np.linalg.norm(p3 - p2),
                        np.linalg.norm(p1 - p3)
                    ])
                
                if edge_lengths:
                    mean_edge = np.mean(edge_lengths)
                    edge_std = np.std(edge_lengths)
                    uniformity = edge_std / mean_edge if mean_edge > 0 else 0
                    
                    self.mean_edge_label.setText(f"Mean edge: {mean_edge:.4f}")
                    self.uniformity_label.setText(f"Uniformity: {uniformity:.4f}")
                else:
                    self.mean_edge_label.setText("Mean edge: 0.0")
                    self.uniformity_label.setText("Uniformity: 0.0")
            else:
                self.num_triangles_label.setText("Triangles: 0")
                self.num_vertices_label.setText("Vertices: 0")
                self.mean_edge_label.setText("Mean edge: 0.0")
                self.uniformity_label.setText("Uniformity: 0.0")
            
            # Intersection statistics
            if hasattr(self, 'surface_intersection_count_label'):
                # Count dataset-specific intersections
                dataset_surface_intersections = 0
                dataset_polyline_intersections = 0
                
                if hasattr(self, 'datasets_intersections') and self.current_dataset_index in self.datasets_intersections:
                    for intersection in self.datasets_intersections[self.current_dataset_index]:
                        if intersection['is_polyline_mesh']:
                            dataset_polyline_intersections += 1
                        else:
                            dataset_surface_intersections += 1
                
                # Update UI with dataset-specific intersection counts
                self.surface_intersection_count_label.setText(f"Surface-Surface: {dataset_surface_intersections}")
                self.polyline_intersection_count_label.setText(f"Polyline-Surface: {dataset_polyline_intersections}")
                self.triple_point_count_label.setText(f"Triple Points: {triple_point_count}")
    
    def _update_dataset_list(self):
        """Update the dataset list widget"""
        # Save current selection
        current_index = self.current_dataset_index
        
        # Clear list
        self.dataset_list_widget.clear()
        
        # Add datasets to list
        for dataset in self.datasets:
            visibility = "✓" if dataset.get('visible', True) else "✗"
            color_square = "■ "
            item_text = f"{color_square}{visibility} {dataset['name']}"
            
            item = QListWidgetItem(item_text)
            
            # Set item color based on dataset color
            color = dataset.get('color', '#000000')
            item.setForeground(QColor(color))
            
            self.dataset_list_widget.addItem(item)
        
        # Restore selection if possible
        if 0 <= current_index < self.dataset_list_widget.count():
            self.dataset_list_widget.setCurrentRow(current_index)
        elif self.dataset_list_widget.count() > 0:
            self.dataset_list_widget.setCurrentRow(0)
            self.current_dataset_index = 0
        else:
            self.current_dataset_index = -1
            
        # Update surface visibility checkboxes if the widget exists
        if hasattr(self, 'surface_checkboxes_layout'):
            self._populate_surface_checkboxes()
    
    def _rename_dataset(self, dataset_index):
        """Rename a dataset"""
        if not (0 <= dataset_index < len(self.datasets)):
            return
            
        dataset = self.datasets[dataset_index]
        old_name = dataset['name']
        
        # Create dialog for renaming
        dialog = QDialog(self)
        dialog.setWindowTitle("Rename Dataset")
        dialog.setFixedSize(300, 100)
        dialog_layout = QVBoxLayout(dialog)
        
        # Add name input
        dialog_layout.addWidget(QLabel("Dataset Name:"))
        name_input = QLineEdit(old_name)
        dialog_layout.addWidget(name_input)
        
        # Add buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        ok_btn = QPushButton("OK")
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(ok_btn)
        dialog_layout.addLayout(button_layout)
        
        # Connect signals
        cancel_btn.clicked.connect(dialog.reject)
        ok_btn.clicked.connect(dialog.accept)
        
        # Show dialog and get result
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            new_name = name_input.text().strip()
            if new_name:
                # Update dataset name
                dataset['name'] = new_name
                
                # Update list
                self._update_dataset_list()
    
    def _toggle_dataset_visibility(self, dataset_index):
        """Toggle visibility of a dataset"""
        if not (0 <= dataset_index < len(self.datasets)):
            return
            
        dataset = self.datasets[dataset_index]
        dataset['visible'] = not dataset.get('visible', True)
        
        # Update list
        self._update_dataset_list()
        
        # Update visualization
        self._update_visualization()
    
    def _change_dataset_color(self, dataset_index):
        """Change color of a dataset"""
        if not (0 <= dataset_index < len(self.datasets)):
            return
            
        dataset = self.datasets[dataset_index]
        old_color = dataset.get('color', '#000000')
        
        # Get new color using color dialog
        new_color = QColorDialog.getColor(QColor(old_color), self, "Select Dataset Color")
        
        if new_color.isValid():
            # Update dataset color
            dataset['color'] = new_color.name()
            
            # Update list
            self._update_dataset_list()
            
            # Update visualization
            self._update_visualization()
    
    def _remove_dataset(self, dataset_index):
        """Remove a dataset"""
        if not (0 <= dataset_index < len(self.datasets)):
            return
            
        # Confirm deletion
        confirm = QMessageBox.question(
            self, 
            "Remove Dataset", 
            f"Are you sure you want to remove dataset '{self.datasets[dataset_index]['name']}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Remove dataset
            self.datasets.pop(dataset_index)
            
            # Update dataset list
            self._update_dataset_list()
            
            # Update statistics and visualization
            self._update_statistics()
            self._update_visualization()
    
    def clear_all_datasets(self):
        """Clear all loaded datasets"""
        if not self.datasets:
            return
            
        # Confirm deletion
        confirm = QMessageBox.question(
            self, 
            "Clear All Datasets", 
            f"Are you sure you want to remove all {len(self.datasets)} datasets?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Clear datasets
            self.datasets = []
            self.current_dataset_index = -1
            
            # Update UI
            self._update_dataset_list()
            self._update_statistics()
            
            # Clear visualizations
            self._clear_visualizations()
            
            self.statusBar().showMessage("All datasets cleared")
    
    def remove_active_dataset(self):
        """Remove the active dataset"""
        if 0 <= self.current_dataset_index < len(self.datasets):
            self._remove_dataset(self.current_dataset_index)
    def _clear_visualizations(self):
        """Clear all visualizations"""
        # ... existing clear calls ...
        self._clear_hull_plot()
        self._clear_segment_plot()
        self._clear_triangulation_plot()
        self._clear_intersection_plot()
        self._clear_refine_mesh_plot() # Add this line
        
        # ... rest of the method ...
        
        # Clear points visualization
        while self.file_viz_layout.count():
            item = self.file_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Reset placeholder
        if HAVE_PYVISTA:
            placeholder_text = "Load data to visualize points in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nLoad data to visualize points in 2D."
            
        self.file_viz_placeholder = QLabel(placeholder_text)
        self.file_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.file_viz_layout.addWidget(self.file_viz_placeholder)
    
    def _update_visualization(self):
        """Update all visualizations based on loaded datasets"""
        # Check if we have any datasets
        if not self.datasets:
            self._clear_visualizations()
            return
        
        # Get visible datasets
        visible_datasets = [d for d in self.datasets if d.get('visible', True)]
        
        if not visible_datasets:
            self._clear_visualizations()
            return
        
        # Update points visualization
        self._visualize_all_points()
        
        # Update hull visualization if we have any hulls
        hulls_exist = any(d.get('hull_points') is not None for d in visible_datasets)
        if hulls_exist:
            self._visualize_all_hulls()
        
        # Update segments visualization if we have any segments
        segments_exist = any(d.get('segments') is not None for d in visible_datasets)
        if segments_exist:
            self._visualize_all_segments()
        
        # Update triangulation visualization if we have any triangulations
        triangulations_exist = any(d.get('triangulation_result') is not None for d in visible_datasets)
        if triangulations_exist:
            self._visualize_all_triangulations()
        
        # Update intersection visualization if we have any intersections
        intersections_exist = hasattr(self, 'datasets_intersections') and bool(self.datasets_intersections)
        if intersections_exist:
            self._visualize_intersections()
    
    def load_file(self):
        """Load a single data file"""
        self.statusBar().showMessage("Loading file...")
        
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a point data file",
            "",
            "Text files (*.txt);;Data files (*.dat);;CSV files (*.csv);;All files (*.*)"
        )
        
        if not file_path:
            self.statusBar().showMessage("File loading canceled")
            return
            
        # Update status with file path
        self.statusBar().showMessage(f"Loading file: {os.path.basename(file_path)}...")
        
        try:
            # Try to read the file
            points = self._read_point_file(file_path)
            
            if points is not None and len(points) > 0:
                # Create a new dataset
                filename = os.path.basename(file_path)
                dataset = {
                    'name': filename,
                    'points': points,
                    'visible': True,
                    'color': self._get_next_color()
                }
                
                # Add to datasets
                self.datasets.append(dataset)
                self.current_dataset_index = len(self.datasets) - 1
                
                # Update UI
                self._update_dataset_list()
                self._update_statistics()
                self._visualize_all_points()
                
                self.statusBar().showMessage(f"Successfully loaded {len(points)} points from {filename}")
            else:
                self.statusBar().showMessage("Error: No valid points found in file")
                QMessageBox.critical(self, "Error", "No valid points found in file")
        
        except Exception as e:
            self.statusBar().showMessage(f"Error loading file: {str(e)}")
            logger.error(f"Error loading file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
    
    def load_multiple_files(self):
        """Load multiple data files"""
        self.statusBar().showMessage("Loading multiple files...")
        
        # Open file dialog
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select point data files",
            "",
            "Text files (*.txt);;Data files (*.dat);;CSV files (*.csv);;All files (*.*)"
        )
        
        if not file_paths:
            self.statusBar().showMessage("File loading canceled")
            return
        
        successful_loads = 0
        
        for file_path in file_paths:
            try:
                # Update status with file path
                filename = os.path.basename(file_path)
                self.statusBar().showMessage(f"Loading file: {filename}...")
                
                # Try to read the file
                points = self._read_point_file(file_path)
                
                if points is not None and len(points) > 0:
                    # Create a new dataset
                    dataset = {
                        'name': filename,
                        'points': points,
                        'visible': True,
                        'color': self._get_next_color()
                    }
                    
                    # Add to datasets
                    self.datasets.append(dataset)
                    self.current_dataset_index = len(self.datasets) - 1
                    successful_loads += 1
                else:
                    logger.warning(f"No valid points found in file: {filename}")
            except Exception as e:
                logger.error(f"Error loading file {filename}: {str(e)}")
        
        if successful_loads > 0:
            # Update UI
            self._update_dataset_list()
            self._update_statistics()
            self._visualize_all_points()
            
            self.statusBar().showMessage(f"Successfully loaded {successful_loads} out of {len(file_paths)} files")
        else:
            self.statusBar().showMessage("Error: No valid points found in any file")
            QMessageBox.critical(self, "Error", "No valid points found in any file")
            
    def _get_next_color(self):
        """Get next color from palette for a new dataset"""
        if not self.datasets:
            return self.DEFAULT_COLORS[0]
        
        # Count existing colors
        used_colors = {}
        for dataset in self.datasets:
            color = dataset.get('color')
            if color:
                used_colors[color] = used_colors.get(color, 0) + 1
        
        # Find first least used color
        for color in self.DEFAULT_COLORS:
            if color not in used_colors:
                return color
        
        # If all colors are used, use the least used one
        min_used = min(used_colors.values())
        least_used_colors = [c for c, count in used_colors.items() if count == min_used]
        
        # Return the first one from palette order
        for color in self.DEFAULT_COLORS:
            if color in least_used_colors:
                return color
        
        # If all else fails, return a random color
        return self.DEFAULT_COLORS[np.random.randint(0, len(self.DEFAULT_COLORS))]

    def _visualize_all_points(self):
        """Visualize all visible datasets' points"""
        # Get visible datasets
        visible_datasets = [d for d in self.datasets if d.get('visible', True)]
        
        if not visible_datasets:
            self._clear_hull_plot()
            return
        
        # Clear existing visualization
        while self.file_viz_layout.count():
            item = self.file_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Calculate global bounds for all visible datasets
        all_points = np.vstack([d['points'] for d in visible_datasets if d['points'] is not None])
        if len(all_points) == 0:
            return
            
        # Calculate bounds, correctly handling 3D coordinates
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        min_x, min_y = min_coords[0], min_coords[1]
        max_x, max_y = max_coords[0], max_coords[1]
        
        # Use 3D visualization if enabled
        if self.view_3d_enabled:
            # Create 3D visualization
            self._create_multi_dataset_3d_visualization(
                self.file_viz_frame,
                visible_datasets,
                "Points Visualization",
                view_type="points"
            )
        else:
            # Fall back to matplotlib if PyVista is not available
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Plot each dataset with its color
            for dataset in visible_datasets:
                points = dataset['points']
                if points is not None and len(points) > 0:
                    color = dataset.get('color', 'blue')
                    name = dataset.get('name', 'Unnamed')
                    ax.scatter(points[:, 0], points[:, 1], s=5, c=color, alpha=0.7, label=name)
            
            ax.set_aspect('equal')
            ax.set_title("Point Clouds")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            
            # Add grid and set limits with some padding
            ax.grid(True, linestyle='--', alpha=0.6)
            padding = max((max_x - min_x), (max_y - min_y)) * 0.05
            ax.set_xlim(min_x - padding, max_x + padding)
            ax.set_ylim(min_y - padding, max_y + padding)
            
            # Create canvas
            canvas = FigureCanvas(fig)
            self.file_viz_layout.addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.file_viz_frame)
            self.file_viz_layout.addWidget(toolbar)
    
    def _visualize_all_hulls(self):
        """Visualize all visible hulls"""
        # Get datasets with hulls
        datasets_with_hull = []
        for dataset in self.datasets:
            if dataset.get('visible', True) and dataset.get('hull_points') is not None:
                datasets_with_hull.append(dataset)
                
        # Clear previous visualization if no datasets with hull
        if not datasets_with_hull:
            self._clear_hull_plot()
            return
            
        # Use 3D visualization or PyVista
        if self.view_3d_enabled:
            self._create_multi_dataset_3d_visualization(
                self.hull_viz_frame, 
                datasets_with_hull,
                "Convex Hull Visualization",
                view_type="hulls"
            )
            return
        
        # Clear previous visualization
        while self.hull_viz_layout.count():
            item = self.hull_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        # Create a matplotlib visualization
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Find global limits for all datasets
        all_points = []
        for dataset in datasets_with_hull:
            points = dataset.get('points')
            if points is not None and len(points) > 0:
                all_points.append(points)
                
        if not all_points:
            return
            
        all_points_np = np.vstack(all_points)
        min_x, min_y = np.min(all_points_np[:, 0:2], axis=0)
        max_x, max_y = np.max(all_points_np[:, 0:2], axis=0)
        
        # Add some margin to the plot
        margin_x = 0.1 * (max_x - min_x)
        margin_y = 0.1 * (max_y - min_y)
        
        # Plot each dataset
        for dataset in datasets_with_hull:
            points = dataset.get('points')
            hull_points = dataset.get('hull_points')
            color = dataset.get('color', '#000000')
            name = dataset.get('name', 'Unnamed')
            
            if points is None or len(points) == 0 or hull_points is None or len(hull_points) == 0:
                continue
            
            # Plot points (with reduced alpha for better visualization)
            ax.scatter(points[:, 0], points[:, 1], s=5, c=color, alpha=0.3)
            
            # Plot hull (using first 2 dimensions)
            ax.plot(hull_points[:, 0], hull_points[:, 1], c=color, linewidth=2, label=name)
            
        ax.set_aspect('equal')
        ax.set_title(f"Hull Visualization: {len(datasets_with_hull)} datasets")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')
        
        # Set limits with margin
        ax.set_xlim(min_x - margin_x, max_x + margin_x)
        ax.set_ylim(min_y - margin_y, max_y + margin_y)
        
        # Create canvas
        canvas = FigureCanvas(fig)
        self.hull_viz_layout.addWidget(canvas)
        
        # Add toolbar
        toolbar = NavigationToolbar(canvas, self.hull_viz_frame)
        self.hull_viz_layout.addWidget(toolbar)
    
    def _visualize_all_segments(self):
        """Visualize all visible datasets' segments"""
        # Get visible datasets with segments
        visible_datasets = [d for d in self.datasets if d.get('visible', True) and d.get('segments') is not None]
        
        if not visible_datasets:
            self._clear_segment_plot()
            return
        
        # Clear existing visualization
        while self.segment_viz_layout.count():
            item = self.segment_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Calculate global bounds for all visible datasets
        all_points = np.vstack([d['points'] for d in visible_datasets if d['points'] is not None])
        if len(all_points) == 0:
            return
            
        # Handle 3D coordinates properly
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        min_x, min_y = min_coords[0], min_coords[1]
        max_x, max_y = max_coords[0], max_coords[1]
        
        # Use 3D visualization if enabled
        if self.view_3d_enabled:
            # Create 3D visualization - pass segments to be drawn as lines
            self._create_multi_dataset_3d_visualization(
                self.segment_viz_frame,
                visible_datasets,
                "Segmentation Visualization",
                view_type="segments" # This already handles plotting segments
            )
        else:
            # Fall back to matplotlib if PyVista is not available
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Plot each dataset with its color
            for dataset in visible_datasets:
                points = dataset['points']
                segments = dataset['segments']
                
                if points is not None and len(points) > 0 and segments is not None and len(segments) > 0:
                    color = dataset.get('color', 'blue')
                    name = dataset.get('name', 'Unnamed')
                    
                    # Plot points
                    ax.scatter(points[:, 0], points[:, 1], s=5, c=color, alpha=0.3)
                    
                    # Collect all segment endpoints
                    segment_endpoints = []
                    for segment in segments:
                        # Add both endpoints of each segment
                        if hasattr(segment[0], 'shape'):  # NumPy array format
                            segment_endpoints.append(segment[0])
                            segment_endpoints.append(segment[1])
                        else:  # List format
                            segment_endpoints.append(np.array(segment[0]))
                            segment_endpoints.append(np.array(segment[1]))
                    
                    # Remove duplicates by converting to tuples and back
                    unique_endpoints = []
                    seen = set()
                    for point in segment_endpoints:
                        point_tuple = tuple(point)
                        if point_tuple not in seen:
                            seen.add(point_tuple)
                            unique_endpoints.append(point)
                    
                    # Convert to numpy array for plotting
                    if unique_endpoints:
                        unique_endpoints_array = np.array(unique_endpoints)
                        # Plot segment endpoints as red points
                        ax.scatter(unique_endpoints_array[:, 0], unique_endpoints_array[:, 1], 
                                s=20, c='red', edgecolor='black', label=f"{name} Segment Points")

                    # Plot segments - make them slightly more prominent
                    for segment in segments:
                        ax.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 
                              color=color, linewidth=1.8, alpha=0.9) # Increased linewidth/alpha
            
            ax.set_aspect('equal')
            ax.set_title("Segmentations (Boundary Defined by Segments)") # Updated title
            # ... rest of matplotlib setup (labels, legend, grid, limits) ...
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            
            # Add grid and set limits with some padding
            ax.grid(True, linestyle='--', alpha=0.6)
            padding = max((max_x - min_x), (max_y - min_y)) * 0.05
            ax.set_xlim(min_x - padding, max_x + padding)
            ax.set_ylim(min_y - padding, max_y + padding)
            
            # Create canvas
            canvas = FigureCanvas(fig)
            self.segment_viz_layout.addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.segment_viz_frame)
            self.segment_viz_layout.addWidget(toolbar)
        
        # Update the segments legend (remains the same)
        # ... existing legend update code ...
    
    def _visualize_all_triangulations(self):
        """Visualize all visible datasets' triangulations"""
        # Get visible datasets with triangulations
        visible_datasets = [d for d in self.datasets if d.get('visible', True) and d.get('triangulation_result') is not None]
        
        if not visible_datasets:
            self._clear_triangulation_plot()
            return
        
        # Clear existing visualization
        while self.tri_viz_layout.count():
            item = self.tri_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Calculate global bounds for all visible datasets
        all_vertices = []
        for dataset in visible_datasets:
            if dataset.get('triangulation_result') is not None:
                vertices = dataset['triangulation_result']['vertices']
                if vertices is not None and len(vertices) > 0:
                    all_vertices.append(vertices)
        
        if not all_vertices:
            return
            
        all_vertices = np.vstack(all_vertices)
        # Handle 3D coordinates properly
        min_coords = np.min(all_vertices, axis=0)
        max_coords = np.max(all_vertices, axis=0)
        min_x, min_y = min_coords[0], min_coords[1]
        max_x, max_y = max_coords[0], max_coords[1]
        
        # Use 3D visualization if enabled
        if self.view_3d_enabled:
            # In 3D, the mesh itself shows the boundary. We can optionally highlight edges.
            # The existing _create_multi_dataset_3d_visualization already shows edges.
            self._create_multi_dataset_3d_visualization(
                self.tri_viz_frame,
                visible_datasets,
                "Triangulation Visualization",
                view_type="triangulation"
            )
        else:
            # Fall back to matplotlib if PyVista is not available
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Plot each dataset with its color
            for dataset in visible_datasets:
                triangulation_result = dataset['triangulation_result']
                # hull_points = dataset.get('hull_points') # Don't use original hull
                
                if triangulation_result is not None:
                    vertices = triangulation_result.get('vertices')
                    triangles = triangulation_result.get('triangles')
                    
                    if vertices is not None and len(vertices) > 0 and triangles is not None and len(triangles) > 0:
                        color = dataset.get('color', 'blue')
                        name = dataset.get('name', 'Unnamed')
                        
                        # Plot triangulation using triplot
                        from matplotlib.tri import Triangulation
                        tri = Triangulation(vertices[:, 0], vertices[:, 1], triangles)
                        ax.triplot(tri, color=color, lw=0.5, alpha=0.7, label=f"{name} Mesh")
                        
                        # --- Plot actual boundary edges ---
                        boundary_edges_indices = self._get_boundary_edges(triangles)
                        if boundary_edges_indices:
                            boundary_lines = []
                            for i1, i2 in boundary_edges_indices:
                                p1 = vertices[i1]
                                p2 = vertices[i2]
                                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1.5, label=f"{name} Boundary" if not ax.get_legend_handles_labels()[1] else "") # Only label once per dataset
                        # --- End plot boundary edges ---

                        # # Plot hull boundary if available - REMOVED
                        # if hull_points is not None and len(hull_points) > 3:
                        #     ax.plot(hull_points[:, 0], hull_points[:, 1], color=color, linewidth=1.5)
            
            ax.set_aspect('equal')
            ax.set_title("Triangulations (Actual Boundary)") # Updated title
            # ... rest of matplotlib setup (labels, legend, grid, limits) ...
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
             # Consolidate legend entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles)) # Remove duplicate labels
            ax.legend(by_label.values(), by_label.keys())
            
            # Add grid and set limits with some padding
            ax.grid(True, linestyle='--', alpha=0.6)
            padding = max((max_x - min_x), (max_y - min_y)) * 0.05
            ax.set_xlim(min_x - padding, max_x + padding)
            ax.set_ylim(min_y - padding, max_y + padding)
            
            # Create canvas
            canvas = FigureCanvas(fig)
            self.tri_viz_layout.addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.tri_viz_frame)
            self.tri_viz_layout.addWidget(toolbar)
        
        # Update the triangulation legend (remains the same)
        # ... existing legend update code ...
    
    def _visualize_refined_intersections(self):
        """Visualize refined intersections with enhanced special point visualization."""
        # Get plotter
        plotter = None
        if hasattr(self, 'refine_mesh_plotter') and self.refine_mesh_plotter:
            plotter = self.refine_mesh_plotter
        elif hasattr(self, 'plotters') and 'refine_mesh' in self.plotters and self.plotters['refine_mesh']:
            plotter = self.plotters['refine_mesh']
            self.refine_mesh_plotter = plotter
        
        if not plotter:
            logger.warning("Refine/Mesh plotter not available for visualization.")
            if HAVE_PYVISTA and hasattr(self, 'refine_mesh_viz_frame') and self.refine_mesh_viz_frame:
                try:
                    from pyvistaqt import QtInteractor
                    new_plotter = QtInteractor(self.refine_mesh_viz_frame)
                    self.refine_mesh_plot_layout.addWidget(new_plotter.interactor)
                    new_plotter.set_background([0.318, 0.341, 0.431])
                    plotter = new_plotter
                    self.refine_mesh_plotter = plotter
                    if hasattr(self, 'plotters'):
                        self.plotters['refine_mesh'] = plotter
                    logger.info("Successfully recreated refine_mesh_plotter")
                except Exception as e:
                    logger.error(f"Error recreating Refine/Mesh plotter: {e}")
                    return
            else:
                return

        plotter.clear()
        plotter.set_background([0.2, 0.2, 0.25])

        if not hasattr(self, 'datasets_intersections') or not self.datasets_intersections:
            plotter.add_text("No refined intersections to display.", position='upper_edge', color='white')
            plotter.reset_camera()
            return

        logger.info("Visualizing refined intersections...")

        # Initialize constraint points collection (consolidated)
        constraint_points = []  # All constraint points for triangulation
        
        involved_dataset_indices = set()
        refined_intersection_lines = []
        original_intersection_lines = []
        plotter_has_content = False

        # Check if original intersections exist for comparison
        has_original = hasattr(self, 'original_intersections_backup') and self.original_intersections_backup

        # Process original intersection lines (for comparison)
        if has_original:
            for dataset_idx_key, intersections_list in self.original_intersections_backup.items():
                for intersection_data in intersections_list:
                    if intersection_data['points'] and len(intersection_data['points']) >= 2:
                        try:
                            points_list = []
                            for p in intersection_data['points']:
                                if isinstance(p, (list, tuple)) and len(p) >= 3:
                                    points_list.append([float(p[0]), float(p[1]), float(p[2])])
                                elif hasattr(p, 'x') and hasattr(p, 'y') and hasattr(p, 'z'):
                                    points_list.append([float(p.x), float(p.y), float(p.z)])
                            
                            if len(points_list) >= 2:
                                original_intersection_lines.append((np.array(points_list), intersection_data.get('is_polyline_mesh', False)))
                        except Exception as e:
                            logger.error(f"Error processing original intersection: {e}")

        # Process refined intersection lines and extract special points
        for dataset_idx_key in self.datasets_intersections.keys():
            intersections_list = self.datasets_intersections[dataset_idx_key]
            for intersection_data in intersections_list:
                if 'points' not in intersection_data or not intersection_data['points'] or len(intersection_data['points']) < 2:
                    continue
                    
                try:
                    # Track involved datasets
                    dataset_id1 = intersection_data.get('dataset_id1', -1)
                    dataset_id2 = intersection_data.get('dataset_id2', -1)
                    involved_dataset_indices.add(dataset_id1)
                    involved_dataset_indices.add(dataset_id2)
                    
                    # Process intersection points
                    points_list = []
                    for p in intersection_data['points']:
                        coord = None
                        point_type = None
                        
                        if isinstance(p, (list, tuple)):
                            if len(p) >= 3:
                                coord = [float(p[0]), float(p[1]), float(p[2])]
                                point_type = p[3] if len(p) > 3 else None
                        elif hasattr(p, 'x') and hasattr(p, 'y') and hasattr(p, 'z'):
                            coord = [float(p.x), float(p.y), float(p.z)]
                            point_type = getattr(p, 'point_type', getattr(p, 'type', None))
                        
                        if coord:
                            points_list.append(coord)
                            
                            # Categorize points by type
                            constraint_points.append(coord)
                    if len(points_list) >= 2:
                        is_polyline = intersection_data.get('is_polyline_mesh', False)
                        refined_intersection_lines.append((np.array(points_list), is_polyline))
                        
                except Exception as e:
                    logger.error(f"Error processing refined intersection: {e}")

        # Add original intersection lines (faded)
        if has_original and original_intersection_lines:
            for line_points, is_polyline in original_intersection_lines:
                try:
                    if len(line_points) >= 2:
                        cells = []
                        for i in range(len(line_points) - 1):
                            cells.extend([2, i, i + 1])
                        original_line = pv.PolyData(line_points, lines=np.array(cells))
                        plotter.add_mesh(original_line, color='gray', opacity=0.3, line_width=2.0)
                        plotter_has_content = True
                except Exception as e:
                    logger.error(f"Error visualizing original line: {e}")

        # Add refined intersection lines
        if refined_intersection_lines:
            surf_surf_color = [0.9, 0.5, 0.1]  # Orange for surface-surface
            surf_poly_color = [0.1, 0.8, 0.9]  # Cyan for surface-polyline
            
            for line_points, is_polyline in refined_intersection_lines:
                try:
                    if len(line_points) >= 2:
                        cells = []
                        for i in range(len(line_points) - 1):
                            cells.extend([2, i, i + 1])
                        refined_line = pv.PolyData(line_points, lines=np.array(cells))
                        
                        color = surf_poly_color if is_polyline else surf_surf_color
                        plotter.add_mesh(refined_line, color=color, line_width=4.0, 
                                    render_lines_as_tubes=True, smooth_shading=True)
                        plotter_has_content = True
                except Exception as e:
                    logger.error(f"Error visualizing refined line: {e}")

        # Process convex hulls and extract corner points
        for original_idx in involved_dataset_indices:
            if original_idx >= 0 and original_idx < len(self.datasets):
                dataset = self.datasets[original_idx]
                if not dataset.get('visible', True):
                    continue
                    
                hull_points = dataset.get('hull_points')
                if hull_points is None or len(hull_points) < 3:
                    continue
                    
                name = dataset.get('name', f'Dataset {original_idx + 1}')
                hull_color = dataset.get('color', '#CCCCCC')
                
                # Extract corner points and junction points from hull
                # Extract constraint points from hull (non-DEFAULT types)
                # Extract ALL hull points as constraint points (including DEFAULT subdivision points)
                for pt_idx, pt in enumerate(hull_points):
                    if len(pt) >= 3:
                        coord = [float(pt[0]), float(pt[1]), float(pt[2])]
                        constraint_points.append(coord)  # ALL hull points are constraints
                
                # Draw convex hull outline
                try:
                    hull_vertices = []
                    for point in hull_points:
                        if len(point) >= 3:
                            hull_vertices.append([point[0], point[1], point[2] if len(point) > 2 else 0.0])
                    
                    if len(hull_vertices) >= 3:
                        hull_vertices_np = np.array(hull_vertices)
                        n_points = len(hull_vertices_np)
                        
                        # Create hull outline
                        line_segments = []
                        for i in range(n_points):
                            line_segments.extend([2, i, (i+1) % n_points])
                        
                        hull_polydata = pv.PolyData(hull_vertices_np)
                        hull_polydata.lines = np.array(line_segments)
                        
                        plotter.add_mesh(hull_polydata, color=hull_color, opacity=0.7,
                                    line_width=3.0, render_lines_as_tubes=True)
                        plotter_has_content = True
                        
                except Exception as e:
                    logger.error(f"Error visualizing convex hull for dataset {original_idx}: {e}")

        # Add triple points from stored triple points
        # Add triple points to constraint points
        if hasattr(self, 'triple_points') and self.triple_points:
            for tp in self.triple_points:
                p = tp['point']
                try:
                    if isinstance(p, (list, tuple)) and len(p) >= 3:
                        constraint_points.append([float(p[0]), float(p[1]), float(p[2])])
                    elif hasattr(p, 'x') and hasattr(p, 'y') and hasattr(p, 'z'):
                        constraint_points.append([float(p.x), float(p.y), float(p.z)])
                except Exception as e:
                    logger.error(f"Error processing triple point: {e}")

        # Visualize all special point types with distinct colors and sizes
        
        # Visualize constraint points (all non-DEFAULT points for triangulation)
        if constraint_points:
            try:
                # Remove duplicates
                unique_constraints = []
                for pt in constraint_points:
                    is_duplicate = False
                    for existing in unique_constraints:
                        if abs(pt[0] - existing[0]) < 1e-6 and abs(pt[1] - existing[1]) < 1e-6 and abs(pt[2] - existing[2]) < 1e-6:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_constraints.append(pt)
                
                points_np = np.array(unique_constraints)
                plotter.add_mesh(points_np, color='red', point_size=10,
                            render_points_as_spheres=True, name="Constraint_Points")
                plotter.add_point_labels(points_np, [f"C{i}" for i in range(len(points_np))],
                                    point_size=5, text_color='black')
                plotter_has_content = True
                logger.info(f"Added {len(unique_constraints)} constraint points (consolidated)")
            except Exception as e:
                logger.error(f"Error adding constraint points: {e}")

        # Add comprehensive legend and statistics
        if plotter_has_content:
            # Statistics text
           # Clean statistics text
            # Clean statistics text
            stats_text = "MeshIt Constraint Points (All Types):\n"
            if constraint_points:
                unique_constraints = []
                for pt in constraint_points:
                    is_duplicate = False
                    for existing in unique_constraints:
                        if abs(pt[0] - existing[0]) < 1e-6 and abs(pt[1] - existing[1]) < 1e-6 and abs(pt[2] - existing[2]) < 1e-6:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_constraints.append(pt)
                
                stats_text += f"• Total Constraint Points: {len(unique_constraints)} (red)\n"
                stats_text += f"• Raw Points Collected: {len(constraint_points)}\n"

            stats_text += f"\nIntersection Lines: {len(refined_intersection_lines)}"
            stats_text += f"\nDatasets Involved: {len(involved_dataset_indices)}"

            # Simple legend
            legend_text = "Constraint Points for Triangulation:\n"
            legend_text += "C = Constraint Point (all special types)\n"
            legend_text += "Red = Ready for triangulation\n"
            
            stats_text += f"\nIntersection Lines: {len(refined_intersection_lines)}"
            stats_text += f"\nDatasets Involved: {len(involved_dataset_indices)}"
            
            plotter.add_text(stats_text, position='lower_left', font_size=10, color='white')
            
          
            plotter.add_text(legend_text, position='upper_right', font_size=9, color='white')
            
            plotter.add_axes()
            plotter.reset_camera()
            
            logger.info(f"Statistics: Constraints={len(constraint_points)}")
            logger.info("Updated refined intersections view with enhanced special point visualization.")
        else:
            plotter.add_text("No valid data to display in refined view.", position='upper_edge', color='white')
            plotter.reset_camera()
    def _clear_refine_mesh_plot(self):
            """Clear the embedded Refine & Mesh PyVista plotter."""
            plotter = None
            if hasattr(self, 'refine_mesh_plotter') and self.refine_mesh_plotter:
                plotter = self.refine_mesh_plotter
            elif 'refine_mesh' in self.plotters and self.plotters['refine_mesh']:
                plotter = self.plotters['refine_mesh']
                self.refine_mesh_plotter = plotter
            
            if plotter:
                plotter.clear()
                plotter.add_text("Refine intersections to visualize or load data.", position='upper_edge', color='white')
                plotter.reset_camera()
            elif hasattr(self, 'refine_mesh_plot_layout'): # Fallback if plotter is None but layout exists
                for i in reversed(range(self.refine_mesh_plot_layout.count())):
                    widget = self.refine_mesh_plot_layout.itemAt(i).widget()
                    if widget:
                        if hasattr(self, 'refine_mesh_plotter') and self.refine_mesh_plotter and widget == self.refine_mesh_plotter.interactor:
                            continue
                        widget.setParent(None)
                        widget.deleteLater()
                if not hasattr(self, 'refine_mesh_plotter') or not self.refine_mesh_plotter:
                    placeholder = QLabel("PyVista required or plot cleared.")
                    placeholder.setAlignment(Qt.AlignCenter)
                    self.refine_mesh_plot_layout.addWidget(placeholder)
    
    def _update_refined_visualization(self):
        """Update the refined intersection visualization based on UI settings"""
        # Simply call visualize method again to refresh with current settings
        self._visualize_refined_intersections()
    
    def consolidate_points_for_triangulation(self):
        """
        Consolidate all refined points following C++ MeshIt logic for tetrahedral meshing:
        - Triple Points: Only TRIPLE_POINT type
        - Special Points: ALL other points INCLUDING DEFAULT subdivision points
        
        DEFAULT points are the refined subdivision points from C++ RefineByLength that are 
        essential for creating manifold structure in tetrahedral meshing.
        """
        # Clear existing consolidated collections
        self.consolidated_special_points = []
        self.consolidated_triple_points = []
        
        logger.info("Consolidating points for triangulation constraints...")
        
        # 1. Process all intersection line points (INCLUDING DEFAULT subdivision points)
        for dataset_idx_key in self.datasets_intersections.keys():
            intersections_list = self.datasets_intersections[dataset_idx_key]
            for intersection_data in intersections_list:
                if 'points' not in intersection_data or not intersection_data['points']:
                    continue
                    
                for p in intersection_data['points']:
                    point_type = None
                    coord = None
                    
                    # Extract coordinates and type
                    if isinstance(p, (list, tuple)):
                        if len(p) >= 3:
                            coord = [float(p[0]), float(p[1]), float(p[2])]
                            point_type = p[3] if len(p) > 3 else None
                    elif hasattr(p, 'x') and hasattr(p, 'y') and hasattr(p, 'z'):
                        coord = [float(p.x), float(p.y), float(p.z)]
                        point_type = getattr(p, 'point_type', getattr(p, 'type', None))
                    
                    if not coord:
                        continue
                        
                    # Consolidate based on corrected C++ logic
                    if point_type:
                        type_str = str(point_type)
                        if type_str == "TRIPLE_POINT":
                            self.consolidated_triple_points.append({
                                'coord': coord, 'type': type_str, 'source': 'intersection'
                            })
                        else:  # ALL other points (including DEFAULT) become special points
                            self.consolidated_special_points.append({
                                'coord': coord, 'type': type_str, 'source': 'intersection'
                            })
                    else:
                        # Points without type are treated as DEFAULT subdivision points
                        self.consolidated_special_points.append({
                            'coord': coord, 'type': 'DEFAULT', 'source': 'intersection'
                        })
        
        # 2. Process all convex hull points (INCLUDING DEFAULT subdivision points)
        for dataset_idx in range(len(self.datasets)):
            dataset = self.datasets[dataset_idx]
            hull_points = dataset.get('hull_points', [])
            
            for pt in hull_points:
                if len(pt) >= 3:  # Process ALL hull points
                    coord = [float(pt[0]), float(pt[1]), float(pt[2])]
                    pt_type = str(pt[3]) if len(pt) > 3 and isinstance(pt[3], str) else 'DEFAULT'
                    
                    # ALL hull points become constraints (including DEFAULT subdivision points)
                    self.consolidated_special_points.append({
                        'coord': coord, 'type': pt_type, 'source': f'hull_dataset_{dataset_idx}'
                    })
        
        # 3. Process stored triple points
        if hasattr(self, 'triple_points') and self.triple_points:
            for tp in self.triple_points:
                p = tp['point']
                try:
                    if isinstance(p, (list, tuple)) and len(p) >= 3:
                        coord = [float(p[0]), float(p[1]), float(p[2])]
                    elif hasattr(p, 'x') and hasattr(p, 'y') and hasattr(p, 'z'):
                        coord = [float(p.x), float(p.y), float(p.z)]
                    else:
                        continue
                        
                    self.consolidated_triple_points.append({
                        'coord': coord, 'type': 'TRIPLE_POINT', 'source': 'stored_triple_points'
                    })
                except Exception as e:
                    logger.error(f"Error processing stored triple point: {e}")
        
        # 4. Remove duplicates (points that might appear in both collections)
        def remove_duplicates(points_list, tolerance=1e-6):
            unique_points = []
            for pt in points_list:
                is_duplicate = False
                for existing_pt in unique_points:
                    if (abs(pt['coord'][0] - existing_pt['coord'][0]) < tolerance and
                        abs(pt['coord'][1] - existing_pt['coord'][1]) < tolerance and  
                        abs(pt['coord'][2] - existing_pt['coord'][2]) < tolerance):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_points.append(pt)
            return unique_points
        
        self.consolidated_special_points = remove_duplicates(self.consolidated_special_points)
        self.consolidated_triple_points = remove_duplicates(self.consolidated_triple_points)
        
        # Log consolidation results with detailed breakdown
        logger.info("=== CONSOLIDATION RESULTS (Including DEFAULT subdivision points) ===")
        logger.info(f"Triple Points: {len(self.consolidated_triple_points)}")
        logger.info(f"Special Points (ALL types including DEFAULT): {len(self.consolidated_special_points)}")
        
        # Break down special points by type
        special_by_type = {}
        special_by_source = {}
        for pt in self.consolidated_special_points:
            pt_type = pt['type']
            pt_source = pt['source']
            special_by_type[pt_type] = special_by_type.get(pt_type, 0) + 1
            special_by_source[pt_source] = special_by_source.get(pt_source, 0) + 1
        
        logger.info("Special Points Breakdown by Type:")
        for pt_type, count in special_by_type.items():
            logger.info(f"  {pt_type}: {count}")
        
        logger.info("Special Points Breakdown by Source:")
        for pt_source, count in special_by_source.items():
            logger.info(f"  {pt_source}: {count}")
        
        logger.info("Ready for constrained triangulation with manifold structure!")
        logger.info("==============================================================")
        
        return self.consolidated_special_points, self.consolidated_triple_points

    def _create_multi_dataset_3d_visualization(self, parent_frame, datasets, title, view_type="points"):
        """Create a 3D visualization of multiple datasets with proper coordinate validation."""
        if not HAVE_PYVISTA:
            # Clear previous content if any
            while parent_frame.layout() and parent_frame.layout().count():
                item = parent_frame.layout().takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

            # Ensure parent_frame has a layout
            if parent_frame.layout() is None:
                parent_frame.setLayout(QVBoxLayout())

            # Create a message if PyVista is not available
            msg_widget = QWidget()
            msg_layout = QVBoxLayout(msg_widget)
            msg = QLabel("PyVista not installed.\nPlease install PyVista for 3D visualization.")
            msg.setAlignment(Qt.AlignCenter)
            msg_layout.addWidget(msg)
            parent_frame.layout().addWidget(msg_widget)
            return msg_widget

        # Close previous plotter if it exists
        if hasattr(self, 'current_plotter') and self.current_plotter is not None:
            try:
                self.current_plotter.close()
            except Exception as e:
                logger.warning(f"Error closing previous plotter: {e}")
            self.current_plotter = None

        # Clear previous content from parent frame layout first
        parent_layout = parent_frame.layout()
        if parent_layout is None:
            parent_layout = QVBoxLayout(parent_frame)
            parent_layout.setContentsMargins(0,0,0,0)

        while parent_layout.count():
            item = parent_layout.takeAt(0)
            widget = item.widget()
            if widget:
                if isinstance(widget, QFrame) and hasattr(widget, 'interactor'):
                    try:
                        widget.interactor.close()
                    except Exception as e:
                        logger.debug(f"Minor issue closing interactor: {e}")
                widget.deleteLater()

        # Create a container widget for the entire visualization area
        vis_container_widget = QWidget()
        vis_container_layout = QVBoxLayout(vis_container_widget)
        vis_container_layout.setContentsMargins(0, 0, 0, 0)

        # Add visualization info header based on view_type
        if view_type == "points":
            info_title = "Point Cloud Visualization"
        elif view_type == "hulls":
            info_title = "Convex Hull Visualization"
        elif view_type == "segments":
            info_title = "Segmentation Visualization"
        elif view_type == "triangulation":
            info_title = "Triangulation Visualization"
        else:
            info_title = title

        info_label = QLabel(info_title)
        info_label.setAlignment(Qt.AlignCenter)
        vis_container_layout.addWidget(info_label)

        # Create legend with colored boxes for each dataset
        legend_widget = QWidget()
        legend_layout = QHBoxLayout(legend_widget)
        legend_layout.setContentsMargins(5, 2, 5, 2)

        visible_datasets_in_list = [d for d in datasets if d.get('visible', True)]
        for dataset in visible_datasets_in_list:
            color = dataset.get('color', '#000000')
            name = dataset.get('name', 'Unnamed')
            
            legend_item = QWidget()
            legend_item_layout = QHBoxLayout(legend_item)
            legend_item_layout.setContentsMargins(0, 0, 5, 0)
            
            # Color box
            color_box = QLabel("■")
            color_box.setStyleSheet(f"color: {color}; font-size: 16px;")
            legend_item_layout.addWidget(color_box)
            
            # Dataset name
            name_label = QLabel(name)
            legend_item_layout.addWidget(name_label)
            
            legend_layout.addWidget(legend_item)

        legend_layout.addStretch()
        vis_container_layout.addWidget(legend_widget)

        try:
            # Create PyVista plotter widget using QtInteractor
            from pyvistaqt import QtInteractor
            import pyvista as pv
            import numpy as np
            
            self.current_plotter = QtInteractor(parent=vis_container_widget)
            vis_container_layout.addWidget(self.current_plotter)
            
            # Set background color
            self.current_plotter.set_background("#383F51")
            
            # Process datasets based on visualization type
            plotter_has_geometry = False
            for i, dataset in enumerate(datasets):
                if not dataset.get('visible', True):
                    continue
                    
                # Get dataset properties
                points = dataset.get('points')
                color = dataset.get('color', '#000000')
                name = dataset.get('name', 'Unnamed')
                
                if points is None or len(points) == 0:
                    continue
                
                # Convert 2D points to 3D with height variation
                if points.shape[1] == 2:
                    points_3d = np.zeros((len(points), 3))
                    points_3d[:, 0] = points[:, 0]
                    points_3d[:, 1] = points[:, 1]
                    # Z coordinate is left as 0 for 2D points
                elif points.shape[1] >= 3:
                    # Use actual Z coordinates from the data
                    points_3d = points.copy()[:, 0:3]
                else:
                    logger.warning(f"Dataset '{name}' has unexpected point dimensions: {points.shape[1]}. Skipping.")
                    continue

                # Add visualization based on type
                if view_type == "points":
                    point_cloud = pv.PolyData(points_3d)
                    self.current_plotter.add_mesh(point_cloud, color=color, render_points_as_spheres=True, 
                                        point_size=8, label=name)
                    plotter_has_geometry = True
                
                elif view_type == "hulls":
                    hull_points = dataset.get('hull_points')
                    if hull_points is not None and len(hull_points) > 3:
                        try:
                            # Ensure hull points are properly formatted as 3D coordinates
                            hull_3d = np.array(hull_points)
                            
                            # Validate dimensions
                            if len(hull_3d.shape) < 2:
                                logger.warning(f"Invalid hull shape for {name}: {hull_3d.shape}")
                                continue
                                
                            # Ensure each point has exactly 3 coordinates
                            if hull_3d.shape[1] < 3:
                                # Pad with zeros if less than 3D
                                padding = np.zeros((hull_3d.shape[0], 3 - hull_3d.shape[1]))
                                hull_3d = np.hstack([hull_3d, padding])
                            elif hull_3d.shape[1] > 3:
                                # Take only first 3 coordinates if more than 3D
                                hull_3d = hull_3d[:, :3]
                            
                            # Create lines for the hull with proper validation
                            for j in range(len(hull_3d)-1):
                                try:
                                    # Convert to proper 3D coordinate lists
                                    point_a = hull_3d[j].tolist() if hasattr(hull_3d[j], 'tolist') else list(hull_3d[j])
                                    point_b = hull_3d[j+1].tolist() if hasattr(hull_3d[j+1], 'tolist') else list(hull_3d[j+1])
                                    
                                    # Ensure exactly 3 coordinates and valid floats
                                    if (len(point_a) >= 3 and len(point_b) >= 3 and
                                        all(isinstance(x, (int, float)) for x in point_a[:3]) and
                                        all(isinstance(x, (int, float)) for x in point_b[:3])):
                                        
                                        hull_line = pv.Line([float(point_a[0]), float(point_a[1]), float(point_a[2])],
                                                        [float(point_b[0]), float(point_b[1]), float(point_b[2])])
                                        self.current_plotter.add_mesh(hull_line, color=color, line_width=3)
                                    else:
                                        logger.warning(f"Skipping invalid hull line {j} for {name}")
                                except Exception as e:
                                    logger.warning(f"Error creating hull line {j} for {name}: {e}")
                                    continue
                                    
                            # Add the original points with reduced opacity for context
                            point_cloud = pv.PolyData(points_3d)
                            self.current_plotter.add_mesh(point_cloud, color=color, opacity=0.3, 
                                            render_points_as_spheres=True, point_size=5)
                            plotter_has_geometry = True
                            
                        except Exception as e:
                            logger.error(f"Error processing hull points for {name}: {e}")
                            # Fallback to just showing points
                            point_cloud = pv.PolyData(points_3d)
                            self.current_plotter.add_mesh(point_cloud, color=color, opacity=0.7, 
                                            render_points_as_spheres=True, point_size=5)
                            plotter_has_geometry = True

                elif view_type == "segments":
                    segments = dataset.get('segments')
                    if segments is not None and len(segments) > 0:
                        # Add points for context
                        point_cloud = pv.PolyData(points_3d)
                        self.current_plotter.add_mesh(point_cloud, color=color, opacity=0.3, render_points_as_spheres=True,
                                        point_size=5)

                        # Draw the actual segments with validation
                        for segment in segments:
                            try:
                                # Ensure segment points are 3D NumPy arrays
                                p1 = np.array(segment[0])
                                p2 = np.array(segment[1])
                                
                                # Pad with zeros if dimension is less than 3
                                if p1.shape[0] < 3: 
                                    p1 = np.append(p1, [0.0] * (3 - p1.shape[0]))
                                if p2.shape[0] < 3: 
                                    p2 = np.append(p2, [0.0] * (3 - p2.shape[0]))

                                # Validate coordinates
                                if (len(p1) >= 3 and len(p2) >= 3 and
                                    all(isinstance(x, (int, float)) for x in p1[:3]) and
                                    all(isinstance(x, (int, float)) for x in p2[:3])):
                                    
                                    segment_line = pv.Line([float(p1[0]), float(p1[1]), float(p1[2])],
                                                        [float(p2[0]), float(p2[1]), float(p2[2])])
                                    self.current_plotter.add_mesh(segment_line, color=color, line_width=2.5)
                                else:
                                    logger.warning(f"Skipping invalid segment for {name}")
                                    
                            except Exception as e:
                                logger.warning(f"Error creating segment for {name}: {e}")
                                continue

                        plotter_has_geometry = True

                elif view_type == "triangulation":
                    triangulation_result = dataset.get('triangulation_result')
                    if triangulation_result is not None:
                        vertices = triangulation_result.get('vertices')
                        triangles = triangulation_result.get('triangles')
                        
                        if vertices is not None and len(vertices) > 0 and triangles is not None and len(triangles) > 0:
                            # Convert vertices to 3D, respecting Z coordinates 
                            if vertices.shape[1] == 2:
                                vertices_3d = np.zeros((len(vertices), 3))
                                vertices_3d[:, 0:2] = vertices
                                
                                # Try to map Z values from original points
                                for j in range(len(vertices)):
                                    vertex_2d = vertices[j]
                                    # Find closest point in original dataset for Z value
                                    distances = np.sum((points_3d[:, 0:2] - vertex_2d)**2, axis=1)
                                    closest_idx = np.argmin(distances)
                                    vertices_3d[j, 2] = points_3d[closest_idx, 2]
                            else:
                                # Use Z values if they're already present
                                vertices_3d = vertices.copy()[:, 0:3]
                            
                            cells = np.hstack([np.full((len(triangles), 1), 3), triangles])
                            mesh = pv.PolyData(vertices_3d, cells)
                            
                            self.current_plotter.add_mesh(mesh, color=color, opacity=0.7, 
                                                    show_edges=True, edge_color=color, 
                                                    line_width=1, specular=0.5, label=name)
                            plotter_has_geometry = True

            # Add axes and reset camera only if something was plotted
            if plotter_has_geometry:
                self.current_plotter.add_axes()
                self.current_plotter.reset_camera()
            else:
                logger.warning("No geometry added to the plotter for the current view.")
            
            # Add controls for adjustment
            controls_widget = QWidget()
            controls_layout = QHBoxLayout(controls_widget)
            controls_layout.setContentsMargins(5, 2, 5, 2)
            
            # Height adjustment slider - for Z axis exaggeration
            controls_layout.addWidget(QLabel("Z Exaggeration:"))
            height_slider = QSlider(Qt.Horizontal)
            height_slider.setMinimum(1)
            height_slider.setMaximum(100)
            height_slider.setValue(int(self.height_factor * 20))
            height_slider.valueChanged.connect(lambda v: self._set_height_factor_and_update(v / 20.0))
            controls_layout.addWidget(height_slider)
            
            # Add zoom controls
            controls_layout.addStretch(1)
            controls_layout.addWidget(QLabel("Zoom:"))
            zoom_in_btn = QPushButton("+")
            zoom_in_btn.setMaximumWidth(30)
            zoom_in_btn.clicked.connect(lambda: self.current_plotter.camera.zoom(1.2))
            controls_layout.addWidget(zoom_in_btn)
            
            zoom_out_btn = QPushButton("-")
            zoom_out_btn.setMaximumWidth(30)
            zoom_out_btn.clicked.connect(lambda: self.current_plotter.camera.zoom(1/1.2))
            controls_layout.addWidget(zoom_out_btn)
            
            # Reset view button
            reset_btn = QPushButton("Reset View")
            reset_btn.clicked.connect(lambda: self.current_plotter.reset_camera())
            controls_layout.addWidget(reset_btn)
            
            vis_container_layout.addWidget(controls_widget)

        except ImportError:
            logger.error("PyVistaQt import failed unexpectedly.")
            error_msg = QLabel("Error: Failed to load PyVistaQt.")
            error_msg.setAlignment(Qt.AlignCenter)
            vis_container_layout.addWidget(error_msg)
        except Exception as e:
            error_msg_text = f"Error creating 3D view: {str(e)}\nCheck logs for details."
            error_msg = QLabel(error_msg_text)
            error_msg.setAlignment(Qt.AlignCenter)
            error_msg.setWordWrap(True)
            vis_container_layout.addWidget(error_msg)
            logger.exception(f"Error creating multi-dataset 3D view:")

        # Add the visualization container widget to the parent frame provided
        parent_layout.addWidget(vis_container_widget)

        return vis_container_widget

    def _set_height_factor_and_update(self, height_factor):
        """Sets the height factor and triggers a visualization update."""
        self.height_factor = height_factor
        # Trigger update for the currently visible tab
        current_index = self.notebook.currentIndex()
        self._on_tab_changed(current_index)
    def _update_embedded_intersection_view(self):
        """Update the embedded intersection view with bulletproof coordinate validation."""
        if not hasattr(self, 'intersection_plotter') or not self.intersection_plotter:
            return

        import pyvista as pv
        import numpy as np

        plotter = self.intersection_plotter
        plotter.clear()
        plotter.set_background([0.318, 0.341, 0.431])
        
        plotter_has_content = False
        
        def validate_and_convert_point(point):
            """Convert any point format to a valid 3D float tuple."""
            try:
                if point is None:
                    return None
                    
                # Convert to list
                if hasattr(point, 'tolist'):
                    point = point.tolist()
                elif isinstance(point, np.ndarray):
                    point = point.flatten().tolist()
                elif not isinstance(point, (list, tuple)):
                    if hasattr(point, '__getitem__') and hasattr(point, '__len__'):
                        try:
                            point = [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
                        except:
                            return None
                    else:
                        return None
                
                point = list(point)
                
                # Ensure exactly 3 coordinates
                if len(point) < 2:
                    return None
                elif len(point) < 3:
                    point.extend([0.0] * (3 - len(point)))
                elif len(point) > 3:
                    point = point[:3]
                
                # Convert to floats and validate
                try:
                    validated_point = [float(point[0]), float(point[1]), float(point[2])]
                    if any(not np.isfinite(x) for x in validated_point):
                        return None
                    return validated_point
                except (ValueError, TypeError):
                    return None
            except Exception:
                return None
        
        # Add mesh surfaces first for context
        for index, dataset in enumerate(self.datasets):
            tri_result = dataset.get('triangulation_result')
            if tri_result:
                vertices = tri_result.get('vertices')
                triangles = tri_result.get('triangles')
                color = dataset.get('color', '#CCCCCC')
                name = dataset.get('name', f'Dataset {index+1}')

                if vertices is not None and len(vertices) > 0 and triangles is not None and len(triangles) > 0:
                    try:
                        vertices = np.array(vertices)
                        if vertices.shape[1] == 2: 
                            temp_vertices = np.zeros((vertices.shape[0], 3))
                            temp_vertices[:,:2] = vertices
                            vertices = temp_vertices
                        elif vertices.shape[1] > 3: 
                            vertices = vertices[:, :3]
                        
                        triangles = np.array(triangles)
                        cells = np.hstack([np.full((len(triangles), 1), 3, dtype=triangles.dtype), triangles])
                        surface_mesh = pv.PolyData(vertices, faces=cells)
                        
                        plotter.add_mesh(surface_mesh, color=color, opacity=0.4, 
                                        show_edges=True, edge_color='black', 
                                        line_width=1, label=name)
                        plotter_has_content = True
                    except Exception as e:
                        logger.error(f"Error creating mesh for dataset {index} ('{name}') in embedded view: {e}")

        # Add intersection lines with robust validation
        all_intersection_lines = []
        if hasattr(self, 'datasets_intersections') and self.datasets_intersections:
            for dataset_index, intersections in self.datasets_intersections.items():
                for intersection in intersections:
                    intersection_points = intersection.get('points', [])
                    
                    if intersection_points and len(intersection_points) >= 2:
                        valid_points = []
                        for point in intersection_points:
                            validated = validate_and_convert_point(point)
                            if validated is not None:
                                valid_points.append(validated)
                        
                        if len(valid_points) >= 2:
                            all_intersection_lines.append(valid_points)

        # Draw intersection lines
        if all_intersection_lines:
            for line_points in all_intersection_lines:
                try:
                    for i in range(len(line_points) - 1):
                        point_a = line_points[i]
                        point_b = line_points[i + 1]
                        
                        if (len(point_a) == 3 and len(point_b) == 3 and
                            all(isinstance(x, (int, float)) and np.isfinite(x) for x in point_a) and
                            all(isinstance(x, (int, float)) and np.isfinite(x) for x in point_b)):
                            
                            segment = pv.Line(point_a, point_b)
                            plotter.add_mesh(segment, color='red', line_width=4)
                            plotter_has_content = True
                            
                except Exception as e:
                    logger.error(f"Error adding intersection line segment in embedded view: {e}")

        # Add triple points with robust validation
        if hasattr(self, 'triple_points') and self.triple_points:
            try:
                valid_triple_points = []
                for tp in self.triple_points:
                    point = tp.get('point')
                    if point is not None:
                        validated = validate_and_convert_point(point)
                        if validated is not None:
                            valid_triple_points.append(validated)
                
                if valid_triple_points:
                    points_array = np.array(valid_triple_points, dtype=np.float64)
                    if (points_array.shape[0] > 0 and points_array.shape[1] == 3 and 
                        np.all(np.isfinite(points_array))):
                        
                        triple_points_poly = pv.PolyData(points_array)
                        plotter.add_points(triple_points_poly, color='black', point_size=8, 
                                        render_points_as_spheres=True, label="Triple Points")
                        plotter_has_content = True
                        
            except Exception as e:
                logger.error(f"Error adding triple points in embedded view: {e}")

        if plotter_has_content:
            plotter.add_legend(bcolor=None, face='circle', border=False, size=(0.15, 0.15))
            plotter.add_axes()
            plotter.reset_camera()
            logger.info("Updated embedded intersection view.")
        else:
            plotter.add_text("No valid intersection data to display.", position='upper_edge', color='white')
            logger.info("Embedded intersection view updated, but no content added.")
    def _on_tab_changed(self, index):
        """Handle tab changes to update visualizations as needed"""
        logger.debug(f"Tab changed to index {index}")
        # Get the widget associated with the current index
        current_tab_widget = self.notebook.widget(index)

        if not self.datasets:
            logger.debug("No datasets loaded, clearing visualizations.")
            self._clear_visualizations() # Clears all known plots
            # Explicitly clear the current tab's specific plot if it has one and _clear_visualizations doesn't cover it
            if current_tab_widget == self.file_tab: pass # Covered
            elif current_tab_widget == self.hull_tab: self._clear_hull_plot()
            elif current_tab_widget == self.segment_tab: self._clear_segment_plot()
            elif current_tab_widget == self.triangulation_tab: self._clear_triangulation_plot()
            elif current_tab_widget == self.intersection_tab: self._clear_intersection_plot()
            elif current_tab_widget == self.refine_mesh_tab: self._clear_refine_mesh_plot() # Add this
            elif current_tab_widget == self.pre_tetramesh_tab: self._clear_pre_tetramesh_plot() # Add this
            elif hasattr(self, 'tetra_mesh_tab') and current_tab_widget == self.tetra_mesh_tab: self._clear_tetra_mesh_plot() # Add this
            return

        if current_tab_widget == self.file_tab:  # File tab
            self._visualize_all_points()
        elif current_tab_widget == self.hull_tab:  # Hull tab
            needs_update = any(d.get('visible', True) and d.get('hull_points') is not None for d in self.datasets)
            if needs_update: self._visualize_all_hulls()
            else: self._clear_hull_plot(); self.statusBar().showMessage("No visible hulls computed.")
        elif current_tab_widget == self.segment_tab:  # Segment tab
            needs_update = any(d.get('visible', True) and d.get('segments') is not None for d in self.datasets)
            if needs_update: self._visualize_all_segments()
            else: self._clear_segment_plot(); self.statusBar().showMessage("No visible segments computed.")
        elif current_tab_widget == self.triangulation_tab:  # Triangulation tab
            needs_update = any(d.get('visible', True) and d.get('triangulation_result') is not None for d in self.datasets)
            if needs_update: self._visualize_all_triangulations()
            else: self._clear_triangulation_plot(); self.statusBar().showMessage("No visible triangulations computed.")
        elif current_tab_widget == self.intersection_tab:  # Intersection tab
            has_intersections = hasattr(self, 'datasets_intersections') and bool(self.datasets_intersections)
            if has_intersections: self._visualize_intersections()
            else: self._clear_intersection_plot(); self.statusBar().showMessage("No intersections computed yet.")
        elif current_tab_widget == self.refine_mesh_tab: # Refine & Mesh tab - Add this block
            has_refined_intersections = hasattr(self, 'datasets_intersections') and bool(self.datasets_intersections) # Check if any intersections exist (refined or not)
            if has_refined_intersections:
                self._visualize_refined_intersections() # This will show current state of intersections
            else:
                self._clear_refine_mesh_plot()
                self.statusBar().showMessage("No intersections to refine or display.")
        elif current_tab_widget == self.pre_tetramesh_tab: # Pre-Tetramesh tab - Add this block
            if any(d.get('constrained_triangulation_result') for d in self.datasets):
                self._visualize_constrained_meshes()
            else:
                self._clear_pre_tetramesh_plot()
                self.statusBar().showMessage("No constrained surface meshes computed yet.")
        elif hasattr(self, 'tetra_mesh_tab') and current_tab_widget == self.tetra_mesh_tab: # Tetra Mesh tab
            # Update surface table when switching to tetra mesh tab
            if hasattr(self, '_update_surface_table'):
                self._update_surface_table()
            # Show existing tetrahedral mesh if available
            if hasattr(self, 'tetrahedral_mesh') and self.tetrahedral_mesh:
                self._visualize_tetrahedral_mesh()
            else:
                # Initialize visualization with data from previous tab
                if hasattr(self, 'tetra_plotter') and self.tetra_plotter:
                    self.tetra_plotter.clear()
                    # Force a full visualization update with all elements
                    self._update_unified_visualization(filter_changed=True)
                    self.statusBar().showMessage("Ready for tetrahedral mesh generation.")
    # ... rest of the class methods (_get_next_color, _create_main_layout, etc.) ...

    def _on_hulls_visibility_changed(self, checked):
        """Handle convex hulls visibility changes dynamically."""
        if not hasattr(self, 'tetra_plotter') or not self.tetra_plotter:
            return
            
        logger.info(f"Convex hulls visibility changed: {checked}")
        self._update_unified_visualization(hulls_changed=True)
    

    
    def _on_intersections_visibility_changed(self, checked):
        """Handle intersections visibility changes dynamically."""
        if not hasattr(self, 'tetra_plotter') or not self.tetra_plotter:
            return
            
        logger.info(f"Intersections visibility changed: {checked}")
        self._update_unified_visualization(intersections_changed=True)
    
    def _on_highlighting_changed(self, checked):
        """Handle selection highlighting changes dynamically."""
        if not hasattr(self, 'tetra_plotter') or not self.tetra_plotter:
            return
            
        logger.info(f"Selection highlighting changed: {checked}")
        self._update_unified_visualization(highlighting_changed=True)
    
    def _on_surface_display_changed(self):
        """Handle surface display mode changes (faces/wireframe/both)."""
        if not hasattr(self, 'tetra_plotter') or not self.tetra_plotter:
            return
            
        # Determine current display mode
        if self.show_faces_radio.isChecked():
            display_mode = "faces"
        elif self.show_wireframe_radio.isChecked():
            display_mode = "wireframe"
        else:
            display_mode = "both"
            
        logger.info(f"Surface display mode changed: {display_mode}")
        
        # Use unified visualization system that respects current filter
        self._update_unified_visualization(display_mode_changed=True)
    
    def _add_convex_hulls_to_plotter(self):
        """Add convex hulls to the plotter."""
        try:
            import pyvista as pv
            import numpy as np
            
            for i, dataset in enumerate(self.datasets):
                if 'hull_points' in dataset:
                    hull_points = dataset.get('hull_points')
                    if self._has_valid_array_data(hull_points):
                        self._add_convex_hull_to_plotter(i, hull_points)
                        
            logger.info("Added convex hulls to visualization")
        except Exception as e:
            logger.error(f"Error adding convex hulls: {e}")
    
    def _remove_convex_hulls_from_plotter(self):
        """Remove convex hulls from the plotter."""
        try:
            # Remove all hull-related actors
            actors_to_remove = []
            for actor_name in self.tetra_plotter.renderer.actors:
                if 'hull' in actor_name.lower() or 'convex' in actor_name.lower():
                    actors_to_remove.append(actor_name)
            
            for actor_name in actors_to_remove:
                try:
                    self.tetra_plotter.remove_actor(actor_name)
                except:
                    pass
                    
            logger.info("Removed convex hulls from visualization")
        except Exception as e:
            logger.error(f"Error removing convex hulls: {e}")
    

    

    
    def _add_intersections_to_plotter(self):
        """Add intersections to the plotter."""
        try:
            self._add_all_intersections_to_plotter()
            logger.info("Added intersections to visualization")
        except Exception as e:
            logger.error(f"Error adding intersections: {e}")
    
    def _remove_intersections_from_plotter(self):
        """Remove intersections from the plotter."""
        try:
            # Remove all intersection-related actors
            actors_to_remove = []
            for actor_name in self.tetra_plotter.renderer.actors:
                if 'intersection' in actor_name.lower() or 'triple' in actor_name.lower():
                    actors_to_remove.append(actor_name)
            
            for actor_name in actors_to_remove:
                try:
                    self.tetra_plotter.remove_actor(actor_name)
                except:
                    pass
                    
            logger.info("Removed intersections from visualization")
        except Exception as e:
            logger.error(f"Error removing intersections: {e}")
    
    def _remove_selection_highlighting(self):
        """Remove selection highlighting from all surfaces."""
        try:
            # Restore normal appearance for all selected surfaces
            for surface_index in self.selected_surfaces:
                self._update_surface_selection_visual(surface_index, False)
                
            logger.info("Removed selection highlighting")
        except Exception as e:
            logger.error(f"Error removing selection highlighting: {e}")
    
    def _update_surface_display_mode(self, display_mode):
        """Update the display mode for all surface actors."""
        try:
            import pyvista as pv
            
            # Get all surface actors
            for i, dataset in enumerate(self.datasets):
                actor_name = f"surface_{i}"
                try:
                    # Get the current mesh data
                    vertices = None
                    triangles = None
                    
                    # Try to get mesh data from different sources
                    if 'constrained_vertices' in dataset and 'constrained_triangles' in dataset:
                        vertices = dataset.get('constrained_vertices')
                        triangles = dataset.get('constrained_triangles')
                    elif 'triangulated_vertices' in dataset and 'triangulated_triangles' in dataset:
                        vertices = dataset.get('triangulated_vertices')
                        triangles = dataset.get('triangulated_triangles')
                    
                    if vertices is not None and triangles is not None:
                        # Remove old actor
                        try:
                            self.tetra_plotter.remove_actor(actor_name)
                        except:
                            pass
                        
                        # Create new mesh with updated display mode
                        self._add_surface_with_display_mode(i, vertices, triangles, display_mode)
                        
                except Exception as e:
                    logger.warning(f"Could not update display mode for surface {i}: {e}")
                    
            logger.info(f"Updated surface display mode to: {display_mode}")
        except Exception as e:
            logger.error(f"Error updating surface display mode: {e}")
    
    def _add_surface_with_display_mode(self, surface_index, vertices, triangles, display_mode):
        """Add a surface with the specified display mode."""
        try:
            import pyvista as pv
            import numpy as np
            
            vertices_array = np.array(vertices)
            triangles_array = np.array(triangles)
            
            # Ensure 3D coordinates
            if vertices_array.shape[1] == 2:
                temp_vertices = np.zeros((vertices_array.shape[0], 3))
                temp_vertices[:, :2] = vertices_array
                vertices_array = temp_vertices
            
            # Create PyVista mesh
            faces = []
            for tri in triangles_array:
                faces.extend([3, tri[0], tri[1], tri[2]])
            
            mesh = pv.PolyData(vertices_array, faces=faces)
            
            # Get surface properties
            surface_type = self._get_surface_type(surface_index)
            color = self._get_surface_color(surface_type)
            dataset = self.datasets[surface_index]
            actor_name = f"surface_{surface_index}"
            
            # Set display properties based on mode
            if display_mode == "faces":
                show_edges = False
                opacity = 0.7
                style = 'surface'
            elif display_mode == "wireframe":
                show_edges = True
                opacity = 1.0
                style = 'wireframe'
            else:  # both
                show_edges = True
                opacity = 0.6
                style = 'surface'
            
            # Add mesh to plotter
            self.tetra_plotter.add_mesh(
                mesh,
                name=actor_name,
                color=color,
                opacity=opacity,
                show_edges=show_edges,
                edge_color='black' if show_edges else None,
                line_width=1 if show_edges else None,
                style=style,
                pickable=True
            )
            
        except Exception as e:
            logger.error(f"Error adding surface {surface_index} with display mode {display_mode}: {e}")

    def _get_boundary_edges(self, triangles):
        """Finds the boundary edges from a list of triangles."""
        if triangles is None or len(triangles) == 0:
            return []

        edges = set()
        boundary_edges = []

        for tri in triangles:
            # Ensure triangle vertices are sorted to make edges canonical
            tri_sorted = sorted(tri)
            # Edges: (v0, v1), (v1, v2), (v0, v2) -> use smallest index first
            current_edges = [
                tuple(sorted((tri[i], tri[(i + 1) % 3]))) for i in range(3)
            ]

            for edge in current_edges:
                if edge in edges:
                    # This edge is shared by another triangle, remove it
                    edges.remove(edge)
                else:
                    # This is the first time we see this edge
                    edges.add(edge)

        # Remaining edges in the set are boundary edges
        return list(edges) # Return as a list of tuples (v1_idx, v2_idx)

    def _create_main_layout(self):
        """Creates the main layout after widgets are initialized."""
        # This assumes self.main_layout and self.notebook exist
        if not hasattr(self, 'main_layout'):
            logger.error("_create_main_layout called before main_layout initialized.")
            self.main_layout = QVBoxLayout(self.central_widget) # Attempt recovery

        if not hasattr(self, 'notebook'):
             logger.error("_create_main_layout called before notebook initialized.")
             self.notebook = QTabWidget() # Attempt recovery
             self.main_layout.addWidget(self.notebook)

        # Ensure tabs are added after the notebook is part of the main layout
        # The setup methods already add the tabs, this function primarily ensures
        # the notebook is correctly placed in the main layout.
        # No additional widget adding needed here if setup methods are correct.
        logger.debug("Main layout structure finalized.")


        


    def _run_batch_computation(self, compute_type, total_items):
        """Runs a batch computation using a worker thread."""
        logger.info(f"Starting batch computation: {compute_type}")
        self._disable_compute_buttons() # Disable buttons first

        # Setup progress dialog
        # For global intersection, total_items is conceptually 1 (the whole task)
        progress_max = 1 if compute_type == "intersections" else total_items
        self.progress_dialog = QProgressDialog(f"Computing {compute_type}...", "Cancel", 0, progress_max, self)
        self.progress_dialog.setWindowTitle(f"Processing {compute_type.capitalize()}")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(False) # We will close it manually
        self.progress_dialog.setAutoReset(False) # We will reset manually
        self.progress_dialog.setValue(0)
        self.progress_dialog.canceled.connect(self.cancel_computation) # Connect cancel signal
        self.progress_dialog.show()

        # Initialize counters for progress tracking
        self.processed_count = 0
        self.total_to_process = progress_max # Use the calculated max value

        # Setup worker and thread
        self.worker = ComputationWorker(self) # Pass GUI instance
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        # Connect signals
        if compute_type == "intersections":
            # Global intersection has only batch_finished and error signals
            self.worker.batch_finished.connect(self.handle_batch_finished)
            self.worker.error_occurred.connect(self.handle_computation_error)
            self.thread.started.connect(self.worker.compute_global_intersections_task)
        else:
            # Other types have per-dataset signals
            self.worker.dataset_finished.connect(self.handle_dataset_finished)
            self.worker.batch_finished.connect(self.handle_batch_finished)
            self.worker.error_occurred.connect(self.handle_computation_error)
            self.thread.started.connect(getattr(self.worker, f"compute_{compute_type}_batch")) # e.g., compute_hulls_batch
        
        # Common signals for thread management
        self.worker.batch_finished.connect(self.thread.quit) # Quit thread when batch finishes
        self.worker.error_occurred.connect(self.thread.quit) # Also quit on error
        self.thread.finished.connect(self.worker.deleteLater) # Schedule worker deletion
        self.thread.finished.connect(self.thread.deleteLater) # Schedule thread deletion

        # Start the thread
        self.thread.start()
        logger.info(f"Worker thread started for {compute_type}.")

    def handle_dataset_finished(self, index, name, success):
        """Handles the completion of computation for a single dataset."""
        # REMOVED old progress update logic here
        
        self.processed_count += 1

        # Update progress dialog
        # --- START EDIT: Check if progress_dialog exists ---
        if self.progress_dialog is not None:
            self.progress_dialog.setLabelText(f"Processed: {name} ({'OK' if success else 'Fail'}) - {self.processed_count}/{self.total_to_process}")
            self.progress_dialog.setValue(self.processed_count)
        # --- END EDIT ---

        # Update dataset list styling
        item = self.dataset_list_widget.item(index)
        if success:
            item.setForeground(QColor(Qt.black))
        else:
            item.setForeground(QColor(Qt.red))

    def handle_batch_finished(self, success_count, total_eligible, elapsed_time):
        """Handles the completion of a batch computation."""
        # Determine if this was the global intersection task
        is_intersection_task = (total_eligible == 1 and hasattr(self.worker, 'compute_global_intersections_task'))
        
        if is_intersection_task:
            task_name = "Global Intersection"
            success_status = "succeeded" if success_count == 1 else "failed"
            logger.info(f"Batch computation finished for {task_name}. Status: {success_status}. Time: {elapsed_time:.2f}s")
            self.statusBar().showMessage(f"{task_name} {success_status} in {elapsed_time:.2f}s.", 10000)
        else:
            task_name = "Batch computation"
            logger.info(f"{task_name} finished. Success: {success_count}/{total_eligible}. Time: {elapsed_time:.2f}s")
            self.statusBar().showMessage(f"Batch finished: {success_count}/{total_eligible} succeeded in {elapsed_time:.2f}s.", 10000)

        # Close progress dialog if it exists
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            # If it was the global intersection task, set value to max to show completion
            if is_intersection_task:
                self.progress_dialog.setValue(self.progress_dialog.maximum())
            self.progress_dialog.close()
            self.progress_dialog = None
            logger.debug("Progress dialog closed.")

        # Thread cleanup (moved thread check inside)
        if hasattr(self, 'thread') and self.thread and self.thread.isRunning():
            logger.info("Signaling worker thread to stop and quit.")
            if hasattr(self, 'worker') and self.worker:
                self.worker.stop() # Signal the worker loop to exit
            self.thread.quit() # Ask the thread's event loop to exit

            # Wait for the thread to finish (increased timeout to 5 seconds)
            if self.thread.wait(5000): # Wait up to 5000 ms
                logger.info("Worker thread finished gracefully.")
            else:
                # This is often a sign that the worker's run loop didn't exit cleanly.
                # Termination can be risky, so we'll log and detach.
                logger.warning("Worker thread did not finish gracefully after 5 seconds. Detaching.")
                # self.thread.terminate() # Avoid terminate() if possible, it's dangerous

        # Ensure references are cleared regardless of wait outcome
        self.thread = None
        self.worker = None
        logger.debug("Thread and worker references cleared.")

        self._enable_compute_buttons()
        # Update visualization and stats - This happens via QTimer in _compute_global_intersections for intersections
        if not is_intersection_task:
            self._update_visualization() 
            self._update_statistics()
        else:
            # For intersections, the updates are already scheduled via QTimer.singleShot
            # but we can ensure stats are updated one last time here just in case.
            self._update_statistics()

    def handle_computation_error(self, error_message):
        """Handles errors reported by the worker thread."""
        logger.error(f"Computation error: {error_message}")
        QMessageBox.critical(self, "Computation Error", f"An error occurred during computation:\n{error_message}")

        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
            logger.debug("Progress dialog closed due to error.")

        if self.thread and self.thread.isRunning():
            logger.info("Attempting to stop worker thread due to error.")
            if self.worker:
                self.worker.stop()
            self.thread.quit()

            if self.thread.wait(5000): # Wait up to 5000 ms
                logger.info("Worker thread finished after error.")
            else:
                logger.warning("Worker thread did not finish gracefully after error. Detaching.")
                # self.thread.terminate() # Avoid terminate()

        # Ensure references are cleared
        self.thread = None
        self.worker = None
        logger.debug("Thread and worker references cleared after error.")

        self._enable_compute_buttons()
        # Fix: Use self.statusBar() instead of self.status_bar
        self.statusBar().showMessage("Computation failed.", 5000)


    def cancel_computation(self):
        """Cancels the ongoing computation."""
        logger.info("Cancel computation requested by user.")
        if hasattr(self, 'worker') and self.worker:
             self.worker.stop() # Signal the worker to stop
             logger.debug("Stop signal sent to worker.")
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
             self.progress_dialog.setLabelText("Canceling computation...")
             self.progress_dialog.setEnabled(False) # Disable further interaction
             logger.debug("Progress dialog updated for cancellation.")
        # The batch_finished or error_occurred signal will handle cleanup


    def _disable_compute_buttons(self):
        """Disables the main compute buttons on all tabs."""
        logger.debug("Disabling compute buttons.")
        tabs_and_buttons = [
            (self.hull_tab, "compute_btn"),
            (self.segment_tab, "compute_btn"),
            (self.triangulation_tab, "run_btn"),
        ]
        for tab_widget, button_name in tabs_and_buttons:
            if tab_widget: # Check if tab exists
                button = tab_widget.findChild(QPushButton, button_name)
                if button:
                    button.setEnabled(False)
                else:
                    logger.warning(f"Could not find button '{button_name}' in tab {tab_widget.objectName()} to disable.")
            else:
                 logger.warning(f"Tab widget not found when trying to disable button '{button_name}'.")
        
        # Handle intersection buttons directly
        if hasattr(self, 'compute_intersections_btn'):
            self.compute_intersections_btn.setEnabled(False)
        
        # Also disable compute all button for intersections if it exists
        if hasattr(self, 'compute_all_intersections_btn'):
            self.compute_all_intersections_btn.setEnabled(False)


    def _enable_compute_buttons(self):
        """Enables the main compute buttons on all tabs."""
        logger.debug("Enabling compute buttons.")
        tabs_and_buttons = [
            (self.hull_tab, "compute_btn"),
            (self.segment_tab, "compute_btn"),
            (self.triangulation_tab, "run_btn"),
        ]
        for tab_widget, button_name in tabs_and_buttons:
             if tab_widget: # Check if tab exists
                 button = tab_widget.findChild(QPushButton, button_name)
                 if button:
                     button.setEnabled(True)
                 else:
                      logger.warning(f"Could not find button '{button_name}' in tab {tab_widget.objectName()} to enable.")
             else:
                 logger.warning(f"Tab widget not found when trying to enable button '{button_name}'.")
        
        # Handle intersection buttons directly
        if hasattr(self, 'compute_intersections_btn'):
            self.compute_intersections_btn.setEnabled(True)
        
        # Also enable compute all button for intersections if it exists
        if hasattr(self, 'compute_all_intersections_btn'):
            self.compute_all_intersections_btn.setEnabled(True)


    def closeEvent(self, event):
        """Handle closing the application, especially if a thread is running."""
        logger.debug("Close event triggered.")
        thread_running = False
        try:
            # Check if a worker thread is running, handle potential AttributeError
            if hasattr(self, 'thread') and self.thread is not None and self.thread.isRunning():
                thread_running = True
        except AttributeError:
            logger.warning("AttributeError accessing thread.isRunning() during closeEvent. Assuming not running.")
            thread_running = False

        if thread_running:
            logger.info("Close event while worker thread is running.")
            reply = QMessageBox.question(self, 'Confirm Exit',
                                       "A computation is running. Stop and exit?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                logger.info("User chose to stop and exit.")
                if hasattr(self, 'worker') and self.worker:
                    self.worker.stop() # Signal the worker to stop

                # Optionally wait briefly for the thread to stop naturally
                if not self.thread.wait(1000): # Wait 1 second
                    logger.warning("Thread did not finish quickly after stop signal. Terminating.")
                    self.thread.terminate() # Force stop if it doesn't finish
                    self.thread.wait() # Wait again after termination

                # Clean up all PyVista plotters and OpenGL contexts
                self._cleanup_pyvista_plotters()
                event.accept() # Allow closing
                logger.info("Application closing after stopping thread.")
            else:
                logger.info("User chose not to exit.")
                event.ignore() # Prevent closing
        else:
            logger.debug("No active thread running. Closing application.")
            # Clean up all PyVista plotters and OpenGL contexts
            self._cleanup_pyvista_plotters()
            event.accept() # No thread running, close normally

    def compute_intersections(self):
        """Compute intersections globally (uses all eligible datasets)."""
        # Check if there are at least two datasets with triangulation
        eligible_datasets = [i for i, d in enumerate(self.datasets) if d.get('triangulation_result') is not None]
        
        if len(eligible_datasets) < 2:
            QMessageBox.warning(self, "Not Enough Data", 
                               "Need at least two triangulated datasets to compute intersections.")
            return
        
        logger.info("Triggering global intersection computation.")
        # Disable compute buttons during computation
        # self._disable_compute_buttons() # This is now handled by _run_batch_computation
        
        # Run the single global computation task in a worker thread
        # total_items is 1 because it's one global task
        self._run_batch_computation("intersections", 1)

    def compute_all_intersections(self):
        """Compute intersections globally (this button now does the same as Compute Intersections)."""
        # This button becomes redundant with the global approach, but we keep the UI
        # and just call the same logic.
        self.compute_intersections()

    def _compute_global_intersections(self):
        """
        Compute intersections globally for all datasets with triangulation data.
        This method is called by the worker thread.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting global intersection computation...")
        try:
            from meshit.intersection_utils import (
                Vector3D, Triangle, Intersection, TriplePoint, 
                run_intersection_workflow
            )
        except ImportError as e:
            logger.error(f"Failed to import intersection utilities: {e}")
            QMessageBox.critical(self, "Import Error", f"Failed to import intersection utilities: {e}\nPlease ensure meshit.intersection_utils is available.")
            return False
        
        # Check if there are any datasets with triangulation
        eligible_dataset_indices = [i for i, d in enumerate(self.datasets) if d.get('triangulation_result') is not None]
        
        if len(eligible_dataset_indices) < 2:
            logger.warning("Need at least two triangulated datasets to compute intersections.")
            QMessageBox.warning(self, "Not Enough Data", "Need at least two triangulated datasets to compute intersections.")
            return False
        
        logger.info(f"Found {len(eligible_dataset_indices)} eligible datasets for intersection.")

        # Create a structure similar to MeshItModel to store all eligible datasets
        class ModelWrapper:
            def __init__(self):
                self.surfaces = []
                self.model_polylines = [] # Future extension: Add polylines if needed
                self.intersections = []
                self.triple_points = []
                self.original_indices = {} # Map model surface index back to original dataset index
        
        model = ModelWrapper()
        
        # Populate model with surfaces from eligible datasets
        for original_index in eligible_dataset_indices:
            dataset = self.datasets[original_index]
            
            # Create a surface object for this dataset
            class SurfaceWrapper:
                def __init__(self, dataset, index):
                    self.name = dataset.get('name', f"Dataset {index+1}")
                    self.vertices = [] # Vertices corresponding to the triangulation
                    self.triangles = [] # Triangle indices referencing self.vertices
                    self.convex_hull = []
                    self.bounds = [Vector3D(), Vector3D()]
                    self.type = "Surface"
                    
                    tri_result = dataset.get('triangulation_result')
                    if not tri_result or 'vertices' not in tri_result or 'triangles' not in tri_result:
                        logger.warning(f"Skipping SurfaceWrapper for dataset {index}: Missing valid triangulation result (vertices or triangles).")
                        return # Skip if no valid triangulation result
                        
                    # --- CORRECTED: Use vertices from triangulation result --- 
                    tri_vertices = tri_result['vertices']
                    self.triangles = tri_result['triangles']
                    
                    if tri_vertices is None or len(tri_vertices) == 0 or self.triangles is None:
                         logger.warning(f"Skipping SurfaceWrapper for dataset {index}: Empty vertices or triangles in triangulation result.")
                         return
                         
                    # Convert triangulation vertices to Vector3D
                    for point in tri_vertices:
                         # Ensure points are 3D
                         if len(point) >= 3:
                             self.vertices.append(Vector3D(point[0], point[1], point[2]))
                         elif len(point) == 2:
                             self.vertices.append(Vector3D(point[0], point[1], 0.0)) # Assume Z=0 for 2D
                         else:
                             logger.warning(f"Skipping invalid vertex in triangulation result for dataset {index}: {point}")
                    
                    if not self.vertices:
                        logger.warning(f"Skipping SurfaceWrapper for dataset {index}: No valid Vector3D vertices created from triangulation result.")
                        return # Skip if no valid vertices could be created
                        
                    # Ensure triangle indices are valid for the created vertices list
                    max_vertex_index = len(self.vertices) - 1
                    valid_triangles = []
                    for tri in self.triangles:
                        if all(0 <= idx <= max_vertex_index for idx in tri):
                            valid_triangles.append(tri)
                        else:
                            logger.warning(f"Skipping invalid triangle in dataset {index} (indices out of bounds): {tri}")
                    self.triangles = valid_triangles
                    
                    if not self.triangles:
                        logger.warning(f"Skipping SurfaceWrapper for dataset {index}: No valid triangles remain after index check.")
                        return
                    # --- END CORRECTION --- 
                    
                    # Add convex hull if available (optional for intersection but good practice)
                    hull_points_data = dataset.get('hull_points')
                    if hull_points_data is not None and len(hull_points_data) > 0:
                        self.convex_hull = [Vector3D(p[0], p[1], p[2]) for p in hull_points_data if len(p) >= 3]
                    
                    # Calculate bounds for early rejection test using the triangulation vertices
                    if self.vertices:
                        min_x = min(v.x for v in self.vertices)
                        min_y = min(v.y for v in self.vertices)
                        min_z = min(v.z for v in self.vertices)
                        max_x = max(v.x for v in self.vertices)
                        max_y = max(v.y for v in self.vertices)
                        max_z = max(v.z for v in self.vertices)
                        
                        self.bounds[0] = Vector3D(min_x, min_y, min_z)
                        self.bounds[1] = Vector3D(max_x, max_y, max_z)
            
            # Add surface to model if valid
            surface = SurfaceWrapper(dataset, original_index)
            # --- START CORRECTED BLOCK ---
            # Check the validity of the surface before adding
            # Use explicit length checks for lists/arrays to avoid ValueError
            vertices_valid = hasattr(surface, 'vertices') and surface.vertices is not None and len(surface.vertices) > 0
            triangles_valid = hasattr(surface, 'triangles') and surface.triangles is not None and len(surface.triangles) > 0

            if vertices_valid and triangles_valid:
                # If checks pass, add the surface
                model_surface_index = len(model.surfaces)
                model.surfaces.append(surface)
                model.original_indices[model_surface_index] = original_index # Store mapping
                logger.debug(f"Added dataset {original_index} as model surface {model_surface_index}")
            else:
                # Log if checks fail
                logger.warning(f"Dataset {original_index} ('{dataset.get('name')}') could not be added to intersection model (missing valid vertices or triangles). Vertices valid: {vertices_valid}, Triangles valid: {triangles_valid}")
            # --- END CORRECTED BLOCK ---

        # Check if we have enough valid surfaces in the model
        if len(model.surfaces) < 2:
            logger.warning("Need at least two valid surfaces in the model to compute intersections.")
            QMessageBox.warning(self, "Not Enough Valid Data", "Need at least two valid triangulated datasets for intersection computation.")
            return False

        logger.info(f"Prepared intersection model with {len(model.surfaces)} surfaces.")

        # Initialize progress reporting function for intersection_utils
        def progress_callback(message):
            # We can log this, but the main progress dialog is handled by batch_finished
            logger.debug(f"Intersection util progress: {message.strip()}")
            QApplication.processEvents() # Keep UI responsive during internal steps
        
        # --- Run the intersection workflow --- 
        try:
            logger.info("Calling run_intersection_workflow...")
            # Configure constraint processing for intersection workflow
            intersection_config = {
                'use_constraint_processing': True,
                'type_based_sizing': True,
                'hierarchical_constraints': True,
                'gradient': 2.0
            }
            model = run_intersection_workflow(model, progress_callback, config=intersection_config)
            logger.info("run_intersection_workflow finished.")
        except Exception as e:
            error_msg = f"Error during run_intersection_workflow: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Intersection Computation Error", error_msg)
            return False
        
        # --- Store the results --- 
        # Clear previous results
        self.datasets_intersections = {} # Store intersections per *original* dataset index
        self.triple_points = [] # Store triple points globally
        
        logger.info(f"Processing {len(model.intersections)} raw intersections and {len(model.triple_points)} triple points from workflow.")

        # Process intersections
        found_intersections_count = 0
        for intersection in model.intersections:
            # Map model surface indices back to original dataset indices
            original_id1 = model.original_indices.get(intersection.id1, -1)
            original_id2 = model.original_indices.get(intersection.id2, -1)
            
            if original_id1 == -1 or original_id2 == -1:
                logger.warning(f"Skipping intersection with invalid original index mapping (IDs: {intersection.id1}, {intersection.id2})")
                continue # Skip invalid mappings
            
            # Convert points to regular lists for storage
            points = []
            for point in intersection.points:
                points.append([point.x, point.y, point.z])
            
            # Create intersection info
            intersection_info = {
                'dataset_id1': original_id1,
                'dataset_id2': original_id2,
                'is_polyline_mesh': intersection.is_polyline_mesh,
                'points': points
            }
            found_intersections_count += 1

            # Store only with the dataset having the lower ID
            if original_id1 <= original_id2:
                if original_id1 not in self.datasets_intersections:
                    self.datasets_intersections[original_id1] = []
                self.datasets_intersections[original_id1].append(intersection_info)
            else:
                if original_id2 not in self.datasets_intersections:
                    self.datasets_intersections[original_id2] = []
                self.datasets_intersections[original_id2].append(intersection_info)
  
        # Process triple points
        for tp in model.triple_points:
            point = [tp.point.x, tp.point.y, tp.point.z]
            # Note: intersection_ids in TriplePoint refer to the indices within model.intersections
            # We might need to map these if we store intersections differently, but for now, store raw indices.
            intersection_ids = tp.intersection_ids 
            self.triple_points.append({
                'point': point,
                'intersection_ids': intersection_ids
            })
        
        logger.info(f"Stored {found_intersections_count} intersections across datasets and {len(self.triple_points)} triple points.")

        # --- Update UI --- 
        # Use QTimer.singleShot to ensure UI updates happen on the main thread
        # after the worker thread finishes processing this method.
        QTimer.singleShot(0, self._update_statistics)
        QTimer.singleShot(0, self._update_intersection_list)
        QTimer.singleShot(0, self._visualize_intersections) # Or visualize selected if preferred
        
        logger.info("Global intersection computation successful.")
        return True # Indicate success
    def _clear_intersection_results(self):
        """Clear all intersection results"""
        # Clear intersection data
        self.datasets_intersections = {}
        self.triple_points = []
        
        # Clear UI
        self.intersection_list.clear()
        self._clear_intersection_plot()
        self._update_statistics()
        
        QMessageBox.information(self, "Results Cleared", "Intersection results have been cleared.")

    def _clear_intersection_plot(self):
        """Clear the embedded intersection PyVista plotter."""
        if hasattr(self, 'intersection_plotter') and self.intersection_plotter:
            self.intersection_plotter.clear()
            # Optionally add placeholder text back if desired
            # self.intersection_plotter.add_text("Compute intersections or select one from the list.", position='upper_edge')
            self.intersection_plotter.reset_camera()
        else:
            # Fallback for non-PyVista or error cases
            if hasattr(self, 'intersection_plot_layout'):
                 # Clear any potential old matplotlib widgets or placeholders
                 for i in reversed(range(self.intersection_plot_layout.count())):
                     widget = self.intersection_plot_layout.itemAt(i).widget()
                     if widget:
                         # Check if it's the plotter interactor itself before deleting
                         if hasattr(self, 'intersection_plotter') and self.intersection_plotter and widget == self.intersection_plotter.interactor:
                             continue # Don't delete the main interactor widget
                         widget.setParent(None)
                         widget.deleteLater()
                 # Add text placeholder if plotter doesn't exist
                 if not hasattr(self, 'intersection_plotter') or not self.intersection_plotter:
                      placeholder = QLabel("PyVista required or plot cleared.")
                      placeholder.setAlignment(Qt.AlignCenter)
                      self.intersection_plot_layout.addWidget(placeholder)

    def _update_intersection_list(self):
        """Update the list of intersections in the UI"""
        self.intersection_list.clear()
        
        # Add all intersections to the list
        for dataset_index, intersections in self.datasets_intersections.items():
            if not intersections:
                continue
                
            dataset_name = self.datasets[dataset_index].get('name', f"Dataset {dataset_index+1}")
            
            # Add a header item for this dataset
            dataset_item = QListWidgetItem(f"{dataset_name} Intersections:")
            dataset_item.setBackground(QColor(240, 240, 240))
            dataset_item.setFlags(dataset_item.flags() & ~Qt.ItemIsSelectable)
            self.intersection_list.addItem(dataset_item)
            
            # Add intersections for this dataset
            for i, intersection in enumerate(intersections):
                other_dataset_id = (intersection['dataset_id1'] 
                                   if intersection['dataset_id1'] != dataset_index 
                                   else intersection['dataset_id2'])
                
                other_dataset_name = self.datasets[other_dataset_id].get('name', f"Dataset {other_dataset_id+1}")
                
                # Create a descriptive name for the intersection
                intersection_type = "Polyline-Surface" if intersection['is_polyline_mesh'] else "Surface-Surface"
                num_points = len(intersection['points'])
                
                item_text = f"  {intersection_type}: with {other_dataset_name} ({num_points} points)"
                
                # Create item with data
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, {
                    'dataset_index': dataset_index,
                    'intersection_index': i
                })
                
                self.intersection_list.addItem(item)

    def _on_intersection_selection_changed(self):
        """Handle selection changes in the intersection list"""
        selected_items = self.intersection_list.selectedItems()
        
        if not selected_items:
            return
            
        item = selected_items[0]
        data = item.data(Qt.UserRole)
        
        if not data:
            return  # Header item or invalid selection
        
        dataset_index = data['dataset_index']
        intersection_index = data['intersection_index']
        
        # Visualize the selected intersection
        self._visualize_selected_intersection(dataset_index, intersection_index)

    def _visualize_selected_intersection(self, dataset_index, intersection_index):
        """Visualize a specific intersection by highlighting it in the embedded plotter."""
        if not hasattr(self, 'intersection_plotter') or not self.intersection_plotter:
            logger.warning("Intersection plotter not available for selection visualization.")
            return

        plotter = self.intersection_plotter
        # Don't clear everything, just update highlights or add emphasis
        # Re-plotting everything might be simpler for now
        logger.info(f"Highlighting intersection {intersection_index} from dataset {dataset_index}")
        
        # --- Re-plot everything, but highlight the selected intersection --- 
        plotter.clear()
        plotter.set_background('white')
        
        if dataset_index not in self.datasets_intersections or intersection_index >= len(self.datasets_intersections[dataset_index]):
            logger.warning("Selected intersection data not found.")
            plotter.add_text("Selected intersection not found.", position='upper_edge', color='red')
            return
            
        selected_intersection = self.datasets_intersections[dataset_index][intersection_index]
        involved_dataset_ids = {selected_intersection['dataset_id1'], selected_intersection['dataset_id2']}
        
        # Add involved dataset meshes (dim others slightly)
        plotter_has_content = False
        for index, dataset in enumerate(self.datasets):
            # Determine opacity: Highlight involved, dim others (or keep all same)
            is_involved = index in involved_dataset_ids
            opacity = 0.7 # Use consistent opacity like triangulation step
            # opacity = 0.7 if is_involved else 0.2 # Alternative: Dim uninvolved

            tri_result = dataset.get('triangulation_result')
            if tri_result:
                vertices = tri_result.get('vertices')
                triangles = tri_result.get('triangles')
                color = dataset.get('color', self.DEFAULT_COLORS[index % len(self.DEFAULT_COLORS)])
                name = dataset.get('name', f'Dataset {index+1}')

                if vertices is not None and len(vertices) > 0 and triangles is not None and len(triangles) > 0:
                    try:
                        vertices = np.array(vertices)
                        if vertices.shape[1] == 2: 
                             temp_vertices = np.zeros((vertices.shape[0], 3)); temp_vertices[:,:2] = vertices; vertices = temp_vertices
                        elif vertices.shape[1] > 3: vertices = vertices[:, :3]
                        triangles = np.array(triangles)
                        cells = np.hstack([np.full((len(triangles), 1), 3, dtype=triangles.dtype), triangles])
                        surface_mesh = pv.PolyData(vertices, faces=cells)
                        # --- Apply Triangulation Style --- 
                        plotter.add_mesh(surface_mesh, color=color, opacity=opacity, 
                                         show_edges=True, edge_color=color, 
                                         line_width=1, specular=0.5, label=name)
                        plotter_has_content = True
                    except Exception as e:
                         logger.error(f"Error creating mesh for dataset {index} ('{name}') during selection view: {e}")

        # Add the selected intersection line prominently (keep distinct style)
        if selected_intersection['points'] and len(selected_intersection['points']) >= 2:
            try:
                # Validate intersection points first
                valid_points = []
                for point in selected_intersection['points']:
                    if hasattr(point, 'tolist'):
                        point = point.tolist()
                    elif isinstance(point, np.ndarray):
                        point = point.flatten().tolist()
                    
                    # Ensure 3D coordinates
                    if len(point) >= 2:
                        if len(point) < 3:
                            point.extend([0.0] * (3 - len(point)))
                        elif len(point) > 3:
                            point = point[:3]
                        
                        # Validate numbers
                        try:
                            validated_point = [float(point[0]), float(point[1]), float(point[2])]
                            if all(np.isfinite(x) for x in validated_point):
                                valid_points.append(validated_point)
                        except (ValueError, TypeError):
                            continue
                
                # Create line segments from valid points
                for i in range(len(valid_points) - 1):
                    segment = pv.Line(valid_points[i], valid_points[i + 1])
                    plotter.add_mesh(segment, color='#FF0000', line_width=6) 
                    plotter_has_content = True
                    
            except Exception as e:
                logger.error(f"Error adding selected intersection line segment: {e}")
                # Optionally add triple points if relevant to this intersection (more complex logic needed)
        # For simplicity, we might omit triple points in single selection view or show all
        if hasattr(self, 'triple_points') and self.triple_points:
            all_triple_points_coords = [tp['point'] for tp in self.triple_points]
            if all_triple_points_coords:
                try:
                    triple_points_poly = pv.PolyData(np.array(all_triple_points_coords))
                    plotter.add_points(triple_points_poly, color='black', point_size=8, render_points_as_spheres=True)
                    plotter_has_content = True
                except Exception as e:
                     logger.error(f"Error adding triple points during selection view: {e}")

        if plotter_has_content:
            plotter.add_legend(bcolor=None, face='circle', border=False, size=(0.15, 0.15))
            plotter.add_axes()
            plotter.reset_camera()
            # Zoom slightly on the selected intersection line (optional)
            if selected_intersection['points'] and len(selected_intersection['points']) >= 2:
                 try:
                      plotter.camera.focal_point = np.mean(line_points, axis=0)
                      # plotter.camera.zoom(1.5) # Adjust zoom factor as needed
                 except Exception as e:
                      logger.warning(f"Could not adjust camera for selected intersection: {e}")
            logger.info("Updated embedded view for selected intersection.")
        else:
            plotter.add_text("Could not display selected intersection.", position='upper_edge', color='white')
            logger.warning("No content added when visualizing selected intersection.")

    def _visualize_intersections(self):
        """Visualize all intersections in the embedded PyVista plotter, matching triangulation style."""
        if not hasattr(self, 'intersection_plotter') or not self.intersection_plotter:
            logger.warning("Intersection plotter not available for visualization.")
            return
            
        plotter = self.intersection_plotter
        plotter.clear() # Start fresh
        # --- Use MeshIt Background --- 
        plotter.set_background([0.318, 0.341, 0.431]) 
        
        # Check if data exists
        if not hasattr(self, 'datasets_intersections') or not self.datasets_intersections:
            plotter.add_text("No intersections computed yet.", position='upper_edge', color='white')
            plotter.reset_camera()
            return

        logger.info("Visualizing all intersections in embedded view...")
        
        involved_dataset_indices = set()
        all_intersection_lines = []
        unique_intersections = set()

        # Collect intersection lines and involved datasets
        for dataset_index, intersections in self.datasets_intersections.items():
            involved_dataset_indices.add(dataset_index)
            for i, intersection in enumerate(intersections):
                involved_dataset_indices.add(intersection['dataset_id1'])
                involved_dataset_indices.add(intersection['dataset_id2'])
                key = tuple(sorted((intersection['dataset_id1'], intersection['dataset_id2']))) + (intersection['is_polyline_mesh'],)
                if key not in unique_intersections:
                    unique_intersections.add(key)
                    if intersection['points'] and len(intersection['points']) >= 2:
                        all_intersection_lines.append(np.array(intersection['points']))

        # Collect and validate triple points
        all_triple_points_coords = []
        if hasattr(self, 'triple_points') and self.triple_points:
            for tp in self.triple_points:
                point = tp.get('point')
                if point is not None:
                    # Validate and convert point to proper format
                    if hasattr(point, 'tolist'):
                        point = point.tolist()
                    elif isinstance(point, np.ndarray):
                        point = point.flatten().tolist()
                    elif not isinstance(point, (list, tuple)):
                        if hasattr(point, '__getitem__') and hasattr(point, '__len__'):
                            try:
                                point = [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
                            except:
                                continue
                        else:
                            continue
                    
                    # Ensure exactly 3 coordinates
                    if len(point) < 2:
                        continue
                    elif len(point) < 3:
                        point = list(point) + [0.0] * (3 - len(point))
                    elif len(point) > 3:
                        point = point[:3]
                    
                    # Validate as finite floats
                    try:
                        validated_point = [float(point[0]), float(point[1]), float(point[2])]
                        if all(np.isfinite(x) for x in validated_point):
                            all_triple_points_coords.append(validated_point)
                    except (ValueError, TypeError):
                        continue

        # Add involved dataset meshes
        plotter_has_content = False
        for index in involved_dataset_indices:
            if 0 <= index < len(self.datasets):
                dataset = self.datasets[index]
                tri_result = dataset.get('triangulation_result')
                if tri_result:
                    vertices = tri_result.get('vertices')
                    triangles = tri_result.get('triangles')
                    # Use a consistent surface color unless dataset has one
                    mesh_color = dataset.get('color', '#CCCCCC') # Default light grey
                    name = dataset.get('name', f'Dataset {index+1}')

                    if vertices is not None and len(vertices) > 0 and triangles is not None and len(triangles) > 0:
                        try:
                            # Basic validation and formatting for PyVista
                            vertices = np.array(vertices)
                            if vertices.shape[1] == 2:
                                 temp_vertices = np.zeros((vertices.shape[0], 3))
                                 temp_vertices[:,:2] = vertices
                                 vertices = temp_vertices
                            elif vertices.shape[1] > 3:
                                 vertices = vertices[:, :3]
                            triangles = np.array(triangles)
                            cells = np.hstack([np.full((len(triangles), 1), 3, dtype=triangles.dtype), triangles])
                            
                            surface_mesh = pv.PolyData(vertices, faces=cells)
                            # --- Apply Style similar to test_intersections.py --- 
                            plotter.add_mesh(surface_mesh, color=mesh_color, opacity=0.7, 
                                             show_edges=True, edge_color='black', 
                                             line_width=1, label=name)
                            plotter_has_content = True
                        except Exception as e:
                             logger.error(f"Error creating mesh for dataset {index} ('{name}') in embedded view: {e}")

        # Add intersection lines (Keep style distinct: red, thicker line)
        if all_intersection_lines:
            for line_points in all_intersection_lines:
                try:
                    # Validate and convert points to proper format for PyVista
                    valid_points = []
                    for point in line_points:
                        if hasattr(point, 'tolist'):
                            point = point.tolist()
                        elif isinstance(point, np.ndarray):
                            point = point.flatten().tolist()
                        elif not isinstance(point, (list, tuple)):
                            if hasattr(point, '__getitem__') and hasattr(point, '__len__'):
                                try:
                                    point = [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
                                except:
                                    continue
                            else:
                                continue
                        
                        # Ensure exactly 3 coordinates
                        if len(point) < 2:
                            continue
                        elif len(point) < 3:
                            point = list(point) + [0.0] * (3 - len(point))
                        elif len(point) > 3:
                            point = point[:3]
                        
                        # Validate as finite floats
                        try:
                            validated_point = [float(point[0]), float(point[1]), float(point[2])]
                            if all(np.isfinite(x) for x in validated_point):
                                valid_points.append(validated_point)
                        except (ValueError, TypeError):
                            continue
                    
                    # Create line segments from valid points
                    if len(valid_points) >= 2:
                        for i in range(len(valid_points) - 1):
                            point_a = valid_points[i]
                            point_b = valid_points[i + 1]
                            
                            # Ensure points are exactly length 3 tuples of floats
                            if (len(point_a) == 3 and len(point_b) == 3 and
                                all(isinstance(x, (int, float)) and np.isfinite(x) for x in point_a) and
                                all(isinstance(x, (int, float)) and np.isfinite(x) for x in point_b)):
                                
                                segment = pv.Line(point_a, point_b)
                                plotter.add_mesh(segment, color='red', line_width=4) # Keep intersection lines distinct
                                plotter_has_content = True
                            
                except Exception as e:
                     logger.error(f"Error adding intersection line segment in embedded view: {e}")

        # Add triple points (Keep style distinct: black spheres)
        if all_triple_points_coords:
            try:
                # Convert to numpy array with proper validation
                if len(all_triple_points_coords) > 0:
                    points_array = np.array(all_triple_points_coords, dtype=np.float64)
                    
                    # Validate array shape and content
                    if (points_array.shape[0] > 0 and points_array.shape[1] == 3 and 
                        np.all(np.isfinite(points_array))):
                        
                        triple_points_poly = pv.PolyData(points_array)
                        plotter.add_points(triple_points_poly, color='black', point_size=8, 
                                           render_points_as_spheres=True, label="Triple Points")
                        plotter_has_content = True
                    else:
                        logger.warning(f"Invalid triple points array shape or content: {points_array.shape}")
            except Exception as e:
                logger.error(f"Error adding triple points in embedded view: {e}")

        if plotter_has_content:
            # Use white text for legend on dark background
            plotter.add_legend(bcolor=None, face='circle', border=False, size=(0.15, 0.15))
            plotter.add_axes()
            plotter.reset_camera()
            logger.info("Updated embedded intersection view.")
        else:
            plotter.add_text("No valid intersection data to display.", position='upper_edge', color='white')
            logger.info("Embedded intersection view updated, but no content added.")

    def _visualize_selected_intersection(self, dataset_index, intersection_index):
        """Visualize a specific intersection by highlighting it in the embedded plotter, matching triangulation style."""
        if not hasattr(self, 'intersection_plotter') or not self.intersection_plotter:
            logger.warning("Intersection plotter not available for selection visualization.")
            return

        plotter = self.intersection_plotter
        plotter.clear()
        # --- Use MeshIt Background --- 
        plotter.set_background([0.318, 0.341, 0.431])
        
        if dataset_index not in self.datasets_intersections or intersection_index >= len(self.datasets_intersections[dataset_index]):
            logger.warning("Selected intersection data not found.")
            plotter.add_text("Selected intersection not found.", position='upper_edge', color='red')
            return
            
        selected_intersection = self.datasets_intersections[dataset_index][intersection_index]
        involved_dataset_ids = {selected_intersection['dataset_id1'], selected_intersection['dataset_id2']}
        
        # Add involved dataset meshes
        plotter_has_content = False
        for index, dataset in enumerate(self.datasets):
            if index not in involved_dataset_ids: continue # Only show involved datasets
            
            opacity = 0.7 # Keep involved ones clearly visible

            tri_result = dataset.get('triangulation_result')
            if tri_result:
                vertices = tri_result.get('vertices')
                triangles = tri_result.get('triangles')
                mesh_color = dataset.get('color', '#CCCCCC') # Use dataset color or default grey
                name = dataset.get('name', f'Dataset {index+1}')

                if vertices is not None and len(vertices) > 0 and triangles is not None and len(triangles) > 0:
                    try:
                        vertices = np.array(vertices)
                        if vertices.shape[1] == 2: 
                             temp_vertices = np.zeros((vertices.shape[0], 3)); temp_vertices[:,:2] = vertices; vertices = temp_vertices
                        elif vertices.shape[1] > 3: vertices = vertices[:, :3]
                        triangles = np.array(triangles)
                        cells = np.hstack([np.full((len(triangles), 1), 3, dtype=triangles.dtype), triangles])
                        surface_mesh = pv.PolyData(vertices, faces=cells)
                        # --- Apply Style similar to test_intersections.py --- 
                        plotter.add_mesh(surface_mesh, color=mesh_color, opacity=opacity, 
                                         show_edges=True, edge_color='black', 
                                         line_width=1, label=name)
                        plotter_has_content = True
                    except Exception as e:
                         logger.error(f"Error creating mesh for dataset {index} ('{name}') during selection view: {e}")

        # Add the selected intersection line prominently (keep distinct style)
        if selected_intersection['points'] and len(selected_intersection['points']) >= 2:
            try:
                # Validate and convert points to proper format for PyVista
                valid_points = []
                for point in selected_intersection['points']:
                    if hasattr(point, 'tolist'):
                        point = point.tolist()
                    elif isinstance(point, np.ndarray):
                        point = point.flatten().tolist()
                    elif not isinstance(point, (list, tuple)):
                        if hasattr(point, '__getitem__') and hasattr(point, '__len__'):
                            try:
                                point = [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
                            except:
                                continue
                        else:
                            continue
                    
                    # Ensure exactly 3 coordinates
                    if len(point) < 2:
                        continue
                    elif len(point) < 3:
                        point = list(point) + [0.0] * (3 - len(point))
                    elif len(point) > 3:
                        point = point[:3]
                    
                    # Validate as finite floats
                    try:
                        validated_point = [float(point[0]), float(point[1]), float(point[2])]
                        if all(np.isfinite(x) for x in validated_point):
                            valid_points.append(validated_point)
                    except (ValueError, TypeError):
                        continue
                
                # Create line segments from valid points
                if len(valid_points) >= 2:
                    for i in range(len(valid_points) - 1):
                        point_a = valid_points[i]
                        point_b = valid_points[i + 1]
                        
                        # Ensure points are exactly length 3 tuples of floats
                        if (len(point_a) == 3 and len(point_b) == 3 and
                            all(isinstance(x, (int, float)) and np.isfinite(x) for x in point_a) and
                            all(isinstance(x, (int, float)) and np.isfinite(x) for x in point_b)):
                            
                            segment = pv.Line(point_a, point_b)
                            # Highlight: thicker and brighter red
                            plotter.add_mesh(segment, color='#FF0000', line_width=6) 
                            plotter_has_content = True
                        
            except Exception as e:
                 logger.error(f"Error adding selected intersection line segment: {e}")

        # Add triple points (keep distinct style)
        if hasattr(self, 'triple_points') and self.triple_points:
            all_triple_points_coords = [tp['point'] for tp in self.triple_points]
            if all_triple_points_coords:
                try:
                    triple_points_poly = pv.PolyData(np.array(all_triple_points_coords))
                    plotter.add_points(triple_points_poly, color='black', point_size=8, render_points_as_spheres=True)
                    plotter_has_content = True
                except Exception as e:
                     logger.error(f"Error adding triple points during selection view: {e}")

        if plotter_has_content:
            # Use white text for legend on dark background
            plotter.add_legend(bcolor=None, face='circle', border=False, size=(0.15, 0.15))
            plotter.add_axes()
            plotter.reset_camera()
            # Zoom slightly on the selected intersection line (optional)
            if selected_intersection['points'] and len(selected_intersection['points']) >= 2:
                 try:
                      plotter.camera.focal_point = np.mean(line_points, axis=0)
                      # plotter.camera.zoom(1.5) # Adjust zoom factor as needed
                 except Exception as e:
                      logger.warning(f"Could not adjust camera for selected intersection: {e}")
            logger.info("Updated embedded view for selected intersection.")
        else:
            plotter.add_text("Could not display selected intersection.", position='upper_edge', color='white')
            logger.warning("No content added when visualizing selected intersection.")

    def show_intersections_3d_view(self):
        """Show a 3D view of the datasets involved in intersections and the intersection lines/points."""
        if not HAVE_PYVISTA:
            QMessageBox.warning(self, "PyVista Needed", "PyVista is required for 3D visualization.")
            return
        
        # Check if there are any intersections to show
        if not hasattr(self, 'datasets_intersections') or not self.datasets_intersections:
            QMessageBox.warning(self, "No Intersections", "No intersections computed yet.")
            return

        logger.info("Preparing data for Intersection 3D View...")
        
        plotter = pv.Plotter(window_size=[1000, 800], off_screen=False) # Create a new plotter instance
        plotter.set_background("white")
        
        involved_dataset_indices = set()
        all_intersection_lines = []
        all_triple_points_coords = []

        # Collect all intersection lines and involved dataset indices
        unique_intersections = set() # Use a set to avoid plotting the same intersection twice
        for dataset_index, intersections in self.datasets_intersections.items():
            involved_dataset_indices.add(dataset_index)
            for i, intersection in enumerate(intersections):
                involved_dataset_indices.add(intersection['dataset_id1'])
                involved_dataset_indices.add(intersection['dataset_id2'])
                
                # Create a unique key for the intersection (order-independent)
                key = tuple(sorted((intersection['dataset_id1'], intersection['dataset_id2']))) + (intersection['is_polyline_mesh'],)
                
                if key not in unique_intersections:
                    unique_intersections.add(key)
                    if intersection['points'] and len(intersection['points']) >= 2:
                        # Convert intersection points to NumPy array for PyVista Line
                        line_points = np.array(intersection['points'])
                        all_intersection_lines.append(line_points)

        # Collect triple points coordinates
        if hasattr(self, 'triple_points') and self.triple_points:
            for tp in self.triple_points:
                 all_triple_points_coords.append(tp['point'])

        # Add involved dataset meshes (triangulated surfaces)
        for index in involved_dataset_indices:
            if 0 <= index < len(self.datasets):
                dataset = self.datasets[index]
                tri_result = dataset.get('triangulation_result')
                if tri_result:
                    vertices = tri_result.get('vertices')
                    triangles = tri_result.get('triangles')
                    color = dataset.get('color', self.DEFAULT_COLORS[index % len(self.DEFAULT_COLORS)])
                    name = dataset.get('name', f'Dataset {index+1}')

                    if vertices is not None and len(vertices) > 0 and triangles is not None and len(triangles) > 0:
                        try:
                            # Ensure vertices are 3D numpy array
                            if isinstance(vertices, list):
                                vertices = np.array(vertices)
                            if vertices.shape[1] == 2:
                                 # Pad with Z=0 if needed (should ideally come from tri_result)
                                 temp_vertices = np.zeros((vertices.shape[0], 3))
                                 temp_vertices[:,:2] = vertices
                                 vertices = temp_vertices
                            elif vertices.shape[1] > 3:
                                 vertices = vertices[:, :3] # Ensure max 3 columns
                                 
                            # Ensure triangles are correctly formatted for PyVista
                            if isinstance(triangles, list):
                                triangles = np.array(triangles)
                            # PyVista needs cells prepended with the number of points (3 for triangles)
                            cells = np.hstack([np.full((len(triangles), 1), 3, dtype=triangles.dtype), triangles])
                            
                            surface_mesh = pv.PolyData(vertices, faces=cells)
                            plotter.add_mesh(surface_mesh, color=color, opacity=0.6, 
                                             show_edges=False, label=name)
                            logger.debug(f"Added surface mesh for dataset {index} ('{name}')")
                        except Exception as e:
                             logger.error(f"Error creating mesh for dataset {index} ('{name}'): {e}", exc_info=True)
                    else:
                         logger.warning(f"Dataset {index} ('{name}') has triangulation result but missing valid vertices/triangles.")

        # Add intersection lines
        if all_intersection_lines:
            for line_points in all_intersection_lines:
                try:
                    # Validate and convert points to proper format for PyVista
                    valid_points = []
                    for point in line_points:
                        if hasattr(point, 'tolist'):
                            point = point.tolist()
                        elif isinstance(point, np.ndarray):
                            point = point.flatten().tolist()
                        elif not isinstance(point, (list, tuple)):
                            if hasattr(point, '__getitem__') and hasattr(point, '__len__'):
                                try:
                                    point = [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
                                except:
                                    continue
                            else:
                                continue
                        
                        # Ensure exactly 3 coordinates
                        if len(point) < 2:
                            continue
                        elif len(point) < 3:
                            point = list(point) + [0.0] * (3 - len(point))
                        elif len(point) > 3:
                            point = point[:3]
                        
                        # Validate as finite floats
                        try:
                            validated_point = [float(point[0]), float(point[1]), float(point[2])]
                            if all(np.isfinite(x) for x in validated_point):
                                valid_points.append(validated_point)
                        except (ValueError, TypeError):
                            continue
                    
                    # Create line segments from valid points
                    if len(valid_points) >= 2:
                        for i in range(len(valid_points) - 1):
                            point_a = valid_points[i]
                            point_b = valid_points[i + 1]
                            
                            # Ensure points are exactly length 3 tuples of floats
                            if (len(point_a) == 3 and len(point_b) == 3 and
                                all(isinstance(x, (int, float)) and np.isfinite(x) for x in point_a) and
                                all(isinstance(x, (int, float)) and np.isfinite(x) for x in point_b)):
                                
                                # Create line segment using PolyData with line connectivity
                                line_points = np.array([point_a, point_b])
                                line_cells = np.array([2, 0, 1])  # Line with 2 points: indices 0 and 1
                                segment = pv.PolyData(line_points)
                                segment.lines = line_cells
                                plotter.add_mesh(segment, color='red', line_width=5) # Thicker red lines
                            
                except Exception as e:
                     logger.error(f"Error adding intersection line segment: {e}", exc_info=True)
            logger.info(f"Added {len(all_intersection_lines)} intersection lines.")

        # Add triple points
        if all_triple_points_coords:
            try:
                # Validate and convert triple points to proper format
                valid_triple_points = []
                for point in all_triple_points_coords:
                    if hasattr(point, 'tolist'):
                        point = point.tolist()
                    elif isinstance(point, np.ndarray):
                        point = point.flatten().tolist()
                    elif not isinstance(point, (list, tuple)):
                        if hasattr(point, '__getitem__') and hasattr(point, '__len__'):
                            try:
                                point = [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
                            except:
                                continue
                        else:
                            continue
                    
                    # Ensure exactly 3 coordinates
                    if len(point) < 2:
                        continue
                    elif len(point) < 3:
                        point = list(point) + [0.0] * (3 - len(point))
                    elif len(point) > 3:
                        point = point[:3]
                    
                    # Validate as finite floats
                    try:
                        validated_point = [float(point[0]), float(point[1]), float(point[2])]
                        if all(np.isfinite(x) for x in validated_point):
                            valid_triple_points.append(validated_point)
                    except (ValueError, TypeError):
                        continue
                
                # Create PyVista PolyData from valid points
                if valid_triple_points:
                    points_array = np.array(valid_triple_points, dtype=np.float64)
                    
                    # Validate array shape and content
                    if (points_array.shape[0] > 0 and points_array.shape[1] == 3 and 
                        np.all(np.isfinite(points_array))):
                        
                        triple_points_poly = pv.PolyData(points_array)
                        plotter.add_points(triple_points_poly, color='black', point_size=10, 
                                           render_points_as_spheres=True, label="Triple Points")
                        logger.info(f"Added {len(valid_triple_points)} triple points.")
                    else:
                        logger.warning(f"Invalid triple points array shape or content: {points_array.shape}")
                else:
                    logger.warning("No valid triple points found after validation.")
            except Exception as e:
                logger.error(f"Error adding triple points: {e}", exc_info=True)

        # Add legend, axes, etc.
        plotter.add_legend(bcolor=None, face='circle') # Simple legend
        plotter.add_axes()
        plotter.reset_camera()
        plotter.show(title="MeshIt Intersections 3D View")
        logger.info("Showing Intersection 3D View.")
    
    def _validate_surfaces_for_tetgen(self):
        """Validate constrained surfaces for tetgen readiness"""
        if not hasattr(self, 'datasets') or not self.datasets:
            self.statusBar().showMessage("No datasets to validate.")
            return
        
        # Check if we have constrained meshes
        datasets_with_constrained_meshes = []
        for dataset in self.datasets:
            if 'constrained_vertices' in dataset and 'constrained_triangles' in dataset:
                datasets_with_constrained_meshes.append(dataset)
        
        if not datasets_with_constrained_meshes:
            self.statusBar().showMessage("No constrained meshes found. Please generate constrained surface meshes first.")
            return
        
        # Import the validation function
        try:
            from meshit.intersection_utils import validate_surfaces_for_tetgen
        except ImportError:
            self.statusBar().showMessage("Validation function not available.")
            return
        
        # Run validation
        logger.info("Starting tetgen surface validation...")
        self.statusBar().showMessage("Validating surfaces for tetgen...")
        
        try:
            validation_results = validate_surfaces_for_tetgen(datasets_with_constrained_meshes)
            
            # Create detailed validation report dialog
            self._show_validation_results_dialog(validation_results)
            
            # Update status bar based on results
            if validation_results['ready_for_tetgen']:
                self.statusBar().showMessage(f"✓ All {validation_results['surface_count']} surfaces are ready for tetgen!")
            elif validation_results['overall_status'] == 'PARTIAL':
                ready_count = validation_results['statistics']['valid_surfaces']
                total_count = validation_results['surface_count']
                self.statusBar().showMessage(f"⚠ {ready_count}/{total_count} surfaces ready for tetgen")
            else:
                self.statusBar().showMessage("✗ Surfaces not ready for tetgen - check validation report")
                
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            self.statusBar().showMessage(f"Validation failed: {str(e)}")
    
    def _setup_tetra_mesh_tab(self):
        """Enhanced tetra mesh tab with advanced constraint selection capabilities"""
        
        # Initialize data structures
        self.tetra_materials = []
        self.selected_surface_parts = {}
        self.tetrahedral_mesh = None
        self.surface_quality_data = {}
        self.interactive_selection_enabled = False
        self.constraint_selection_enabled = False
        self.constraint_visualization_mode = "NORMAL"  # "NORMAL" or "SELECTION"
        
        # Initialize constraint manager
        self.constraint_manager = SurfaceConstraintManager()
        self.constraint_manager._parent_gui = self  # Pass reference for accessing triple points
        
        # Border recognition
        self.border_surface_indices = self._identify_border_surfaces()
        self.unit_surface_indices = self._identify_unit_surfaces()
        self.fault_surface_indices = self._identify_fault_surfaces()
        
        # Create main layout
        tab_layout = QHBoxLayout(self.tetra_mesh_tab)
        
        # === Control Panel (Left) ===
        control_panel = QWidget()
        control_panel.setMaximumWidth(400)
        control_layout = QVBoxLayout(control_panel)
        
        # --- Advanced Selection Group ---
        selection_group = QGroupBox("Advanced Constraint Selection")
        selection_layout = QVBoxLayout(selection_group)
        
        # Selection tools (like C++ SelectionTool enum)
        self.selection_tool_combo = QComboBox()
        self.selection_tool_combo.addItems([
            "SINGLE - Click individual constraints",
            "MULTI - Rectangle selection", 
            "BUCKET - Flood fill selection",
            "POLYGON - Polygon selection",
            "ALL - Select all constraints"
        ])
        selection_layout.addWidget(QLabel("Selection Tool:"))
        selection_layout.addWidget(self.selection_tool_combo)
        
        # Selection modes (like C++ SelectionMode enum)
        self.selection_mode_combo = QComboBox()
        self.selection_mode_combo.addItems([
            "MARK - Add to selection",
            "UNMARK - Remove from selection", 
            "HOLE - Mark as holes",
            "INVERT - Toggle selection"
        ])
        selection_layout.addWidget(QLabel("Selection Mode:"))
        selection_layout.addWidget(self.selection_mode_combo)
        
        # Selection target
        self.selection_target_combo = QComboBox()
        self.selection_target_combo.addItems([
            "Surface Constraints",
            "Hull Boundaries", 
            "Intersection Lines",
            "All Elements"
        ])
        selection_layout.addWidget(QLabel("Select:"))
        selection_layout.addWidget(self.selection_target_combo)
        
        # Enable/disable constraint selection
        self.enable_constraint_selection_btn = QPushButton("🎯 Enable Constraint Selection")
        self.enable_constraint_selection_btn.setCheckable(True)
        self.enable_constraint_selection_btn.clicked.connect(self._toggle_constraint_selection)
        selection_layout.addWidget(self.enable_constraint_selection_btn)
        
        # Generate constraints button
        self.generate_constraints_btn = QPushButton("🔗 Generate Constraints from Pre-Tetra Data")
        self.generate_constraints_btn.clicked.connect(self._generate_constraints_from_data)
        selection_layout.addWidget(self.generate_constraints_btn)
        
        # View mode selection
        view_mode_layout = QHBoxLayout()
        self.view_mode_label = QLabel("View Mode:")
        self.show_all_surfaces_btn = QPushButton("Show All")
        self.show_selected_only_btn = QPushButton("Selected Only")
        
        self.show_all_surfaces_btn.clicked.connect(self._on_show_all_surfaces)
        self.show_selected_only_btn.clicked.connect(self._on_show_selected_only)
        
        # Make buttons toggleable style
        self.show_all_surfaces_btn.setCheckable(True)
        self.show_selected_only_btn.setCheckable(True)
        self.show_selected_only_btn.setChecked(True)  # Default to selected only
        self.show_all_surfaces_btn.setEnabled(False)  # Initially disabled
        self.show_selected_only_btn.setEnabled(False)
        
        view_mode_layout.addWidget(self.view_mode_label)
        view_mode_layout.addWidget(self.show_all_surfaces_btn)
        view_mode_layout.addWidget(self.show_selected_only_btn)
        selection_layout.addLayout(view_mode_layout)
        
        # Selection summary
        self.constraint_selection_summary = QLabel("No constraints generated yet")
        self.constraint_selection_summary.setWordWrap(True)
        self.constraint_selection_summary.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;")
        selection_layout.addWidget(self.constraint_selection_summary)
        
        control_layout.addWidget(selection_group)
        
        # --- Surface & Constraint Tree ---
        surface_list_group = QGroupBox("Surfaces & Constraints")
        surface_list_layout = QVBoxLayout(surface_list_group)
        
        self.surface_constraint_tree = QTreeWidget()
        self.surface_constraint_tree.setHeaderLabels(["Element", "Type", "State", "Details"])
        self.surface_constraint_tree.itemChanged.connect(self._on_constraint_tree_item_changed)
        surface_list_layout.addWidget(self.surface_constraint_tree)
        
        # Tree control buttons
        tree_control_layout = QHBoxLayout()
        
        # Selection buttons
        select_all_constraints_btn = QPushButton("Select All")
        unselect_all_constraints_btn = QPushButton("Unselect All")
        select_all_constraints_btn.clicked.connect(self._select_all_constraints)
        unselect_all_constraints_btn.clicked.connect(self._unselect_all_constraints)
        
        # Expansion control buttons
        expand_all_btn = QPushButton("Expand All")
        collapse_all_btn = QPushButton("Collapse All")
        expand_all_btn.clicked.connect(self._expand_all_tree_items)
        collapse_all_btn.clicked.connect(self._collapse_all_tree_items)
        
        # Add buttons to layout
        tree_control_layout.addWidget(select_all_constraints_btn)
        tree_control_layout.addWidget(unselect_all_constraints_btn)
        tree_control_layout.addWidget(expand_all_btn)
        tree_control_layout.addWidget(collapse_all_btn)
        surface_list_layout.addLayout(tree_control_layout)
        
        control_layout.addWidget(surface_list_group)
        
        # Visualization controls removed - using only constraint selection viewer
        
        # --- Generation Controls ---
        generate_group = QGroupBox("Tetrahedral Mesh Generation")
        generate_layout = QVBoxLayout(generate_group)
        
        self.generate_tetra_mesh_btn = QPushButton("🔧 Generate Tetrahedral Mesh")
        self.generate_tetra_mesh_btn.clicked.connect(self._generate_tetrahedral_mesh_action)
        generate_layout.addWidget(self.generate_tetra_mesh_btn)
        
        self.export_mesh_btn = QPushButton("💾 Export Mesh")
        self.export_mesh_btn.clicked.connect(self._export_tetrahedral_mesh)
        self.export_mesh_btn.setEnabled(False)
        generate_layout.addWidget(self.export_mesh_btn)
        
        control_layout.addWidget(generate_group)
        
        tab_layout.addWidget(control_panel)
        
        # === 3D Viewer (Right) ===
        viewer_group = QGroupBox("3D Viewer - Constraint Selection")
        viewer_layout = QVBoxLayout(viewer_group)
        
        # Setup enhanced 3D viewer with constraint visualization
        self._setup_constraint_3d_viewer(viewer_group, viewer_layout)
        
        tab_layout.addWidget(viewer_group, 1)
        
        # Initialize
        logger.info("Enhanced tetra mesh tab with constraint selection initialized")
        
        # Initialize with delay
        QTimer.singleShot(200, self._initialize_tetra_visualization)

    def _on_show_all_surfaces(self):
        """Handle show all surfaces button"""
        self.show_all_surfaces_btn.setChecked(True)
        self.show_selected_only_btn.setChecked(False)
        self._visualize_constraints(selection_mode=self.constraint_selection_enabled)
        logger.info("Switched to show all surfaces mode")

    def _on_show_selected_only(self):
        """Handle show selected only button"""
        self.show_selected_only_btn.setChecked(True)
        self.show_all_surfaces_btn.setChecked(False)
        self._visualize_selected_surfaces_only()
        logger.info("Switched to show selected surfaces only mode")

    def _setup_constraint_3d_viewer(self, parent_group, layout):
        """Setup 3D viewer with constraint selection capabilities"""
        
        if HAVE_PYVISTA:
            from pyvistaqt import QtInteractor
            
            self.constraint_plotter = QtInteractor(parent_group)
            layout.addWidget(self.constraint_plotter.interactor)
            
            self.constraint_plotter.set_background([0.15, 0.15, 0.2])
            
            logger.info("Constraint selection 3D viewer initialized")
        else:
            placeholder = QLabel("PyVista required for constraint selection")
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder)
            self.constraint_plotter = None

    def _generate_constraints_from_data(self):
        """Generate constraints from pre-tetra mesh data"""
        if not self.datasets:
            QMessageBox.warning(self, "No Data", "No datasets loaded. Load data first.")
            return
        
        try:
            self.constraint_manager.generate_constraints_from_pre_tetra_data(self.datasets)
            self._populate_constraint_tree()
            self._update_constraint_summary()
            
            # Initialize visualization with all constraints in "Show All" mode
            # This ensures all actors exist for efficient updates later
            if self.constraint_manager.surface_constraints:
                logger.info("Initializing constraint visualization...")
                self._visualize_constraints(selection_mode=False)  # Show all constraints initially
                
                self.enable_constraint_selection_btn.setEnabled(True)
                self.generate_tetra_mesh_btn.setEnabled(True)
                self.show_all_surfaces_btn.setEnabled(True)
                self.show_selected_only_btn.setEnabled(True)
                logger.info("Constraints generated successfully from pre-tetra mesh data")
            else:
                # Start with no visualization only if no constraints were generated
                self.constraint_plotter.clear()
                self.constraint_plotter.add_text("Constraints generated!\nSelect constraints in the tree to visualize them.", 
                                                position='upper_edge', font_size=12)
                QMessageBox.warning(self, "No Constraints", 
                                  "No constraints could be generated. Ensure pre-tetra mesh tab is completed first.")
                
        except Exception as e:
            logger.error(f"Error generating constraints: {e}")
            QMessageBox.critical(self, "Error", f"Failed to generate constraints: {str(e)}")

    def _toggle_constraint_selection(self, enabled):
        """Toggle constraint selection mode"""
        
        self.constraint_selection_enabled = enabled
        
        if enabled:
            self.enable_constraint_selection_btn.setText("🎯 Constraint Selection ON")
            self.enable_constraint_selection_btn.setStyleSheet("background-color: #4CAF50; color: white;")
            # Use current view mode setting
            if self.show_all_surfaces_btn.isChecked():
                self._visualize_constraints(selection_mode=True)
            else:
                self._visualize_selected_surfaces_only()
            logger.info("Constraint selection mode enabled")
        else:
            self.enable_constraint_selection_btn.setText("🎯 Enable Constraint Selection")
            self.enable_constraint_selection_btn.setStyleSheet("")
            # Use current view mode setting
            if self.show_all_surfaces_btn.isChecked():
                self._visualize_constraints(selection_mode=False)
            else:
                self._visualize_selected_surfaces_only()
            logger.info("Constraint selection mode disabled")

    def _visualize_constraints(self, selection_mode=False):
        """Visualize constraints with color-coded selection capability"""
        
        if not self.constraint_plotter:
            return
            
        self.constraint_plotter.clear()
        
        # Set visualization mode
        self.constraint_visualization_mode = "SELECTION" if selection_mode else "NORMAL"
        
        for surface_idx, constraints in self.constraint_manager.surface_constraints.items():
            dataset = self.datasets[surface_idx] if surface_idx < len(self.datasets) else None
            if not dataset:
                continue
                
            surface_name = dataset.get('name', f'Surface_{surface_idx}')
            
            # Visualize constrained triangulated surface (background)
            self._add_surface_mesh_to_constraint_plotter(surface_idx, dataset)
            
            # Visualize constraint segments
            for constraint_idx, constraint in enumerate(constraints):
                self._add_constraint_to_plotter(
                    surface_idx, constraint_idx, constraint, selection_mode
                )
        
        # Setup picking callback for selection mode
        if selection_mode:
            try:
                self.constraint_plotter.enable_mesh_picking(
                    callback=self._on_constraint_picked,
                    show=False
                )
            except Exception as e:
                logger.warning(f"Could not enable mesh picking: {e}")
        else:
            try:
                self.constraint_plotter.disable_picking()
            except:
                pass
        
        self.constraint_plotter.reset_camera()

    def _visualize_single_surface_constraints(self, surface_idx, update_summary=False):
        """Efficiently visualize only a specific surface and its constraints"""
        
        if not self.constraint_plotter:
            return
        
        # Get the dataset for this surface
        dataset = self.datasets[surface_idx] if surface_idx < len(self.datasets) else None
        if not dataset:
            return
        
        # Clear only actors related to this surface
        self._clear_surface_actors(surface_idx)
        
        # Add surface mesh as background
        self._add_surface_mesh_to_constraint_plotter(surface_idx, dataset)
        
        # Add constraint segments for this surface only
        constraints = self.constraint_manager.surface_constraints.get(surface_idx, [])
        for constraint_idx, constraint in enumerate(constraints):
            self._add_constraint_to_plotter(
                surface_idx, constraint_idx, constraint, self.constraint_selection_enabled
            )
        
        # Update summary if requested
        if update_summary:
            self._update_constraint_summary()
        
        # Don't reset camera for single surface updates to maintain view
        logger.debug(f"Updated visualization for surface {surface_idx} only")

    def _clear_surface_actors(self, surface_idx):
        """Clear only the actors related to a specific surface"""
        if not self.constraint_plotter:
            return
        
        # Remove surface mesh
        surface_actor_name = f"surface_{surface_idx}"
        if surface_actor_name in self.constraint_plotter.actors:
            self.constraint_plotter.remove_actor(surface_actor_name)
        
        # Remove constraint actors for this surface
        actors_to_remove = []
        for actor_name in self.constraint_plotter.actors.keys():
            if actor_name.startswith(f"constraint_{surface_idx}_"):
                actors_to_remove.append(actor_name)
        
        for actor_name in actors_to_remove:
            self.constraint_plotter.remove_actor(actor_name)

    def _visualize_selected_surfaces_only(self):
        """Visualize only surfaces that have selected constraints (very efficient)"""
        try:
            if not self.constraint_plotter:
                return
            
            self.constraint_plotter.clear()
            
            # Get surfaces with selected constraints
            selected_surfaces = self.constraint_manager.get_selected_surfaces()
            
            if not selected_surfaces:
                # Show message when nothing is selected
                self.constraint_plotter.add_text("No constraints selected.\nUse the tree or selection mode to select constraints.", 
                                               position='upper_edge', font_size=12)
                return
            
            # Only visualize selected surfaces
            for surface_idx in selected_surfaces:
                dataset = self.datasets[surface_idx] if surface_idx < len(self.datasets) else None
                if not dataset:
                    continue
                
                # Add surface mesh as background
                self._add_surface_mesh_to_constraint_plotter(surface_idx, dataset)
                
                # Add constraint segments for this surface
                constraints = self.constraint_manager.surface_constraints.get(surface_idx, [])
                for constraint_idx, constraint in enumerate(constraints):
                    self._add_constraint_to_plotter(
                        surface_idx, constraint_idx, constraint, self.constraint_selection_enabled
                    )
            
            # Setup picking if in selection mode
            if self.constraint_selection_enabled:
                try:
                    self.constraint_plotter.enable_mesh_picking(
                        callback=self._on_constraint_picked,
                        show=False
                    )
                except Exception as e:
                    logger.warning(f"Could not enable mesh picking: {e}")
            
            self.constraint_plotter.reset_camera()
            logger.info(f"Visualizing {len(selected_surfaces)} selected surfaces only")
            
        except Exception as e:
            logger.error(f"Error in _visualize_selected_surfaces_only: {e}")
            return

    def _add_surface_mesh_to_constraint_plotter(self, surface_idx, dataset):
        """Add surface mesh as background to constraint plotter"""
        constrained_vertices = dataset.get('constrained_vertices')
        constrained_triangles = dataset.get('constrained_triangles')
        
        if (constrained_vertices is not None and constrained_triangles is not None and 
            len(constrained_vertices) > 0 and len(constrained_triangles) > 0):
            
            import numpy as np
            vertices_array = np.array(constrained_vertices)
            triangles_array = np.array(constrained_triangles)
            
            # Create faces array for PyVista
            faces = []
            for tri in triangles_array:
                faces.extend([3, tri[0], tri[1], tri[2]])
            
            mesh = pv.PolyData(vertices_array, faces=faces)
            
            # Add as semi-transparent background
            surface_type = self._get_surface_type(surface_idx)
            color = self._get_surface_color(surface_type)
            
            self.constraint_plotter.add_mesh(
                mesh,
                name=f"surface_{surface_idx}",
                color=color,
                opacity=0.3,
                style='surface',
                pickable=False
            )

    def _add_constraint_to_plotter(self, surface_idx, constraint_idx, constraint, selection_mode):
        """Add individual constraint segments to plotter"""
        
        constraint_key = (surface_idx, constraint_idx)
        constraint_state = self.constraint_manager.constraint_states.get(constraint_key, "UNDEFINED")
        
        # Handle new constraint format (direct points) vs old format (segments list)
        if 'segments' in constraint:
            # Old format compatibility
            segments_to_render = constraint['segments']
        else:
            # New format - constraint is the segment itself
            segments_to_render = [constraint]
        
        for segment_idx, segment in enumerate(segments_to_render):
            # Extract points safely
            if 'points' in segment:
                points_data = segment['points']
            else:
                logger.warning(f"No points found in constraint segment {constraint_idx}")
                continue
                
            try:
                points = np.array([[p[0], p[1], p[2]] for p in points_data])
            except (IndexError, TypeError, ValueError) as e:
                logger.warning(f"Error extracting points from constraint: {e}")
                continue
            
            if len(points) < 2:
                continue
                
            # Create line mesh
            line_mesh = pv.PolyData(points)
            line_mesh.lines = np.hstack([[2, i, i+1] for i in range(len(points)-1)])
            
            # Color and style based on mode and state
            if selection_mode:
                # Selection mode: use RGB color for picking
                rgb_color = segment.get('rgb', [255, 255, 255])
                color = [c/255.0 for c in rgb_color]  # Normalize to [0,1]
                line_width = 3
                opacity = 1.0
            else:
                # Normal mode: color by constraint state (like C++ makeConstraints)
                if constraint_state == "SEGMENTS":
                    color = 'blue'     # Selected constraints
                    line_width = 5
                elif constraint_state == "HOLES":
                    color = 'red'      # Hole constraints
                    line_width = 5
                else:  # "UNDEFINED"
                    color = 'white'    # Unselected constraints
                    line_width = 3
                opacity = 0.9
            
            # Add to plotter
            actor_name = f"constraint_{surface_idx}_{constraint_idx}_{segment_idx}"
            self.constraint_plotter.add_mesh(
                line_mesh,
                name=actor_name,
                color=color,
                line_width=line_width,
                opacity=opacity,
                style='wireframe',
                pickable=selection_mode
            )

    def _on_constraint_picked(self, mesh):
        """Handle constraint selection via mouse picking (like C++ findSelection)"""
        
        if not self.constraint_selection_enabled:
            return
        
        # Extract RGB from mesh (simplified - in practice you'd use a more robust method)
        # For now, we'll use mesh bounds and actor name to identify the constraint
        try:
            # Get the actor name to identify which constraint was picked
            actor_name = None
            for actor in self.constraint_plotter.renderer.actors.values():
                if hasattr(actor, 'GetMapper') and actor.GetMapper():
                    mapper = actor.GetMapper()
                    if hasattr(mapper, 'GetInput') and mapper.GetInput() == mesh:
                        # Found the actor, extract name from our naming convention
                        # constraint_{surface_idx}_{constraint_idx}_{segment_idx}
                        for name, mesh_obj in self.constraint_plotter.actors.items():
                            if mesh_obj == mesh:
                                actor_name = name
                                break
                        break
            
            if actor_name and actor_name.startswith("constraint_"):
                parts = actor_name.split("_")
                if len(parts) >= 4:
                    surface_idx = int(parts[1])
                    constraint_idx = int(parts[2])
                    
                    # Apply selection based on current mode
                    selection_mode = self.selection_mode_combo.currentText().split(" - ")[0]
                    constraint_key = (surface_idx, constraint_idx)
                    
                    if selection_mode == "MARK":
                        self.constraint_manager.constraint_states[constraint_key] = "SEGMENTS"
                    elif selection_mode == "UNMARK":
                        self.constraint_manager.constraint_states[constraint_key] = "UNDEFINED"
                    elif selection_mode == "HOLE":
                        self.constraint_manager.constraint_states[constraint_key] = "HOLES"
                    elif selection_mode == "INVERT":
                        current_state = self.constraint_manager.constraint_states.get(constraint_key, "UNDEFINED")
                        new_state = "SEGMENTS" if current_state == "UNDEFINED" else "UNDEFINED"
                        self.constraint_manager.constraint_states[constraint_key] = new_state
                    
                    # Update visualization and UI
                    self._visualize_constraints(selection_mode=False)  # Switch back to normal mode to see colors
                    self._update_constraint_tree()
                    self._update_constraint_summary()
                    
                    new_state = self.constraint_manager.constraint_states[constraint_key]
                    logger.info(f"Constraint {surface_idx}_{constraint_idx} marked as {new_state}")
                    
                    # Switch back to selection mode after brief normal view
                    QTimer.singleShot(500, lambda: self._visualize_constraints(selection_mode=True))
        
        except Exception as e:
            logger.warning(f"Error processing constraint selection: {e}")

    def _populate_constraint_tree(self):
        """Populate the constraint tree widget with hierarchical surface and constraint information"""
        
        self.surface_constraint_tree.clear()
        
        for surface_idx, constraints in self.constraint_manager.surface_constraints.items():
            dataset = self.datasets[surface_idx] if surface_idx < len(self.datasets) else None
            if not dataset:
                continue
                
            surface_name = dataset.get('name', f'Surface_{surface_idx}')
            surface_item = QTreeWidgetItem([surface_name, "Surface", "Visible", f"{len(constraints)} constraint segments"])
            surface_item.setData(0, Qt.UserRole, ('surface', surface_idx))
            
            # Add visibility checkbox for surface
            surface_item.setFlags(surface_item.flags() | Qt.ItemIsUserCheckable)
            
            # Set checkbox based on current visibility state
            is_visible = self.constraint_manager.surface_visibility.get(surface_idx, True)
            surface_item.setCheckState(0, Qt.Checked if is_visible else Qt.Unchecked)
            surface_item.setText(2, "Visible" if is_visible else "Hidden")
            
            # Group constraints by parent type for hierarchical display
            hull_segments = []
            intersection_groups = {}  # line_id -> list of segments
            
            for constraint_idx, constraint in enumerate(constraints):
                constraint_type = constraint.get('type', 'UNKNOWN')
                if constraint_type == 'hull':
                    hull_segments.append((constraint_idx, constraint))
                elif constraint_type.startswith('intersection_'):
                    # Extract line_id from type like "intersection_0", "intersection_1", etc.
                    try:
                        line_id = int(constraint_type.split('_')[1])
                    except (IndexError, ValueError):
                        line_id = 0
                    if line_id not in intersection_groups:
                        intersection_groups[line_id] = []
                    intersection_groups[line_id].append((constraint_idx, constraint))
            
            # Add Hull Boundary parent with individual segments as children
            if hull_segments:
                hull_parent = QTreeWidgetItem(["Hull Boundary", "HULL_BOUNDARY", "Container", f"{len(hull_segments)} segments"])
                hull_parent.setData(0, Qt.UserRole, ('parent', 'HULL_BOUNDARY', surface_idx))
                
                # Add selection checkbox for hull parent to select all hull segments
                hull_parent.setFlags(hull_parent.flags() | Qt.ItemIsUserCheckable)
                hull_parent.setCheckState(0, Qt.Unchecked)  # Start unchecked
                
                for constraint_idx, constraint in hull_segments:
                    constraint_key = (surface_idx, constraint_idx)
                    constraint_state = self.constraint_manager.constraint_states.get(constraint_key, "UNDEFINED")
                    
                    segment_id = constraint.get('segment_id', constraint_idx)
                    segment_name = f"Hull Segment {segment_id}"
                    
                    # Get point count from constraint data
                    total_points = len(constraint.get('points', []))
                    segment_item = QTreeWidgetItem([
                        segment_name, 
                        "HULL_SEGMENT", 
                        constraint_state, 
                        f"{total_points} points"
                    ])
                    segment_item.setData(0, Qt.UserRole, ('constraint', surface_idx, constraint_idx))
                    segment_item.setFlags(segment_item.flags() | Qt.ItemIsUserCheckable)
                    
                    # Set checkbox state based on constraint state
                    if constraint_state == "SEGMENTS":
                        segment_item.setCheckState(0, Qt.Checked)
                    else:
                        segment_item.setCheckState(0, Qt.Unchecked)
                    
                    hull_parent.addChild(segment_item)
                
                surface_item.addChild(hull_parent)
                
                # Update hull parent checkbox state based on children
                self._initialize_parent_checkbox_state(hull_parent)
            
            # Add Intersection Lines with individual segments as children
            for line_id, line_segments in intersection_groups.items():
                intersection_parent = QTreeWidgetItem([
                    f"Intersection Line {line_id}", 
                    "INTERSECTION_LINE", 
                    "Container", 
                    f"{len(line_segments)} segments"
                ])
                intersection_parent.setData(0, Qt.UserRole, ('parent', 'INTERSECTION_LINE', surface_idx, line_id))
                
                # Add selection checkbox for intersection parent to select all intersection segments
                intersection_parent.setFlags(intersection_parent.flags() | Qt.ItemIsUserCheckable)
                intersection_parent.setCheckState(0, Qt.Unchecked)  # Start unchecked
                
                for constraint_idx, constraint in line_segments:
                    constraint_key = (surface_idx, constraint_idx)
                    constraint_state = self.constraint_manager.constraint_states.get(constraint_key, "UNDEFINED")
                    
                    segment_id = constraint.get('segment_id', 0)
                    segment_name = f"Intersection Segment {segment_id}"
                    
                    # Get point count from constraint data
                    total_points = len(constraint.get('points', []))
                    segment_item = QTreeWidgetItem([
                        segment_name, 
                        "INTERSECTION_SEGMENT", 
                        constraint_state, 
                        f"{total_points} points"
                    ])
                    segment_item.setData(0, Qt.UserRole, ('constraint', surface_idx, constraint_idx))
                    segment_item.setFlags(segment_item.flags() | Qt.ItemIsUserCheckable)
                    
                    # Set checkbox state based on constraint state
                    if constraint_state == "SEGMENTS":
                        segment_item.setCheckState(0, Qt.Checked)
                    else:
                        segment_item.setCheckState(0, Qt.Unchecked)
                    
                    intersection_parent.addChild(segment_item)
                
                surface_item.addChild(intersection_parent)
                
                # Update intersection parent checkbox state based on children
                self._initialize_parent_checkbox_state(intersection_parent)
            
            self.surface_constraint_tree.addTopLevelItem(surface_item)
        
        # Default expansion: Expand surfaces and parent categories, but not individual segments
        self._set_default_expansion_state()

        # ------------------------------------------------------------------ #
    # tree-widget callback                                               #
    # ------------------------------------------------------------------ #
    def _on_constraint_tree_item_changed(self, item, column):
        """
        Called when the user toggles any checkbox in the surface /
        constraint tree.

        • surface items (top-level) act as *exclusive* visibility toggles:
          checking one surface automatically hides all others.
        • parent items (“Hull Boundary”, “Intersection Line n”) forward the
          state to their children.
        • leaf items (individual constraints) update only their own
          appearance.
        """
        item_data = item.data(0, Qt.UserRole)
        if not item_data:
            return

        # ------------------------------------------------------------------
        # 1)  SURFACE visibility  (independent, no auto-hiding)
        # ------------------------------------------------------------------
        if item_data[0] == 'surface':
            surface_idx = item_data[1]
            is_visible  = item.checkState(0) == Qt.Checked

            # store & apply
            self.constraint_manager.surface_visibility[surface_idx] = is_visible
            self._update_surface_visibility(surface_idx, is_visible)
            item.setText(2, "Visible" if is_visible else "Hidden")
            return

        # ------------------------------------------------------------------
        # 2)  PARENT level (Hull Boundary / Intersection Line)
        # ------------------------------------------------------------------
        if item_data[0] == 'parent':
            parent_type  = item_data[1]
            surface_idx  = item_data[2]
            if parent_type == 'HULL_BOUNDARY':
                self._handle_hull_parent_selection(item, surface_idx)
            elif parent_type == 'INTERSECTION_LINE' and len(item_data) >= 4:
                line_id = item_data[3]
                self._handle_intersection_parent_selection(item, surface_idx, line_id)
            return

        # ------------------------------------------------------------------
        # 3)  CONSTRAINT (leaf)                                             #
        # ------------------------------------------------------------------
        if item_data[0] == 'constraint':
            surface_idx, constraint_idx = item_data[1], item_data[2]
            constraint_key = (surface_idx, constraint_idx)

            # update state
            if item.checkState(0) == Qt.Checked:
                self.constraint_manager.constraint_states[constraint_key] = "SEGMENTS"
            else:
                self.constraint_manager.constraint_states[constraint_key] = "UNDEFINED"

            # update label in tree
            item.setText(2, self.constraint_manager.constraint_states[constraint_key])

            # visual appearance – update only this constraint
            logger.info(f"Updating constraint appearance: "
                        f"Surface {surface_idx}, Constraint {constraint_idx}, "
                        f"State: {self.constraint_manager.constraint_states[constraint_key]}")
            self._update_constraint_appearance(surface_idx, constraint_idx)

            # refresh parent checkbox
            self._update_parent_checkbox_state(item)

            # refresh summary text
            self._update_constraint_summary()

    def _handle_hull_parent_selection(self, parent_item, surface_idx):
        """Handle selection of Hull Boundary parent - select/unselect all hull segments"""
        is_checked = parent_item.checkState(0) == Qt.Checked
        
        # Block signals to prevent recursion
        self.surface_constraint_tree.blockSignals(True)
        
        try:
            # Update all hull segment children
            updated_constraints = []
            for i in range(parent_item.childCount()):
                child_item = parent_item.child(i)
                child_data = child_item.data(0, Qt.UserRole)
                
                if child_data and child_data[0] == 'constraint':
                    constraint_idx = child_data[2]
                    constraint_key = (surface_idx, constraint_idx)
                    
                    # Update child checkbox
                    child_item.setCheckState(0, Qt.Checked if is_checked else Qt.Unchecked)
                    
                    # Update constraint state
                    self.constraint_manager.constraint_states[constraint_key] = "SEGMENTS" if is_checked else "UNDEFINED"
                    child_item.setText(2, self.constraint_manager.constraint_states[constraint_key])
                    
                    updated_constraints.append(constraint_idx)
            
            # Update parent status text
            parent_item.setText(2, f"All Selected" if is_checked else "Container")
            
        finally:
            self.surface_constraint_tree.blockSignals(False)
        
        # Batch update visual appearance
        for constraint_idx in updated_constraints:
            self._update_constraint_appearance(surface_idx, constraint_idx)
        
        self._update_constraint_summary()
        logger.info(f"Hull boundary parent selection: {is_checked} - updated {len(updated_constraints)} segments")

    def _handle_intersection_parent_selection(self, parent_item, surface_idx, line_id):
        """Handle selection of Intersection Line parent - select/unselect all intersection segments"""
        is_checked = parent_item.checkState(0) == Qt.Checked
        
        # Block signals to prevent recursion
        self.surface_constraint_tree.blockSignals(True)
        
        try:
            # Update all intersection segment children
            updated_constraints = []
            for i in range(parent_item.childCount()):
                child_item = parent_item.child(i)
                child_data = child_item.data(0, Qt.UserRole)
                
                if child_data and child_data[0] == 'constraint':
                    constraint_idx = child_data[2]
                    constraint_key = (surface_idx, constraint_idx)
                    
                    # Update child checkbox
                    child_item.setCheckState(0, Qt.Checked if is_checked else Qt.Unchecked)
                    
                    # Update constraint state
                    self.constraint_manager.constraint_states[constraint_key] = "SEGMENTS" if is_checked else "UNDEFINED"
                    child_item.setText(2, self.constraint_manager.constraint_states[constraint_key])
                    
                    updated_constraints.append(constraint_idx)
            
            # Update parent status text
            parent_item.setText(2, f"All Selected" if is_checked else "Container")
            
        finally:
            self.surface_constraint_tree.blockSignals(False)
        
        # Batch update visual appearance
        for constraint_idx in updated_constraints:
            self._update_constraint_appearance(surface_idx, constraint_idx)
        
        self._update_constraint_summary()
        logger.info(f"Intersection line {line_id} parent selection: {is_checked} - updated {len(updated_constraints)} segments")

    def _update_parent_checkbox_state(self, child_item):
        """Update parent checkbox state based on children selection status"""
        parent_item = child_item.parent()
        if not parent_item:
            return
        
        parent_data = parent_item.data(0, Qt.UserRole)
        if not parent_data or parent_data[0] != 'parent':
            return
        
        # Block signals to prevent recursion
        self.surface_constraint_tree.blockSignals(True)
        
        try:
            # Count checked/total children
            checked_count = 0
            total_count = parent_item.childCount()
            
            for i in range(total_count):
                child = parent_item.child(i)
                if child.checkState(0) == Qt.Checked:
                    checked_count += 1
            
            # Update parent checkbox state
            if checked_count == 0:
                parent_item.setCheckState(0, Qt.Unchecked)
                parent_item.setText(2, "Container")
            elif checked_count == total_count:
                parent_item.setCheckState(0, Qt.Checked)
                parent_item.setText(2, "All Selected")
            else:
                parent_item.setCheckState(0, Qt.PartiallyChecked)
                parent_item.setText(2, f"{checked_count}/{total_count} Selected")
                
        finally:
            self.surface_constraint_tree.blockSignals(False)

    def _initialize_parent_checkbox_state(self, parent_item):
        """Initialize parent checkbox state based on current children states during tree population"""
        checked_count = 0
        total_count = parent_item.childCount()
        
        for i in range(total_count):
            child = parent_item.child(i)
            if child.checkState(0) == Qt.Checked:
                checked_count += 1
        
        # Set initial parent checkbox state
        if checked_count == 0:
            parent_item.setCheckState(0, Qt.Unchecked)
            parent_item.setText(2, "Container")
        elif checked_count == total_count:
            parent_item.setCheckState(0, Qt.Checked)
            parent_item.setText(2, "All Selected")
        else:
            parent_item.setCheckState(0, Qt.PartiallyChecked)
            parent_item.setText(2, f"{checked_count}/{total_count} Selected")

    def _update_surface_visibility(self, surface_idx, is_visible):
        """EFFICIENTLY update visibility of an entire surface and all its constraints"""
        if not self.constraint_plotter:
            return
        
        try:
            # Update surface mesh actor visibility
            surface_actor_name = f"surface_{surface_idx}"
            if surface_actor_name in self.constraint_plotter.actors:
                actor = self.constraint_plotter.actors[surface_actor_name]
                actor.SetVisibility(is_visible)
                actor.Modified()
            
             # -------- constraint actors for this surface ---------------
            constraints = self.constraint_manager.surface_constraints.get(surface_idx, [])
            for constraint_idx, constraint in enumerate(constraints):
                # new format: each constraint *is* one segment
                # old format (if any): constraint['segments'] → list[…]
                if 'segments' in constraint:
                    seg_iter = constraint['segments']
                else:
                    seg_iter = [constraint]

                for segment_idx, _unused in enumerate(seg_iter):
                    actor_name = f"constraint_{surface_idx}_{constraint_idx}_{segment_idx}"
                    if actor_name in self.constraint_plotter.actors:
                        actor = self.constraint_plotter.actors[actor_name]
                        actor.SetVisibility(is_visible)
                        actor.Modified()
            
            # Single render call for all visibility updates
            self.constraint_plotter.render()
            
            logger.info(f"Surface {surface_idx} visibility set to: {is_visible}")
            
        except Exception as e:
            logger.error(f"Error updating surface visibility: {e}")

    def _update_constraint_appearance(self, surface_idx, constraint_idx):
        """EFFICIENTLY update only the visual appearance of a specific constraint without re-rendering"""
        if not self.constraint_plotter:
            return
        
        try:
            # Check if surface is visible first
            surface_visible = self.constraint_manager.surface_visibility.get(surface_idx, True)
            if not surface_visible:
                return  # Don't update appearance of hidden surfaces
            
            # Get constraint data
            constraints = self.constraint_manager.surface_constraints.get(surface_idx, [])
            if constraint_idx >= len(constraints):
                return
            
            constraint = constraints[constraint_idx]
            constraint_key = (surface_idx, constraint_idx)
            constraint_state = self.constraint_manager.constraint_states.get(constraint_key, "UNDEFINED")
            
            # Handle new constraint format (direct segment) vs old format (segments list)
            if 'segments' in constraint:
                # Old format compatibility
                segments_to_update = constraint['segments']
            else:
                # New format - constraint is the segment itself
                segments_to_update = [constraint]
            
            # Update each segment of this constraint
            for segment_idx, segment in enumerate(segments_to_update):
                actor_name = f"constraint_{surface_idx}_{constraint_idx}_{segment_idx}"
                
                # Check if actor exists in the plotter
                if actor_name in self.constraint_plotter.actors:
                    actor = self.constraint_plotter.actors[actor_name]
                    
                    # Update visual properties based on state WITHOUT recreating the mesh
                    # Check if we're in selection mode or normal visualization mode
                    if hasattr(self, 'constraint_visualization_mode') and self.constraint_visualization_mode == "SELECTION":
                        # In selection mode - use RGB colors for picking
                        rgb_color = self._get_rgb_color_for_constraint(surface_idx, constraint_idx, segment_idx)
                        normalized_color = [c / 255.0 for c in rgb_color]
                        logger.debug(f"Setting actor {actor_name} to RGB {rgb_color} (SELECTION mode)")
                        actor.GetProperty().SetColor(*normalized_color)
                        actor.GetProperty().SetLineWidth(4)
                        actor.GetProperty().SetOpacity(1.0)
                    # Normal mode - use state-based colors
                    elif constraint_state == "SEGMENTS":
                        # Selected: blue color, thick line
                        logger.debug(f"Setting actor {actor_name} to BLUE (SEGMENTS)")
                        actor.GetProperty().SetColor(0, 0, 1)  # Blue
                        actor.GetProperty().SetLineWidth(5)
                        actor.GetProperty().SetOpacity(0.9)
                    elif constraint_state == "HOLES":
                        # Holes: red color, thick line
                        logger.debug(f"Setting actor {actor_name} to RED (HOLES)")
                        actor.GetProperty().SetColor(1, 0, 0)  # Red
                        actor.GetProperty().SetLineWidth(5)
                        actor.GetProperty().SetOpacity(0.9)
                    else:  # "UNDEFINED"
                        # Unselected: white color, thin line
                        logger.debug(f"Setting actor {actor_name} to WHITE (UNDEFINED)")
                        actor.GetProperty().SetColor(1, 1, 1)  # White
                        actor.GetProperty().SetLineWidth(3)
                        actor.GetProperty().SetOpacity(0.7)
                    
                    # Force update of the actor
                    actor.Modified()
                else:
                    logger.warning(f"Actor {actor_name} not found in plotter.actors")
            
            # Only render once after updating all segments of this constraint
            self.constraint_plotter.render()
            
        except Exception as e:
            logger.error(f"Error updating constraint appearance: {e}")
            # Fallback to full update only if the efficient method fails
            self._visualize_single_surface_constraints(surface_idx, update_summary=False)

    def _batch_update_all_constraint_appearances(self, target_state):
        """EFFICIENTLY update all constraint appearances in a single batch operation"""
        if not self.constraint_plotter:
            return
        
        try:
            # Check if we have any actors at all - if not, need to visualize first
            total_actors = len([name for name in self.constraint_plotter.actors.keys() 
                              if name.startswith("constraint_")])
            
            if total_actors == 0:
                logger.info("No constraint actors found, initializing visualization first...")
                self._visualize_constraints(selection_mode=False)
            
            # Get visual properties for the target state
            if target_state == "SEGMENTS":
                color = (0, 0, 1)  # Blue
                line_width = 5
                opacity = 0.9
            elif target_state == "HOLES":
                color = (1, 0, 0)  # Red
                line_width = 5
                opacity = 0.9
            else:  # "UNDEFINED"
                color = (1, 1, 1)  # White
                line_width = 3
                opacity = 0.7
            
            # Update all constraint actors in one go (only for visible surfaces)
            updated_count = 0
            for surface_idx, constraints in self.constraint_manager.surface_constraints.items():
                # Skip hidden surfaces
                if not self.constraint_manager.surface_visibility.get(surface_idx, True):
                    continue
                    
                for constraint_idx, constraint in enumerate(constraints):
                    # Handle new constraint format (direct segment) vs old format (segments list)
                    if 'segments' in constraint:
                        segments_to_update = constraint['segments']
                    else:
                        segments_to_update = [constraint]
                    
                    for segment_idx, segment in enumerate(segments_to_update):
                        actor_name = f"constraint_{surface_idx}_{constraint_idx}_{segment_idx}"
                        
                        if actor_name in self.constraint_plotter.actors:
                            actor = self.constraint_plotter.actors[actor_name]
                            
                            # Update visual properties
                            actor.GetProperty().SetColor(*color)
                            actor.GetProperty().SetLineWidth(line_width)
                            actor.GetProperty().SetOpacity(opacity)
                            actor.Modified()
                            updated_count += 1
            
            # Single render call for all updates
            self.constraint_plotter.render()
            
            logger.info(f"Batch updated {updated_count} constraint actors to state: {target_state}")
            
        except Exception as e:
            logger.error(f"Error in batch update: {e}")
            # Fallback to full visualization if batch update fails
            logger.info("Falling back to full visualization...")
            if target_state == "UNDEFINED":
                self._visualize_constraints(selection_mode=False)  # Show all in white
            else:
                self._visualize_constraints(selection_mode=False)  # Show all with proper colors

    def _update_constraint_summary(self):
        """Update the constraint selection summary display"""
        summary = self.constraint_manager.get_constraint_summary()
        
        if summary['total'] == 0:
            text = "No constraints generated yet"
        else:
            text = f"Constraints: {summary['total']} total\n"
            text += f"• {summary['selected']} selected (SEGMENTS)\n"
            text += f"• {summary['holes']} holes\n"
            text += f"• {summary['undefined']} undefined"
            
            selected_surfaces = len(self.constraint_manager.get_selected_surfaces())
            text += f"\n\nSelected surfaces: {selected_surfaces}"
        
        self.constraint_selection_summary.setText(text)

    def _select_all_constraints(self):
        """Check every surface + every constraint → SEGMENTS & Visible"""
        # block signals while we tick check-boxes
        self.surface_constraint_tree.blockSignals(True)
        try:
            # iterate tree – tick every leaf & every surface
            for i in range(self.surface_constraint_tree.topLevelItemCount()):
                surface_item = self.surface_constraint_tree.topLevelItem(i)
                surface_idx  = surface_item.data(0, Qt.UserRole)[1]
                surface_item.setCheckState(0, Qt.Checked)
                surface_item.setText(2, "Visible")
                self.constraint_manager.surface_visibility[surface_idx] = True
                self._update_surface_visibility(surface_idx, True)

                # tick every descendant
                def _tick_children(parent):
                    for j in range(parent.childCount()):
                        child = parent.child(j)
                        child.setCheckState(0, Qt.Checked)
                        _tick_children(child)
                _tick_children(surface_item)
        finally:
            self.surface_constraint_tree.blockSignals(False)

        # set every constraint state to SEGMENTS
        for key in self.constraint_manager.constraint_states:
            self.constraint_manager.constraint_states[key] = "SEGMENTS"

        # fast appearance update
        self._batch_update_all_constraint_appearances("SEGMENTS")
        self._update_constraint_summary()

    def _unselect_all_constraints(self):
        """Uncheck every surface + every constraint → UNDEFINED & Hidden"""
        self.surface_constraint_tree.blockSignals(True)
        try:
            for i in range(self.surface_constraint_tree.topLevelItemCount()):
                surface_item = self.surface_constraint_tree.topLevelItem(i)
                surface_idx  = surface_item.data(0, Qt.UserRole)[1]
                surface_item.setCheckState(0, Qt.Unchecked)
                surface_item.setText(2, "Hidden")
                self.constraint_manager.surface_visibility[surface_idx] = False
                self._update_surface_visibility(surface_idx, False)

                def _untick(parent):
                    for j in range(parent.childCount()):
                        child = parent.child(j)
                        child.setCheckState(0, Qt.Unchecked)
                        _untick(child)
                _untick(surface_item)
        finally:
            self.surface_constraint_tree.blockSignals(False)

        for key in self.constraint_manager.constraint_states:
            self.constraint_manager.constraint_states[key] = "UNDEFINED"

        self._batch_update_all_constraint_appearances("UNDEFINED")
        self._update_constraint_summary()

    def _expand_all_tree_items(self):
        """Expand all items in the constraint tree"""
        self.surface_constraint_tree.expandAll()
        logger.info("Expanded all tree items")

    def _collapse_all_tree_items(self):
        """Collapse all items in the constraint tree"""
        self.surface_constraint_tree.collapseAll()
        logger.info("Collapsed all tree items")

    def _set_default_expansion_state(self):
        """Set a smart default expansion state for the tree"""
        # Expand all top-level surface items
        for i in range(self.surface_constraint_tree.topLevelItemCount()):
            surface_item = self.surface_constraint_tree.topLevelItem(i)
            surface_item.setExpanded(True)
            
            # Expand parent categories (Hull Boundary, Intersection Lines) 
            # but keep individual segments collapsed for cleaner view
            for j in range(surface_item.childCount()):
                parent_item = surface_item.child(j)
                parent_data = parent_item.data(0, Qt.UserRole)
                
                if parent_data and parent_data[0] == 'parent':
                    # Expand parent categories
                    parent_item.setExpanded(True)
                    
                    # If a parent has few children (<=3), expand those too
                    if parent_item.childCount() <= 3:
                        for k in range(parent_item.childCount()):
                            child_item = parent_item.child(k)
                            child_item.setExpanded(True)
                    # Otherwise keep children collapsed for cleaner view
        
        logger.info("Set default expansion state: surfaces and categories expanded")

    def _generate_tetrahedral_mesh_action(self):
        """Generate tetrahedral mesh using selected constraints"""
        
        # First, ensure constraints are generated from pre-tetra mesh data
        if not hasattr(self, 'constraint_manager') or not self.constraint_manager.surface_constraints:
            QMessageBox.information(self, "Generate Constraints", 
                                  "Please generate constraints from pre-tetra data first.")
            return
        
        # Get selected surfaces (only those with SEGMENTS constraints)
        selected_surfaces = self.constraint_manager.get_selected_surfaces()
        
        if not selected_surfaces:
            QMessageBox.warning(self, "No Selection", 
                              "No constraints marked as SEGMENTS. Please select constraints first.")
            return
        
        # Generate mesh using existing TetrahedralMeshGenerator with selected surfaces
        try:
            from meshit.tetra_mesh_utils import TetrahedralMeshGenerator
            
            generator = TetrahedralMeshGenerator(
                datasets=self.datasets,
                selected_surfaces=selected_surfaces  # Only selected surfaces
            )
            
            # Get TetGen switches
            tetgen_switches = "pq1.414aA"  # Default switches
            
            result = generator.generate_tetrahedral_mesh(tetgen_switches)
            
            if result:
                self.tetrahedral_mesh = result
                self._visualize_tetrahedral_result(result)
                self.export_mesh_btn.setEnabled(True)
                logger.info(f"Generated tetrahedral mesh using {len(selected_surfaces)} surfaces with selected constraints")
                QMessageBox.information(self, "Success", 
                                      f"Tetrahedral mesh generated successfully using {len(selected_surfaces)} selected surfaces!")
            else:
                QMessageBox.warning(self, "Generation Failed", "Failed to generate tetrahedral mesh.")
                
        except Exception as e:
            logger.error(f"Error generating tetrahedral mesh: {e}")
            QMessageBox.critical(self, "Error", f"Generation failed: {str(e)}")

    def _visualize_tetrahedral_result(self, result):
        """Visualize the generated tetrahedral mesh"""
        if not self.constraint_plotter or not result:
            return
        
        try:
            # Clear current visualization
            self.constraint_plotter.clear()
            
            # Get mesh data
            vertices = result.get('vertices')
            tetrahedra = result.get('tetrahedra')
            
            if vertices is not None and tetrahedra is not None:
                # Create PyVista unstructured grid
                import pyvista as pv
                import numpy as np
                
                vertices_array = np.array(vertices)
                tetrahedra_array = np.array(tetrahedra)
                
                # Create cells array for PyVista (tetrahedra)
                cells = []
                for tet in tetrahedra_array:
                    cells.extend([4, tet[0], tet[1], tet[2], tet[3]])
                
                # Create unstructured grid
                cell_types = [pv.CellType.TETRA] * len(tetrahedra_array)
                mesh = pv.UnstructuredGrid(cells, cell_types, vertices_array)
                
                # Visualize edges of tetrahedra
                edges = mesh.extract_all_edges()
                self.constraint_plotter.add_mesh(
                    edges,
                    name="tetrahedral_mesh",
                    color='cyan',
                    line_width=1,
                    style='wireframe'
                )
                
                # Also show surface
                surface = mesh.extract_surface()
                self.constraint_plotter.add_mesh(
                    surface,
                    name="tetrahedral_surface",
                    color='lightblue',
                    opacity=0.6,
                    style='surface'
                )
                
                self.constraint_plotter.reset_camera()
                logger.info(f"Visualized tetrahedral mesh: {len(vertices)} vertices, {len(tetrahedra)} tetrahedra")
            
        except Exception as e:
            logger.error(f"Error visualizing tetrahedral mesh: {e}")

    def _initialize_tetra_visualization(self):
        """Initialize the tetra mesh tab visualization with default settings."""
        try:
            # Set default visualization options
            if hasattr(self, 'show_convex_hulls_check'):
                self.show_convex_hulls_check.setChecked(True)
            if hasattr(self, 'show_intersections_check'):
                self.show_intersections_check.setChecked(True)

            if hasattr(self, 'highlight_selected_check'):
                self.highlight_selected_check.setChecked(True)
            
            # Add all visualization elements
            self._add_selectable_elements_to_plotter()
            
            logger.info("Tetra mesh visualization initialized with default settings")
        except Exception as e:
            logger.error(f"Error initializing tetra visualization: {e}")

    # _setup_tetra_visualization_controls removed - using only constraint selection viewer

    # Display mode and visualization options handlers removed - using only constraint selection viewer





    def _has_valid_visualization_data(self, dataset):
        """Check if dataset has valid data for visualization."""
        # Check for constrained mesh data first
        if ('constrained_vertices' in dataset and 'constrained_triangles' in dataset):
            vertices = dataset.get('constrained_vertices')
            triangles = dataset.get('constrained_triangles')
            if self._has_valid_array_data(vertices) and self._has_valid_array_data(triangles):
                return True
        
        # Check for triangulated data
        if ('triangulated_vertices' in dataset and 'triangulated_triangles' in dataset):
            vertices = dataset.get('triangulated_vertices')
            triangles = dataset.get('triangulated_triangles')
            if self._has_valid_array_data(vertices) and self._has_valid_array_data(triangles):
                return True
        
        # Check for hull data
        if 'hull_points' in dataset:
            hull_points = dataset.get('hull_points')
            if self._has_valid_array_data(hull_points):
                return True
        
        return False

    def _identify_border_surfaces(self):
        """Identify which datasets are borders based on naming patterns (like C++ version)."""
        border_indices = []
        for i, dataset in enumerate(self.datasets):
            name = dataset['name'].lower()
            # ONLY border-specific patterns (no overlap with units)
            border_patterns = [
                'border', 'bound', 'boundary', 'side'
            ]
            
            if any(pattern in name for pattern in border_patterns):
                border_indices.append(i)
                logger.info(f"Detected BORDER surface: {name} (index {i})")
        
        logger.info(f"Found {len(border_indices)} border surfaces: {border_indices}")
        return border_indices
    

    def _export_tetrahedral_mesh(self):
        """Export tetrahedral mesh using the utility module."""
        if not hasattr(self, 'tetra_mesh_generator') or not self.tetrahedral_mesh:
            QMessageBox.warning(self, "No Mesh", "No tetrahedral mesh to export.")
            return
        
        # Get export file path
        file_path, file_filter = QFileDialog.getSaveFileName(
            self,
            "Export Tetrahedral Mesh",
            "tetrahedral_mesh.vtk",
            "VTK files (*.vtk);;VTU files (*.vtu);;PLY files (*.ply);;STL files (*.stl)"
        )
        
        if not file_path:
            return
        
        try:
            success = self.tetra_mesh_generator.export_mesh(file_path, self.tetrahedral_mesh)
            if success:
                QMessageBox.information(self, "Export Successful", f"Tetrahedral mesh exported to:\n{file_path}")
            else:
                QMessageBox.critical(self, "Export Error", "Failed to export mesh. Check logs for details.")
                
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            QMessageBox.critical(self, "Export Error", f"Failed to export mesh:\n{str(e)}")
    def _identify_unit_surfaces(self):
        """Identify which datasets are geological units."""
        unit_indices = []
        for i, dataset in enumerate(self.datasets):
            name = dataset['name'].lower()
            # ONLY unit-specific patterns (geological layers)
            unit_patterns = [
                'top', 'bottom', 'middle', 'unit', 'layer', 'formation', 
                'geol', 'rock', 'strat', 'horizon'
            ]
            
            # Make sure it's not already classified as border or fault
            is_border = any(border_pattern in name for border_pattern in ['border', 'bound', 'boundary', 'side'])
            is_fault = any(fault_pattern in name for fault_pattern in ['fault', 'fracture', 'break'])
            
            if not is_border and not is_fault and any(pattern in name for pattern in unit_patterns):
                unit_indices.append(i)
                logger.info(f"Detected UNIT surface: {name} (index {i})")
        
        logger.info(f"Found {len(unit_indices)} unit surfaces: {unit_indices}")
        return unit_indices
    def _generate_tetrahedral_mesh(self):
        """Generate tetrahedral mesh using the new utility module."""
        # Get visible surfaces instead of selected surfaces
        visible_surfaces = set()
        for i, dataset in enumerate(self.datasets):
            if dataset.get('visible', True):
                visible_surfaces.add(i)
        
        if not visible_surfaces:
            QMessageBox.warning(self, "No Visible Surfaces", "Please make some surfaces visible for mesh generation.")
            return
        
        try:
            # Create tetrahedral mesh generator
            generator = TetrahedralMeshGenerator(
                datasets=self.datasets,
                selected_surfaces=visible_surfaces,
                border_surface_indices=self.border_surface_indices,
                unit_surface_indices=self.unit_surface_indices,
                fault_surface_indices=self.fault_surface_indices,
                materials=self.materials
            )
            
            # Get TetGen switches from UI
            tetgen_switches = self.tetgen_switches_input.text().strip() or "pq1.414aA"
            
            # Generate mesh
            tetrahedral_grid = generator.generate_tetrahedral_mesh(tetgen_switches)
            
            if tetrahedral_grid is not None:
                self.tetrahedral_mesh = tetrahedral_grid
                self.tetra_mesh_generator = generator  # Store for later use
                
                # Update visualization and statistics
                self._update_tetra_stats()
                self._visualize_tetrahedral_mesh()
                
                # Enable export button
                if hasattr(self, 'export_tetra_mesh_btn'):
                    self.export_tetra_mesh_btn.setEnabled(True)
                
                logger.info("Tetrahedral mesh generation completed successfully")
                QMessageBox.information(self, "Success", "Tetrahedral mesh generated successfully!")
            else:
                logger.error("TetGen failed to generate mesh")
                QMessageBox.critical(self, "TetGen Failed", "Failed to generate tetrahedral mesh. Check surface selection and quality.")
        
        except Exception as e:
            logger.error(f"Mesh generation failed: {str(e)}")
            QMessageBox.critical(self, "Generation Failed", f"Mesh generation failed: {str(e)}")
    def _on_selection_mode_changed(self, mode):
        """Handle selection mode changes."""
        logger.info(f"Selection mode changed to: {mode}")
        
        # Update mouse selection state based on mode
        if mode == "Mouse Selection":
            if not self.interactive_selection_enabled:
                self._toggle_mouse_selection(True)
        else:
            if self.interactive_selection_enabled:
                self._toggle_mouse_selection(False)
        
        # Update visual indicators
        self._update_visualization()
        
        # Update status
        if hasattr(self, 'status_bar') and self.status_bar:
            self.status_bar.showMessage(f"Selection mode: {mode}", 3000)



    def _update_surface_selection_visual(self, surface_index, selected):
        """Update visual representation of surface selection."""
        if not hasattr(self, 'tetra_plotter') or not self.tetra_plotter:
            return
            
        actor_name = f"surface_{surface_index}"
        
        try:
            # Get mesh data from dataset
            dataset = self.datasets[surface_index]
            vertices = None
            triangles = None
            
            # Try different mesh data sources
            if 'constrained_vertices' in dataset and 'constrained_triangles' in dataset:
                vertices = dataset.get('constrained_vertices')
                triangles = dataset.get('constrained_triangles')
            elif 'triangulated_vertices' in dataset and 'triangulated_triangles' in dataset:
                vertices = dataset.get('triangulated_vertices')
                triangles = dataset.get('triangulated_triangles')
            
            if vertices is not None and triangles is not None and len(vertices) > 0 and len(triangles) > 0:
                import pyvista as pv
                import numpy as np
                
                # Create mesh
                vertices_array = np.array(vertices)
                triangles_array = np.array(triangles)
                
                # Ensure 3D coordinates
                if vertices_array.shape[1] == 2:
                    temp_vertices = np.zeros((vertices_array.shape[0], 3))
                    temp_vertices[:, :2] = vertices_array
                    vertices_array = temp_vertices
                
                # Create faces array
                faces = []
                for tri in triangles_array:
                    faces.extend([3, tri[0], tri[1], tri[2]])
                
                mesh = pv.PolyData(vertices_array, faces=faces)
                
                if selected:
                    # Highlight selected surface
                    self.tetra_plotter.add_mesh(
                        mesh,
                        name=actor_name,
                        color='yellow',
                        opacity=0.8,
                        pickable=True
                    )
                else:
                    # Reset to default appearance
                    surface_type = self._get_surface_type(surface_index)
                    color = self._get_surface_color(surface_type)
                    
                    self.tetra_plotter.add_mesh(
                        mesh,
                        name=actor_name,
                        color=color,
                        opacity=0.6,
                        pickable=True
                    )
            else:
                logger.warning(f"No valid mesh data for surface {surface_index}")
                
        except Exception as e:
            logger.error(f"Error updating surface {surface_index} visual: {e}")

    def _get_surface_color(self, surface_type):
        """Get color for surface type."""
        color_map = {
            'Border': 'red',
            'Unit': 'blue', 
            'Fault': 'green',
            'Intersection': 'orange',
            'Unknown': 'gray'
        }
        return color_map.get(surface_type, 'gray')
    def _identify_fault_surfaces(self):
        """Identify which datasets are faults."""
        fault_indices = []
        for i, dataset in enumerate(self.datasets):
            name = dataset['name'].lower()
            # More comprehensive fault naming patterns
            fault_patterns = [
                'fault', 'fracture', 'shear', 'joint', 'break', 'fault1', 'fault2', 'fault3', 'fault4'
            ]
            
            if any(pattern in name for pattern in fault_patterns):
                fault_indices.append(i)
                logger.info(f"Detected FAULT surface: {name} (index {i})")
        
        logger.info(f"Found {len(fault_indices)} fault surfaces: {fault_indices}")
        return fault_indices
    def _get_surface_type(self, surface_index):
        """Get the type of a surface (BORDER, UNIT, FAULT) like C++ version."""
        # Check for manual override first
        if surface_index < len(self.datasets):
            manual_type = self.datasets[surface_index].get('manual_surface_type')
            if manual_type and manual_type in ['BORDER', 'UNIT', 'FAULT', 'UNKNOWN']:
                return manual_type
        
        # Use automatic detection based on index lists
        if surface_index in self.border_surface_indices:
            return "BORDER"
        elif surface_index in self.fault_surface_indices:
            return "FAULT"
        elif surface_index in self.unit_surface_indices:
            return "UNIT"
        else:
            # Fallback: analyze filename if no type detected
            if surface_index < len(self.datasets):
                name = self.datasets[surface_index]['name'].lower()
                if any(p in name for p in ['border', 'bound', 'top.', 'bottom.', 'side']):
                    return "BORDER"
                elif any(p in name for p in ['fault', 'fracture']):
                    return "FAULT"
                elif any(p in name for p in ['unit', 'layer', 'middle.']):
                    return "UNIT"
            
            return "UNKNOWN"
    def _setup_interactive_3d_viewer(self, parent_group, layout):
        """Set up 3D viewer with mouse selection capabilities."""
        
        # Viewer controls
        viewer_controls = QHBoxLayout()
        
        self.reset_view_btn = QPushButton("🔄 Reset View")
        self.fit_view_btn = QPushButton("📐 Fit All")
        self.enable_mouse_selection_btn = QPushButton("🖱️ Enable Mouse Selection")
        self.selection_info_label = QLabel("Click elements to select")
        self.selection_info_label.setStyleSheet("color: #666; font-style: italic;")
        
        viewer_controls.addWidget(self.reset_view_btn)
        viewer_controls.addWidget(self.fit_view_btn)
        viewer_controls.addWidget(self.enable_mouse_selection_btn)
        viewer_controls.addStretch()
        viewer_controls.addWidget(self.selection_info_label)
        
        layout.addLayout(viewer_controls)
        
        # 3D Visualization frame with selection
        self.tetra_viz_frame = QFrame()
        self.tetra_viz_frame.setFrameShape(QFrame.StyledPanel)
        self.tetra_plot_layout = QVBoxLayout(self.tetra_viz_frame)
        self.tetra_plot_layout.setContentsMargins(0, 0, 0, 0)
        
        if HAVE_PYVISTA:
            from pyvistaqt import QtInteractor
            self.tetra_plotter = QtInteractor(self.tetra_viz_frame)
            self.tetra_plot_layout.addWidget(self.tetra_plotter.interactor)
            
            # Configure for interactive selection
            self.tetra_plotter.set_background([0.2, 0.2, 0.25])
            
            # Initialize color cycle for surfaces
            self.color_cycle = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
                              'lightpink', 'lightcyan', 'lightgray', 'wheat', 'plum', 'khaki']
            
            # Connect mouse selection button
            self.enable_mouse_selection_btn.clicked.connect(
                lambda: self._toggle_mouse_selection(not self.interactive_selection_enabled)
            )
            
            # Connect view controls
            self.reset_view_btn.clicked.connect(self.tetra_plotter.reset_camera)
            self.fit_view_btn.clicked.connect(lambda: self.tetra_plotter.reset_camera(render=True))
            
            logger.info("Interactive 3D viewer with selection capability setup completed")
        else:
            placeholder = QLabel("PyVista is required for interactive selection.")
            placeholder.setAlignment(Qt.AlignCenter)
            self.tetra_plot_layout.addWidget(placeholder)
            self.tetra_plotter = None
            
            # Disable selection button
            self.enable_mouse_selection_btn.setEnabled(False)
        
        layout.addWidget(self.tetra_viz_frame, 1)

    def _add_selectable_elements_to_plotter(self):
        """Add all selectable elements from pre-tetra mesh tab to the plotter."""
        if not self.tetra_plotter:
            return
        
        # Use the unified visualization system
        logger.info("Initializing unified visualization system")
        self._update_unified_visualization(filter_changed=True)
    

    def _add_all_intersections_to_plotter(self):
        """Add all intersection lines to the plotter."""
        try:
            import pyvista as pv
            
            all_intersection_points = []
            all_intersection_lines = []
            line_start_idx = 0
            
            for dataset_idx, dataset in enumerate(self.datasets):
                intersections = dataset.get('intersections', [])
                
                for intersection_idx, intersection in enumerate(intersections):
                    points = intersection.get('points', [])
                    
                    if len(points) >= 2:
                        # Add points to global list
                        intersection_points = np.array([[p[0], p[1], p[2]] for p in points])
                        all_intersection_points.extend(intersection_points)
                        
                        # Create line connectivity
                        for i in range(len(points) - 1):
                            all_intersection_lines.extend([2, line_start_idx + i, line_start_idx + i + 1])
                        
                        line_start_idx += len(points)
            
            if all_intersection_points and all_intersection_lines:
                intersection_mesh = pv.PolyData(np.array(all_intersection_points), lines=all_intersection_lines)
                
                self.tetra_plotter.add_mesh(
                    intersection_mesh,
                    color='yellow',
                    line_width=3,
                    render_lines_as_tubes=False,
                    name='intersection_lines'
                )
                
                logger.info(f"Added {len(all_intersection_points)} intersection points")
                
        except Exception as e:
            logger.warning(f"Failed to add intersection lines: {e}")




    def _add_surface_to_plotter(self, surface_index, vertices, triangles):
        """Add a surface with vertices and triangles to the plotter."""
        if not self.tetra_plotter:
            return
            
        try:
            import pyvista as pv
            import numpy as np
            
            # Convert to numpy arrays
            vertices_array = np.array(vertices)
            
            # Create faces array for PyVista (prepend face size)
            faces = []
            for tri in triangles:
                faces.extend([3, tri[0], tri[1], tri[2]])
            
            # Create mesh
            surface_mesh = pv.PolyData(vertices_array, faces)
            
            # Add surface metadata
            surface_mesh.field_data['surface_id'] = np.array([surface_index])
            surface_mesh.field_data['element_type'] = np.array(['surface'], dtype='U10')
            
            # Get surface type for coloring
            surface_type = self._get_surface_type(surface_index)
            
            # Color based on surface type
            if surface_type == "BORDER":
                color = 'lightblue'
                opacity = 0.7
            elif surface_type == "FAULT":
                color = 'lightcoral'
                opacity = 0.8
            elif surface_type == "UNIT":
                color = 'lightgreen'
                opacity = 0.6
            else:
                color = self.color_cycle[surface_index % len(self.color_cycle)]
                opacity = 0.7
            
            # Add to plotter
            actor = self.tetra_plotter.add_mesh(
                surface_mesh,
                color=color,
                opacity=opacity,
                show_edges=True,
                edge_color='black',
                line_width=1,
                name=f'surface_{surface_index}'
            )
            
            logger.debug(f"Added surface {surface_index} to plotter with {len(vertices)} vertices, {len(triangles)} triangles")
            
        except Exception as e:
            logger.error(f"Failed to add surface {surface_index} to plotter: {e}")

    def _add_surface_with_display_mode(self, surface_index, vertices, triangles, display_mode):
        """Add a surface with the specified display mode."""
        if not self.tetra_plotter:
            return
            
        try:
            import pyvista as pv
            import numpy as np
            
            # Convert to numpy arrays
            vertices_array = np.array(vertices)
            
            # Ensure 3D coordinates
            if vertices_array.shape[1] == 2:
                temp_vertices = np.zeros((vertices_array.shape[0], 3))
                temp_vertices[:, :2] = vertices_array
                vertices_array = temp_vertices
            
            # Create faces array for PyVista (prepend face size)
            faces = []
            for tri in triangles:
                faces.extend([3, tri[0], tri[1], tri[2]])
            
            # Create mesh
            surface_mesh = pv.PolyData(vertices_array, faces)
            
            # Add surface metadata
            surface_mesh.field_data['surface_id'] = np.array([surface_index])
            surface_mesh.field_data['element_type'] = np.array(['surface'], dtype='U10')
            
            # Get surface type for coloring
            surface_type = self._get_surface_type(surface_index)
            color = self._get_surface_color(surface_type)
            
            # Apply display mode
            if display_mode == "faces":
                self.tetra_plotter.add_mesh(
                    surface_mesh,
                    color=color,
                    opacity=0.7,
                    show_edges=False,
                    name=f'surface_faces_{surface_index}',
                    pickable=True
                )
            elif display_mode == "wireframe":
                self.tetra_plotter.add_mesh(
                    surface_mesh,
                    color=color,
                    style='wireframe',
                    line_width=2,
                    name=f'surface_wireframe_{surface_index}',
                    pickable=True
                )
            elif display_mode == "both":
                # Add faces with transparency
                self.tetra_plotter.add_mesh(
                    surface_mesh,
                    color=color,
                    opacity=0.5,
                    show_edges=False,
                    name=f'surface_faces_{surface_index}',
                    pickable=True
                )
                # Add wireframe
                self.tetra_plotter.add_mesh(
                    surface_mesh,
                    color='black',
                    style='wireframe',
                    line_width=1,
                    name=f'surface_wireframe_{surface_index}',
                    pickable=True
                )
            
            logger.debug(f"Added surface {surface_index} with display mode '{display_mode}': {len(vertices)} vertices, {len(triangles)} triangles")
            
        except Exception as e:
            logger.error(f"Failed to add surface {surface_index} with display mode '{display_mode}': {e}")

    def _add_convex_hull_to_plotter(self, surface_idx, hull_points):
        """Add convex hull as simple boundary lines - SIMPLIFIED like pre-tetramesh."""
        try:
            import pyvista as pv
            from scipy.spatial import ConvexHull
        except Exception as e:
            logger.error(f"Failed to import pyvista or scipy: {e}")
            return
        
        logger.debug(f"Adding simple convex hull for surface {surface_idx}")
        
        if len(hull_points) < 3:
            logger.warning(f"Not enough points ({len(hull_points)}) for convex hull")
            return
            
        plotter = self.tetra_plotter
        if not plotter:
            logger.error("No plotter available for convex hull visualization")
            return
        
        # Extract coordinates (handle type information if present) 
        if hasattr(hull_points, 'dtype') and hull_points.dtype == object:
            coords = np.array([[p[0], p[1], p[2]] for p in hull_points])
        else:
            coords = np.array(hull_points)[:, :3]
        
        try:
            # Simple approach: get convex hull vertices and create boundary line loop (like pre-tetramesh)
            hull = ConvexHull(coords)
            hull_boundary_points = coords[hull.vertices]
            
            # Close the loop by adding first point at the end
            hull_boundary_closed = np.vstack([hull_boundary_points, hull_boundary_points[0:1]])
            
            # Create simple line mesh (just like pre-tetramesh does)
            line_mesh = pv.PolyData(hull_boundary_closed)
            lines = []
            for i in range(len(hull_boundary_closed) - 1):
                lines.extend([2, i, i + 1])
            line_mesh.lines = np.array(lines)
            
            # Add to plotter with simple red lines
            plotter.add_mesh(
                line_mesh,
                color='red',
                line_width=3,
                style='wireframe',
                name=f'convex_hull_{surface_idx}',
                pickable=True
            )
            
            logger.info(f"Successfully added simple convex hull boundary for surface {surface_idx}")
            
        except Exception as e:
            logger.warning(f"Standard ConvexHull failed for surface {surface_idx}: {e}")
            
            # Fallback 1: Try with QJ option (joggle input to avoid precision issues)
            try:
                hull = ConvexHull(coords, qhull_options='QJ')
                hull_boundary_points = coords[hull.vertices]
                
                # Close the loop by adding first point at the end
                hull_boundary_closed = np.vstack([hull_boundary_points, hull_boundary_points[0:1]])
                
                # Create simple line mesh
                line_mesh = pv.PolyData(hull_boundary_closed)
                lines = []
                for i in range(len(hull_boundary_closed) - 1):
                    lines.extend([2, i, i + 1])
                line_mesh.lines = np.array(lines)
                
                plotter.add_mesh(
                    line_mesh,
                    color='red',
                    line_width=3,
                    style='wireframe',
                    name=f'convex_hull_{surface_idx}',
                    pickable=True
                )
                
                logger.info(f"Successfully added simple QJ convex hull boundary for surface {surface_idx}")
                
            except Exception as e2:
                logger.warning(f"QJ ConvexHull also failed for surface {surface_idx}: {e2}")
                
                # Final fallback: Use all points as a wireframe mesh
                try:
                    # Create a simple point cloud and connect nearest neighbors
                    line_mesh = pv.PolyData(coords)
                    
                    # Create lines connecting consecutive points (simple boundary)
                    lines = []
                    n_points = len(coords)
                    for i in range(n_points):
                        lines.extend([2, i, (i + 1) % n_points])
                    line_mesh.lines = np.array(lines)
                    
                    plotter.add_mesh(
                        line_mesh,
                        color='orange',
                        line_width=1,
                        style='wireframe',
                        name=f'convex_hull_{surface_idx}',
                        pickable=True
                    )
                    
                    logger.info(f"Added fallback wireframe for surface {surface_idx}")
                    
                except Exception as e3:
                    logger.error(f"All hull methods failed for surface {surface_idx}: {e3}")

    def _on_mesh_picked(self, mesh_data):
        """Handle mesh picking events."""
        if not self.interactive_selection_enabled:
            return
        
        try:
            # Get picked mesh info
            if hasattr(mesh_data, 'field_data') and mesh_data.field_data:
                surface_id = None
                element_type = 'unknown'
                
                # Try to get surface_id from field_data
                if 'surface_id' in mesh_data.field_data:
                    surface_id = mesh_data.field_data['surface_id'][0]
                
                # Try to get element_type from field_data
                if 'element_type' in mesh_data.field_data:
                    element_type = mesh_data.field_data['element_type'][0]
                    if isinstance(element_type, bytes):
                        element_type = element_type.decode('utf-8')
                
                if surface_id is not None:
                    self._handle_tetra_element_selection(surface_id, element_type, mesh_data)
                else:
                    # Try to identify surface by mesh properties
                    surface_id = self._identify_clicked_surface(mesh_data)
                    if surface_id is not None:
                        self._handle_tetra_element_selection(surface_id, 'surface', mesh_data)
                    
        except Exception as e:
            logger.warning(f"Error in mesh picking: {e}")

    def _handle_tetra_element_selection(self, surface_id, element_type, mesh_data):
        """Handle selection of a specific element in tetra mesh tab."""
        try:
            # Toggle surface selection
            if surface_id in self.selected_surfaces:
                self.selected_surfaces.remove(surface_id)
                action = "deselected"
            else:
                self.selected_surfaces.add(surface_id)
                action = "selected"
                
            logger.info(f"Surface {surface_id} {action} in tetra mesh tab")
            
            # Update visualization and UI
            self._update_tetra_selection_stats()
            self._update_unified_visualization(highlighting_changed=True)
            
            # Generate button is always enabled since we use visible surfaces
                
        except Exception as e:
            logger.error(f"Error handling tetra element selection: {e}")

    def _handle_element_selection(self, surface_id, element_type, mesh_data):
        """Handle selection of a specific element."""
        current_mode = self.selection_mode_combo.currentText()
        
        if current_mode == "Whole Surfaces":
            if surface_id in self.selected_surfaces:
                self.selected_surfaces.remove(surface_id)
                action = "deselected"
            else:
                self.selected_surfaces.add(surface_id)
                action = "selected"
                
            logger.info(f"Surface {surface_id} {action}")
            
        elif current_mode == "Surface Parts":
            # Select specific triangles/parts
            if surface_id not in self.selected_surface_parts:
                self.selected_surface_parts[surface_id] = set()
                
            # Add picked triangles to selection
            if hasattr(mesh_data, 'GetNumberOfCells'):
                for i in range(mesh_data.GetNumberOfCells()):
                    self.selected_surface_parts[surface_id].add(i)
                    
            logger.info(f"Added parts to surface {surface_id}")
            
        elif current_mode == "Convex Hull Parts":
            # Select hull segments
            logger.info(f"Selected hull part of surface {surface_id}")
        
        # Update visualization and UI
        self._update_selection_visualization()
        self._update_selection_summary()
        self._populate_selection_tree()

    def _toggle_mouse_selection(self, enabled):
        """Toggle mouse selection mode with proper picker setup."""
        self.interactive_selection_enabled = enabled
        
        if not self.tetra_plotter:
            return
        
        if enabled:
            try:
                # First disable any existing picking
                self.tetra_plotter.disable_picking()
                
                # Then enable mesh picking callback
                self.tetra_plotter.enable_mesh_picking(
                    callback=self._on_mesh_picked, 
                    show=False,
                    style='wireframe'
                )
                self.enable_mouse_selection_btn.setText("🖱️ Mouse Selection ON")
                self.enable_mouse_selection_btn.setStyleSheet("background-color: #4CAF50; color: white;")
                self.selection_info_label.setText("Click elements to select/deselect")
                logger.info("Mouse selection enabled")
            except Exception as e:
                logger.error(f"Failed to enable mouse selection: {e}")
        else:
            try:
                self.tetra_plotter.disable_picking()
                self.enable_mouse_selection_btn.setText("🖱️ Enable Mouse Selection")
                self.enable_mouse_selection_btn.setStyleSheet("")
                self.selection_info_label.setText("Mouse selection disabled")
                logger.info("Mouse selection disabled")
            except Exception as e:
                logger.error(f"Failed to disable mouse selection: {e}")
    
    def _update_selection_visualization(self):
        """Update the visual representation of the current selection."""
        if not hasattr(self, 'tetra_plotter') or not self.tetra_plotter:
            return
            
        try:
            # Update highlighting to reflect current selection
            if hasattr(self, 'highlight_selected_check') and self.highlight_selected_check.isChecked():
                self._update_unified_visualization(highlighting_changed=True)
            else:
                # Just refresh the current visualization
                self._update_unified_visualization()
        except Exception as e:
            logger.error(f"Error updating selection visualization: {e}")

    def _update_surface_table_checkboxes(self):
        """Update checkboxes in the surface table to match current selection."""
        for i in range(self.surface_table.rowCount()):
            name_item = self.surface_table.item(i, 1)
            if name_item:
                surface_name = name_item.text()
                # Find corresponding surface index
                surface_index = None
                for idx, dataset in enumerate(self.datasets):
                    if dataset.get('name', f'Surface_{idx}') == surface_name:
                        surface_index = idx
                        break
                
                if surface_index is not None:
                    checkbox_widget = self.surface_table.cellWidget(i, 0)
                    if checkbox_widget:
                        checkbox = checkbox_widget.findChild(QCheckBox)
                        if checkbox:
                            # Temporarily disconnect to prevent recursion
                            checkbox.blockSignals(True)
                            checkbox.setChecked(surface_index in self.selected_surfaces)
                            checkbox.blockSignals(False)

    
    def _identify_clicked_surface(self, mesh):
        """Identify which surface index corresponds to the clicked mesh."""
        # This is a simplified approach - you may need to store mesh metadata
        # to properly identify the surface
        try:
            # Check mesh center against known surface centers
            mesh_center = mesh.center if hasattr(mesh, 'center') else [0, 0, 0]
            
            for i, dataset in enumerate(self.datasets):
                vertices = dataset.get('constrained_vertices')
                if self._has_valid_array_data(vertices):
                    try:
                        import numpy as np
                        surface_center = np.mean(vertices, axis=0)
                        distance = np.linalg.norm(np.array(mesh_center) - surface_center)
                        if distance < 10:  # Threshold for matching
                            return i
                    except:
                        continue
            
            return None
        except Exception as e:
            logger.warning(f"Failed to identify clicked surface: {e}")
            return None
    def _update_selection_summary(self):
        """Update the selection summary display."""
        total_surfaces = len(self.selected_surfaces)
        total_parts = sum(len(parts) for parts in self.selected_surface_parts.values())
        
        if total_surfaces == 0 and total_parts == 0:
            summary = "No elements selected"
        else:
            summary = f"Selected: {total_surfaces} surfaces"
            if total_parts > 0:
                summary += f", {total_parts} surface parts"
        
        self.selection_summary_label.setText(summary)

    def _populate_selection_tree(self):
        """Populate the hierarchical selection tree."""
        self.selection_tree.clear()
        
        for i, dataset in enumerate(self.datasets):
            if not ('constrained_vertices' in dataset and 'constrained_triangles' in dataset):
                continue
                
            # Create surface item
            surface_name = dataset.get('name', f'Surface_{i}')
            triangle_count = len(dataset.get('constrained_triangles', []))
            quality = self._calculate_surface_quality(
                dataset.get('constrained_vertices'), 
                dataset.get('constrained_triangles')
            )
            
            surface_item = QTreeWidgetItem([
                surface_name,
                'Surface', 
                '✓ Selected' if i in self.selected_surfaces else '○ Available',
                str(triangle_count),
                f"{quality:.3f}"
            ])
            
            # Add convex hull child
            if 'hull_points' in dataset:
                hull_item = QTreeWidgetItem([
                    'Convex Hull',
                    'Hull',
                    '○ Available',
                    str(len(dataset.get('hull_points', []))),
                    '-'
                ])
                surface_item.addChild(hull_item)
            
            # Add intersection constraints child
            if 'intersection_constraints' in dataset:
                constraints = dataset.get('intersection_constraints', [])
                if constraints:
                    constraint_item = QTreeWidgetItem([
                        'Intersection Constraints',
                        'Constraints',
                        '○ Available', 
                        str(len(constraints)),
                        '-'
                    ])
                    surface_item.addChild(constraint_item)
            
            self.selection_tree.addTopLevelItem(surface_item)
        
        self.selection_tree.expandAll()
    def _auto_select_by_quality(self):
        """Auto-select surfaces based on quality threshold."""
        try:
            # Get threshold value (use default if widget doesn't exist)
            if hasattr(self, 'quality_threshold'):
                threshold = self.quality_threshold.value()
            else:
                threshold = 0.5  # Default threshold
                
            selected_count = 0
            
            for i, dataset in enumerate(self.datasets):
                vertices = dataset.get('constrained_vertices')
                triangles = dataset.get('constrained_triangles')
                
                if self._has_valid_array_data(vertices) and self._has_valid_array_data(triangles):
                    quality = self._calculate_surface_quality(vertices, triangles)
                    
                    if quality >= threshold:
                        self.selected_surfaces.add(i)
                        selected_count += 1
                        
                        # Update checkbox in table
                        for row in range(self.surface_table.rowCount()):
                            name_item = self.surface_table.item(row, 1)
                            if name_item and name_item.text() == dataset.get('name', f'Surface_{i}'):
                                checkbox_widget = self.surface_table.cellWidget(row, 0)
                                if checkbox_widget:
                                    checkbox = checkbox_widget.findChild(QCheckBox)
                                    if checkbox:
                                        checkbox.blockSignals(True)
                                        checkbox.setChecked(True)
                                        checkbox.blockSignals(False)
                                break
            
            # Update visualization and statistics
            self._update_selection_statistics()
            self._update_unified_visualization(highlighting_changed=True)
            
            logger.info(f"Auto-selected {selected_count} surfaces with quality >= {threshold:.2f}")
            
        except Exception as e:
            logger.error(f"Error in auto-select by quality: {e}")
    def _filter_surfaces_by_type(self, filter_type):
        """Filter surfaces in the table based on type."""
        for i in range(self.surface_table.rowCount()):
            type_item = self.surface_table.item(i, 2)  # Type column
            constraints_item = self.surface_table.item(i, 5)  # Constraints column
            
            show_row = True
            if filter_type == "Constrained Only":
                show_row = type_item and "Constrained" in type_item.text()
            elif filter_type == "With Intersections":
                show_row = constraints_item and constraints_item.text() != "0"
            elif filter_type == "Boundary Surfaces":
                # Add logic to identify boundary surfaces
                pass
            
            self.surface_table.setRowHidden(i, not show_row)

    def _invert_surface_selection(self):
        """Invert the current surface selection."""
        for i in range(self.surface_table.rowCount()):
            checkbox_widget = self.surface_table.cellWidget(i, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(not checkbox.isChecked())
    def _has_valid_array_data(self, data):
        """Helper function to safely check if array-like data is valid, non-empty, and numeric."""
        if data is None:
            return False
        if not hasattr(data, '__len__'):
            return False
        try:
            if len(data) == 0:
                return False
            
            # For object or string arrays, we need to check if they can be converted
            import numpy as np
            arr = np.array(data)
            
            # If it's already numeric, check normally
            if np.issubdtype(arr.dtype, np.number):
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    return False
                return True
            
            # For non-numeric arrays, try to validate if they can be converted
            # This is a quick check - the actual conversion happens in _validate_and_convert_points
            if arr.dtype == object or arr.dtype.kind in ['U', 'S']:
                # Special handling for hull_points format: object array with [x, y, z, type] elements
                if arr.dtype == object:
                    # Check if this looks like hull points data
                    try:
                        # Sample first few elements to check format
                        sample_size = min(3, len(arr.flat))
                        valid_hull_points = 0
                        
                        for i, item in enumerate(arr.flat):
                            if i >= sample_size:
                                break
                            
                            # Handle different formats of hull point data
                            if hasattr(item, '__len__') and not isinstance(item, str):
                                # Could be [x, y, z, type] or similar format - check if it's a numpy array first
                                if hasattr(item, 'shape') and len(item) >= 3:
                                    try:
                                        # For numpy arrays like [-40.0 90.0 65.0 'DEFAULT']
                                        coords = [float(item[j]) for j in range(3)]
                                        valid_hull_points += 1
                                        logger.debug(f"Validated numpy array hull point: {coords}")
                                    except (ValueError, TypeError, IndexError):
                                        continue
                                elif len(item) >= 3:
                                    try:
                                        # Try to convert first 3 elements to float (x, y, z)
                                        coords = [float(item[j]) for j in range(3)]
                                        valid_hull_points += 1
                                        logger.debug(f"Validated list hull point: {coords}")
                                    except (ValueError, TypeError, IndexError):
                                        continue
                            elif hasattr(item, 'x') and hasattr(item, 'y') and hasattr(item, 'z'):
                                # Handle Vector3D-like objects
                                try:
                                    x, y, z = float(item.x), float(item.y), float(item.z)
                                    valid_hull_points += 1
                                    logger.debug(f"Validated Vector3D hull point: {x}, {y}, {z}")
                                except (ValueError, TypeError, AttributeError):
                                    continue
                            elif isinstance(item, str):
                                # Handle string representations like "(-79.180, -85.410, 5.332)"
                                import re
                                numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', str(item))
                                if len(numbers) >= 3:
                                    try:
                                        coords = [float(n) for n in numbers[:3]]
                                        valid_hull_points += 1
                                        logger.debug(f"Validated string hull point: {coords}")
                                    except ValueError:
                                        continue
                        
                        # If we found valid hull point format, it's valid
                        logger.debug(f"Found {valid_hull_points} valid hull points out of {sample_size} sampled")
                        if valid_hull_points > 0:
                            return True
                    except Exception as e:
                        logger.debug(f"Error checking object array format: {e}")
                        pass
                
                # Sample a few items to see if they look like coordinate data
                sample_size = min(5, len(arr.flat))
                for i, item in enumerate(arr.flat):
                    if i >= sample_size:
                        break
                        
                    if isinstance(item, str):
                        import re
                        # Check if string contains numbers
                        numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', str(item))
                        if numbers:
                            return True
                    elif hasattr(item, '__len__') and not isinstance(item, str):
                        # Check nested structures
                        try:
                            nested_arr = np.array(item)
                            if np.issubdtype(nested_arr.dtype, np.number):
                                return True
                        except:
                            pass
                
                return False  # No valid numeric data found in sample
                
            return False
            
        except Exception as e:
            logger.debug(f"Data validation failed: {e}")
            return False
    def _validate_and_convert_points(self, points, name="points"):
        """Validate and convert points to proper numeric numpy array format."""
        try:
            import numpy as np
            import ast
            import re
            
            if points is None:
                return None
                
            # Handle different data formats
            if isinstance(points, str):
                # Try to parse string representation of coordinates
                try:
                    # Remove any extra whitespace and brackets
                    cleaned = points.strip()
                    if cleaned.startswith('[') and cleaned.endswith(']'):
                        # Try to parse as literal
                        parsed = ast.literal_eval(cleaned)
                        points = np.array(parsed)
                    else:
                        # Try to extract numbers using regex
                        numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', cleaned)
                        if numbers:
                            numbers = [float(n) for n in numbers]
                            points = np.array(numbers)
                        else:
                            logger.error(f"{name} string format not parseable: {cleaned[:100]}")
                            return None
                except Exception as e:
                    logger.error(f"Failed to parse {name} string: {e}")
                    return None
            
            # Convert to numpy array if needed
            if not isinstance(points, np.ndarray):
                points = np.array(points)
            
            # Handle object arrays (nested lists/tuples)
            if points.dtype == object:
                try:
                    # Try to extract numeric data from nested structure
                    flat_points = []
                    for item in points.flat:
                        if isinstance(item, (list, tuple, np.ndarray)):
                            # Flatten nested arrays
                            flat_points.extend(self._flatten_nested_data(item))
                        elif isinstance(item, str):
                            # Parse string coordinates
                            try:
                                parsed = ast.literal_eval(item)
                                if isinstance(parsed, (list, tuple)):
                                    flat_points.extend(parsed)
                                else:
                                    flat_points.append(float(parsed))
                            except:
                                # Try regex extraction
                                numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', item)
                                flat_points.extend([float(n) for n in numbers])
                        elif isinstance(item, (int, float)):
                            flat_points.append(float(item))
                    
                    if flat_points:
                        points = np.array(flat_points)
                    else:
                        logger.error(f"{name} object array contains no extractable numeric data")
                        return None
                except Exception as e:
                    logger.error(f"Failed to extract data from {name} object array: {e}")
                    return None
            
            # Handle string arrays
            if points.dtype.kind in ['U', 'S']:  # Unicode or byte strings
                try:
                    flat_points = []
                    for item in points.flat:
                        item_str = str(item)
                        try:
                            # Try to parse as literal first
                            parsed = ast.literal_eval(item_str)
                            if isinstance(parsed, (list, tuple)):
                                flat_points.extend(parsed)
                            else:
                                flat_points.append(float(parsed))
                        except:
                            # Extract numbers using regex
                            numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', item_str)
                            flat_points.extend([float(n) for n in numbers])
                    
                    if flat_points:
                        points = np.array(flat_points)
                    else:
                        logger.error(f"{name} string array contains no extractable numeric data")
                        return None
                except Exception as e:
                    logger.error(f"Failed to parse {name} string array: {e}")
                    return None
            
            # Ensure numeric type
            if not np.issubdtype(points.dtype, np.number):
                try:
                    points = points.astype(np.float64)
                except:
                    logger.error(f"{name} cannot be converted to numeric type")
                    return None
                    
            # Check for invalid values
            if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                logger.error(f"{name} contains NaN or infinite values")
                return None
                
            # Ensure proper shape for 3D points
            if len(points.shape) == 1:
                if len(points) % 3 == 0:
                    points = points.reshape(-1, 3)
                elif len(points) % 2 == 0:
                    # Assume 2D points, add z=0
                    points_2d = points.reshape(-1, 2)
                    points = np.column_stack((points_2d, np.zeros(len(points_2d))))
                else:
                    logger.error(f"{name} has invalid shape for 3D coordinates: {points.shape}")
                    return None
            elif len(points.shape) == 2:
                if points.shape[1] == 2:
                    # Add z=0 for 2D points
                    points = np.column_stack((points, np.zeros(len(points))))
                elif points.shape[1] > 3:
                    # Take only x,y,z coordinates
                    points = points[:, :3]
                elif points.shape[1] != 3:
                    logger.error(f"{name} has invalid number of columns: {points.shape[1]}")
                    return None
            else:
                logger.error(f"{name} has invalid dimensionality: {points.shape}")
                return None
                
            # Final validation
            if points.size == 0 or len(points) == 0:
                return None
                
            return points.astype(np.float64)
            
        except Exception as e:
            logger.error(f"Error validating {name}: {e}")
            return None
    def _flatten_nested_data(self, data):
        """Recursively flatten nested data structures to extract numeric values."""
        import numpy as np
        
        result = []
        try:
            if isinstance(data, (int, float)):
                result.append(float(data))
            elif isinstance(data, str):
                import re
                # Extract numbers from string
                numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', data)
                result.extend([float(n) for n in numbers])
            elif isinstance(data, (list, tuple, np.ndarray)):
                for item in data:
                    result.extend(self._flatten_nested_data(item))
        except Exception as e:
            logger.debug(f"Error flattening data: {e}")
        
        return result

    def _validate_surface_selection(self):
        """Validate the current surface selection for mesh generation."""
        if not self.selected_surfaces:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Selection", "Please select at least one surface.")
            return
        
        # Check for manifold surfaces
        validation_results = []
        valid_count = 0
        total_count = len(self.selected_surfaces)
        
        for surface_index in self.selected_surfaces:
            if surface_index < len(self.datasets):
                dataset = self.datasets[surface_index]
                surface_name = dataset.get('name', f'Surface_{surface_index}')
                
                # Basic validation with proper numpy array handling
                constrained_vertices = dataset.get('constrained_vertices')
                constrained_triangles = dataset.get('constrained_triangles')
                
                # Fix: Handle numpy arrays properly
                has_vertices = (constrained_vertices is not None and 
                            hasattr(constrained_vertices, '__len__') and 
                            len(constrained_vertices) > 0)
                
                has_triangles = (constrained_triangles is not None and 
                            hasattr(constrained_triangles, '__len__') and 
                            len(constrained_triangles) > 0)
                
                if not has_vertices or not has_triangles:
                    validation_results.append(f"❌ {surface_name}: Missing constrained mesh data")
                else:
                    validation_results.append(f"✅ {surface_name}: Ready for meshing")
                    valid_count += 1
        
        # Create validation results dictionary in the expected format
        validation_dict = {
            'ready_for_tetgen': valid_count == total_count,
            'overall_status': 'VALID' if valid_count == total_count else 'PARTIAL' if valid_count > 0 else 'INVALID',
            'surface_count': total_count,
            'statistics': {
                'valid_surfaces': valid_count,
                'invalid_surfaces': total_count - valid_count
            },
            'details': validation_results
        }
        
        # Show validation dialog
        self._show_validation_results_dialog(validation_dict)

    def _update_selection_statistics(self):
        """Update the selection statistics label."""
        selected_count = len(self.selected_surfaces)
        
        # Count datasets with valid constrained vertices
        total_count = sum(1 for d in self.datasets if self._has_valid_array_data(d.get('constrained_vertices')))
        
        if selected_count == 0:
            self.selection_stats_label.setText("No surfaces selected")
        else:
            total_triangles = 0
            total_vertices = 0
            for surface_index in self.selected_surfaces:
                if surface_index < len(self.datasets):
                    dataset = self.datasets[surface_index]
                    triangles = dataset.get('constrained_triangles', [])
                    vertices = dataset.get('constrained_vertices', [])
                    
                    # Safe counting using helper function
                    if self._has_valid_array_data(triangles):
                        total_triangles += len(triangles)
                    
                    if self._has_valid_array_data(vertices):
                        total_vertices += len(vertices)
            
            self.selection_stats_label.setText(
                f"Selected: {selected_count}/{total_count} surfaces "
                f"({total_vertices:,} vertices, {total_triangles:,} triangles)"
            )

    def _calculate_surface_quality(self, vertices, triangles):
        """Calculate quality metrics for a surface."""
        # Fix: Handle numpy arrays properly
        if vertices is None or triangles is None:
            return 0.0
        
        # Convert to numpy arrays if needed and check length
        if hasattr(vertices, '__len__') and len(vertices) == 0:
            return 0.0
        if hasattr(triangles, '__len__') and len(triangles) == 0:
            return 0.0
        
        try:
            vertices_np = np.array(vertices)
            triangles_np = np.array(triangles)
            
            # Additional safety checks
            if vertices_np.size == 0 or triangles_np.size == 0:
                return 0.0
            if len(vertices_np.shape) < 2 or vertices_np.shape[1] < 3:
                return 0.0
            if len(triangles_np.shape) < 2 or triangles_np.shape[1] < 3:
                return 0.0
            
            # Calculate basic quality metrics
            qualities = []
            for triangle in triangles_np:
                if len(triangle) >= 3:
                    try:
                        # Get triangle vertices
                        v0, v1, v2 = vertices_np[triangle[:3]]
                        
                        # Calculate edge lengths
                        a = np.linalg.norm(v1 - v0)
                        b = np.linalg.norm(v2 - v1)
                        c = np.linalg.norm(v0 - v2)
                        
                        # Calculate area using cross product
                        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                        
                        # Calculate quality (area-to-perimeter ratio)
                        if area > 1e-12 and (a + b + c) > 1e-12:  # Use small epsilon instead of 0
                            quality = (4 * np.sqrt(3) * area) / ((a + b + c) ** 2)
                            if 0 <= quality <= 1:  # Sanity check
                                qualities.append(quality)
                    except (IndexError, ValueError):
                        # Skip invalid triangles
                        continue
            
            return np.mean(qualities) if qualities else 0.0
        except Exception as e:
            logger.warning(f"Error calculating surface quality: {e}")
            return 0.0
    
    def _update_surface_table(self):
        """Update the surface selection table with available constrained surfaces and quality metrics."""
        if not hasattr(self, 'surface_table'):
            return
            
        # Clear existing rows
        self.surface_table.setRowCount(0)
        
        # Check for constrained surfaces
        available_surfaces = []
        for i, dataset in enumerate(self.datasets):
            try:
                surface_name = dataset.get('name', f'Surface_{i}')
                surface_type = self._get_surface_type(i)
                
                # Check for constrained mesh data first (highest priority)
                if ('constrained_vertices' in dataset and 'constrained_triangles' in dataset):
                    constrained_vertices = dataset['constrained_vertices']
                    constrained_triangles = dataset['constrained_triangles']
                    
                    if (constrained_vertices is not None and constrained_triangles is not None and 
                        hasattr(constrained_vertices, '__len__') and hasattr(constrained_triangles, '__len__') and
                        len(constrained_vertices) > 0 and len(constrained_triangles) > 0):
                        
                        triangle_count = len(constrained_triangles)
                        
                        # Calculate quality with error handling
                        try:
                            quality = self._calculate_surface_quality(constrained_vertices, constrained_triangles)
                        except Exception as e:
                            logger.warning(f"Failed to calculate quality for surface {i}: {e}")
                            quality = 0.0
                        
                        # Count constraints
                        constraints_count = len(dataset.get('intersection_constraints', []))
                        
                        available_surfaces.append({
                            'index': i,
                            'name': surface_name,
                            'type': f'{surface_type} (Refined)',
                            'triangles': triangle_count,
                            'quality': quality,
                            'constraints': constraints_count
                        })
                        continue
                
                # Check for triangulated data (fallback)
                if ('triangulated_vertices' in dataset and 'triangulated_triangles' in dataset):
                    triangulated_vertices = dataset['triangulated_vertices']
                    triangulated_triangles = dataset['triangulated_triangles']
                    
                    if (triangulated_vertices is not None and triangulated_triangles is not None and 
                        hasattr(triangulated_vertices, '__len__') and hasattr(triangulated_triangles, '__len__') and
                        len(triangulated_vertices) > 0 and len(triangulated_triangles) > 0):
                        
                        triangle_count = len(triangulated_triangles)
                        
                        try:
                            quality = self._calculate_surface_quality(triangulated_vertices, triangulated_triangles)
                        except Exception as e:
                            quality = 0.0
                        
                        available_surfaces.append({
                            'index': i,
                            'name': surface_name,
                            'type': f'{surface_type} (Triangulated)',
                            'triangles': triangle_count,
                            'quality': quality,
                            'constraints': 0
                        })
                        continue
                
                # Check for hull data (last fallback)
                if 'hull_points' in dataset:
                    hull_points = dataset['hull_points']
                    if (hull_points is not None and hasattr(hull_points, '__len__') and len(hull_points) > 3):
                        available_surfaces.append({
                            'index': i,
                            'name': surface_name,
                            'type': f'{surface_type} (Hull)',
                            'triangles': 'Auto',
                            'quality': 0.5,  # Default quality for hull
                            'constraints': 0
                        })
                        continue
                        
            except Exception as e:
                logger.warning(f"Error processing dataset {i} for surface table: {e}")
                continue
        
        # Populate table
        self.surface_table.setRowCount(len(available_surfaces))
        
        from PyQt5.QtWidgets import QCheckBox, QWidget, QHBoxLayout, QComboBox
        from PyQt5.QtGui import QColor
        from PyQt5.QtCore import Qt
        
        for row, surface in enumerate(available_surfaces):
            try:
                # Checkbox for selection
                checkbox_widget = QWidget()
                checkbox_layout = QHBoxLayout(checkbox_widget)
                checkbox = QCheckBox()
                checkbox.setChecked(surface['index'] in self.selected_surfaces)
                checkbox.stateChanged.connect(lambda state, idx=surface['index']: self._on_surface_selection_changed(idx, state))
                checkbox_layout.addWidget(checkbox)
                checkbox_layout.setContentsMargins(0, 0, 0, 0)
                checkbox_layout.setAlignment(Qt.AlignCenter)
                self.surface_table.setCellWidget(row, 0, checkbox_widget)
                
                # Surface info with type coloring
                name_item = QTableWidgetItem(surface['name'])
                
                # Create combobox for surface type
                type_combobox = QComboBox()
                type_combobox.addItems(['BORDER', 'UNIT', 'FAULT', 'UNKNOWN'])
                
                # Extract base type from the surface type string (remove modifiers like "(Refined)")
                base_type = surface['type'].split(' ')[0]
                if base_type in ['BORDER', 'UNIT', 'FAULT', 'UNKNOWN']:
                    type_combobox.setCurrentText(base_type)
                else:
                    type_combobox.setCurrentText('UNKNOWN')
                
                # Set combobox style based on type
                self._style_type_combobox(type_combobox, base_type)
                
                # Connect change signal
                type_combobox.currentTextChanged.connect(
                    lambda new_type, idx=surface['index']: self._on_surface_type_changed(idx, new_type)
                )
                
                self.surface_table.setItem(row, 1, name_item)
                self.surface_table.setCellWidget(row, 2, type_combobox)
                self.surface_table.setItem(row, 3, QTableWidgetItem(str(surface['triangles'])))
                self.surface_table.setItem(row, 4, QTableWidgetItem(f"{surface['quality']:.3f}"))
                self.surface_table.setItem(row, 5, QTableWidgetItem(str(surface['constraints'])))
                
                # Color code quality
                quality_item = self.surface_table.item(row, 4)
                if surface['quality'] > 0.5:
                    quality_item.setBackground(QColor(200, 255, 200))  # Light green
                elif surface['quality'] > 0.3:
                    quality_item.setBackground(QColor(255, 255, 200))  # Light yellow
                else:
                    quality_item.setBackground(QColor(255, 200, 200))  # Light red
                    
            except Exception as e:
                logger.warning(f"Error creating table row {row}: {e}")
                continue
        
        self._update_selection_statistics()
        
        # Update the plotter visualization
        self._add_selectable_elements_to_plotter()

    def _style_type_combobox(self, combobox, surface_type):
        """Apply color styling to surface type combobox based on type."""
        from PyQt5.QtWidgets import QComboBox
        
        # Set background color based on surface type
        if surface_type == 'BORDER':
            combobox.setStyleSheet("QComboBox { background-color: rgb(173, 216, 230); }")  # Light blue
        elif surface_type == 'FAULT':
            combobox.setStyleSheet("QComboBox { background-color: rgb(255, 182, 193); }")  # Light pink
        elif surface_type == 'UNIT':
            combobox.setStyleSheet("QComboBox { background-color: rgb(144, 238, 144); }")  # Light green
        else:  # UNKNOWN
            combobox.setStyleSheet("QComboBox { background-color: rgb(211, 211, 211); }")  # Light gray

    def _on_surface_type_changed(self, surface_index, new_type):
        """Handle manual surface type changes from the dropdown."""
        try:
            # Update the surface type in the appropriate index list
            old_type = self._get_surface_type(surface_index)
            
            # Remove from old type lists
            if surface_index in self.border_surface_indices:
                self.border_surface_indices.remove(surface_index)
            if surface_index in self.unit_surface_indices:
                self.unit_surface_indices.remove(surface_index)
            if surface_index in self.fault_surface_indices:
                self.fault_surface_indices.remove(surface_index)
            
            # Add to new type list
            if new_type == 'BORDER':
                self.border_surface_indices.append(surface_index)
            elif new_type == 'UNIT':
                self.unit_surface_indices.append(surface_index)
            elif new_type == 'FAULT':
                self.fault_surface_indices.append(surface_index)
            # For UNKNOWN, don't add to any list
            
            # Update combobox styling
            sender = self.sender()
            if sender and isinstance(sender, QComboBox):
                self._style_type_combobox(sender, new_type)
            
            # Store the manual override in the dataset
            if surface_index < len(self.datasets):
                self.datasets[surface_index]['manual_surface_type'] = new_type
            
            logger.info(f"Surface {surface_index} type changed from {old_type} to {new_type}")
            
            # Update visualization if needed
            self._update_unified_visualization(filter_changed=True)
            
        except Exception as e:
            logger.error(f"Error changing surface type for surface {surface_index}: {e}")

    
    def _update_unified_visualization(self, filter_changed=False, display_mode_changed=False, 
                                     hulls_changed=False, intersections_changed=False):
        """Stub function since visualization controls have been removed - using only constraint selection viewer"""
        logger.debug("Unified visualization update bypassed - using only constraint selection viewer")
    
    # Visualization helper functions removed - using only constraint selection viewer
    def _validate_hull_points(self, hull_points):
        """Validate and convert hull points to proper numeric format."""
        try:
            if hull_points is None:
                return None
                
            # Convert to numpy array if needed
            if not isinstance(hull_points, np.ndarray):
                hull_points = np.array(hull_points)
                
            # Ensure it's numeric type
            if hull_points.dtype.kind not in 'fiu':  # float, int, unsigned int
                hull_points = hull_points.astype(np.float64)
                
            # Ensure it's 2D array with 3 columns (x, y, z)
            if len(hull_points.shape) == 1:
                hull_points = hull_points.reshape(-1, 3)
            elif hull_points.shape[1] != 3:
                logger.warning(f"Hull points have {hull_points.shape[1]} columns, expected 3")
                return None
                
            return hull_points
            
        except Exception as e:
            logger.error(f"Error validating hull points: {e}")
            return None
    def _add_filtered_convex_hulls(self, surface_indices):
        """Add convex hulls using the simple approach from pre-tetra mesh tab."""
        try:
            import pyvista as pv
            import numpy as np
            
            for i in surface_indices:
                dataset = self.datasets[i]
                dataset_name = dataset.get('name', f'Dataset_{i}')
                
                # Get hull points from dataset (like pre-tetra mesh tab does)
                hull_points = dataset.get('hull_points')
                if hull_points is None or len(hull_points) == 0:
                    logger.warning(f"No hull_points found for surface {i}")
                    continue
                
                # Convert hull points to numpy array, handling different formats safely
                try:
                    # First, let's see what we're dealing with
                    if len(hull_points) == 0:
                        continue
                    
                    # Handle [x,y,z,type] format where type might be string
                    if isinstance(hull_points[0], (list, tuple, np.ndarray)) and len(hull_points[0]) >= 3:
                        # Extract just the numeric x,y,z coordinates (ignore type at index 3)
                        constraint_points = []
                        for pt in hull_points:
                            try:
                                # Convert first 3 elements to float, ignore the rest
                                x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
                                constraint_points.append([x, y, z])
                            except (ValueError, TypeError, IndexError) as e:
                                logger.debug(f"Skipping invalid point {pt}: {e}")
                                continue
                        constraint_points = np.array(constraint_points, dtype=np.float64)
                    else:
                        # Try direct conversion for simple numeric arrays
                        constraint_points = np.array(hull_points, dtype=np.float64)
                except Exception as e:
                    logger.warning(f"Failed to convert hull_points for surface {i}: {e}")
                    continue
                
                if len(constraint_points) < 2:
                    continue
                
                # Close the loop for hull (exactly like pre-tetra mesh tab does)
                constraint_points_closed = np.vstack([constraint_points, constraint_points[0:1]])
                
                # Create line mesh exactly like pre-tetra mesh tab
                line_mesh = pv.PolyData(constraint_points_closed)
                line_mesh.lines = np.hstack([[2, i, i+1] for i in range(len(constraint_points_closed)-1)])
                
                # Add to plotter with same styling as pre-tetra mesh tab
                self.tetra_plotter.add_mesh(
                    line_mesh,
                    color='red',  # Hull boundary color from pre-tetra mesh tab
                    line_width=4,  # Hull boundary width from pre-tetra mesh tab
                    name=f'{dataset_name}_hull_boundary',
                    style='wireframe'
                )
                
                logger.info(f"Added simple convex hull for surface {i} with {len(constraint_points)} points")
            
        except Exception as e:
            logger.error(f"Error adding convex hulls: {e}")



    def _removed_boundary_hull_method(self, surface_index, coords):
        """UNUSED - Create a simple boundary wireframe when convex hull fails."""
        # This method is no longer used - the new simple convex hull approach works better
        return
        try:
            import pyvista as pv
            import numpy as np
            
            # Find boundary points by taking every Nth point to create a rough boundary
            if len(coords) > 20:
                # Sample points for large datasets
                step = max(1, len(coords) // 20)
                boundary_coords = coords[::step]
            else:
                boundary_coords = coords
            
            # Create lines connecting consecutive boundary points
            lines = []
            n_points = len(boundary_coords)
            for i in range(n_points):
                lines.extend([2, i, (i + 1) % n_points])
            
            # Create simple boundary mesh
            boundary_mesh = pv.PolyData(boundary_coords)
            if lines:
                boundary_mesh.lines = np.array(lines, dtype=np.int32)
            
            # Add to tetra plotter
            self.tetra_plotter.add_mesh(
                boundary_mesh,
                color='orange',  # Different color to indicate it's a simple boundary
                line_width=2,
                style='wireframe',
                name=f'convex_hull_{surface_index}',
                pickable=True
            )
            
            logger.info(f"Added simple boundary wireframe for surface {surface_index}")
            
        except Exception as e:
            logger.error(f"Even simple boundary failed for surface {surface_index}: {e}")

    def _parse_hull_points_manually(self, hull_points, surface_index):
        """Simplified hull points parsing that skips problematic data."""
        try:
            import numpy as np
            
            # Skip parsing complex object arrays for now to avoid errors
            if isinstance(hull_points, np.ndarray) and hull_points.dtype == object:
                logger.debug(f"Skipping object dtype hull data for surface {surface_index} to avoid parsing issues")
                return None
            
            # For simple numeric arrays, try basic conversion
            if isinstance(hull_points, (list, tuple, np.ndarray)):
                try:
                    points_array = np.array(hull_points, dtype=np.float64)
                    
                    # Ensure it's a 2D array
                    if points_array.ndim == 1:
                        if len(points_array) % 2 == 0:
                            points_array = points_array.reshape(-1, 2)
                        else:
                            return None
                    
                    # Add z=0 if only 2D coordinates
                    if points_array.shape[1] == 2:
                        z_column = np.zeros((points_array.shape[0], 1))
                        points_array = np.hstack([points_array, z_column])
                    
                    if points_array.shape[1] >= 3 and len(points_array) >= 3:
                        logger.info(f"Successfully parsed {len(points_array)} hull points for surface {surface_index}")
                        return points_array[:, :3]
                        
                except Exception as e:
                    logger.debug(f"Failed basic hull parsing for surface {surface_index}: {e}")
            
            return None
            
        except Exception as e:
            logger.debug(f"Error parsing hull points for surface {surface_index}: {e}")
            return None




                
    


        
    def _add_filtered_intersections(self, surface_indices):
        """Add intersections only for filtered surfaces."""
        try:
            # First try refined intersections from pre-tetramesh
            if hasattr(self, 'refined_intersections') and self.refined_intersections:
                for i, intersection in enumerate(self.refined_intersections):
                    # Handle both dictionary and object-like intersection formats
                    surface_index = intersection.get('surface_index') if isinstance(intersection, dict) else getattr(intersection, 'surface_index', i)
                    if surface_index in surface_indices:
                        self._add_intersection_to_plotter(intersection, surface_index, i)
            elif hasattr(self, 'datasets_intersections') and self.datasets_intersections:
                # Fall back to original intersections
                # datasets_intersections is a dict where keys are dataset indices and values are lists of intersections
                for dataset_index, intersections in self.datasets_intersections.items():
                    if dataset_index in surface_indices:
                        for i, intersection in enumerate(intersections):
                            self._add_intersection_to_plotter(intersection, dataset_index, i)
                            
                            # If this intersection involves another surface, show it for that surface too
                            other_surface = intersection.get('other_surface_index') if isinstance(intersection, dict) else getattr(intersection, 'other_surface_index', None)
                            if other_surface is not None and other_surface in surface_indices:
                                self._add_intersection_to_plotter(intersection, other_surface, i)
                                
            # Also check for intersection lines from pre-tetramesh
            for i in surface_indices:
                dataset = self.datasets[i]
                if 'intersection_lines' in dataset:
                    intersection_lines = dataset.get('intersection_lines', [])
                    for line_idx, line in enumerate(intersection_lines):
                        if line is not None and self._has_valid_array_data(line):
                            self._add_intersection_line_to_plotter(i, line_idx, line)
        except Exception as e:
            logger.error(f"Error adding filtered intersections: {e}")

                
    def _add_intersection_line_to_plotter(self, surface_index, line_idx, points):
        """Add an intersection line to the plotter with proper validation."""
        try:
            import pyvista as pv
            import numpy as np
            
            # Validate and convert points
            validated_points = self._validate_and_convert_points(points, f"intersection_line_{surface_index}_{line_idx}")
            if validated_points is None:
                logger.warning(f"Invalid intersection line points for surface {surface_index}, line {line_idx}")
                return
                
            # Create line cells
            n_points = len(validated_points)
            if n_points < 2:
                return
                
            # Create cells array for lines connecting consecutive points
            cells = []
            for i in range(n_points - 1):
                cells.extend([2, i, i + 1])
            
            # Create line mesh using explicit cell type
            cells_array = np.array(cells, dtype=np.int32)
            line = pv.UnstructuredGrid({pv.CellType.LINE: cells_array}, validated_points)
            
            # Add to plotter with unique name
            actor_name = f"intersection_line_{surface_index}_{line_idx}"
            self.tetra_plotter.add_mesh(line, color='green', line_width=2, name=actor_name)
            
        except Exception as e:
            logger.error(f"Error adding intersection line to plotter: {e}")

            
    def _add_intersection_to_plotter(self, intersection, surface_index, index):
        """Add a single intersection to the plotter with proper validation."""
        try:
            import pyvista as pv
            import numpy as np
            
            # Get intersection points based on data structure
            if isinstance(intersection, dict):
                # Try different possible keys for intersection points
                points = (
                    intersection.get('points', []) or 
                    intersection.get('intersection_points', []) or 
                    intersection.get('line_points', [])
                )
                triple_points = intersection.get('triple_points', [])
            else:
                # Handle object-like intersection
                points = (
                    getattr(intersection, 'points', None) or 
                    getattr(intersection, 'intersection_points', None) or 
                    getattr(intersection, 'line_points', None) or 
                    []
                )
                triple_points = getattr(intersection, 'triple_points', [])                # Validate and convert intersection points
            if points and self._has_valid_array_data(points):
                validated_points = self._validate_and_convert_points(points, f"intersection_{surface_index}_{index}")
                if validated_points is not None and len(validated_points) >= 2:
                    # Create line from points
                    n_points = len(validated_points)
                    
                    # Create cells array for lines connecting consecutive points
                    # PyVista expects a flattened array: [n_points, idx1, idx2, n_points, idx3, idx4, ...]
                    cells = []
                    for i in range(n_points - 1):
                        cells.extend([2, i, i + 1])  # 2 points per line segment
                    
                    # Create line mesh using PolyData with lines
                    cells_array = np.array(cells, dtype=np.int32)
                    line = pv.PolyData(validated_points)
                    line.lines = cells_array
                    
                    # Add to plotter with unique name
                    actor_name = f"intersection_{surface_index}_{index}"
                    self.tetra_plotter.add_mesh(line, color='green', line_width=3, name=actor_name)
                    logger.debug(f"Added intersection line for surface {surface_index}: {n_points} points")
            
            # Add triple points if they exist
            if triple_points and self._has_valid_array_data(triple_points):
                validated_triple_points = self._validate_and_convert_points(triple_points, f"triple_points_{surface_index}_{index}")
                if validated_triple_points is not None and len(validated_triple_points) > 0:
                    # Create point cloud for triple points
                    point_cloud = pv.PolyData(validated_triple_points)
                    
                    # Add to plotter
                    actor_name = f"triple_points_{surface_index}_{index}"
                    self.tetra_plotter.add_mesh(point_cloud, color='red', point_size=8, render_points_as_spheres=True, name=actor_name)
                    logger.debug(f"Added triple points for surface {surface_index}: {len(validated_triple_points)} points")
                    
        except Exception as e:
            logger.error(f"Error adding intersection to plotter: {e}")
        
    def _apply_filtered_highlighting(self, surface_indices):
        """Apply highlighting only to filtered and selected surfaces."""
        try:
            for surface_index in self.selected_surfaces:
                if surface_index in surface_indices:  # Only highlight if also visible due to filter
                    self._update_surface_selection_visual(surface_index, True)
        except Exception as e:
            logger.error(f"Error applying filtered highlighting: {e}")

    def _remove_convex_hulls_from_plotter(self):
        """Remove all convex hulls from the plotter."""
        if not self.tetra_plotter:
            return
            
        try:
            # Remove all convex hull actors (both old and new naming patterns)
            actors_to_remove = []
            for actor_name in self.tetra_plotter.renderer.actors:
                actor_str = str(actor_name)
                if ('convex_hull' in actor_str or 
                    'hull_boundary' in actor_str or 
                    'hull_wireframe' in actor_str or 
                    'hull_faces' in actor_str):
                    actors_to_remove.append(actor_name)
            
            for actor_name in actors_to_remove:
                try:
                    self.tetra_plotter.remove_actor(actor_name)
                except:
                    pass
            
            logger.info(f"Removed {len(actors_to_remove)} convex hull actors from plotter")
        except Exception as e:
            logger.error(f"Error removing convex hulls: {e}")



    def _remove_intersections_from_plotter(self):
        """Remove all intersections from the plotter."""
        if not self.tetra_plotter:
            return
            
        try:
            # Remove all intersection actors
            actors_to_remove = []
            for actor_name in self.tetra_plotter.renderer.actors:
                if 'intersection' in str(actor_name) or 'triple_points' in str(actor_name):
                    actors_to_remove.append(actor_name)
            
            for actor_name in actors_to_remove:
                try:
                    self.tetra_plotter.remove_actor(actor_name)
                except:
                    pass
            
            logger.debug("Removed intersections from plotter")
        except Exception as e:
            logger.error(f"Error removing intersections: {e}")

    def _remove_selection_highlighting(self):
        """Remove all selection highlighting from the plotter."""
        if not self.tetra_plotter:
            return
            
        try:
            # Remove all highlighting actors
            actors_to_remove = []
            for actor_name in self.tetra_plotter.renderer.actors:
                if 'highlight' in str(actor_name):
                    actors_to_remove.append(actor_name)
            
            for actor_name in actors_to_remove:
                try:
                    self.tetra_plotter.remove_actor(actor_name)
                except:
                    pass
            
            logger.debug("Removed selection highlighting from plotter")
        except Exception as e:
            logger.error(f"Error removing selection highlighting: {e}")
    
    def _add_hull_surface_to_plotter_with_mode(self, surface_index, hull_points, display_mode):
        """Add hull surface to plotter with specified display mode."""
        try:
            # Validate hull points first
            validated_points = self._validate_hull_points(hull_points)
            if validated_points is None:
                logger.warning(f"Invalid hull points for surface {surface_index}")
                return
                
            # Create convex hull
            from scipy.spatial import ConvexHull
            hull = ConvexHull(validated_points)
            
            # Get surface color
            surface_type = self._get_surface_type(surface_index)
            color = self._get_surface_color(surface_type)
            
            # Create boundary edges only (no internal triangulation for wireframe)
            boundary_edges = set()
            for simplex in hull.simplices:
                # Add each edge of the triangle to the boundary edge set
                edges = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]
                for edge in edges:
                    # Sort edge vertices to ensure consistent representation
                    sorted_edge = tuple(sorted(edge))
                    boundary_edges.add(sorted_edge)
            
            # Create hull faces for solid display mode
            hull_faces = []
            for simplex in hull.simplices:
                hull_faces.extend([3, simplex[0], simplex[1], simplex[2]])
            
            # Create line connectivity for wireframe display
            lines = []
            for edge in boundary_edges:
                lines.extend([2, edge[0], edge[1]])
            
            # Create meshes for different display modes
            hull_mesh_faces = pv.PolyData(validated_points, faces=hull_faces)
            hull_mesh_lines = pv.PolyData(validated_points, lines=lines)
            
            # Add to plotter based on display mode
            if display_mode == "faces":
                self.tetra_plotter.add_mesh(
                    hull_mesh_faces,
                    color=color,
                    opacity=0.7,
                    name=f'hull_faces_{surface_index}'
                )
            elif display_mode == "wireframe":
                self.tetra_plotter.add_mesh(
                    hull_mesh_lines,
                    color=color,
                    line_width=2,
                    name=f'hull_wireframe_{surface_index}'
                )
            elif display_mode == "both":
                # Add faces
                self.tetra_plotter.add_mesh(
                    hull_mesh_faces,
                    color=color,
                    opacity=0.5,
                    name=f'hull_faces_{surface_index}'
                )
                # Add wireframe (boundary edges only)
                self.tetra_plotter.add_mesh(
                    hull_mesh_lines,
                    color='black',
                    line_width=1,
                    name=f'hull_wireframe_{surface_index}'
                )
                
        except Exception as e:
            logger.error(f"Error adding hull surface {surface_index} with mode {display_mode}: {e}")
    def _update_surface_highlighting(self):
        """Update surface highlighting in the visualization."""
        try:
            if not hasattr(self, 'tetra_plotter') or not self.tetra_plotter:
                return
                
            # Remove existing highlighting
            self._remove_selection_highlighting()
            
            # Add highlighting for selected surfaces
            for surface_index in self.selected_surfaces:
                if surface_index < len(self.datasets):
                    self._highlight_surface_in_plotter(surface_index)
                    
        except Exception as e:
            logger.error(f"Error updating surface highlighting: {e}")

    def _highlight_surface_in_plotter(self, surface_index):
        """Highlight a specific surface in the plotter."""
        try:
            if not hasattr(self, 'tetra_plotter') or not self.tetra_plotter:
                return
                
            dataset = self.datasets[surface_index]
            vertices = dataset.get('constrained_vertices')
            triangles = dataset.get('constrained_triangles')
            
            if self._has_valid_array_data(vertices) and self._has_valid_array_data(triangles):
                # Create highlighting mesh
                faces = []
                for triangle in triangles:
                    if len(triangle) >= 3:
                        faces.extend([3, triangle[0], triangle[1], triangle[2]])
                
                if faces:
                    highlight_mesh = pv.PolyData(vertices, faces=faces)
                    self.tetra_plotter.add_mesh(
                        highlight_mesh,
                        style='wireframe',
                        color='yellow',
                        line_width=3,
                        name=f'highlight_{surface_index}',
                        opacity=0.8
                    )
                    
        except Exception as e:
            logger.error(f"Error highlighting surface {surface_index}: {e}")
    def _select_all_surfaces(self):
        """Select all available surfaces."""
        for i in range(self.surface_table.rowCount()):
            checkbox_widget = self.surface_table.cellWidget(i, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(True)
    
    def _deselect_all_surfaces(self):
        """Deselect all surfaces."""
        for i in range(self.surface_table.rowCount()):
            checkbox_widget = self.surface_table.cellWidget(i, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(False)
    
    def _add_default_material(self):
        """Add a default material region."""
        if not self.tetra_materials:
            # Calculate default location as center of all surfaces
            default_location = self._calculate_default_location()
            
            material = {
                'name': 'Material_1',
                'locations': [default_location],
                'attribute': 1
            }
            self.tetra_materials.append(material)
            self._update_material_list()
    
    def _calculate_default_location(self):
        """Calculate a default location based on surface centers."""
        if not self.datasets:
            return [0.0, 0.0, 0.0]
        
        # Use center of all constrained surfaces
        all_coords = []
        for dataset in self.datasets:
            if 'constrained_vertices' in dataset:
                vertices = dataset['constrained_vertices']
                if vertices is not None and len(vertices) > 0:
                    if isinstance(vertices, list):
                        vertices = np.array(vertices)
                    if len(vertices.shape) == 2 and vertices.shape[1] >= 3:
                        all_coords.extend(vertices[:, :3])
        
        if all_coords:
            center = np.mean(all_coords, axis=0)
            return [float(center[0]), float(center[1]), float(center[2])]
        
        return [0.0, 0.0, 0.0]
    
    def _add_material(self):
        """Add a new material region."""
        from PyQt5.QtWidgets import QInputDialog
        
        material_name, ok = QInputDialog.getText(
            self, 'Add Material', 'Enter material name:',
            text=f'Material_{len(self.tetra_materials) + 1}'
        )
        
        if ok and material_name:
            # Calculate a default location
            default_location = self._calculate_default_location()
            
            material = {
                'name': material_name,
                'locations': [default_location],
                'attribute': len(self.tetra_materials) + 1
            }
            
            self.tetra_materials.append(material)
            self._update_material_list()
            
            # Select the new material
            self.material_list.setCurrentRow(len(self.tetra_materials) - 1)

    
    def _remove_material(self):
        """Remove the selected material."""
        current_row = self.material_list.currentRow()
        if current_row >= 0 and current_row < len(self.tetra_materials):
            from PyQt5.QtWidgets import QMessageBox
            
            material_name = self.tetra_materials[current_row].get('name', f'Material_{current_row+1}')
            reply = QMessageBox.question(
                self, 'Remove Material',
                f'Are you sure you want to remove "{material_name}"?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                del self.tetra_materials[current_row]
                self._update_material_list()
                
                # Clear location list
                if hasattr(self, 'material_location_list'):
                    self.material_location_list.clear()
    def _update_material_list(self):
        """Update the material list widget."""
        if not hasattr(self, 'material_list'):
            return
            
        self.material_list.clear()
        for i, material in enumerate(self.tetra_materials):
            material_name = material.get('name', f'Material_{i+1}')
            location_count = len(material.get('locations', []))
            self.material_list.addItem(f"{material_name} ({location_count} locations)")

    
    def _on_material_selection_changed(self, current, previous=None):
        """Handle material selection change."""
        if current >= 0 and current < len(self.tetra_materials):
            # Update the location list for the selected material
            self._update_location_list(current)
        else:
            # Clear location list if no valid material selected
            if hasattr(self, 'material_location_list'):
                self.material_location_list.clear()
    
    def _update_location_list(self, material_index):
        """Update the location list for the specified material."""
        if not hasattr(self, 'material_location_list'):
            return
            
        self.material_location_list.clear()
        
        if material_index >= 0 and material_index < len(self.tetra_materials):
            material = self.tetra_materials[material_index]
            locations = material.get('locations', [])
            
            for i, location in enumerate(locations):
                location_text = f"Location {i+1}: ({location[0]:.2f}, {location[1]:.2f}, {location[2]:.2f})"
                self.material_location_list.addItem(location_text)

    
    def _add_material_location(self):
        """Add a new location to the selected material."""
        current_material = self.material_list.currentRow()
        if current_material >= 0 and current_material < len(self.tetra_materials):
            from PyQt5.QtWidgets import QInputDialog
            
            # Get location coordinates
            coords_text, ok = QInputDialog.getText(
                self, 'Add Location', 
                'Enter coordinates (x, y, z):',
                text='0.0, 0.0, 0.0'
            )
            
            if ok and coords_text:
                try:
                    # Parse coordinates
                    coords = [float(x.strip()) for x in coords_text.split(',')]
                    if len(coords) >= 3:
                        location = [coords[0], coords[1], coords[2]]
                        self.tetra_materials[current_material]['locations'].append(location)
                        self._update_location_list(current_material)
                    else:
                        from PyQt5.QtWidgets import QMessageBox
                        QMessageBox.warning(self, "Invalid Input", "Please enter three coordinates separated by commas.")
                except ValueError:
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric coordinates.")

    
    def _remove_material_location(self):
        """Remove the selected location from the current material."""
        current_material = self.material_list.currentRow()
        current_location = self.material_location_list.currentRow()
        
        if (current_material >= 0 and current_material < len(self.tetra_materials) and
            current_location >= 0):
            
            material = self.tetra_materials[current_material]
            locations = material.get('locations', [])
            
            if current_location < len(locations):
                del locations[current_location]
                self._update_location_list(current_material)
    def _auto_place_materials(self):
        """Automatically place materials based on surface geometry."""
        if not self.selected_surfaces:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Surfaces", "Please select surfaces first.")
            return
        
        # Clear existing materials except the first one
        self.tetra_materials = self.tetra_materials[:1] if self.tetra_materials else []
        
        # Calculate bounding box of selected surfaces
        all_vertices = []
        for surface_index in self.selected_surfaces:
            if surface_index < len(self.datasets):
                dataset = self.datasets[surface_index]
                vertices = dataset.get('constrained_vertices')
                if self._has_valid_array_data(vertices):
                    all_vertices.extend(vertices)
        
        if all_vertices:
            vertices_np = np.array(all_vertices)
            
            # Calculate center and bounds
            center = np.mean(vertices_np, axis=0)
            min_coords = np.min(vertices_np, axis=0)
            max_coords = np.max(vertices_np, axis=0)
            
            # Place materials at strategic locations
            locations = [
                center.tolist(),  # Center
                ((min_coords + center) / 2).tolist(),  # Quarter points
                ((max_coords + center) / 2).tolist(),
            ]
            
            # Update first material or create new one
            if self.tetra_materials:
                self.tetra_materials[0]['locations'] = locations
            else:
                material = {
                    'name': 'Auto_Material_1',
                    'locations': locations,
                    'attribute': 1
                }
                self.tetra_materials.append(material)
            
            self._update_material_list()
            
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "Auto-Placement", f"Placed {len(locations)} material locations automatically.")
        
    
    

    # Surface visibility functions removed - using only constraint selection viewer



    def _clear_current_selection(self):
        """Clear current selection."""
        self.selected_surfaces.clear()
        self._update_surface_table()
        self._update_selection_statistics()





    def _assign_materials_to_mesh(self, tetrahedral_grid):
        """Assign material IDs to tetrahedral mesh."""
        if not self.tetra_materials:
            return
            
        try:
            # Check if materials were already assigned by TetGen
            if hasattr(tetrahedral_grid, 'cell_data') and 'MaterialID' in tetrahedral_grid.cell_data:
                material_data = tetrahedral_grid.cell_data['MaterialID']
                if not np.all(material_data == 0):
                    logger.info("Materials already assigned by TetGen")
                    return
            
            # Manual material assignment
            logger.info("Assigning materials manually...")
            num_cells = tetrahedral_grid.n_cells
            material_ids = np.ones(num_cells, dtype=int)
            
            # Simple assignment based on material point locations
            for i, material in enumerate(self.tetra_materials):
                material_id = i + 1
                assigned_count = 0
                
                for location in material['locations']:
                    loc_point = np.array(location)
                    cell_id = tetrahedral_grid.find_containing_cell(loc_point)
                    
                    if cell_id >= 0:
                        material_ids[cell_id] = material_id
                        assigned_count += 1
                
                logger.info("Assigned material %d to %d cells", material_id, assigned_count)
            
            # Assign remaining cells to nearest material
            unassigned = material_ids == 1
            if np.any(unassigned):
                logger.info("Assigning %d unassigned cells to nearest materials", np.sum(unassigned))
                # Simple approach: assign all to material 1
                # Could be improved with nearest neighbor search
            
            tetrahedral_grid.cell_data['MaterialID'] = material_ids
            
        except Exception as e:
            logger.warning("Material assignment failed: %s", e)


    def _handle_mesh_error(self, error):
        """Handle mesh generation errors."""
        error_msg = f"Tetrahedral mesh generation failed: {str(error)}"
        logger.error(error_msg)
        QMessageBox.critical(self, "Mesh Generation Error", error_msg)

    
    def _update_tetra_stats(self):
        """Update tetrahedral mesh statistics using the utility module."""
        if not hasattr(self, 'tetra_mesh_generator') or not self.tetrahedral_mesh:
            return
        
        try:
            # Get statistics from the utility module
            stats = self.tetra_mesh_generator.get_mesh_statistics(self.tetrahedral_mesh)
            
            # Update UI labels
            if hasattr(self, 'tetra_vertices_label'):
                self.tetra_vertices_label.setText(f"Vertices: {stats.get('n_vertices', 0):,}")
            
            if hasattr(self, 'tetra_cells_label'):
                self.tetra_cells_label.setText(f"Tetrahedra: {stats.get('n_tetrahedra', 0):,}")
            
            if hasattr(self, 'tetra_volume_label'):
                volume = stats.get('volume', 0.0)
                self.tetra_volume_label.setText(f"Volume: {volume:.6f}")
            
            logger.info(f"Mesh statistics: {stats.get('n_vertices', 0)} vertices, {stats.get('n_tetrahedra', 0)} tetrahedra, volume: {stats.get('volume', 0.0):.6f}")
            
        except Exception as e:
            logger.error(f"Failed to update statistics: {str(e)}")
    
    def _visualize_tetrahedral_mesh(self):
        """Visualize the generated tetrahedral mesh."""
        if not hasattr(self, 'tetra_plotter') or not self.tetra_plotter:
            return
        
        plotter = self.tetra_plotter
        plotter.clear()
        plotter.set_background([0.2, 0.2, 0.25])
        
        if not self.tetrahedral_mesh:
            plotter.add_text("No tetrahedral mesh to display.", position='upper_edge', color='white')
            return
        
        # Show surfaces if enabled
        if self.show_surfaces_check.isChecked():
            self._add_surfaces_to_plotter(plotter)
        
        # Show material points if enabled
        if self.show_materials_check.isChecked():
            self._add_material_points_to_plotter(plotter)
        
        # Show wireframe if enabled
        if self.show_wireframe_check.isChecked():
            self._add_wireframe_to_plotter(plotter)
        
        plotter.add_axes()
        plotter.reset_camera()
        logger.info("Updated tetrahedral mesh visualization")
    
    def _add_surfaces_to_plotter(self, plotter):
        """Add surface meshes to the plotter."""
        if self.tetrahedral_mesh and 'pyvista_grid' in self.tetrahedral_mesh:
            # Show the surface of the tetrahedral mesh
            pyvista_grid = self.tetrahedral_mesh['pyvista_grid']
            surface_mesh = pyvista_grid.extract_surface()
            
            plotter.add_mesh(surface_mesh, color='lightblue', opacity=0.8, 
                           show_edges=True, edge_color='black', 
                           line_width=1, label='Tetrahedral Surface')
        else:
            # Show individual constrained surfaces
            for i, surface_index in enumerate(self.selected_surfaces):
                if surface_index < len(self.datasets):
                    dataset = self.datasets[surface_index]
                    vertices = dataset.get('constrained_vertices')
                    triangles = dataset.get('constrained_triangles')
                    
                    if vertices is not None and triangles is not None:
                        vertices = np.array(vertices)
                        triangles = np.array(triangles)
                        
                        # Create PyVista mesh
                        import pyvista as pv
                        cells = np.hstack([np.full((len(triangles), 1), 3), triangles])
                        surface_mesh = pv.PolyData(vertices, faces=cells)
                        
                        # Get color for this surface
                        color = dataset.get('color', self.DEFAULT_COLORS[surface_index % len(self.DEFAULT_COLORS)])
                        surface_name = dataset.get('name', f'Surface_{surface_index}')
                        
                        plotter.add_mesh(surface_mesh, color=color, opacity=0.8, 
                                       show_edges=True, edge_color='black', 
                                       line_width=1, label=surface_name)
    
    def _add_material_points_to_plotter(self, plotter):
        """Add material location points to the plotter."""
        import pyvista as pv
        
        # Define colors for different materials
        material_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'orange']
        
        for i, material in enumerate(self.tetra_materials):
            locations = np.array(material['locations'])
            if len(locations) > 0:
                color = material_colors[i % len(material_colors)]
                
                # Create point cloud
                points_mesh = pv.PolyData(locations)
                plotter.add_points(points_mesh, color=color, point_size=10, 
                                 render_points_as_spheres=True, 
                                 label=material['name'])
    
    def _add_wireframe_to_plotter(self, plotter):
        """Add wireframe representation to the plotter."""
        if self.tetrahedral_mesh and 'pyvista_grid' in self.tetrahedral_mesh:
            # Show wireframe of tetrahedral mesh
            pyvista_grid = self.tetrahedral_mesh['pyvista_grid']
            
            # Extract edges for wireframe
            edges = pyvista_grid.extract_all_edges()
            plotter.add_mesh(edges, color='red', line_width=2, 
                           style='wireframe', label='Tetrahedral Wireframe')
        else:
            # Show wireframe of surface meshes if no tetrahedral mesh
            for i, surface_index in enumerate(self.selected_surfaces):
                if surface_index < len(self.datasets):
                    dataset = self.datasets[surface_index]
                    vertices = dataset.get('constrained_vertices')
                    triangles = dataset.get('constrained_triangles')
                    
                    if vertices is not None and triangles is not None:
                        vertices = np.array(vertices)
                        triangles = np.array(triangles)
                        
                        # Create PyVista mesh
                        import pyvista as pv
                        cells = np.hstack([np.full((len(triangles), 1), 3), triangles])
                        surface_mesh = pv.PolyData(vertices, faces=cells)
                        
                        # Show wireframe
                        plotter.add_mesh(surface_mesh, style='wireframe', 
                                       color='darkred', line_width=1)
    
    def _update_tetra_visualization(self):
        """Update the tetrahedral mesh visualization based on current settings."""
        self._visualize_tetrahedral_mesh()
    
    def _clear_tetra_mesh_plot(self):
        """Clear the tetrahedral mesh plot."""
        if hasattr(self, 'tetra_plotter') and self.tetra_plotter:
            self.tetra_plotter.clear()
            self.tetra_plotter.add_text("Select surfaces and generate tetrahedral mesh to visualize.", 
                                      position='upper_edge', color='white')
        
        # Reset mesh data
        self.tetrahedral_mesh = None
        
        # Reset statistics
        if hasattr(self, 'tetra_stats_label'):
            self.tetra_stats_label.setText("No tetrahedral mesh generated yet.")
        
        # Reset surface visibility to default
        for dataset in self.datasets:
            dataset['visible'] = True
        
        # Update surface table
        if hasattr(self, '_update_surface_table'):
            self._update_surface_table()

    def _show_validation_results_dialog(self, validation_results):
        """Show validation results in a dialog."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QLabel
        from PyQt5.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Surface Validation Results")
        dialog.setModal(True)
        dialog.resize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Handle both dictionary and list formats
        if isinstance(validation_results, dict):
            # Dictionary format (from tetgen validation)
            if validation_results.get('ready_for_tetgen', False):
                status_text = f"✅ All {validation_results['surface_count']} surfaces are ready for tetgen!"
                status_color = "green"
            elif validation_results.get('overall_status') == 'PARTIAL':
                valid_count = validation_results['statistics']['valid_surfaces']
                total_count = validation_results['surface_count']
                status_text = f"⚠️ {valid_count}/{total_count} surfaces ready for tetgen"
                status_color = "orange"
            else:
                status_text = "❌ Surfaces not ready for tetgen"
                status_color = "red"
            
            status_label = QLabel(status_text)
            status_label.setStyleSheet(f"color: {status_color}; font-weight: bold; font-size: 14px; padding: 10px;")
            status_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(status_label)
            
            # Create text area for detailed results
            text_area = QTextEdit()
            text_area.setReadOnly(True)
            
            # Add detailed information
            details_text = "Validation Details:\n\n"
            
            if 'details' in validation_results:
                for detail in validation_results['details']:
                    details_text += f"{detail}\n"
            elif 'surface_reports' in validation_results:
                for surface_report in validation_results['surface_reports']:
                    surface_name = surface_report.get('surface_name', 'Unknown')
                    status = surface_report.get('status', 'Unknown')
                    details_text += f"Surface: {surface_name} - Status: {status}\n"
                    
                    if 'issues' in surface_report and surface_report['issues']:
                        for issue in surface_report['issues']:
                            details_text += f"  • {issue}\n"
                    details_text += "\n"
            
            text_area.setPlainText(details_text)
            layout.addWidget(text_area)
            
        else:
            # List format (from surface selection validation)
            status_text = "Surface Selection Validation"
            status_label = QLabel(status_text)
            status_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
            status_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(status_label)
            
            # Create text area for results
            text_area = QTextEdit()
            text_area.setReadOnly(True)
            
            results_text = "Validation Results:\n\n"
            for result in validation_results:
                results_text += f"{result}\n"
            
            text_area.setPlainText(results_text)
            layout.addWidget(text_area)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        dialog.exec_()

    def _cleanup_pyvista_plotters(self):
        """Comprehensive cleanup of all PyVista plotters and OpenGL contexts."""
        logger.info("Starting comprehensive PyVista plotters cleanup...")
        
        # List of all possible plotter attributes
        plotter_attributes = [
            'tetra_plotter',
            'intersection_plotter', 
            'refine_mesh_plotter',
            'pre_tetramesh_plotter',
            'current_plotter',
            'pv_plotter'
        ]
        
        # Dictionary of plotters from self.plotters if it exists
        plotters_dict = getattr(self, 'plotters', {})
        
        # Clean up individual plotter attributes
        for attr_name in plotter_attributes:
            if hasattr(self, attr_name):
                plotter = getattr(self, attr_name)
                if plotter is not None:
                    try:
                        logger.debug(f"Cleaning up {attr_name}...")
                        
                        # Clear the plotter content first
                        if hasattr(plotter, 'clear'):
                            plotter.clear()
                        
                        # Close the plotter properly
                        if hasattr(plotter, 'close'):
                            plotter.close()
                        
                        # For QtInteractor objects, also clean up the interactor
                        if hasattr(plotter, 'interactor'):
                            try:
                                if hasattr(plotter.interactor, 'close'):
                                    plotter.interactor.close()
                                if hasattr(plotter.interactor, 'finalize'):
                                    plotter.interactor.finalize()
                            except Exception as e:
                                logger.warning(f"Error cleaning up {attr_name} interactor: {e}")
                        
                        # Clean up render window if available
                        if hasattr(plotter, 'render_window'):
                            try:
                                if hasattr(plotter.render_window, 'finalize'):
                                    plotter.render_window.finalize()
                            except Exception as e:
                                logger.warning(f"Error finalizing {attr_name} render window: {e}")
                        
                        # Set attribute to None
                        setattr(self, attr_name, None)
                        logger.debug(f"Successfully cleaned up {attr_name}")
                        
                    except Exception as e:
                        logger.warning(f"Error cleaning up {attr_name}: {e}")
                        # Still set to None even if cleanup failed
                        setattr(self, attr_name, None)
        
        # Clean up plotters from self.plotters dictionary
        for plotter_name, plotter in plotters_dict.items():
            if plotter is not None:
                try:
                    logger.debug(f"Cleaning up plotter from dictionary: {plotter_name}...")
                    
                    # Clear and close the plotter
                    if hasattr(plotter, 'clear'):
                        plotter.clear()
                    if hasattr(plotter, 'close'):
                        plotter.close()
                    
                    # Clean up interactor
                    if hasattr(plotter, 'interactor'):
                        try:
                            if hasattr(plotter.interactor, 'close'):
                                plotter.interactor.close()
                            if hasattr(plotter.interactor, 'finalize'):
                                plotter.interactor.finalize()
                        except Exception as e:
                            logger.warning(f"Error cleaning up {plotter_name} interactor: {e}")
                    
                    # Clean up render window
                    if hasattr(plotter, 'render_window'):
                        try:
                            if hasattr(plotter.render_window, 'finalize'):
                                plotter.render_window.finalize()
                        except Exception as e:
                            logger.warning(f"Error finalizing {plotter_name} render window: {e}")
                    
                    logger.debug(f"Successfully cleaned up plotter: {plotter_name}")
                    
                except Exception as e:
                    logger.warning(f"Error cleaning up plotter {plotter_name}: {e}")
        
        # Clear the plotters dictionary
        if hasattr(self, 'plotters'):
            self.plotters.clear()
        
        # Force garbage collection to help clean up VTK objects
        try:
            import gc
            gc.collect()
            logger.debug("Forced garbage collection after plotter cleanup")
        except Exception as e:
            logger.warning(f"Error during garbage collection: {e}")
        
        # Try to finalize VTK if available
        try:
            import vtk
            if hasattr(vtk, 'vtkObject'):
                vtk.vtkObject.GlobalWarningDisplayOff()
            logger.debug("Disabled VTK warnings")
        except Exception as e:
            logger.debug(f"VTK cleanup not available: {e}")
        
        logger.info("Completed PyVista plotters cleanup")
    
    def _ensure_vtk_cleanup_on_exit(self):
        """Ensure VTK cleanup is performed when Python exits."""
        import atexit
        
        def vtk_cleanup():
            try:
                self._cleanup_pyvista_plotters()
                # Force VTK cleanup
                import vtk
                if hasattr(vtk, 'vtkRenderWindow'):
                    # Clean up any remaining render windows
                    try:
                        vtk.vtkRenderWindow.Finalize()
                    except:
                        pass
            except Exception:
                pass  # Silent cleanup on exit
        
        atexit.register(vtk_cleanup)
    
    def __del__(self):
        """Destructor to ensure cleanup even if closeEvent is not called."""
        try:
            self._cleanup_pyvista_plotters()
        except Exception as e:
            # Use print since logger might not be available during destruction
            print(f"Warning: Error during destructor cleanup: {e}")
    
    def _safe_create_pyvista_plotter(self, parent_widget, background_color='white'):
        """Safely create a PyVista plotter with proper error handling."""
        if not HAVE_PYVISTA:
            return None
            
        try:
            from pyvistaqt import QtInteractor
            plotter = QtInteractor(parent_widget)
            
            # Set background with error handling
            try:
                if isinstance(background_color, str):
                    plotter.set_background(background_color)
                elif isinstance(background_color, (list, tuple)) and len(background_color) == 3:
                    plotter.set_background(background_color)
                else:
                    plotter.set_background('white')  # fallback
            except Exception as e:
                logger.warning(f"Error setting plotter background: {e}")
                
            # Disable VTK warnings for cleaner output
            try:
                import vtk
                if hasattr(vtk, 'vtkObject'):
                    vtk.vtkObject.GlobalWarningDisplayOff()
            except Exception:
                pass  # VTK warnings control is optional
                
            return plotter
            
        except Exception as e:
            logger.error(f"Error creating PyVista plotter: {e}")
            return None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MeshItWorkflowGUI()
    window._ensure_vtk_cleanup_on_exit()  # Register VTK cleanup on exit
    window.show()
    sys.exit(app.exec_())