"""
Headless PyMeshIt workflow helpers.

This module provides a small programmatic API for simple batch meshing cases.
It mirrors the GUI workflow without constructing Qt widgets:

1. build surface hulls
2. create initial surface triangulations
3. compute and refine intersections
4. generate conforming surface meshes from selected constraints
5. pass the conforming PLC to TetGen

The first supported constraint-selection mode mirrors the GUI workflow.
``constraint_mode="intersections"`` behaves like the GUI "Select intersections"
button in the Refine & Mesh tab: it keeps each surface hull as the outer PLC
boundary and adds selected intersection constraints inside that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from Pymeshit.intersection_utils import (
    Intersection,
    TriplePoint,
    Vector3D,
    align_intersections_to_convex_hull,
    calculate_triple_points,
    insert_triple_points,
    make_corners_special,
    refine_hull_with_interpolation,
    refine_intersection_line_by_length,
    refine_well_by_length,
    run_constrained_triangulation_py,
    run_intersection_workflow,
)


logger = logging.getLogger("PyMeshIt-Headless")

ConstraintMode = Literal["all", "intersections", "hulls"]
SurfaceRole = Literal["border", "unit", "fault"]


@dataclass
class SurfaceSpec:
    """A surface input for headless meshing."""

    name: str
    points: Union[str, os.PathLike, Sequence[Sequence[float]], np.ndarray]
    role: SurfaceRole = "unit"
    target_size: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WellSpec:
    """A 1D well/polyline input used as a TetGen edge constraint."""

    name: str
    points: Union[str, os.PathLike, Sequence[Sequence[float]], np.ndarray]
    target_size: Optional[float] = None
    marker: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterialSpec:
    """A TetGen region material seed."""

    name: str
    locations: Union[Sequence[float], Sequence[Sequence[float]]]
    attribute: int = 0
    type: Literal["FORMATION", "FAULT"] = "FORMATION"
    max_volume: float = 0.0

    def to_tetra_dict(self) -> Dict[str, Any]:
        locs = _normalize_locations(self.locations)
        return {
            "name": self.name,
            "locations": locs,
            "attribute": int(self.attribute),
            "type": str(self.type).upper(),
            "max_volume": float(self.max_volume),
        }


@dataclass
class MeshOptions:
    """Headless workflow options."""

    target_size: float = 1.0
    min_angle: float = 20.0
    gradient: float = 2.0
    uniform: bool = True
    interpolation: str = "Thin Plate Spline (TPS)"
    smoothing: float = 0.0
    hull_method: Literal["delaunay", "alpha"] = "delaunay"
    hull_alpha_factor: float = 1.0
    constraint_mode: ConstraintMode = "intersections"
    preserve_boundary_hulls: bool = True
    tetgen_switches: str = "pq1.414aAY"
    generate_volume: bool = True
    include_hull_fallback_faults_in_volume: bool = False
    intersection_tolerance: float = 1e-5
    keep_intermediate: bool = True


@dataclass
class MeshCase:
    """Complete input for one headless mesh generation run."""

    surfaces: Sequence[SurfaceSpec]
    materials: Sequence[MaterialSpec] = field(default_factory=list)
    wells: Sequence[WellSpec] = field(default_factory=list)
    options: MeshOptions = field(default_factory=MeshOptions)


@dataclass
class MeshResult:
    """Outputs from a headless run."""

    datasets: List[Dict[str, Any]]
    conforming_surface_data: Dict[int, Dict[str, Any]]
    tetra_mesh: Any = None
    intersections: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    triple_points: List[Dict[str, Any]] = field(default_factory=list)
    tetra_generator: Optional[Any] = None
    failures: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.tetra_mesh is not None if self.tetra_generator is not None else not self.failures

    def export_mesh(
        self,
        file_path: Union[str, os.PathLike],
        *,
        custom_block_names: Optional[Dict[int, str]] = None,
        custom_sideset_names: Optional[Dict[int, str]] = None,
        custom_well_names: Optional[Dict[int, str]] = None,
        well_export_type: str = "Node Sets",
    ) -> bool:
        """
        Export the generated tetrahedral mesh using the same exporter as the GUI.

        EXODUS/NetCDF paths (``.exo``, ``.e``, ``.nc``, ``.nc4``, ``.cdf``) are
        written with material element blocks, boundary/fault sidesets, and well
        node sets when that metadata is available from TetGen.
        """
        if self.tetra_mesh is None:
            raise ValueError("No tetrahedral mesh is available to export")

        return _export_tetra_mesh(
            file_path,
            self.tetra_mesh,
            tetra_generator=self.tetra_generator,
            custom_block_names=custom_block_names,
            custom_sideset_names=custom_sideset_names,
            custom_well_names=custom_well_names,
            well_export_type=well_export_type,
        )


def run_mesh_case(
    mesh_case: MeshCase,
    *,
    output_path: Optional[Union[str, os.PathLike]] = None,
    custom_block_names: Optional[Dict[int, str]] = None,
    custom_sideset_names: Optional[Dict[int, str]] = None,
    custom_well_names: Optional[Dict[int, str]] = None,
    well_export_type: str = "Node Sets",
) -> MeshResult:
    """
    Run a simple headless PyMeshIt workflow.

    Parameters
    ----------
    mesh_case:
        Surfaces, wells, materials, and meshing options.
    output_path:
        Optional path where the generated tetrahedral mesh should be saved. The
        GUI exporter is used when available, so EXODUS/NetCDF outputs include
        material blocks, sidesets, and well node sets.

    Returns
    -------
    MeshResult
        Contains the intermediate datasets, conforming surface meshes, and the
        final PyVista/TetGen volume mesh when ``generate_volume`` is enabled.
    """
    if not mesh_case.surfaces:
        raise ValueError("MeshCase requires at least one surface")

    options = mesh_case.options
    _validate_options(options)

    datasets = _build_datasets(mesh_case.surfaces, mesh_case.wells)
    surface_count = len(mesh_case.surfaces)

    logger.info("Headless meshing started: %d surfaces, %d wells", surface_count, len(mesh_case.wells))

    for s_idx in range(surface_count):
        _compute_hull_for_dataset(datasets[s_idx], options)
        _compute_segments_for_dataset(datasets[s_idx], _target_size_for_surface(datasets[s_idx], options), options)
        _run_initial_triangulation_for_dataset(
            datasets[s_idx],
            _target_size_for_surface(datasets[s_idx], options),
            options,
        )

    intersections, triple_points = _compute_global_intersections(datasets, surface_count, options)
    refined_intersections = {}
    if intersections:
        intersections, refined_intersections, triple_points = _refine_intersections(
            datasets,
            intersections,
            surface_count,
            options,
        )
        _refine_wells(datasets, surface_count, options)
    else:
        logger.info("No intersections found. Conforming meshes will use selected hull constraints only where enabled.")

    conforming_surface_data, failures = _generate_conforming_surface_meshes(
        datasets,
        mesh_case.surfaces,
        refined_intersections,
        options,
    )

    tetra_mesh = None
    tetra_generator = None
    if options.generate_volume:
        if not conforming_surface_data:
            raise RuntimeError("No conforming surface meshes were generated; cannot run TetGen")

        from Pymeshit.tetra_mesh_utils import TetrahedralMeshGenerator

        selected_surfaces = set(conforming_surface_data.keys())
        border_indices = _indices_by_role(mesh_case.surfaces, "border") & selected_surfaces
        fault_indices = _indices_by_role(mesh_case.surfaces, "fault") & selected_surfaces
        materials = [m.to_tetra_dict() for m in mesh_case.materials]
        well_data = _build_well_data(datasets, surface_count, mesh_case.wells)
        hull_fallback_faults = {
            idx
            for idx in fault_indices
            if conforming_surface_data.get(idx, {}).get("constraint_source") == "hull_fallback"
        }

        def build_tetra_generator(active_surfaces: set):
            active_border = border_indices & active_surfaces
            active_faults = fault_indices & active_surfaces
            active_units = active_surfaces - active_border - active_faults
            return TetrahedralMeshGenerator(
                datasets=datasets,
                selected_surfaces=active_surfaces,
                border_surface_indices=active_border,
                unit_surface_indices=active_units,
                fault_surface_indices=active_faults,
                materials=materials,
                surface_data={idx: conforming_surface_data[idx] for idx in active_surfaces},
                holes=[],
                well_data=well_data,
            )

        volume_surfaces = set(selected_surfaces)
        if hull_fallback_faults and not options.include_hull_fallback_faults_in_volume:
            volume_surfaces -= hull_fallback_faults
            logger.warning(
                "Fault surfaces %s used hull fallback, so they are kept as surface outputs "
                "but excluded from TetGen volume constraints. Set "
                "include_hull_fallback_faults_in_volume=True to force insertion.",
                sorted(hull_fallback_faults),
            )

        tetra_generator = build_tetra_generator(volume_surfaces)
        tetra_mesh = tetra_generator.generate_tetrahedral_mesh(options.tetgen_switches)

        if tetra_mesh is None:
            if hull_fallback_faults and options.include_hull_fallback_faults_in_volume:
                retry_surfaces = volume_surfaces - hull_fallback_faults
                logger.warning(
                    "TetGen rejected the PLC with hull-fallback fault surfaces %s. "
                    "Retrying volume mesh without those fault surfaces.",
                    sorted(hull_fallback_faults),
                )
                retry_generator = build_tetra_generator(retry_surfaces)
                retry_mesh = retry_generator.generate_tetrahedral_mesh(options.tetgen_switches)
                if retry_mesh is not None:
                    tetra_generator = retry_generator
                    tetra_mesh = retry_mesh

        if tetra_mesh is None:
            failures.append(("tetgen", "TetGen did not produce a mesh"))
        elif output_path is not None:
            saved = _export_tetra_mesh(
                output_path,
                tetra_mesh,
                tetra_generator=tetra_generator,
                custom_block_names=custom_block_names,
                custom_sideset_names=custom_sideset_names,
                custom_well_names=custom_well_names,
                well_export_type=well_export_type,
            )
            if saved:
                logger.info("Saved headless tetrahedral mesh to %s", output_path)
            else:
                failures.append(("export", f"Failed to export mesh to {output_path}"))

    return MeshResult(
        datasets=datasets if options.keep_intermediate else [],
        conforming_surface_data=conforming_surface_data if options.keep_intermediate else {},
        tetra_mesh=tetra_mesh,
        intersections=intersections,
        triple_points=triple_points,
        tetra_generator=tetra_generator,
        failures=failures,
    )


def generate_tetrahedral_mesh_from_surfaces(
    surface_meshes: Dict[int, Dict[str, Any]],
    *,
    datasets: Optional[List[Dict[str, Any]]] = None,
    selected_surfaces: Optional[Iterable[int]] = None,
    border_surface_indices: Optional[Iterable[int]] = None,
    unit_surface_indices: Optional[Iterable[int]] = None,
    fault_surface_indices: Optional[Iterable[int]] = None,
    materials: Optional[Sequence[Union[MaterialSpec, Dict[str, Any]]]] = None,
    tetgen_switches: str = "pq1.414aAY",
):
    """
    Run only the final TetGen stage from already conforming surface meshes.

    This is useful when geometry is generated externally and users only need
    PyMeshIt's PLC assembly and TetGen/material handling.
    """
    if not surface_meshes:
        raise ValueError("surface_meshes cannot be empty")

    from Pymeshit.tetra_mesh_utils import TetrahedralMeshGenerator

    selected = set(selected_surfaces or surface_meshes.keys())
    max_idx = max(selected | set(surface_meshes.keys()))
    if datasets is None:
        datasets = [{"name": f"Surface_{idx}", "type": "SURFACE"} for idx in range(max_idx + 1)]

    mats = []
    for material in materials or []:
        if isinstance(material, MaterialSpec):
            mats.append(material.to_tetra_dict())
        else:
            mats.append(dict(material))

    generator = TetrahedralMeshGenerator(
        datasets=datasets,
        selected_surfaces=selected,
        border_surface_indices=set(border_surface_indices or selected),
        unit_surface_indices=set(unit_surface_indices or set()),
        fault_surface_indices=set(fault_surface_indices or set()),
        materials=mats,
        surface_data=surface_meshes,
    )
    return generator.generate_tetrahedral_mesh(tetgen_switches)


def _export_tetra_mesh(
    file_path: Union[str, os.PathLike],
    tetra_mesh: Any,
    *,
    tetra_generator: Optional[Any] = None,
    custom_block_names: Optional[Dict[int, str]] = None,
    custom_sideset_names: Optional[Dict[int, str]] = None,
    custom_well_names: Optional[Dict[int, str]] = None,
    well_export_type: str = "Node Sets",
) -> bool:
    output = Path(file_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if tetra_generator is not None and hasattr(tetra_generator, "export_mesh"):
        return bool(
            tetra_generator.export_mesh(
                str(output),
                tetra_mesh,
                custom_block_names=custom_block_names,
                custom_sideset_names=custom_sideset_names,
                custom_well_names=custom_well_names,
                well_export_type=well_export_type,
            )
        )

    exodus_like = output.suffix.lower() in {".exo", ".e", ".nc", ".nc4", ".cdf"}
    if exodus_like:
        logger.error("EXODUS/NetCDF export requires the TetGen generator metadata")
        return False

    tetra_mesh.save(str(output))
    return True


def read_points(path: Union[str, os.PathLike]) -> np.ndarray:
    """Read a text/CSV/VTU point file into an ``(N, 3)`` float array."""
    file_path = Path(path)
    ext = file_path.suffix.lower()
    if ext in {".vtu", ".vtk", ".vtp"}:
        import pyvista as pv

        mesh = pv.read(str(file_path))
        if mesh.points is None:
            raise ValueError(f"No points found in {file_path}")
        return _as_xyz_array(mesh.points)

    points: List[List[float]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("Bounds:"):
                points.extend(_parse_bounds_line(line))
                continue
            cleaned = line.replace("\t", " ").replace(",", " ").replace(";", " ")
            cleaned = re.sub(r"(?<![eE\d.-])\+", "", cleaned)
            cleaned = re.sub(r"(?<![eE\d.-])-", " -", cleaned)
            parts = " ".join(cleaned.split()).split()
            if len(parts) < 2:
                logger.debug("Skipping line %d in %s: insufficient columns", line_num, file_path)
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2]) if len(parts) >= 3 else 0.0
            except ValueError:
                logger.debug("Skipping line %d in %s: non-numeric data", line_num, file_path)
                continue
            points.append([x, y, z])

    if points:
        return _as_xyz_array(points)

    loaded = np.loadtxt(str(file_path), comments="#", delimiter="," if ext == ".csv" else None)
    return _as_xyz_array(loaded)


def _validate_options(options: MeshOptions) -> None:
    if options.target_size <= 0:
        raise ValueError("MeshOptions.target_size must be positive")
    if options.constraint_mode not in {"all", "intersections", "hulls"}:
        raise ValueError("constraint_mode must be one of: all, intersections, hulls")


def _build_datasets(surfaces: Sequence[SurfaceSpec], wells: Sequence[WellSpec]) -> List[Dict[str, Any]]:
    datasets: List[Dict[str, Any]] = []
    for idx, spec in enumerate(surfaces):
        role = str(spec.role).lower()
        if role not in {"border", "unit", "fault"}:
            raise ValueError(f"Invalid role for surface {spec.name!r}: {spec.role!r}")
        ds = {
            "name": spec.name,
            "type": "SURFACE",
            "role": role,
            "points": _load_points_like(spec.points),
            "target_size": spec.target_size,
            "metadata": dict(spec.metadata),
            "surface_index": idx,
        }
        datasets.append(ds)

    for local_idx, spec in enumerate(wells):
        ds = {
            "name": spec.name,
            "type": "WELL",
            "points": _load_points_like(spec.points),
            "target_size": spec.target_size,
            "marker": spec.marker,
            "metadata": dict(spec.metadata),
            "well_index": local_idx,
        }
        datasets.append(ds)
    return datasets


def _load_points_like(points: Union[str, os.PathLike, Sequence[Sequence[float]], np.ndarray]) -> np.ndarray:
    if isinstance(points, (str, os.PathLike)):
        return read_points(points)
    return _as_xyz_array(points)


def _as_xyz_array(points: Union[Sequence[Sequence[float]], np.ndarray]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("points must be an (N, 2) or (N, 3+) array")
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.zeros(len(arr), dtype=float)])
    return np.asarray(arr[:, :3], dtype=float)


def _normalize_locations(locations: Union[Sequence[float], Sequence[Sequence[float]]]) -> List[List[float]]:
    arr = np.asarray(locations, dtype=float)
    if arr.ndim == 1:
        if arr.size < 3:
            raise ValueError("material location must have at least 3 coordinates")
        return [arr[:3].astype(float).tolist()]
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("material locations must be (3,) or (N, 3+)")
    return arr[:, :3].astype(float).tolist()


def _parse_bounds_line(line: str) -> List[List[float]]:
    x_range = re.search(r"X\[([-+]?\d+\.?\d*),([-+]?\d+\.?\d*)\]", line)
    y_range = re.search(r"Y\[([-+]?\d+\.?\d*),([-+]?\d+\.?\d*)\]", line)
    z_range = re.search(r"Z\[([-+]?\d+\.?\d*),([-+]?\d+\.?\d*)\]", line)
    if not x_range or not y_range:
        return []
    x_min, x_max = float(x_range.group(1)), float(x_range.group(2))
    y_min, y_max = float(y_range.group(1)), float(y_range.group(2))
    z_min, z_max = (float(z_range.group(1)), float(z_range.group(2))) if z_range else (0.0, 0.0)
    return [
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ]


def _target_size_for_surface(dataset: Dict[str, Any], options: MeshOptions) -> float:
    value = dataset.get("target_size")
    if value is None:
        return float(options.target_size)
    value = float(value)
    return value if value > 1e-9 else float(options.target_size)


def _target_size_for_well(dataset: Dict[str, Any], options: MeshOptions) -> float:
    value = dataset.get("target_size")
    if value is not None and float(value) > 1e-9:
        return float(value)
    points = dataset.get("points")
    if points is None or len(points) < 2:
        return float(options.target_size)
    diagonal = float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))
    return max(float(options.target_size), diagonal / 16.0) if diagonal > 0 else float(options.target_size)


def _compute_hull_for_dataset(dataset: Dict[str, Any], options: MeshOptions) -> None:
    from scipy.spatial import ConvexHull, Delaunay

    pts = _as_xyz_array(dataset["points"])
    if len(pts) < 3:
        raise ValueError(f"Surface {dataset['name']!r} has fewer than 3 points")

    hull_pts_np = None
    if np.allclose(pts[:, 2], pts[0, 2]):
        input_2d = pts[:, :2]
        hull = ConvexHull(input_2d)
        hull_pts_np = pts[hull.vertices]
    else:
        _, projected = _pca_project(pts)
        if options.hull_method == "alpha":
            ordered = _compute_alpha_shape_boundary_robust(projected, options.hull_alpha_factor)
            if ordered is not None and len(ordered) >= 3:
                hull_pts_np = pts[ordered]

        if hull_pts_np is None:
            tri = Delaunay(projected)
            edges = set()
            for simplex in tri.simplices:
                for j in range(3):
                    edge = tuple(sorted((simplex[j], simplex[(j + 1) % 3])))
                    if edge in edges:
                        edges.remove(edge)
                    else:
                        edges.add(edge)
            if edges:
                ordered = _order_boundary_edges(edges)
                if len(ordered) >= 3:
                    hull_pts_np = pts[ordered]
            if hull_pts_np is None:
                hull = ConvexHull(projected)
                hull_pts_np = pts[hull.vertices]

    if len(hull_pts_np) > 0 and not np.allclose(hull_pts_np[0], hull_pts_np[-1]):
        hull_pts_np = np.vstack([hull_pts_np, hull_pts_np[0:1]])

    hull_vec = [Vector3D(p[0], p[1], p[2], point_type="DEFAULT") for p in hull_pts_np]
    hull_vec = make_corners_special(hull_vec)
    dataset["hull_points"] = np.array(
        [[p.x, p.y, p.z, getattr(p, "point_type", "DEFAULT")] for p in hull_vec],
        dtype=object,
    )
    dataset.pop("segments", None)
    dataset.pop("triangulation_result", None)
    logger.info("Computed hull for %s with %d points", dataset["name"], len(hull_vec))


def _pca_project(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2]
    projected = centered @ basis.T
    return centroid, projected


def _surface_projection_params(points: np.ndarray) -> Dict[str, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2]
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    return {"centroid": centroid, "basis": basis, "normal": normal}


def _order_boundary_edges(edges: Iterable[Tuple[int, int]]) -> List[int]:
    remaining = set(edges)
    if not remaining:
        return []
    first = remaining.pop()
    ordered = [first[0], first[1]]
    while remaining:
        last = ordered[-1]
        found = None
        for edge in list(remaining):
            if edge[0] == last:
                found = edge
                ordered.append(edge[1])
                break
            if edge[1] == last:
                found = edge
                ordered.append(edge[0])
                break
        if found is None:
            break
        remaining.remove(found)
    return ordered


def _compute_alpha_shape_boundary_robust(points_2d: np.ndarray, alpha_factor: float) -> Optional[np.ndarray]:
    from scipy.spatial import cKDTree

    if len(points_2d) < 4:
        return None
    tree = cKDTree(points_2d)
    k = min(8, len(points_2d))
    dists, _ = tree.query(points_2d, k=k)
    avg_nn = float(np.median(dists[:, 1:].mean(axis=1))) if dists.ndim > 1 and dists.shape[1] > 1 else float(np.median(dists))
    best = None
    best_score = -1
    for mult in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        radius = max(avg_nn * mult * alpha_factor, 1e-12)
        boundary = _extract_alpha_shape_boundary(points_2d, 1.0 / radius)
        if boundary is not None and len(boundary) >= 3:
            score = len(boundary)
            if score > best_score:
                best = boundary
                best_score = score
    return best


def _extract_alpha_shape_boundary(points_2d: np.ndarray, alpha: float) -> Optional[np.ndarray]:
    from collections import Counter, defaultdict
    from scipy.spatial import Delaunay

    tri = Delaunay(points_2d)
    threshold = 1.0 / alpha if alpha > 1e-12 else float("inf")
    kept = []
    for simplex in tri.simplices:
        pts = points_2d[simplex]
        a = np.linalg.norm(pts[1] - pts[0])
        b = np.linalg.norm(pts[2] - pts[1])
        c = np.linalg.norm(pts[0] - pts[2])
        s = (a + b + c) / 2.0
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 1e-20:
            continue
        radius = (a * b * c) / (4.0 * math.sqrt(area_sq))
        if radius < threshold:
            kept.append(tuple(simplex))
    if not kept:
        return None
    counts = Counter()
    for ia, ib, ic in kept:
        for edge in ((ia, ib), (ib, ic), (ic, ia)):
            counts[tuple(sorted(edge))] += 1
    boundary_edges = [edge for edge, count in counts.items() if count == 1]
    if len(boundary_edges) < 3:
        return None
    adjacency = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    start = min(adjacency.keys(), key=lambda idx: (points_2d[idx, 0], points_2d[idx, 1]))
    ring = [start]
    previous = None
    current = start
    for _ in range(len(boundary_edges) + 2):
        neighbors = adjacency[current]
        candidates = [n for n in neighbors if n != previous]
        if not candidates:
            break
        nxt = candidates[0]
        if nxt == start:
            break
        ring.append(nxt)
        previous, current = current, nxt
    return np.asarray(ring, dtype=int) if len(ring) >= 3 else None


def _compute_segments_for_dataset(dataset: Dict[str, Any], target_length: float, options: MeshOptions) -> None:
    hull_points = dataset.get("hull_points")
    if hull_points is None or len(hull_points) < 4:
        raise ValueError(f"Surface {dataset['name']!r} has no valid hull")

    class HullLine:
        def __init__(self, points_with_types):
            self.points = [Vector3D(p[0], p[1], p[2], point_type=p[3] if len(p) > 3 else "DEFAULT") for p in points_with_types]

    refined = refine_intersection_line_by_length(
        HullLine(hull_points),
        float(target_length),
        float(options.min_angle),
        bool(options.uniform),
    )
    segments = []
    if len(refined) >= 2:
        count = len(refined)
        if (refined[0] - refined[-1]).length_squared() < 1e-24:
            count -= 1
        for i in range(count):
            p1 = refined[i]
            p2 = refined[(i + 1) % len(refined)]
            segments.append([p1.to_numpy(), p2.to_numpy()])
    dataset["segments"] = segments
    dataset.pop("triangulation_result", None)
    logger.info("Computed %d hull segments for %s", len(segments), dataset["name"])


def _run_initial_triangulation_for_dataset(dataset: Dict[str, Any], target_size: float, options: MeshOptions) -> None:
    from scipy.spatial import cKDTree
    from scipy.interpolate import RBFInterpolator
    from Pymeshit.triangle_direct import DirectTriangleWrapper

    segments_data = dataset.get("segments")
    if segments_data is None or len(segments_data) < 3:
        raise ValueError(f"Surface {dataset['name']!r} has no segmentation")

    all_pts = np.asarray(dataset["points"], dtype=float)
    centered = all_pts - all_pts.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = -normal if normal[2] < 0.0 else normal
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)

    def build_rot(n, onto_z=True):
        axis = np.cross(n, z_axis) if onto_z else np.cross(z_axis, n)
        c = float(np.dot(z_axis, n)) if onto_z else float(np.dot(n, z_axis))
        an = np.linalg.norm(axis)
        if 1.0 - abs(c) < 1e-12 or an < 1e-15:
            return np.eye(3)
        axis = axis / an
        s = np.sqrt(max(0.0, 1.0 - c * c))
        C = 1.0 - c
        return np.array([
            [axis[0] * axis[0] * C + c, axis[0] * axis[1] * C - axis[2] * s, axis[0] * axis[2] * C + axis[1] * s],
            [axis[1] * axis[0] * C + axis[2] * s, axis[1] * axis[1] * C + c, axis[1] * axis[2] * C - axis[0] * s],
            [axis[2] * axis[0] * C - axis[1] * s, axis[2] * axis[1] * C + axis[0] * s, axis[2] * axis[2] * C + c],
        ])

    R = build_rot(normal, onto_z=True)
    R_inv = build_rot(normal, onto_z=False)
    all_pts_rot = (R @ all_pts.T).T

    unique_points, point_to_index, segment_indices = [], {}, []
    for seg in segments_data:
        p0, p1 = np.asarray(seg[0], float), np.asarray(seg[1], float)
        keys = [tuple(np.round(p0, 12)), tuple(np.round(p1, 12))]
        indices = []
        for key, point in zip(keys, (p0, p1)):
            if key not in point_to_index:
                point_to_index[key] = len(unique_points)
                unique_points.append(point)
            indices.append(point_to_index[key])
        if indices[0] != indices[1]:
            segment_indices.append(indices)

    boundary_xyz = np.asarray(unique_points, float)
    boundary_segs = np.asarray(segment_indices, int)
    boundary_rot = (R @ boundary_xyz.T).T
    boundary_xy = boundary_rot[:, :2]

    triangulator = DirectTriangleWrapper(
        gradient=float(options.gradient),
        min_angle=float(options.min_angle),
        base_size=float(target_size),
    )
    tri_res = triangulator.triangulate(
        points=boundary_xy,
        segments=boundary_segs,
        uniform=bool(options.uniform),
        create_transition=True,
    )
    if tri_res is None or "vertices" not in tri_res or "triangles" not in tri_res:
        raise RuntimeError(f"Initial triangulation failed for {dataset['name']!r}")

    vertices_xy = np.asarray(tri_res["vertices"], dtype=float)
    triangles = np.asarray(tri_res["triangles"], dtype=int)
    sample_xy = all_pts_rot[:, :2]
    sample_z = all_pts_rot[:, 2]

    def interp_idw(q, power=2):
        tree = cKDTree(sample_xy)
        k = min(64, len(sample_xy))
        d, idx = tree.query(q, k=k)
        if k == 1:
            d, idx = d[:, None], idx[:, None]
        r2 = np.maximum(d * d, 1e-24)
        w = 1.0 / (r2**power)
        return np.sum(w * sample_z[idx], axis=1) / np.sum(w, axis=1)

    def interp_tps(q):
        try:
            rbf = RBFInterpolator(sample_xy, sample_z, kernel="thin_plate_spline", smoothing=float(options.smoothing))
            return rbf(q)
        except Exception:
            return interp_idw(q)

    if "Thin Plate" in options.interpolation:
        z_out = interp_tps(vertices_xy)
    elif "IDW" in options.interpolation or "Legacy" in options.interpolation:
        z_out = interp_idw(vertices_xy)
    else:
        z_out = interp_idw(vertices_xy)

    vertices_rot3d = np.zeros((len(vertices_xy), 3), dtype=float)
    vertices_rot3d[:, :2] = vertices_xy
    vertices_rot3d[:, 2] = z_out
    dataset["triangulation_result"] = {"vertices": (R_inv @ vertices_rot3d.T).T, "triangles": triangles}
    logger.info(
        "Initial triangulation for %s: %d vertices, %d triangles",
        dataset["name"],
        len(vertices_xy),
        len(triangles),
    )


def _compute_global_intersections(
    datasets: List[Dict[str, Any]],
    surface_count: int,
    options: MeshOptions,
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    class ModelWrapper:
        def __init__(self):
            self.surfaces = []
            self.model_polylines = []
            self.intersections = []
            self.triple_points = []
            self.original_surface_indices = {}
            self.original_polyline_indices = {}

    class SurfaceWrapper:
        def __init__(self, dataset, index):
            self.name = dataset.get("name", f"Surface_{index}")
            self.vertices = []
            tri_result = dataset.get("triangulation_result") or {}
            tri_vertices = np.asarray(tri_result.get("vertices", []), dtype=float)
            self.triangles = np.asarray(tri_result.get("triangles", []), dtype=int)
            for point in tri_vertices:
                self.vertices.append(Vector3D(point[0], point[1], point[2]))
            hull = dataset.get("hull_points", [])
            self.convex_hull = [Vector3D(p[0], p[1], p[2]) for p in hull if len(p) >= 3]
            if self.vertices:
                xs = [p.x for p in self.vertices]
                ys = [p.y for p in self.vertices]
                zs = [p.z for p in self.vertices]
                self.bounds = [Vector3D(min(xs), min(ys), min(zs)), Vector3D(max(xs), max(ys), max(zs))]
            else:
                self.bounds = [Vector3D(), Vector3D()]
            self.type = "Surface"

    class PolylineWrapper:
        def __init__(self, dataset, index):
            self.name = dataset.get("name", f"Well_{index}")
            self.vertices = [Vector3D(p[0], p[1], p[2]) for p in np.asarray(dataset["points"], dtype=float)]
            self.segments = [(i, i + 1) for i in range(len(self.vertices) - 1)]
            xs = [p.x for p in self.vertices]
            ys = [p.y for p in self.vertices]
            zs = [p.z for p in self.vertices]
            self.bounds = [Vector3D(min(xs), min(ys), min(zs)), Vector3D(max(xs), max(ys), max(zs))]
            self.type = "Polyline"

    model = ModelWrapper()
    for idx in range(surface_count):
        dataset = datasets[idx]
        tri_result = dataset.get("triangulation_result")
        if not tri_result:
            continue
        surface = SurfaceWrapper(dataset, idx)
        if surface.vertices and len(surface.triangles) > 0:
            model_idx = len(model.surfaces)
            model.surfaces.append(surface)
            model.original_surface_indices[model_idx] = idx

    for idx in range(surface_count, len(datasets)):
        dataset = datasets[idx]
        points = dataset.get("points")
        if points is None or len(points) < 2:
            continue
        model_idx = len(model.model_polylines)
        model.model_polylines.append(PolylineWrapper(dataset, idx))
        model.original_polyline_indices[model_idx] = idx

    if len(model.surfaces) + len(model.model_polylines) < 2:
        return {}, []

    config = {
        "use_constraint_processing": True,
        "type_based_sizing": True,
        "hierarchical_constraints": True,
        "gradient": float(options.gradient),
        "use_enhanced_curved_detection": True,
        "adaptive_sampling": True,
        "max_subdivisions": 3,
    }
    model = run_intersection_workflow(
        model,
        progress_callback=lambda message: logger.debug("Intersection workflow: %s", str(message).strip()),
        tolerance=float(options.intersection_tolerance),
        config=config,
    )

    intersections: Dict[int, List[Dict[str, Any]]] = {}
    well_intersection_points: Dict[int, List[List[Any]]] = {}
    for intersection in model.intersections:
        if getattr(intersection, "is_polyline_mesh", False):
            original_poly = model.original_polyline_indices.get(intersection.id1, -1)
            original_surf = model.original_surface_indices.get(intersection.id2, -1)
            original_id1, original_id2 = original_poly, original_surf
            if original_poly >= 0 and intersection.points:
                well_intersection_points.setdefault(original_poly, [])
                for point in intersection.points:
                    well_intersection_points[original_poly].append([point.x, point.y, point.z, "INTERSECTION_POINT"])
        else:
            original_id1 = model.original_surface_indices.get(intersection.id1, -1)
            original_id2 = model.original_surface_indices.get(intersection.id2, -1)

        if original_id1 == -1 or original_id2 == -1:
            continue
        points = [[p.x, p.y, p.z] for p in intersection.points]
        info = {
            "dataset_id1": original_id1,
            "dataset_id2": original_id2,
            "is_polyline_mesh": bool(getattr(intersection, "is_polyline_mesh", False)),
            "points": points,
        }
        intersections.setdefault(min(original_id1, original_id2), []).append(info)

    for well_idx, points in well_intersection_points.items():
        if 0 <= well_idx < len(datasets):
            datasets[well_idx]["intersection_points"] = points

    triple_points = [
        {"point": [tp.point.x, tp.point.y, tp.point.z], "intersection_ids": list(tp.intersection_ids)}
        for tp in model.triple_points
    ]
    logger.info("Computed %d intersections and %d triple points", sum(len(v) for v in intersections.values()), len(triple_points))
    return intersections, triple_points


def _refine_intersections(
    datasets: List[Dict[str, Any]],
    intersections: Dict[int, List[Dict[str, Any]]],
    surface_count: int,
    options: MeshOptions,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    class TempModelWrapper:
        def __init__(self):
            self.surfaces = []
            self.polylines = []
            self.intersections = []
            self.triple_points = []
            self.original_indices_map = {}
            self.is_polyline = {}
            self.surface_original_to_temp_idx_map = {}
            self.polyline_original_to_temp_idx_map = {}
            self.surface_list_to_combined_idx_map = {}
            self.polyline_list_to_combined_idx_map = {}

    class TempDataWrapper:
        def __init__(self, name, size):
            self.convex_hull = []
            self.size = size
            self.name = name

    temp_model = TempModelWrapper()
    involved = set()
    for intersections_list in intersections.values():
        for entry in intersections_list:
            involved.add(entry["dataset_id1"])
            involved.add(entry["dataset_id2"])

    temp_counter = 0
    for original_idx, dataset in enumerate(datasets):
        if original_idx not in involved:
            continue
        data_wrapper = TempDataWrapper(dataset.get("name", f"Dataset_{original_idx}"), _target_for_index(datasets, original_idx, options))
        hull = dataset.get("hull_points")
        if hull is not None and len(hull) > 0:
            for hp in hull:
                ptype = hp[3] if len(hp) > 3 and isinstance(hp[3], str) else "DEFAULT"
                data_wrapper.convex_hull.append(Vector3D(hp[0], hp[1], hp[2], point_type=ptype))
        else:
            points = dataset.get("points")
            if points is None or len(points) == 0:
                continue
            data_wrapper.convex_hull = [Vector3D(p[0], p[1], p[2]) for p in points]

        is_polyline = str(dataset.get("type", "")).upper() in {"WELL", "POLYLINE"}
        temp_model.original_indices_map[temp_counter] = original_idx
        temp_model.is_polyline[temp_counter] = is_polyline
        if is_polyline:
            poly_idx = len(temp_model.polylines)
            temp_model.polyline_original_to_temp_idx_map[original_idx] = poly_idx
            temp_model.polyline_list_to_combined_idx_map[poly_idx] = temp_counter
            temp_model.polylines.append(data_wrapper)
        else:
            surface_idx = len(temp_model.surfaces)
            temp_model.surface_original_to_temp_idx_map[original_idx] = surface_idx
            temp_model.surface_list_to_combined_idx_map[surface_idx] = temp_counter
            temp_model.surfaces.append(data_wrapper)
        temp_counter += 1

    original_to_temp = {original: temp for temp, original in temp_model.original_indices_map.items()}
    for intersections_list in intersections.values():
        for entry in intersections_list:
            temp_id1 = original_to_temp.get(entry["dataset_id1"])
            temp_id2 = original_to_temp.get(entry["dataset_id2"])
            if temp_id1 is None or temp_id2 is None:
                continue
            new_intersection = Intersection(temp_id1, temp_id2, bool(entry.get("is_polyline_mesh", False)))
            for pt in entry.get("points", []):
                new_intersection.add_point(Vector3D(pt[0], pt[1], pt[2] if len(pt) > 2 else 0.0))
            temp_model.intersections.append(new_intersection)

    if not temp_model.intersections:
        return intersections, {}, []

    for i in range(len(temp_model.intersections) - 1):
        for j in range(i + 1, len(temp_model.intersections)):
            for point in calculate_triple_points(i, j, temp_model, tolerance=float(options.intersection_tolerance)):
                triple = TriplePoint(point)
                triple.add_intersection(i)
                triple.add_intersection(j)
                point.point_type = "TRIPLE_POINT"
                temp_model.triple_points.append(triple)
    insert_triple_points(temp_model, tolerance=float(options.intersection_tolerance))

    for temp_surface in temp_model.surfaces:
        if len(getattr(temp_surface, "convex_hull", [])) >= 3:
            temp_surface.convex_hull = make_corners_special(temp_surface.convex_hull, angle_threshold_deg=135.0)

    temp_surface_to_original = {temp: original for original, temp in temp_model.surface_original_to_temp_idx_map.items()}
    for temp_idx, temp_surface in enumerate(temp_model.surfaces):
        if len(getattr(temp_surface, "convex_hull", [])) < 3:
            continue
        original_idx = temp_surface_to_original.get(temp_idx, -1)
        target = _target_for_index(datasets, original_idx, options)

        class HullLine:
            def __init__(self, pts):
                self.points = [Vector3D(p.x, p.y, p.z, point_type=getattr(p, "point_type", "DEFAULT")) for p in pts]

        refined = refine_intersection_line_by_length(
            HullLine(temp_surface.convex_hull),
            float(target),
            float(options.min_angle),
            bool(options.uniform),
            tag_endpoints=False,
        )
        if len(refined) >= 2 and (refined[0] - refined[-1]).length_squared() < 1e-24:
            refined = refined[:-1]
        temp_surface.convex_hull = refined

    for intersection in temp_model.intersections:
        original1 = temp_model.original_indices_map.get(intersection.id1, -1)
        original2 = temp_model.original_indices_map.get(intersection.id2, -1)
        target = min(_target_for_index(datasets, original1, options), _target_for_index(datasets, original2, options))
        intersection.points = refine_intersection_line_by_length(
            intersection,
            float(target),
            float(options.min_angle),
            bool(options.uniform),
        )

    interp_config = {"interp": options.interpolation, "smoothing": float(options.smoothing)}
    for temp_idx, temp_surface in enumerate(temp_model.surfaces):
        if len(getattr(temp_surface, "convex_hull", [])) < 3:
            continue
        original_idx = temp_surface_to_original.get(temp_idx)
        raw_hull = [Vector3D(p.x, p.y, p.z, point_type=getattr(p, "point_type", "DEFAULT")) for p in temp_surface.convex_hull]
        refined_hull = raw_hull
        if original_idx is not None and "points" in datasets[original_idx]:
            scattered = [Vector3D(p[0], p[1], p[2]) for p in datasets[original_idx]["points"]]
            refined_hull = refine_hull_with_interpolation(raw_hull, scattered, interp_config)
        rebuilt = []
        for original_pt, refined_pt in zip(raw_hull, refined_hull):
            if isinstance(refined_pt, Vector3D):
                new_pt = Vector3D(refined_pt.x, refined_pt.y, refined_pt.z)
            else:
                new_pt = Vector3D(refined_pt[0], refined_pt[1], refined_pt[2])
            new_pt.point_type = getattr(original_pt, "point_type", "DEFAULT")
            if hasattr(original_pt, "type"):
                new_pt.type = original_pt.type
            rebuilt.append(new_pt)
        temp_surface.convex_hull = rebuilt

    for temp_surface_idx in range(len(temp_model.surfaces)):
        align_intersections_to_convex_hull(temp_surface_idx, temp_model)

    for dataset in datasets:
        dataset["stored_constraints"] = []

    for intersection in temp_model.intersections:
        sid1 = temp_model.original_indices_map.get(intersection.id1)
        sid2 = temp_model.original_indices_map.get(intersection.id2)
        if sid1 is None or sid2 is None:
            continue
        points = [[p.x, p.y, p.z, getattr(p, "point_type", "DEFAULT")] for p in intersection.points]
        if len(points) < 2:
            continue
        base = {
            "type": "intersection_line",
            "points": points,
            "is_polyline_mesh": bool(getattr(intersection, "is_polyline_mesh", False)),
        }
        for ds_idx, other_idx in ((sid1, sid2), (sid2, sid1)):
            if ds_idx is None or ds_idx >= len(datasets):
                continue
            entry = dict(base)
            entry["other_surface_id"] = other_idx
            entry["size"] = _target_for_index(datasets, ds_idx, options)
            datasets[ds_idx]["stored_constraints"].append(entry)
            datasets[ds_idx]["needs_constraint_update"] = True

    for temp_idx, temp_surface in enumerate(temp_model.surfaces):
        original_idx = temp_surface_to_original.get(temp_idx)
        if original_idx is None:
            continue
        hull = []
        for point in getattr(temp_surface, "convex_hull", []):
            ptype = getattr(point, "point_type", getattr(point, "type", "DEFAULT")) or "DEFAULT"
            hull.append([point.x, point.y, point.z, ptype])
        if hull:
            datasets[original_idx]["hull_points"] = np.array(hull, dtype=object)

    new_intersections: Dict[int, List[Dict[str, Any]]] = {}
    refined_for_visualization: Dict[int, List[Dict[str, Any]]] = {}
    for intersection in temp_model.intersections:
        original1 = temp_model.original_indices_map.get(intersection.id1)
        original2 = temp_model.original_indices_map.get(intersection.id2)
        if original1 is None or original2 is None:
            continue
        points = []
        for point in intersection.points:
            points.append([point.x, point.y, point.z, getattr(point, "point_type", "DEFAULT")])
        entry = {
            "dataset_id1": original1,
            "dataset_id2": original2,
            "is_polyline_mesh": bool(getattr(intersection, "is_polyline_mesh", False)),
            "points": points,
        }
        new_intersections.setdefault(min(original1, original2), []).append(entry)
        refined_for_visualization.setdefault(original1, []).append(entry.copy())
        refined_for_visualization.setdefault(original2, []).append(entry.copy())

    triple_points = [
        {"point": [tp.point.x, tp.point.y, tp.point.z], "intersection_ids": list(tp.intersection_ids)}
        for tp in temp_model.triple_points
    ]
    logger.info("Refined intersections: %d lines", sum(len(v) for v in new_intersections.values()))
    return new_intersections, refined_for_visualization, triple_points


def _target_for_index(datasets: List[Dict[str, Any]], index: int, options: MeshOptions) -> float:
    if 0 <= index < len(datasets):
        if str(datasets[index].get("type", "")).upper() == "WELL":
            return _target_size_for_well(datasets[index], options)
        return _target_size_for_surface(datasets[index], options)
    return float(options.target_size)


def _refine_wells(datasets: List[Dict[str, Any]], surface_count: int, options: MeshOptions) -> None:
    for idx in range(surface_count, len(datasets)):
        dataset = datasets[idx]
        if str(dataset.get("type", "")).upper() != "WELL":
            continue
        points = dataset.get("points")
        if points is None or len(points) < 2:
            continue
        refined = refine_well_by_length(
            points=points,
            target_length=_target_size_for_well(dataset, options),
            intersection_points=dataset.get("intersection_points", []),
            trim_to_intersections=False,
        )
        dataset["refined_well_points"] = [
            [p.x, p.y, p.z, getattr(p, "point_type", "DEFAULT")] if isinstance(p, Vector3D) else list(p)
            for p in refined
        ]


def _generate_conforming_surface_meshes(
    datasets: List[Dict[str, Any]],
    surface_specs: Sequence[SurfaceSpec],
    refined_intersections: Dict[int, List[Dict[str, Any]]],
    options: MeshOptions,
) -> Tuple[Dict[int, Dict[str, Any]], List[Tuple[str, str]]]:
    surface_data: Dict[int, Dict[str, Any]] = {}
    failures: List[Tuple[str, str]] = []
    for s_idx, spec in enumerate(surface_specs):
        dataset = datasets[s_idx]
        name = dataset.get("name", f"Surface_{s_idx}")
        try:
            target = _target_size_for_surface(dataset, options)
            cfg = {
                "target_size": target,
                "min_angle": float(options.min_angle),
                "max_area": float(target) ** 2 * 1.5,
                "gradient": float(options.gradient),
                "interp": options.interpolation,
                "smoothing": float(options.smoothing),
            }
            surface_context = _prepare_surface_data_for_triangulation(s_idx, dataset, cfg)
            proj = surface_context["projection_params"]
            centroid = np.asarray(proj["centroid"])
            basis = np.asarray(proj["basis"])

            def triangulate_entries(entries: List[Dict[str, Any]], label: str):
                pts3d, seg_arr, holes, plc_meta = _build_plc_from_entries(entries, target)
                if len(pts3d) < 3 or len(seg_arr) < 3:
                    raise RuntimeError(f"{label} PLC has insufficient constraints")

                pts2d = (np.array([[p.x, p.y, p.z] for p in pts3d]) - centroid) @ basis.T
                pts2d = pts2d[:, :2]
                holes_2d = np.empty((0, 2), dtype=float)
                if holes:
                    holes_3d = np.array([[h.x, h.y, h.z] for h in holes], dtype=float)
                    holes_2d = (holes_3d - centroid) @ basis.T
                    holes_2d = holes_2d[:, :2]

                cfg_for_triangulation = dict(cfg)
                well_feature_points = plc_meta.get("well_feature_points", [])
                well_feature_sizes = plc_meta.get("well_feature_sizes", [])
                if well_feature_points and len(well_feature_points) == len(well_feature_sizes):
                    wf3d = np.array([[p.x, p.y, p.z] for p in well_feature_points], dtype=float)
                    wf2d = (wf3d - centroid) @ basis.T
                    cfg_for_triangulation["triunsuitable_feature_points_2d"] = wf2d[:, :2]
                    cfg_for_triangulation["triunsuitable_feature_sizes"] = np.asarray(well_feature_sizes, dtype=float)
                    cfg_for_triangulation["enable_well_gradient_refinement"] = True
                    cfg_for_triangulation["triunsuitable_max_iterations"] = 8
                    cfg_for_triangulation["triunsuitable_max_new_points"] = int(min(2500, max(900, 600 + 8 * len(well_feature_points))))
                    cfg_for_triangulation["triunsuitable_max_feature_points"] = 800
                    cfg_for_triangulation["triunsuitable_min_local_size_ratio"] = 0.0

                hull = surface_context.get("hull_points") or []
                if len(hull) >= 3:
                    hull3d = np.array([[p.x, p.y, p.z] for p in hull], dtype=float)
                    hull2d = (hull3d - centroid) @ basis.T
                    cfg_for_triangulation["hull_polygon_2d"] = hull2d[:, :2]

                vertices, triangles, _ = run_constrained_triangulation_py(
                    pts2d,
                    np.asarray(seg_arr, dtype=int),
                    holes_2d,
                    proj,
                    np.array([[p.x, p.y, p.z] for p in pts3d], dtype=float),
                    cfg_for_triangulation,
                )
                if vertices is None or len(vertices) == 0 or triangles is None or len(triangles) == 0:
                    raise RuntimeError(f"{label} triangulation returned no cells")
                return vertices, triangles, holes

            entries = _selected_constraint_entries(
                s_idx,
                datasets,
                surface_specs,
                refined_intersections,
                options,
            )
            constraint_source = "selected"
            try:
                vertices, triangles, holes = triangulate_entries(entries, "selected")
            except Exception as selected_exc:
                if options.constraint_mode != "intersections":
                    raise
                hull_entries = _hull_constraint_entries(datasets[s_idx], target)
                if not hull_entries:
                    raise
                logger.warning(
                    "Conforming mesh for %s failed with intersection-only constraints (%s). "
                    "Retrying with hull constraints for this surface.",
                    name,
                    selected_exc,
                )
                vertices, triangles, holes = triangulate_entries(hull_entries, "hull fallback")
                constraint_source = "hull_fallback"

            dataset["conforming_mesh"] = {
                "vertices": vertices,
                "triangles": triangles,
                "holes": [[h.x, h.y, h.z] for h in holes],
                "constraint_source": constraint_source,
            }
            marker = (1000 + s_idx) if str(spec.role).lower() == "fault" else s_idx + 1
            surface_data[s_idx] = {
                "name": name,
                "vertices": vertices,
                "triangles": triangles,
                "facet_markers": np.full(len(triangles), marker, dtype=np.int32),
                "original_dataset_index": s_idx,
                "conforming_mesh_source": True,
                "constraint_source": constraint_source,
                "mesh_metadata": dataset["conforming_mesh"],
            }
            logger.info("Conforming mesh for %s: %d vertices, %d triangles", name, len(vertices), len(triangles))
        except Exception as exc:
            failures.append((name, str(exc)))
            logger.error("Conforming mesh generation failed for %s: %s", name, exc, exc_info=True)
    return surface_data, failures


def _prepare_surface_data_for_triangulation(dataset_idx: int, dataset: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    hull_data = dataset.get("hull_points", [])
    if hull_data is None or len(hull_data) < 3:
        raise ValueError(f"Surface {dataset_idx} has no hull points")
    hull_points = [Vector3D(p[0], p[1], p[2], point_type=p[3] if len(p) > 3 else "DEFAULT") for p in hull_data if len(p) >= 3]
    hull_np = np.array([[p.x, p.y, p.z] for p in hull_points], dtype=float)
    bbox_diag = float(np.linalg.norm(np.max(hull_np, axis=0) - np.min(hull_np, axis=0)))
    target = float(config.get("target_size", 1.0))
    surface_size = target if target <= bbox_diag or bbox_diag <= 0 else bbox_diag / 20.0
    return {
        "hull_points": hull_points,
        "size": surface_size,
        "projection_params": _surface_projection_params(hull_np),
        "name": dataset.get("name", f"Surface_{dataset_idx}"),
        "index": dataset_idx,
    }


def _selected_constraint_entries(
    surface_idx: int,
    datasets: List[Dict[str, Any]],
    surface_specs: Sequence[SurfaceSpec],
    refined_intersections: Dict[int, List[Dict[str, Any]]],
    options: MeshOptions,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    surface_role = str(surface_specs[surface_idx].role).lower()
    target = _target_size_for_surface(datasets[surface_idx], options)

    include_hulls = options.constraint_mode in {"all", "hulls"} or (
        options.constraint_mode == "intersections"
        and options.preserve_boundary_hulls
        and surface_role in {"border", "unit", "fault"}
    )
    include_intersections = options.constraint_mode in {"all", "intersections"}

    if include_hulls:
        hull = datasets[surface_idx].get("hull_points", [])
        for segment in _segment_hull_by_special_points(hull):
            entries.append({
                "points": segment,
                "type": "HULL",
                "target_size": target,
                "is_well_intersection": False,
            })

    if include_intersections:
        for inter in refined_intersections.get(surface_idx, []):
            raw_pts = inter.get("points", [])
            if not raw_pts:
                continue
            is_well = bool(inter.get("is_polyline_mesh", False))
            if is_well:
                seen = set()
                segments = []
                for point in raw_pts:
                    key = _coord_key(point)
                    if key not in seen:
                        seen.add(key)
                        segments.append([point])
            else:
                segments = _segment_by_triples(raw_pts)
            inter_target = min(
                target,
                _target_for_index(datasets, int(inter.get("dataset_id1", surface_idx)), options),
                _target_for_index(datasets, int(inter.get("dataset_id2", surface_idx)), options),
            )
            for segment in segments:
                entries.append({
                    "points": segment,
                    "type": "INTERSECTION",
                    "target_size": inter_target,
                    "is_well_intersection": is_well,
                    "well_target_size": inter_target,
                })
    return entries


def _hull_constraint_entries(dataset: Dict[str, Any], target_size: float) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    hull = dataset.get("hull_points", [])
    for segment in _segment_hull_by_special_points(hull):
        entries.append({
            "points": segment,
            "type": "HULL",
            "target_size": float(target_size),
            "is_well_intersection": False,
        })
    return entries


def _build_plc_from_entries(entries: List[Dict[str, Any]], surface_target_size: float):
    pts: List[Vector3D] = []
    segments: List[List[int]] = []
    holes: List[Vector3D] = []
    point_sizes: Dict[int, float] = {}
    well_feature_sizes: Dict[int, float] = {}
    index_by_key: Dict[Tuple[float, float, float], int] = {}

    def add(point_like, target_size=None, well_feature=False) -> int:
        point = _to_vector3d(point_like)
        key = _coord_key(point)
        idx = index_by_key.get(key)
        if idx is None:
            idx = len(pts)
            index_by_key[key] = idx
            pts.append(point)
        if target_size is not None and float(target_size) > 1e-9:
            old = point_sizes.get(idx)
            if old is None or float(target_size) < old:
                point_sizes[idx] = float(target_size)
        if well_feature:
            old = well_feature_sizes.get(idx)
            if old is None or float(target_size or surface_target_size) < old:
                well_feature_sizes[idx] = float(target_size or surface_target_size)
        return idx

    for entry in entries:
        seg_pts = entry.get("points", [])
        if not seg_pts:
            continue
        target = float(entry.get("target_size", surface_target_size))
        is_well = bool(entry.get("is_well_intersection", False))
        well_target = float(entry.get("well_target_size", target))

        if len(seg_pts) == 1:
            idx = add(seg_pts[0], target_size=target, well_feature=False)
            if is_well:
                well_feature_sizes[idx] = min(well_feature_sizes.get(idx, well_target), well_target)
            continue

        feature_count_before = len(well_feature_sizes)
        for p1, p2 in zip(seg_pts[:-1], seg_pts[1:]):
            i1 = add(p1, target_size=target)
            i2 = add(p2, target_size=target)
            if is_well:
                if _point_type_name(p1) == "INTERSECTION_POINT":
                    well_feature_sizes[i1] = min(well_feature_sizes.get(i1, well_target), well_target)
                if _point_type_name(p2) == "INTERSECTION_POINT":
                    well_feature_sizes[i2] = min(well_feature_sizes.get(i2, well_target), well_target)
            if i1 != i2:
                segments.append([i1, i2])
        if is_well and len(well_feature_sizes) == feature_count_before:
            first_idx = add(seg_pts[0], target_size=target)
            well_feature_sizes[first_idx] = min(well_feature_sizes.get(first_idx, well_target), well_target)

    meta = {
        "point_sizes": [float(point_sizes.get(i, surface_target_size)) for i in range(len(pts))],
        "well_feature_points": [pts[i] for i in sorted(well_feature_sizes.keys())],
        "well_feature_sizes": [float(well_feature_sizes[i]) for i in sorted(well_feature_sizes.keys())],
    }
    return pts, np.asarray(segments, dtype=int), holes, meta


def _segment_by_triples(points: Sequence[Any]) -> List[List[Any]]:
    if len(points) < 2:
        return []
    triples = [i for i, point in enumerate(points) if _point_type_name(point) == "TRIPLE_POINT"]
    closed = _is_closed_loop(points)
    if not triples:
        segment = list(points)
        if closed and _coord_key(segment[0]) != _coord_key(segment[-1]):
            segment.append(segment[0])
        return [segment]
    if closed:
        result = []
        unique = sorted(set(triples))
        if len(unique) == 1:
            return [list(points)]
        for a, b in zip(unique, unique[1:] + unique[:1]):
            segment = list(points[a:b + 1]) if a <= b else list(points[a:]) + list(points[:b + 1])
            if len(segment) >= 2:
                result.append(segment)
        return result
    splits = []
    for split in [0] + sorted(triples) + [len(points) - 1]:
        if not splits or splits[-1] != split:
            splits.append(split)
    return [list(points[a:b + 1]) for a, b in zip(splits[:-1], splits[1:]) if b - a >= 1]


def _segment_hull_by_special_points(points: Sequence[Any]) -> List[List[Any]]:
    if points is None or len(points) < 3:
        return []
    hull = list(points)
    if len(hull) > 1 and _coord_key(hull[0]) == _coord_key(hull[-1]):
        hull = hull[:-1]
    specials = [i for i, point in enumerate(hull) if _point_type_name(point) in {"CORNER", "CORNER_POINT", "COMMON_INTERSECTION_CONVEXHULL_POINT"}]
    if not specials:
        return [hull + [hull[0]]]
    specials = sorted(set(specials))
    if len(specials) == 1:
        idx = specials[0]
        return [hull[idx:] + hull[: idx + 1]]
    result = []
    for idx, start in enumerate(specials):
        end = specials[(idx + 1) % len(specials)]
        segment = hull[start:end + 1] if start < end else hull[start:] + hull[: end + 1]
        if len(segment) >= 2:
            result.append(segment)
    return result


def _is_closed_loop(points: Sequence[Any]) -> bool:
    if len(points) < 3:
        return False
    if _coord_distance(points[0], points[-1]) < 1e-4:
        return True
    return "LOOP_END" in _point_type_name(points[-1]) or "CIRCULAR" in _point_type_name(points[-1])


def _point_type_name(point_like: Any) -> str:
    if hasattr(point_like, "point_type") and point_like.point_type is not None:
        return str(point_like.point_type)
    if hasattr(point_like, "type") and point_like.type is not None:
        return str(point_like.type)
    try:
        if len(point_like) > 3:
            value = point_like[3]
            if hasattr(value, "item"):
                value = value.item()
            return str(value) if value else "DEFAULT"
    except Exception:
        pass
    return "DEFAULT"


def _to_vector3d(point_like: Any) -> Vector3D:
    if isinstance(point_like, Vector3D):
        return point_like
    ptype = "DEFAULT"
    if len(point_like) > 3:
        ptype = point_like[3]
        if hasattr(ptype, "item"):
            ptype = ptype.item()
    return Vector3D(float(point_like[0]), float(point_like[1]), float(point_like[2]), point_type=str(ptype) if ptype else "DEFAULT")


def _coord_key(point_like: Any, precision: int = 9) -> Tuple[float, float, float]:
    if hasattr(point_like, "x"):
        return (round(float(point_like.x), precision), round(float(point_like.y), precision), round(float(point_like.z), precision))
    return (round(float(point_like[0]), precision), round(float(point_like[1]), precision), round(float(point_like[2]), precision))


def _coord_distance(a: Any, b: Any) -> float:
    ak = np.asarray(_coord_key(a, precision=12), dtype=float)
    bk = np.asarray(_coord_key(b, precision=12), dtype=float)
    return float(np.linalg.norm(ak - bk))


def _indices_by_role(surfaces: Sequence[SurfaceSpec], role: str) -> set:
    return {idx for idx, surface in enumerate(surfaces) if str(surface.role).lower() == role}


def _build_well_data(
    datasets: List[Dict[str, Any]],
    surface_count: int,
    wells: Sequence[WellSpec],
) -> Dict[int, Dict[str, Any]]:
    well_data = {}
    for local_idx, spec in enumerate(wells):
        dataset_idx = surface_count + local_idx
        if dataset_idx >= len(datasets):
            continue
        dataset = datasets[dataset_idx]
        points = dataset.get("refined_well_points") or dataset.get("points")
        if points is None or len(points) < 2:
            continue
        marker = spec.marker if spec.marker is not None else dataset_idx + 2
        vector_points = [_to_vector3d(p) if not hasattr(p, "x") else p for p in points]
        edges = [(vector_points[i], vector_points[i + 1]) for i in range(len(vector_points) - 1)]
        well_data[dataset_idx] = {
            "points": vector_points,
            "edges": edges,
            "marker": int(marker),
            "name": dataset.get("name", f"Well_{dataset_idx}"),
        }
    return well_data


__all__ = [
    "SurfaceSpec",
    "WellSpec",
    "MaterialSpec",
    "MeshOptions",
    "MeshCase",
    "MeshResult",
    "read_points",
    "run_mesh_case",
    "generate_tetrahedral_mesh_from_surfaces",
]
