"""Core one-click 2D meshing workflow for the compact GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math

import numpy as np


@dataclass
class TwoDInputFeature:
    """Input feature loaded from a user file."""

    name: str
    coords: np.ndarray
    is_closed: bool
    source_path: str = ""


@dataclass
class TwoDRunConfig:
    """Fixed defaults for the one-click compact 2D workflow."""

    snap_tol: float = 1e-8
    min_angle: float = 26.0
    target_edge_length: float = 0.0
    max_area_factor: float = 0.50


@dataclass
class TwoDRunResult:
    """One-click 2D run output consumed by the compact GUI."""

    vertices: np.ndarray
    triangles: np.ndarray
    boundary_loops: List[np.ndarray]
    constraint_lines: List[np.ndarray]
    regions: List[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)
    plc_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    plc_segments: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    plc_holes: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))


def _iter_polygons(geom) -> Iterable[Any]:
    """Yield Polygon parts from a shapely geometry."""
    if geom is None or geom.is_empty:
        return

    gtype = geom.geom_type
    if gtype == "Polygon":
        yield geom
    elif gtype in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            yield from _iter_polygons(part)


def _iter_lines(geom) -> Iterable[Any]:
    """Yield LineString parts from a shapely geometry."""
    if geom is None or geom.is_empty:
        return

    gtype = geom.geom_type
    if gtype == "LineString":
        yield geom
    elif gtype == "LinearRing":
        from shapely.geometry import LineString

        yield LineString(geom)
    elif gtype in ("MultiLineString", "GeometryCollection"):
        for part in geom.geoms:
            yield from _iter_lines(part)


def _as_xy_array(coords: Any, feature_name: str, warnings: List[str]) -> np.ndarray:
    """Normalize user coordinates to an (N,2) float array."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        warnings.append(f"{feature_name}: skipped (requires at least two 2D points)")
        return np.empty((0, 2), dtype=float)
    return arr[:, :2]


def _is_closed_ring(xy: np.ndarray, tol: float) -> bool:
    """True when first and last points are effectively identical."""
    if xy.shape[0] < 3:
        return False
    return float(np.linalg.norm(xy[0] - xy[-1])) <= tol


def _to_polygon_parts(xy: np.ndarray, feature_name: str, warnings: List[str], tol: float) -> List[Any]:
    """Build one or more valid polygons from an input ring."""
    from shapely.geometry import Polygon

    if not _is_closed_ring(xy, tol):
        xy = np.vstack([xy, xy[0]])

    poly = Polygon(xy)
    if poly.is_empty or poly.area <= 0.0:
        warnings.append(f"{feature_name}: skipped (empty/zero-area polygon)")
        return []

    if not poly.is_valid:
        repaired = None
        try:
            from shapely.validation import make_valid

            repaired = make_valid(poly)
        except Exception:
            repaired = None

        if repaired is None or repaired.is_empty:
            repaired = poly.buffer(0)

        if repaired is None or repaired.is_empty:
            warnings.append(f"{feature_name}: skipped (invalid polygon could not be repaired)")
            return []

        poly = repaired

    parts = [p for p in _iter_polygons(poly) if p.area > 0.0]
    if not parts:
        warnings.append(f"{feature_name}: skipped (no valid polygon parts after repair)")
    elif len(parts) > 1:
        warnings.append(f"{feature_name}: repaired into {len(parts)} polygon parts")
    return parts


def _collect_domain_boundaries(domain_geom) -> Tuple[List[Any], List[np.ndarray], np.ndarray]:
    """Return boundary lines, display loops and hole points for Triangle."""
    from shapely.geometry import LineString, Polygon

    boundary_lines: List[Any] = []
    boundary_loops: List[np.ndarray] = []
    hole_points: List[List[float]] = []

    for poly in _iter_polygons(domain_geom):
        ext = np.asarray(poly.exterior.coords, dtype=float)
        if ext.shape[0] >= 2:
            boundary_lines.append(LineString(ext))
            boundary_loops.append(ext)

        for ring in poly.interiors:
            inner = np.asarray(ring.coords, dtype=float)
            if inner.shape[0] >= 2:
                boundary_lines.append(LineString(inner))
                boundary_loops.append(inner)
                hole_seed = Polygon(inner).representative_point()
                hole_points.append([float(hole_seed.x), float(hole_seed.y)])

    holes = np.asarray(hole_points, dtype=float) if hole_points else np.empty((0, 2), dtype=float)
    return boundary_lines, boundary_loops, holes


def _node_lines_to_segments(lines: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Node linework and convert to unique vertices + segment index pairs."""
    from shapely.ops import unary_union

    if not lines:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=int)

    noded = unary_union(list(lines))

    vertices: List[List[float]] = []
    vertex_map: Dict[Tuple[float, float], int] = {}
    seg_pairs: List[List[int]] = []
    seg_seen: set[Tuple[int, int]] = set()

    def add_vertex(x: float, y: float) -> int:
        key = (round(float(x), 12), round(float(y), 12))
        idx = vertex_map.get(key)
        if idx is None:
            idx = len(vertices)
            vertices.append([float(x), float(y)])
            vertex_map[key] = idx
        return idx

    for line in _iter_lines(noded):
        coords = np.asarray(line.coords, dtype=float)
        if coords.shape[0] < 2:
            continue
        for i in range(coords.shape[0] - 1):
            p1 = coords[i]
            p2 = coords[i + 1]
            i1 = add_vertex(p1[0], p1[1])
            i2 = add_vertex(p2[0], p2[1])
            if i1 == i2:
                continue
            edge_key = (min(i1, i2), max(i1, i2))
            if edge_key in seg_seen:
                continue
            seg_seen.add(edge_key)
            seg_pairs.append([i1, i2])

    pts = np.asarray(vertices, dtype=float) if vertices else np.empty((0, 2), dtype=float)
    segs = np.asarray(seg_pairs, dtype=int) if seg_pairs else np.empty((0, 2), dtype=int)
    return pts, segs


def _auto_target_edge(plc_points: np.ndarray) -> float:
    """Compute a stable default target edge length from PLC extent."""
    if plc_points.size == 0:
        return 1.0
    mins = np.min(plc_points, axis=0)
    maxs = np.max(plc_points, axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    if diag <= 1e-12:
        return 1.0
    return max(diag / 90.0, 1e-5)


def run_two_d_one_click(
    features: Sequence[TwoDInputFeature],
    config: Optional[TwoDRunConfig] = None,
) -> TwoDRunResult:
    """Execute the compact one-click 2D workflow."""
    if config is None:
        config = TwoDRunConfig()

    warnings: List[str] = []

    try:
        import triangle as tr
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("triangle dependency is required for 2D meshing") from exc

    try:
        from shapely.geometry import LineString
        from shapely.ops import polygonize, unary_union
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("shapely dependency is required for 2D boolean/containment workflow") from exc

    if not features:
        raise ValueError("No 2D inputs loaded")

    closed_polygons: List[Tuple[str, Any]] = []
    open_lines_raw: List[Tuple[str, Any]] = []

    for feat in features:
        xy = _as_xy_array(feat.coords, feat.name, warnings)
        if xy.size == 0:
            continue

        is_closed = bool(feat.is_closed)
        if is_closed and xy.shape[0] >= 3:
            parts = _to_polygon_parts(xy, feat.name, warnings, config.snap_tol)
            for poly in parts:
                closed_polygons.append((feat.name, poly))
        else:
            open_lines_raw.append((feat.name, LineString(xy)))

    if not closed_polygons:
        raise ValueError("No valid closed polygon inputs found")

    # Build arrangement from all polygon boundaries, then use odd-depth parity.
    all_boundaries = []
    for _, poly in closed_polygons:
        all_boundaries.append(poly.exterior)
        all_boundaries.extend(list(poly.interiors))

    merged_boundaries = unary_union(all_boundaries)
    arrangement_faces = [p for p in polygonize(merged_boundaries) if p.area > 0.0]

    odd_regions: List[Dict[str, Any]] = []
    for face in arrangement_faces:
        rp = face.representative_point()
        depth = sum(1 for _, poly in closed_polygons if poly.covers(rp))
        if depth % 2 == 1:
            odd_regions.append(
                {
                    "polygon": face,
                    "depth": int(depth),
                    "seed": (float(rp.x), float(rp.y)),
                    "area": float(face.area),
                }
            )

    if not odd_regions:
        warnings.append("Parity split produced no odd regions; falling back to polygon union")
        union_geom = unary_union([p for _, p in closed_polygons])
        for poly in _iter_polygons(union_geom):
            rp = poly.representative_point()
            odd_regions.append(
                {
                    "polygon": poly,
                    "depth": 1,
                    "seed": (float(rp.x), float(rp.y)),
                    "area": float(poly.area),
                }
            )

    if not odd_regions:
        raise RuntimeError("No meshable domain could be constructed from closed polygons")

    domain_geom = unary_union([entry["polygon"] for entry in odd_regions])
    if domain_geom.is_empty:
        raise RuntimeError("Constructed 2D domain is empty")

    boundary_lines, boundary_loops, hole_points = _collect_domain_boundaries(domain_geom)

    constraint_lines: List[np.ndarray] = []
    clip_linework: List[Any] = []
    for feat_name, raw_line in open_lines_raw:
        if raw_line.is_empty or raw_line.length <= 0.0:
            continue
        clipped = raw_line.intersection(domain_geom)
        parts = list(_iter_lines(clipped))
        if not parts:
            warnings.append(f"{feat_name}: open line outside domain and ignored")
            continue
        for part in parts:
            coords = np.asarray(part.coords, dtype=float)
            if coords.shape[0] < 2:
                continue
            constraint_lines.append(coords)
            clip_linework.append(part)

    plc_points, plc_segments = _node_lines_to_segments(boundary_lines + clip_linework)
    if plc_points.shape[0] < 3 or plc_segments.shape[0] < 3:
        raise RuntimeError("Failed to build a valid PLC for 2D triangulation")

    target_edge = float(config.target_edge_length)
    if target_edge <= 0.0:
        target_edge = _auto_target_edge(plc_points)

    max_area = max(target_edge * target_edge * float(config.max_area_factor), 1e-12)

    regions_for_triangle = []
    exported_regions: List[Dict[str, Any]] = []
    for ridx, region in enumerate(odd_regions, start=1):
        seed = region["seed"]
        attr = float(ridx)
        regions_for_triangle.append([seed[0], seed[1], attr, max_area])
        exported_regions.append(
            {
                "id": ridx,
                "seed": seed,
                "area": float(region["area"]),
                "depth": int(region["depth"]),
                "max_area": max_area,
            }
        )

    tri_input: Dict[str, Any] = {
        "vertices": plc_points,
        "segments": plc_segments,
    }
    if hole_points.size > 0:
        tri_input["holes"] = hole_points
    if regions_for_triangle:
        tri_input["regions"] = np.asarray(regions_for_triangle, dtype=float)

    options = f"pq{float(config.min_angle):.2f}a{max_area:.12g}A"
    tri_result = tr.triangulate(tri_input, opts=options)

    vertices = np.asarray(tri_result.get("vertices", np.empty((0, 2))), dtype=float)
    triangles = np.asarray(tri_result.get("triangles", np.empty((0, 3))), dtype=int)
    if vertices.size == 0 or triangles.size == 0:
        raise RuntimeError("Triangle could not generate a valid 2D mesh")

    return TwoDRunResult(
        vertices=vertices,
        triangles=triangles,
        boundary_loops=boundary_loops,
        constraint_lines=constraint_lines,
        regions=exported_regions,
        warnings=warnings,
        plc_points=plc_points,
        plc_segments=plc_segments,
        plc_holes=hole_points,
    )
