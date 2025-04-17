"""
MeshIt Intersection Utilities

This module provides functionality for computing intersections between 
surfaces and polylines, following the MeshIt workflow.
"""

import numpy as np
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Union

class Vector3D:
    """Simple 3D vector class compatible with MeshIt's Vector3D"""
    
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar):
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        return np.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def normalized(self):
        length = self.length()
        if length < 1e-10:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / length, self.y / length, self.z / length)
    
    def to_numpy(self):
        return np.array([self.x, self.y, self.z])
    
    @staticmethod
    def from_numpy(array):
        return Vector3D(array[0], array[1], array[2])
    
    def __repr__(self):
        return f"Vector3D({self.x}, {self.y}, {self.z})"


class Intersection:
    """Represents an intersection between two objects (surfaces or polylines)"""
    
    def __init__(self, id1: int, id2: int, is_polyline_mesh: bool = False):
        self.id1 = id1
        self.id2 = id2
        self.is_polyline_mesh = is_polyline_mesh
        self.points: List[Vector3D] = []
    
    def add_point(self, point: Vector3D):
        """Add intersection point"""
        self.points.append(point)
    
    def __repr__(self):
        return f"Intersection(id1={self.id1}, id2={self.id2}, points={len(self.points)})"


class TriplePoint:
    """Represents a triple point where three or more intersections meet"""
    
    def __init__(self, point: Vector3D):
        self.point = point
        self.intersection_ids: List[int] = []
    
    def add_intersection(self, intersection_id: int):
        """Associate intersection with this triple point"""
        if intersection_id not in self.intersection_ids:
            self.intersection_ids.append(intersection_id)
    
    def __repr__(self):
        return f"TriplePoint(point={self.point}, intersections={len(self.intersection_ids)})"


class Triangle:
    """Triangle in 3D space"""
    
    def __init__(self, v1: Vector3D, v2: Vector3D, v3: Vector3D):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
    
    def normal(self) -> Vector3D:
        """Calculate the normal vector of the triangle"""
        edge1 = self.v2 - self.v1
        edge2 = self.v3 - self.v1
        return edge1.cross(edge2).normalized()
    
    def centroid(self) -> Vector3D:
        """Calculate the centroid of the triangle"""
        return (self.v1 + self.v2 + self.v3) * (1.0/3.0)
    
    def contains_point(self, point: Vector3D) -> bool:
        """Check if the point lies within the triangle (approximate)"""
        # Barycentric coordinate approach
        v0 = self.v2 - self.v1
        v1 = self.v3 - self.v1
        v2 = point - self.v1
        
        d00 = v0.dot(v0)
        d01 = v0.dot(v1)
        d11 = v1.dot(v1)
        d20 = v2.dot(v0)
        d21 = v2.dot(v1)
        
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            return False
        
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        # Point is in triangle if u, v, w are all between 0 and 1
        return (u >= -1e-5) and (v >= -1e-5) and (w >= -1e-5) and (u + v + w <= 1.0 + 1e-5)


def triangle_triangle_intersection(tri1: Triangle, tri2: Triangle) -> List[Vector3D]:
    """
    Calculate intersection between two triangles.
    
    This implementation finds the actual line of intersection between two triangles
    by checking for intersections between the edges of each triangle and the face of the other.
    
    Args:
        tri1: First triangle
        tri2: Second triangle
        
    Returns:
        List of intersection points (empty if no intersection)
    """
    # Implementation based on Möller's algorithm
    # First, get the edges of both triangles
    tri1_edges = [
        (tri1.v1, tri1.v2),
        (tri1.v2, tri1.v3),
        (tri1.v3, tri1.v1)
    ]
    
    tri2_edges = [
        (tri2.v1, tri2.v2),
        (tri2.v2, tri2.v3),
        (tri2.v3, tri2.v1)
    ]
    
    # Check if the triangles are coplanar (within a small tolerance)
    normal1 = tri1.normal()
    normal2 = tri2.normal()
    if abs(normal1.dot(normal2)) > 0.999:  # Nearly parallel normals
        # Coplanar triangles - we'd need to find the polygon intersection
        # This is complex, so for now return empty list
        return []
    
    intersection_points = []
    
    # Check for intersections between edges of tri1 and the face of tri2
    for edge in tri1_edges:
        p1, p2 = edge
        intersection = line_triangle_intersection(p1, p2, tri2)
        if intersection:
            # Check if this point is already in our list (avoid duplicates)
            is_duplicate = False
            for point in intersection_points:
                if (point - intersection).length() < 1e-8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                intersection_points.append(intersection)
    
    # Check for intersections between edges of tri2 and the face of tri1
    for edge in tri2_edges:
        p1, p2 = edge
        intersection = line_triangle_intersection(p1, p2, tri1)
        if intersection:
            # Check if this point is already in our list (avoid duplicates)
            is_duplicate = False
            for point in intersection_points:
                if (point - intersection).length() < 1e-8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                intersection_points.append(intersection)
    
    # Sort the intersection points to form a line segment if there are exactly two points
    if len(intersection_points) == 2:
        return intersection_points
    elif len(intersection_points) > 2:
        # If we have more than 2 points (rare, but possible due to numerical issues),
        # find the two furthest points to define the intersection line
        max_dist = -1
        furthest_pair = (0, 1)
        for i in range(len(intersection_points)):
            for j in range(i+1, len(intersection_points)):
                dist = (intersection_points[i] - intersection_points[j]).length()
                if dist > max_dist:
                    max_dist = dist
                    furthest_pair = (i, j)
        return [intersection_points[furthest_pair[0]], intersection_points[furthest_pair[1]]]
    
    return intersection_points


def line_triangle_intersection(
    p1: Vector3D, p2: Vector3D, triangle: Triangle
) -> Optional[Vector3D]:
    """
    Calculate intersection between a line segment and a triangle.
    
    Args:
        p1: First endpoint of the line segment
        p2: Second endpoint of the line segment
        triangle: Triangle to check for intersection
        
    Returns:
        Intersection point or None if no intersection
    """
    # Line direction vector
    dir_vec = p2 - p1
    
    # Triangle normal
    normal = triangle.normal()
    
    # Check if line and triangle are parallel (dot product of normal and line direction is zero)
    dot_product = normal.dot(dir_vec)
    if abs(dot_product) < 1e-10:
        return None  # Line and triangle are parallel
    
    # Calculate distance from p1 to triangle plane
    plane_point = triangle.v1
    d = normal.dot(plane_point - p1) / dot_product
    
    # Check if intersection is within line segment bounds
    if d < 0 or d > 1:
        return None  # Intersection outside line segment
    
    # Calculate intersection point
    intersection = p1 + dir_vec * d
    
    # Check if intersection point is inside triangle
    if triangle.contains_point(intersection):
        return intersection
    
    return None


def calculate_surface_surface_intersection(surface1_idx: int, surface2_idx: int, model) -> Optional[Intersection]:
    """
    Calculate intersections between two surfaces.
    
    Args:
        surface1_idx: Index of first surface
        surface2_idx: Index of second surface
        model: MeshItModel instance containing surfaces
        
    Returns:
        Intersection object or None if no intersections found
    """
    surface1 = model.surfaces[surface1_idx]
    surface2 = model.surfaces[surface2_idx]
    
    # Early rejection test using bounding boxes
    if hasattr(surface1, 'bounds') and hasattr(surface2, 'bounds'):
        if (surface1.bounds[1].x < surface2.bounds[0].x or 
            surface1.bounds[0].x > surface2.bounds[1].x or
            surface1.bounds[1].y < surface2.bounds[0].y or 
            surface1.bounds[0].y > surface2.bounds[1].y or
            surface1.bounds[1].z < surface2.bounds[0].z or 
            surface1.bounds[0].z > surface2.bounds[1].z):
            return None  # No intersection possible
    
    # Find all intersections between triangles in both surfaces
    intersection_points = []
    
    # Convert surface triangles to Triangle objects
    tri1_list = []
    for tri_idx in surface1.triangles:
        if len(tri_idx) >= 3:
            v1 = surface1.vertices[tri_idx[0]]
            v2 = surface1.vertices[tri_idx[1]]
            v3 = surface1.vertices[tri_idx[2]]
            tri1_list.append(Triangle(v1, v2, v3))
    
    tri2_list = []
    for tri_idx in surface2.triangles:
        if len(tri_idx) >= 3:
            v1 = surface2.vertices[tri_idx[0]]
            v2 = surface2.vertices[tri_idx[1]]
            v3 = surface2.vertices[tri_idx[2]]
            tri2_list.append(Triangle(v1, v2, v3))
    
    # Check all triangle pairs for intersections
    for tri1 in tri1_list:
        for tri2 in tri2_list:
            # Find intersections between triangles
            points = triangle_triangle_intersection(tri1, tri2)
            
            # Add new points to the intersection list
            for point in points:
                # Check if this point is already in our intersection list (with tolerance)
                is_duplicate = False
                for existing_point in intersection_points:
                    dist = (existing_point - point).length()
                    if dist < 1e-8:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    intersection_points.append(point)
    
    # If we found any intersections, create an Intersection object
    if intersection_points:
        intersection = Intersection(surface1_idx, surface2_idx, False)
        
        # Sort points to form a continuous polyline
        sorted_points = sort_intersection_points_pca(intersection_points)
        
        # Add sorted points to the intersection
        for point in sorted_points:
            intersection.add_point(point)
            
        return intersection
    
    return None


def sort_intersection_points_pca(points: List[Vector3D]) -> List[Vector3D]:
    """
    Sort intersection points using PCA projection for better linearity.
    """
    if len(points) <= 2:
        return points

    # Convert points to numpy array
    points_np = np.array([p.to_numpy() for p in points])

    # Center the points
    centroid = np.mean(points_np, axis=0)
    centered_points = points_np - centroid

    # Perform SVD to find the principal axis (first singular vector)
    try:
        _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
        principal_axis = vh[0] # Direction of greatest variance
    except np.linalg.LinAlgError:
         # Fallback if SVD fails (e.g., all points are identical)
         print("Warning: SVD failed in sort_intersection_points_pca. Using original order.")
         return points

    # Project points onto the principal axis
    projected_distances = centered_points @ principal_axis

    # Sort original points based on projected distances
    sorted_indices = np.argsort(projected_distances)
    sorted_points_list = [points[i] for i in sorted_indices]

    return sorted_points_list


def calculate_polyline_surface_intersection(polyline_idx: int, surface_idx: int, model) -> Optional[Intersection]:
    """
    Calculate intersections between a polyline and a surface.
    
    Args:
        polyline_idx: Index of polyline
        surface_idx: Index of surface
        model: MeshItModel instance containing polylines and surfaces
        
    Returns:
        Intersection object or None if no intersections found
    """
    polyline = model.model_polylines[polyline_idx]
    surface = model.surfaces[surface_idx]
    
    # Early rejection test using bounding boxes
    if hasattr(polyline, 'bounds') and hasattr(surface, 'bounds'):
        if (polyline.bounds[1].x < surface.bounds[0].x or 
            polyline.bounds[0].x > surface.bounds[1].x or
            polyline.bounds[1].y < surface.bounds[0].y or 
            polyline.bounds[0].y > surface.bounds[1].y or
            polyline.bounds[1].z < surface.bounds[0].z or 
            polyline.bounds[0].z > surface.bounds[1].z):
            return None  # No intersection possible
    
    # Find intersections between line segments and triangles
    intersection_points = []
    
    # Convert surface triangles to Triangle objects
    triangles = []
    for tri_idx in surface.triangles:
        if len(tri_idx) >= 3:
            v1 = surface.vertices[tri_idx[0]]
            v2 = surface.vertices[tri_idx[1]]
            v3 = surface.vertices[tri_idx[2]]
            triangles.append(Triangle(v1, v2, v3))
    
    # For each segment in the polyline
    for segment in polyline.segments:
        if len(segment) < 2:
            continue
        
        # Get segment endpoints
        p1 = polyline.vertices[segment[0]]
        p2 = polyline.vertices[segment[1]]
        
        # Check for intersection with each triangle
        for triangle in triangles:
            intersection = line_triangle_intersection(p1, p2, triangle)
            if intersection:
                # Check if this point is already in our intersection list (with tolerance)
                is_duplicate = False
                for existing_point in intersection_points:
                    dist = (existing_point - intersection).length()
                    if dist < 1e-8:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    intersection_points.append(intersection)
    
    # If we found any intersections, create an Intersection object
    if intersection_points:
        intersection = Intersection(polyline_idx, surface_idx, True)
        
        # For polyline-surface intersections, we sort the points along the polyline
        # to maintain the original structure of the polyline
        sorted_points = sort_intersection_points_pca(intersection_points)
        
        for point in sorted_points:
            intersection.add_point(point)
        return intersection
    
    return None


def closest_point_on_segment(p: Vector3D, a: Vector3D, b: Vector3D) -> Vector3D:
    """Find the closest point on the line segment [a, b] to point p."""
    ap = p - a
    ab = b - a
    ab_len_sq = ab.dot(ab)
    if ab_len_sq < 1e-10:
        return a # Segment is a point
    t = ap.dot(ab) / ab_len_sq
    t = max(0, min(1, t)) # Clamp t to [0, 1] for segment
    return a + ab * t


def segment_segment_distance(a1: Vector3D, b1: Vector3D, a2: Vector3D, b2: Vector3D) -> Tuple[float, Vector3D, Vector3D]:
    """Calculate the minimum distance between two 3D line segments [a1, b1] and [a2, b2],
       and the closest points on each segment.

       Uses the algorithm described by Dan Sunday:
       http://geomalgorithms.com/a07-_distance.html#dist3D_Segment_to_Segment
    """
    u = b1 - a1
    v = b2 - a2
    w = a1 - a2

    a = u.dot(u)  # always >= 0
    b = u.dot(v)
    c = v.dot(v)  # always >= 0
    d = u.dot(w)
    e = v.dot(w)
    D = a * c - b * b  # always >= 0
    sc, sN, sD = D, D, D  # sc = sN / sD, default sD = D >= 0
    tc, tN, tD = D, D, D  # tc = tN / tD, default tD = D >= 0

    # compute the line parameters of the two closest points
    if D < 1e-10:  # the lines are almost parallel
        sN = 0.0  # force using point a1 on segment S1
        sD = 1.0  # to prevent possible division by 0
        tN = e
        tD = c
    else:  # get the closest points on the infinite lines
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if sN < 0.0:  # sc < 0 => the s=0 edge is visible
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:  # sc > 1 => the s=1 edge is visible
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:  # tc < 0 => the t=0 edge is visible
        tN = 0.0
        # recompute sc for this edge
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:  # tc > 1 => the t=1 edge is visible
        tN = tD
        # recompute sc for this edge
        if (-d + b) < 0.0:
            sN = 0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    # finally do the division to get sc and tc
    sc = 0.0 if abs(sN) < 1e-10 else sN / sD
    tc = 0.0 if abs(tN) < 1e-10 else tN / tD

    # get the difference of the two closest points
    dP = w + (u * sc) - (v * tc)
    closest_p1 = a1 + u * sc
    closest_p2 = a2 + v * tc

    return dP.length(), closest_p1, closest_p2


def calculate_triple_points(intersection1_idx: int, intersection2_idx: int, model, tolerance=1e-5) -> List[TriplePoint]:
    """
    Calculate triple points between two intersection polylines using
    segment-segment intersection.

    Args:
        intersection1_idx: Index of first intersection polyline in model.intersections
        intersection2_idx: Index of second intersection polyline in model.intersections
        model: MeshItModel instance containing intersections
        tolerance: Distance tolerance to consider segments intersecting

    Returns:
        List of TriplePoint objects found at the intersections.
    """
    intersection1 = model.intersections[intersection1_idx]
    intersection2 = model.intersections[intersection2_idx]

    # Check if the intersections share a common surface/polyline ID
    # This is a basic check, more robust checks might involve surface indices
    ids1 = {intersection1.id1, intersection1.id2}
    ids2 = {intersection2.id1, intersection2.id2}
    if not ids1.intersection(ids2):
        return [] # No common parent object, cannot form a triple point

    found_triple_points = []

    # Iterate through segments of the first intersection polyline
    for i in range(len(intersection1.points) - 1):
        p1a = intersection1.points[i]
        p1b = intersection1.points[i+1]

        # Iterate through segments of the second intersection polyline
        for j in range(len(intersection2.points) - 1):
            p2a = intersection2.points[j]
            p2b = intersection2.points[j+1]

            # Calculate the distance between the two segments
            dist, closest1, closest2 = segment_segment_distance(p1a, p1b, p2a, p2b)

            # If the distance is within tolerance, consider it an intersection
            if dist < tolerance:
                # The intersection point is the midpoint of the closest points
                intersection_point = (closest1 + closest2) * 0.5

                # Create a triple point
                # Check for duplicates before adding
                is_duplicate = False
                for existing_tp in found_triple_points:
                    if (existing_tp.point - intersection_point).length() < tolerance:
                        # Add intersection indices to existing triple point if needed
                        existing_tp.add_intersection(intersection1_idx)
                        existing_tp.add_intersection(intersection2_idx)
                        is_duplicate = True
                        break

                if not is_duplicate:
                    triple_point = TriplePoint(intersection_point)
                    triple_point.add_intersection(intersection1_idx)
                    triple_point.add_intersection(intersection2_idx)
                    found_triple_points.append(triple_point)

    return found_triple_points


def insert_triple_points(model, tolerance=1e-5):
    """
    Insert triple points into the corresponding intersection polylines.

    Args:
        model: MeshItModel instance containing intersections and triple_points
        tolerance: Tolerance for merging close points and finding segments
    """
    if not hasattr(model, 'triple_points') or not model.triple_points:
        return # Nothing to insert

    # 1. Merge close triple points
    merged_triple_points = []
    used_indices = set()

    for i in range(len(model.triple_points)):
        if i in used_indices:
            continue

        current_tp = model.triple_points[i]
        merged_tp = TriplePoint(current_tp.point)
        for int_id in current_tp.intersection_ids:
            merged_tp.add_intersection(int_id)
        used_indices.add(i)

        for j in range(i + 1, len(model.triple_points)):
            if j in used_indices:
                continue

            other_tp = model.triple_points[j]
            if (current_tp.point - other_tp.point).length() < tolerance:
                # Merge intersection IDs
                for int_id in other_tp.intersection_ids:
                    merged_tp.add_intersection(int_id)
                used_indices.add(j)

        merged_triple_points.append(merged_tp)

    model.triple_points = merged_triple_points # Replace with merged list
    if not model.triple_points:
         return # Nothing left after merging

    # 2. Insert merged triple points into intersection polylines
    for tp in model.triple_points:
        for intersection_idx in tp.intersection_ids:
            if not (0 <= intersection_idx < len(model.intersections)):
                print(f"Warning: Invalid intersection index {intersection_idx} in triple point.")
                continue

            intersection = model.intersections[intersection_idx]
            points = intersection.points
            inserted = False

            # Check if point already exists (within tolerance)
            for k, existing_point in enumerate(points):
                if (tp.point - existing_point).length() < tolerance:
                    # Update existing point to the exact triple point coordinate? Optional.
                    # points[k] = tp.point
                    inserted = True
                    break
            if inserted:
                continue # Already exists (or is very close)

            # Find the segment the triple point lies on
            best_segment_idx = -1
            min_dist_to_segment = float('inf')

            for k in range(len(points) - 1):
                p_a = points[k]
                p_b = points[k+1]
                closest_on_seg = closest_point_on_segment(tp.point, p_a, p_b)
                dist_sq = (tp.point - closest_on_seg).length() # Use length directly here

                # Check if the point projects onto the segment line itself (collinear check)
                if dist_sq < tolerance:
                    # Check if it's between endpoints (already handled by closest_point_on_segment clamping)
                    # Store the segment index if it's the best fit so far
                    # We need to make sure it's *on* the segment, not just near it.
                    # Recalculate distance to ensure it's the segment, not just the line projection
                    dist_to_segment_line = (tp.point - closest_on_seg).length()
                    if dist_to_segment_line < min_dist_to_segment:
                         min_dist_to_segment = dist_to_segment_line
                         best_segment_idx = k

            # Insert the point if a suitable segment was found within tolerance
            if best_segment_idx != -1 and min_dist_to_segment < tolerance:
                 # Insert after the starting point of the segment
                 points.insert(best_segment_idx + 1, tp.point)
                 inserted = True
            else:
                 # Fallback: If no segment is found (e.g., TP is outside polyline bounds
                 # or exactly at an endpoint already checked), we might append it,
                 # but ideally, it should lie on a segment. For now, we log a warning.
                 print(f"Warning: Could not find segment for triple point {tp.point} on intersection {intersection_idx}. Min dist: {min_dist_to_segment}")
                 # As a simple fallback, add to the beginning (less ideal)
                 # points.insert(0, tp.point)


def align_intersections_to_convex_hull(surface_idx: int, model):
    """
    Align intersection points to the convex hull of a surface.
    
    Args:
        surface_idx: Index of the surface
        model: MeshItModel instance
    """
    surface = model.surfaces[surface_idx]
    
    # If no convex hull, calculate it first
    if not hasattr(surface, 'convex_hull') or not surface.convex_hull:
        if hasattr(surface, 'calculate_convex_hull'):
            surface.calculate_convex_hull()
        else:
            print(f"Warning: Surface {surface_idx} has no convex hull and no method to calculate it")
            return
    
    # Triangulate the convex hull if not already done
    if not hasattr(surface, 'convex_hull_triangles') or not surface.convex_hull_triangles:
        # Simple triangulation by connecting points to the center
        hull_points = surface.convex_hull
        if len(hull_points) < 3:
            print(f"Warning: Surface {surface_idx} convex hull has fewer than 3 points")
            return
        
        # Calculate centroid of convex hull
        centroid = Vector3D(0, 0, 0)
        for p in hull_points:
            centroid = centroid + p
        centroid = centroid * (1.0 / len(hull_points))
        
        # Create triangles by connecting consecutive hull points to centroid
        surface.convex_hull_triangles = []
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i+1) % len(hull_points)]
            surface.convex_hull_triangles.append([p1, p2, centroid])
    
    # For each intersection involving this surface
    for intersection in model.intersections:
        if intersection.id1 == surface_idx or intersection.id2 == surface_idx:
            # For each point in the intersection
            for i, point in enumerate(intersection.points):
                # Find closest point on convex hull
                min_dist = float('inf')
                closest_point = None
                
                # Check against each triangle in the convex hull
                for tri_points in surface.convex_hull_triangles:
                    triangle = Triangle(tri_points[0], tri_points[1], tri_points[2])
                    
                    # Project point onto triangle plane
                    normal = triangle.normal()
                    plane_point = triangle.v1
                    
                    # Vector from plane point to our point
                    v = point - plane_point
                    
                    # Distance from point to plane
                    dist = normal.dot(v)
                    
                    # Projected point
                    projected = point - normal * dist
                    
                    # Check if projected point is in triangle
                    if triangle.contains_point(projected):
                        # Calculate distance to the projected point
                        projection_dist = abs(dist)
                        if projection_dist < min_dist:
                            min_dist = projection_dist
                            closest_point = projected
                
                # If we found a projection on a triangle, use it
                if closest_point:
                    intersection.points[i] = closest_point


def calculate_size_of_intersections(model):
    """
    Calculate sizes for intersections based on the associated objects.
    
    Args:
        model: MeshItModel instance
    """
    for intersection in model.intersections:
        # Get the associated objects
        if intersection.is_polyline_mesh:
            # Polyline-surface intersection
            polyline = model.model_polylines[intersection.id1]
            surface = model.surfaces[intersection.id2]
            
            # Set size based on objects
            intersection_size = min(
                getattr(polyline, 'size', 1.0),
                getattr(surface, 'size', 1.0)
            ) * 0.5  # Smaller size for intersections
            
            # Store size with each point
            for point in intersection.points:
                setattr(point, 'size', intersection_size)
        else:
            # Surface-surface intersection
            surface1 = model.surfaces[intersection.id1]
            surface2 = model.surfaces[intersection.id2]
            
            # Set size based on surfaces
            intersection_size = min(
                getattr(surface1, 'size', 1.0),
                getattr(surface2, 'size', 1.0)
            ) * 0.5  # Smaller size for intersections
            
            # Store size with each point
            for point in intersection.points:
                setattr(point, 'size', intersection_size)


def run_intersection_workflow(model, progress_callback=None):
    """
    Run the complete intersection workflow as in MeshIt.
    
    Args:
        model: MeshItModel instance
        progress_callback: Optional callback for progress updates
        
    Returns:
        Updated model with intersection information
    """
    def report_progress(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(message, end='')
    
    # 1. Surface-surface intersections
    report_progress(">Start calculating surface-surface intersections...\n")
    
    # Clear existing intersections
    model.intersections = []
    
    # Calculate total number of combinations
    n_surfaces = len(model.surfaces)
    
    # Process each pair of surfaces
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        
        for s1 in range(n_surfaces - 1):
            for s2 in range(s1 + 1, n_surfaces):
                futures.append(
                    executor.submit(calculate_surface_surface_intersection, s1, s2, model)
                )
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            intersection = future.result()
            if intersection:
                model.intersections.append(intersection)
    
    report_progress(">...finished\n")
    
    # 2. Polyline-surface intersections
    report_progress(">Start calculating polyline-surface intersections...\n")
    
    n_polylines = len(model.model_polylines)
    
    # Process each polyline-surface pair
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        
        for p in range(n_polylines):
            for s in range(n_surfaces):
                futures.append(
                    executor.submit(calculate_polyline_surface_intersection, p, s, model)
                )
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            intersection = future.result()
            if intersection:
                model.intersections.append(intersection)
    
    report_progress(">...finished\n")
    
    # 3. Calculate size of intersections
    calculate_size_of_intersections(model)
    
    # 4. Triple points
    report_progress(">Start calculating intersection triplepoints...\n")
    
    # Clear existing triple points
    model.triple_points = []
    
    # Calculate triple points
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        
        for i1 in range(len(model.intersections) - 1):
            for i2 in range(i1 + 1, len(model.intersections)):
                futures.append(
                    executor.submit(calculate_triple_points, i1, i2, model)
                )
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            triple_points = future.result()
            model.triple_points.extend(triple_points)
    
    # Insert triple points into intersections
    insert_triple_points(model)
    
    report_progress(">...finished\n")
    
    # 5. Align intersections to convex hulls
    report_progress(">Start aligning Convex Hulls to Intersections...\n")
    
    for s in range(n_surfaces):
        report_progress(f"   >({s + 1}/{n_surfaces}) {model.surfaces[s].name}\n")
        align_intersections_to_convex_hull(s, model)
    
    report_progress(">...finished\n")
    
    return model 