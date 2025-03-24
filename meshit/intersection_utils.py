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
        sorted_points = sort_intersection_points(intersection_points)
        
        # Add sorted points to the intersection
        for point in sorted_points:
            intersection.add_point(point)
            
        return intersection
    
    return None


def sort_intersection_points(points: List[Vector3D]) -> List[Vector3D]:
    """
    Sort intersection points to form a continuous polyline.
    
    This function uses a nearest-neighbor approach to connect points
    into a continuous polyline by always moving to the closest unvisited point.
    
    Args:
        points: List of unsorted intersection points
        
    Returns:
        Sorted list of points forming a continuous polyline
    """
    if len(points) <= 2:
        return points  # Already a line or single point
        
    # Start with a random point (first one)
    sorted_points = [points[0]]
    remaining_points = points[1:]
    
    # Keep adding the nearest point until all are used
    while remaining_points:
        current = sorted_points[-1]
        
        # Find closest remaining point
        min_dist = float('inf')
        closest_idx = -1
        
        for i, point in enumerate(remaining_points):
            dist = (current - point).length()
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        # Add the closest point to our sorted list
        sorted_points.append(remaining_points.pop(closest_idx))
    
    return sorted_points


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
        sorted_points = sort_intersection_points(intersection_points)
        
        for point in sorted_points:
            intersection.add_point(point)
        return intersection
    
    return None


def calculate_triple_points(intersection1_idx: int, intersection2_idx: int, model) -> List[TriplePoint]:
    """
    Calculate triple points between two intersections.
    
    Args:
        intersection1_idx: Index of first intersection
        intersection2_idx: Index of second intersection
        model: MeshItModel instance containing intersections
        
    Returns:
        List of TriplePoint objects
    """
    intersection1 = model.intersections[intersection1_idx]
    intersection2 = model.intersections[intersection2_idx]
    
    # Check if the intersections share a common surface/polyline
    common_object = False
    if intersection1.id1 == intersection2.id1 or intersection1.id1 == intersection2.id2:
        common_object = True
    elif intersection1.id2 == intersection2.id1 or intersection1.id2 == intersection2.id2:
        common_object = True
    
    if not common_object:
        return []  # No common object, no triple points possible
    
    # Find closest points between the two intersection sets
    triple_points = []
    
    for point1 in intersection1.points:
        for point2 in intersection2.points:
            # Calculate distance
            distance = (point1 - point2).length()
            
            # If points are close enough, they form a triple point
            if distance < 1e-5:  # Tolerance for considering points as coincident
                # Create a new triple point at the midpoint
                midpoint = (point1 + point2) * 0.5
                triple_point = TriplePoint(midpoint)
                triple_point.add_intersection(intersection1_idx)
                triple_point.add_intersection(intersection2_idx)
                triple_points.append(triple_point)
    
    return triple_points


def insert_triple_points(model):
    """
    Insert triple points into model.
    
    This function merges close triple points and adds them to appropriate intersections.
    
    Args:
        model: MeshItModel instance
    """
    if not hasattr(model, 'triple_points') or not model.triple_points:
        return
    
    # Merge triple points that are close to each other
    i = 0
    while i < len(model.triple_points):
        j = i + 1
        while j < len(model.triple_points):
            # Check if triple points are close
            distance = (model.triple_points[i].point - model.triple_points[j].point).length()
            
            if distance < 1e-5:  # Close enough to merge
                # Merge j into i
                for intersection_id in model.triple_points[j].intersection_ids:
                    model.triple_points[i].add_intersection(intersection_id)
                
                # Remove triple point j
                model.triple_points.pop(j)
            else:
                j += 1
        i += 1
    
    # Add triple points to the intersections they belong to
    for tp_idx, triple_point in enumerate(model.triple_points):
        for intersection_id in triple_point.intersection_ids:
            # Insert triple point into the intersection's point list
            intersection = model.intersections[intersection_id]
            
            # Find the best position to insert the triple point (closest to existing points)
            best_pos = 0
            min_dist = float('inf')
            
            for i, point in enumerate(intersection.points):
                dist = (point - triple_point.point).length()
                if dist < min_dist:
                    min_dist = dist
                    best_pos = i
            
            # Insert the triple point at the best position
            # (In practice, we might want a more sophisticated algorithm to maintain the
            # order of points along the intersection curve)
            if min_dist < 1e-5:  # If very close, replace the existing point
                intersection.points[best_pos] = triple_point.point
            else:
                # Otherwise, insert at the beginning
                intersection.points.insert(0, triple_point.point)


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