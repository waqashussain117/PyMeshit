"""
MeshIt Intersection Utilities

This module provides functionality for computing intersections between 
surfaces and polylines, following the MeshIt workflow.
"""

import numpy as np
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Union
import math # Ensure math is imported for floor
import logging

logger = logging.getLogger(__name__)

class Vector3D:
    """Simple 3D vector class compatible with MeshIt's Vector3D"""
    
    def __init__(self, x=0.0, y=0.0, z=0.0, point_type=None):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.type = "DEFAULT" 
        self.type = point_type  # Store type information # Point type: DEFAULT, CORNER, INTERSECTION_POINT, TRIPLE_POINT, COMMON_INTERSECTION_CONVEXHULL_POINT
    
    # Add property for compatibility with MeshIt's Vector3D
    @property
    def type(self):
        return self.point_type
    
    @type.setter
    def type(self, value):
        self.point_type = value

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
        return f"Vector3D({self.x}, {self.y}, {self.z}, {self.type})"


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


class Box:
    """Spatial subdivision box for efficient intersection calculations"""
    
    def __init__(self):
        self.min = Vector3D()
        self.max = Vector3D()
        self.center = Vector3D()
        self.T1s = []  # Triangles from surface 1
        self.T2s = []  # Triangles from surface 2
        self.N1s = []  # Segments from intersection 1
        self.N2s = []  # Segments from intersection 2
        self.Box = [None] * 8  # Subboxes for octree subdivision
    
    def calculate_center(self):
        """Calculate the center of the box"""
        self.center = (self.min + self.max) * 0.5
    
    def generate_subboxes(self):
        """Create 8 child boxes (octree subdivision)"""
        self.calculate_center()
        
        # Create 8 subboxes spanning the octants, octant means the 8 subboxes of the box
        for i in range(8):
            self.Box[i] = Box()
            
            # Set min coordinates based on octant
            self.Box[i].min.x = self.min.x if (i & 1) == 0 else self.center.x # if i is 0 or 1, set min.x to min.x, otherwise set it to center.x
            self.Box[i].min.y = self.min.y if (i & 2) == 0 else self.center.y # if i is 0 or 2, set min.y to min.y, otherwise set it to center.y
            self.Box[i].min.z = self.min.z if (i & 4) == 0 else self.center.z # if i is 0 or 4, set min.z to min.z, otherwise set it to center.z
            
            # Set max coordinates based on octant
            self.Box[i].max.x = self.center.x if (i & 1) == 0 else self.max.x # if i is 0 or 1, set max.x to center.x, otherwise set it to max.x
            self.Box[i].max.y = self.center.y if (i & 2) == 0 else self.max.y # if i is 0 or 2, set max.y to center.y, otherwise set it to max.y
            self.Box[i].max.z = self.center.z if (i & 4) == 0 else self.max.z # if i is 0 or 4, set max.z to center.z, otherwise set it to max.z
    
    def tri_in_box(self, triangle):
        """Check if a triangle intersects this box"""
        # Simple AABB-triangle overlap test
        # First check if any vertex is inside the box
        for vertex in [triangle.v1, triangle.v2, triangle.v3]:
            if (self.min.x <= vertex.x <= self.max.x and
                self.min.y <= vertex.y <= self.max.y and
                self.min.z <= vertex.z <= self.max.z):
                return True
        
        # TODO: Implement more sophisticated AABB-triangle intersection test
        # For simplicity, we'll use a conservative test checking if triangle's bounding box
        # overlaps with our box
        tri_min = Vector3D(
            min(triangle.v1.x, triangle.v2.x, triangle.v3.x),
            min(triangle.v1.y, triangle.v2.y, triangle.v3.y),
            min(triangle.v1.z, triangle.v2.z, triangle.v3.z)
        )
        tri_max = Vector3D(
            max(triangle.v1.x, triangle.v2.x, triangle.v3.x),
            max(triangle.v1.y, triangle.v2.y, triangle.v3.y),
            max(triangle.v1.z, triangle.v2.z, triangle.v3.z)
        )
        
        # Check if bounding boxes overlap
        return not (tri_max.x < self.min.x or tri_min.x > self.max.x or
                    tri_max.y < self.min.y or tri_min.y > self.max.y or
                    tri_max.z < self.min.z or tri_min.z > self.max.z)
    
    def seg_in_box(self, v1, v2):
        """Check if a line segment intersects this box"""
        # Simple AABB-segment overlap test
        # First check if either endpoint is inside the box
        for vertex in [v1, v2]:
            if (self.min.x <= vertex.x <= self.max.x and
                self.min.y <= vertex.y <= self.max.y and
                self.min.z <= vertex.z <= self.max.z):
                return True
        
        # TODO: Implement more sophisticated AABB-segment intersection test
        # For simplicity, we'll use a conservative test checking if segment's bounding box
        # overlaps with our box
        seg_min = Vector3D(
            min(v1.x, v2.x),
            min(v1.y, v2.y),
            min(v1.z, v2.z)
        )
        seg_max = Vector3D(
            max(v1.x, v2.x),
            max(v1.y, v2.y),
            max(v1.z, v2.z)
        )
        
        # Check if bounding boxes overlap
        return not (seg_max.x < self.min.x or seg_min.x > self.max.x or
                    seg_max.y < self.min.y or seg_min.y > self.max.y or
                    seg_max.z < self.min.z or seg_min.z > self.max.z)
    
    def too_much_tri(self):
        """Check if this box contains too many triangles for direct testing"""
        # Threshold for subdivision - adjust based on performance testing
        return len(self.T1s) > 10 and len(self.T2s) > 10
    
    def too_much_seg(self):
        """Check if this box contains too many segments for direct testing"""
        # Threshold for subdivision - adjust based on performance testing
        return len(self.N1s) > 10 and len(self.N2s) > 10
    
    def split_tri(self, int_segments):
        """
        Recursively subdivide box and test triangle intersections.
        
        This implements the spatial subdivision approach of the C++ version.
        
        Args:
            int_segments: A collection to store intersection segments
        """
        self.generate_subboxes()
        
        # Place triangles in appropriate subboxes
        for tri1 in self.T1s:
            for b in range(8):
                if self.Box[b].tri_in_box(tri1):
                    self.Box[b].T1s.append(tri1)
        
        for tri2 in self.T2s:
            for b in range(8):
                if self.Box[b].tri_in_box(tri2):
                    self.Box[b].T2s.append(tri2)
        
        # Process each subbox
        for b in range(8):
            # Only process if both lists have triangles
            if self.Box[b].T1s and self.Box[b].T2s:
                if self.Box[b].too_much_tri():
                    # Further subdivide this box
                    self.Box[b].split_tri(int_segments)
                else:
                    # Perform direct triangle-triangle tests
                    for tri1 in self.Box[b].T1s:
                        for tri2 in self.Box[b].T2s:
                            # Use the optimized triangle-triangle intersection test
                            isectpt1, isectpt2 = tri_tri_intersect_with_isectline(tri1, tri2)
                            if isectpt1 and isectpt2:
                                # Avoid duplicate segments
                                append_non_existing_segment(int_segments, isectpt1, isectpt2)
    
    def split_seg(self, triple_points, i1, i2):
        """
        Recursively subdivide box and test segment intersections for triple points.
        
        This implements the spatial subdivision approach of the C++ version.
        
        Args:
            triple_points: Collection to store triple points
            i1: Index of first intersection
            i2: Index of second intersection
        """
        self.generate_subboxes()
        
        # Place segments in appropriate subboxes
        for seg1 in self.N1s:
            for b in range(8):
                if self.Box[b].seg_in_box(seg1[0], seg1[1]):
                    self.Box[b].N1s.append(seg1)
        
        for seg2 in self.N2s:
            for b in range(8):
                if self.Box[b].seg_in_box(seg2[0], seg2[1]):
                    self.Box[b].N2s.append(seg2)
        
        # Process each subbox
        for b in range(8):
            # Only process if both lists have segments
            if self.Box[b].N1s and self.Box[b].N2s:
                if self.Box[b].too_much_seg():
                    # Further subdivide this box
                    self.Box[b].split_seg(triple_points, i1, i2)
                else:
                    # Perform direct segment-segment tests
                    tolerance = 1e-5 # Use the same default tolerance
                    for seg1 in self.Box[b].N1s:
                        p1a, p1b = seg1[0], seg1[1]
                        for seg2 in self.Box[b].N2s:
                            p2a, p2b = seg2[0], seg2[1]

                            # Calculate distance and closest points between segments FIRST
                            dist, closest1, closest2 = segment_segment_distance(p1a, p1b, p2a, p2b)

                            # Check if distance is within tolerance
                            if dist < tolerance:
                                # Calculate triple point as the midpoint
                                tp_point = (closest1 + closest2) * 0.5
                                # Just append the raw point coordinate to the list passed by reference
                                triple_points.append(tp_point)

                                # --- REMOVED duplicate check and TriplePoint object creation ---
                                # # Check for duplicates within the accumulating list
                                # is_duplicate = False
                                # for existing_tp in triple_points:
                                #     # This check is problematic here, should be done after collecting all points
                                #     # if (existing_tp.point - tp_point).length() < tolerance:
                                #     #     # Merge intersection IDs into the existing TP
                                #     #     existing_tp.add_intersection(i1)
                                #     #     existing_tp.add_intersection(i2)
                                #     #     is_duplicate = True
                                #     #     break
                                #
                                # if not is_duplicate:
                                #     # Create a new TriplePoint object
                                #     # This creation should happen after merging
                                #     # triple_point_obj = TriplePoint(tp_point)
                                #     # triple_point_obj.add_intersection(i1)
                                #     # triple_point_obj.add_intersection(i2)
                                #     # found_triple_points.append(triple_point_obj)
                                # --- END REMOVAL ---


def append_non_existing_segment(segments, p1, p2):
    """
    Add a segment to the collection if it doesn't already exist.
    
    Args:
        segments: Collection of segments
        p1: First endpoint
        p2: Second endpoint
    """
    # Check if segment already exists
    for existing_segment in segments:
        # Check if either (p1,p2) or (p2,p1) already exists
        if (((existing_segment[0] - p1).length() < 1e-8 and 
             (existing_segment[1] - p2).length() < 1e-8) or
            ((existing_segment[0] - p2).length() < 1e-8 and 
             (existing_segment[1] - p1).length() < 1e-8)):
            return  # Segment already exists
    
    # Add new segment
    segments.append((p1, p2))


def tri_tri_intersect_with_isectline(tri1, tri2):
    """
    Fast triangle-triangle intersection test that returns the intersection line.
    
    This is a more efficient implementation that directly computes the
    intersection line between two triangles without checking all edge-face pairs.
    
    Args:
        tri1: First triangle
        tri2: Second triangle
        
    Returns:
        Tuple of (point1, point2) defining the intersection line, or (None, None) if no intersection
    """
    # Implementation based on Tomas Möller's fast triangle-triangle intersection algorithm
    
    # 1. Compute plane equations for both triangles
    n1 = tri1.normal()
    n2 = tri2.normal()
    
    # Triangle 1 plane equation: n1·X - d1 = 0
    d1 = n1.dot(tri1.v1)
    
    # Triangle 2 plane equation: n2·X - d2 = 0
    d2 = n2.dot(tri2.v1)
    
    # 2. Check if triangles are coplanar
    if abs(n1.dot(n2)) > 0.999 and abs(d1 - d2) < 1e-6:
        # Coplanar triangles - we'd need polygon clipping
        # For simplicity, return no intersection
        return None, None
    
    # 3. Compute the line of intersection between the two planes
    # The direction of the line is perpendicular to both normals
    line_dir = n1.cross(n2).normalized()
    
    if line_dir.length() < 1e-6:
        # Parallel planes, no intersection
        return None, None
    
    # 4. Compute distances from each vertex of tri1 to plane of tri2
    dists1 = []
    for v in [tri1.v1, tri1.v2, tri1.v3]:
        dists1.append(n2.dot(v) - d2)
    
    # 5. Compute distances from each vertex of tri2 to plane of tri1
    dists2 = []
    for v in [tri2.v1, tri2.v2, tri2.v3]:
        dists2.append(n1.dot(v) - d1)
    
    # 6. Check if triangles intersect the opposite plane
    if all(d > 0 for d in dists1) or all(d < 0 for d in dists1):
        return None, None  # Tri1 entirely on one side of tri2's plane
    
    if all(d > 0 for d in dists2) or all(d < 0 for d in dists2):
        return None, None  # Tri2 entirely on one side of tri1's plane
    
    # 7. Find the two points of intersection on tri1
    isect1 = []
    for i in range(3):
        j = (i + 1) % 3
        if dists1[i] * dists1[j] <= 0 and abs(dists1[i] - dists1[j]) > 1e-6:
            # This edge crosses the plane, compute intersection point
            t = dists1[i] / (dists1[i] - dists1[j])
            vertices = [tri1.v1, tri1.v2, tri1.v3]
            point = vertices[i] + (vertices[j] - vertices[i]) * t
            isect1.append(point)
    
    # 8. Find the two points of intersection on tri2
    isect2 = []
    for i in range(3):
        j = (i + 1) % 3
        if dists2[i] * dists2[j] <= 0 and abs(dists2[i] - dists2[j]) > 1e-6:
            # This edge crosses the plane, compute intersection point
            t = dists2[i] / (dists2[i] - dists2[j])
            vertices = [tri2.v1, tri2.v2, tri2.v3]
            point = vertices[i] + (vertices[j] - vertices[i]) * t
            isect2.append(point)
    
    # 9. Check if we found intersections
    if len(isect1) != 2 or len(isect2) != 2:
        # This can happen due to numerical issues or edge cases
        # Fall back to the original edge-face intersection method
        return None, None
    
    # 10. Now we need to find the overlap between the two line segments
    # Project both segments onto a common axis
    # We'll use the intersection line direction as the axis
    
    # Project points onto line direction
    proj1 = [line_dir.dot(p - tri1.v1) for p in isect1]
    proj2 = [line_dir.dot(p - tri1.v1) for p in isect2]
    
    # Sort projections
    if proj1[0] > proj1[1]:
        proj1.reverse()
        isect1.reverse()
    
    if proj2[0] > proj2[1]:
        proj2.reverse()
        isect2.reverse()
    
    # Find overlap
    if proj1[1] < proj2[0] or proj2[1] < proj1[0]:
        return None, None  # No overlap
    
    # Compute intersection segment
    start_t = max(proj1[0], proj2[0])
    end_t = min(proj1[1], proj2[1])
    
    # Convert back to 3D points
    if abs(proj1[1] - proj1[0]) < 1e-6:
        # Use isect2 as reference if isect1 is degenerate
        start_idx = 0 if abs(proj2[0] - start_t) < 1e-6 else 1
        end_idx = 0 if abs(proj2[0] - end_t) < 1e-6 else 1
        p1 = isect2[start_idx]
        p2 = isect2[end_idx]
    else:
        # Interpolate along isect1
        t1 = (start_t - proj1[0]) / (proj1[1] - proj1[0])
        t2 = (end_t - proj1[0]) / (proj1[1] - proj1[0])
        p1 = isect1[0] + (isect1[1] - isect1[0]) * t1
        p2 = isect1[0] + (isect1[1] - isect1[0]) * t2
    
    # Set intersection point types
    p1.type = "INTERSECTION_POINT"
    p2.type = "INTERSECTION_POINT"
    
    return p1, p2


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
    Calculate intersections between two surfaces using spatial subdivision for efficiency.
    
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
    
    # Set up the spatial subdivision box
    box = Box()
    
    # Initialize box bounds to encompass both surfaces
    if tri1_list and tri2_list:
        # Get min/max for surface 1
        s1_min = Vector3D(
            min(min(t.v1.x, t.v2.x, t.v3.x) for t in tri1_list),
            min(min(t.v1.y, t.v2.y, t.v3.y) for t in tri1_list),
            min(min(t.v1.z, t.v2.z, t.v3.z) for t in tri1_list)
        )
        s1_max = Vector3D(
            max(max(t.v1.x, t.v2.x, t.v3.x) for t in tri1_list),
            max(max(t.v1.y, t.v2.y, t.v3.y) for t in tri1_list),
            max(max(t.v1.z, t.v2.z, t.v3.z) for t in tri1_list)
        )
        
        # Get min/max for surface 2
        s2_min = Vector3D(
            min(min(t.v1.x, t.v2.x, t.v3.x) for t in tri2_list),
            min(min(t.v1.y, t.v2.y, t.v3.y) for t in tri2_list),
            min(min(t.v1.z, t.v2.z, t.v3.z) for t in tri2_list)
        )
        s2_max = Vector3D(
            max(max(t.v1.x, t.v2.x, t.v3.x) for t in tri2_list),
            max(max(t.v1.y, t.v2.y, t.v3.y) for t in tri2_list),
            max(max(t.v1.z, t.v2.z, t.v3.z) for t in tri2_list)
        )
        
        # Set box to intersection of bounding boxes
        box.min.x = max(s1_min.x, s2_min.x)
        box.min.y = max(s1_min.y, s2_min.y)
        box.min.z = max(s1_min.z, s2_min.z)
        box.max.x = min(s1_max.x, s2_max.x)
        box.max.y = min(s1_max.y, s2_max.y)
        box.max.z = min(s1_max.z, s2_max.z)
        
        # Check if there's no overlap in the bounding boxes
        if (box.min.x > box.max.x or 
            box.min.y > box.max.y or 
            box.min.z > box.max.z):
            return None  # No overlap, cannot have intersections
    else:
        return None  # No triangles in one of the surfaces
    
    # Add triangles to the box
    for tri in tri1_list:
        if box.tri_in_box(tri):
            box.T1s.append(tri)
    
    for tri in tri2_list:
        if box.tri_in_box(tri):
            box.T2s.append(tri)
    
    # If no triangles in the intersection box, return None
    if not box.T1s or not box.T2s:
        return None
    
    # Container for intersection segments
    intersection_segments = []
    
    # Use spatial subdivision to find intersections
    if box.too_much_tri():
        box.split_tri(intersection_segments)
    else:
        # Direct testing for small number of triangles
        for tri1 in box.T1s:
            for tri2 in box.T2s:
                isectpt1, isectpt2 = tri_tri_intersect_with_isectline(tri1, tri2)
                if isectpt1 and isectpt2:
                    append_non_existing_segment(intersection_segments, isectpt1, isectpt2)
    
    # If we found any intersections, create an Intersection object
    if intersection_segments:
        intersection = Intersection(surface1_idx, surface2_idx, False)
        
        # Flatten segments into points while preserving order
        intersection_points = []
        for segment in intersection_segments:
            for point in segment:
                # Check if this point is already in our list (with tolerance)
                is_duplicate = False
                for existing_point in intersection_points:
                    if (existing_point - point).length() < 1e-8:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    intersection_points.append(point)
        
        # Sort points to form a continuous polyline
        sorted_points = sort_intersection_points(intersection_points)
        
        # Add sorted points to the intersection
        for point in sorted_points:
            intersection.add_point(point)
            
        return intersection
    
    return None


def calculate_skew_line_transversal(p1: Vector3D, p2: Vector3D, p3: Vector3D, p4: Vector3D) -> Optional[Vector3D]:
    """
    Calculate the skew line transversal between two 3D line segments.
    
    This is a port of C_Line::calculateSkewLineTransversal from C++.
    It calculates the point of closest approach between two non-coplanar segments.
    
    Args:
        p1: First point of first segment
        p2: Second point of first segment
        p3: First point of second segment
        p4: Second point of second segment
        
    Returns:
        Vector3D representing the point of closest approach, or None if the lines are parallel
    """
    # Direction vectors for the two lines
    d1 = p2 - p1
    d2 = p4 - p3
    
    # Check if the lines are parallel
    cross_d1d2 = d1.cross(d2)
    len_cross = cross_d1d2.length()
    if len_cross < 1e-10:
        return None  # Lines are parallel, no unique transversal
    
    # Calculate parameters for the closest point
    n = cross_d1d2.normalized()
    
    # Calculate distance between the lines
    p_diff = p3 - p1
    
    # Calculate t values for closest points on the two lines
    # Compute determinants for the linear system
    det1 = p_diff.dot(d2.cross(n))
    det2 = p_diff.dot(d1.cross(n))
    
    # Denominator is the square of the sin of the angle between d1 and d2
    denom = len_cross * len_cross
    
    # Parameters along the two lines for the closest points
    t1 = det1 / denom
    t2 = det2 / denom
    
    # Check if the closest points are within the segments
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        # Calculate the two closest points
        c1 = p1 + d1 * t1
        c2 = p3 + d2 * t2
        
        # Check if the points are close enough to be considered an intersection
        if (c1 - c2).length() < 1e-5:
            # Return midpoint
            return (c1 + c2) * 0.5
    
    return None


def sort_intersection_points(points: List[Vector3D]) -> List[Vector3D]:
    """
    Sort intersection points to form a continuous polyline.
    
    This is a port of C_Line::SortByType from C++. It sorts intersection points
    spatially rather than using PCA projection.
    
    Args:
        points: List of intersection points to sort
        
    Returns:
        Sorted list of intersection points
    """
    if len(points) <= 2:
        return points
        
    # Find special points (endpoints or triple points) to use as fixed anchors
    special_points = []
    normal_points = []
    
    for p in points:
        if p.type != "DEFAULT" and p.type != "INTERSECTION_POINT":
            special_points.append(p)
        else:
            normal_points.append(p)
    
    # If we have special points, use them as anchors
    if len(special_points) >= 2:
        # Start with the first special point
        result = [special_points[0]]
        used_points = {0}
        
        # Find the next closest point until we've used all points
        while len(result) < len(points):
            last_point = result[-1]
            
            # Find closest point among remaining points
            best_dist = float('inf')
            best_idx = -1
            best_point = None
            
            # First priority: check remaining special points
            for i, p in enumerate(special_points):
                if i not in used_points:
                    dist = (p - last_point).length()
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                        best_point = p
            
            # If we found a special point, use it
            if best_idx != -1:
                result.append(best_point)
                used_points.add(best_idx)
                continue
            
            # Second priority: find closest normal point
            best_dist = float('inf')
            best_idx = -1
            
            for i, p in enumerate(normal_points):
                if i + len(special_points) not in used_points:
                    dist = (p - last_point).length()
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
            
            if best_idx != -1:
                result.append(normal_points[best_idx])
                used_points.add(best_idx + len(special_points))
            else:
                # This shouldn't happen, but just in case
                break
        
        return result
    else:
        # No special points, use the old PCA method as fallback
        return sort_intersection_points_pca(points)


def sort_intersection_points_pca(points: List[Vector3D]) -> List[Vector3D]:
    """
    Sort intersection points using PCA projection for better linearity.
    This is a fallback method when no special points are available.
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
    sorted_points = [points[i] for i in sorted_indices]

    return sorted_points


def calculate_polyline_surface_intersection(polyline_idx: int, surface_idx: int, model) -> Optional[Intersection]:
    """
    Calculate intersections between a polyline and a surface using spatial subdivision.
    
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
    
    # Convert surface triangles to Triangle objects
    triangles = []
    for tri_idx in surface.triangles:
        if len(tri_idx) >= 3:
            v1 = surface.vertices[tri_idx[0]]
            v2 = surface.vertices[tri_idx[1]]
            v3 = surface.vertices[tri_idx[2]]
            triangles.append(Triangle(v1, v2, v3))
    
    # Convert polyline segments to segment pairs
    segments = []
    for segment_idx in polyline.segments:
        if len(segment_idx) < 2:
            continue
        
        p1 = polyline.vertices[segment_idx[0]]
        p2 = polyline.vertices[segment_idx[1]]
        segments.append((p1, p2))
    
    # If no segments or triangles, return None
    if not segments or not triangles:
        return None
    
    # Set up the spatial subdivision box
    box = Box()
    
    # Initialize box bounds to the intersection of polyline and surface bounds
    # Get min/max for polyline
    p_min = Vector3D(
        min(min(s[0].x, s[1].x) for s in segments),
        min(min(s[0].y, s[1].y) for s in segments),
        min(min(s[0].z, s[1].z) for s in segments)
    )
    p_max = Vector3D(
        max(max(s[0].x, s[1].x) for s in segments),
        max(max(s[0].y, s[1].y) for s in segments),
        max(max(s[0].z, s[1].z) for s in segments)
    )
    
    # Get min/max for surface
    s_min = Vector3D(
        min(min(t.v1.x, t.v2.x, t.v3.x) for t in triangles),
        min(min(t.v1.y, t.v2.y, t.v3.y) for t in triangles),
        min(min(t.v1.z, t.v2.z, t.v3.z) for t in triangles)
    )
    s_max = Vector3D(
        max(max(t.v1.x, t.v2.x, t.v3.x) for t in triangles),
        max(max(t.v1.y, t.v2.y, t.v3.y) for t in triangles),
        max(max(t.v1.z, t.v2.z, t.v3.z) for t in triangles)
    )
    
    # Set box to intersection of bounding boxes
    box.min.x = max(p_min.x, s_min.x)
    box.min.y = max(p_min.y, s_min.y)
    box.min.z = max(p_min.z, s_min.z)
    box.max.x = min(p_max.x, s_max.x)
    box.max.y = min(p_max.y, s_max.y)
    box.max.z = min(p_max.z, s_max.z)
    
    # Check if there's no overlap in the bounding boxes
    if (box.min.x > box.max.x or 
        box.min.y > box.max.y or 
        box.min.z > box.max.z):
        return None  # No overlap, cannot have intersections
    
    # Add segments and triangles to the box if they intersect
    for segment in segments:
        if box.seg_in_box(segment[0], segment[1]):
            box.N1s.append(segment)
    
    for tri in triangles:
        if box.tri_in_box(tri):
            box.T2s.append(tri)
    
    # If no segments or triangles in the intersection box, return None
    if not box.N1s or not box.T2s:
        return None
    
    # Find intersections between line segments and triangles
    intersection_points = []
    
    # Check each segment against each triangle in the box
    for segment in box.N1s:
        p1, p2 = segment
        
        for triangle in box.T2s:
            intersection = line_triangle_intersection(p1, p2, triangle)
            if intersection:
                # Mark as intersection point
                intersection.type = "INTERSECTION_POINT"
                
                # Check if this point is already in our intersection list (with tolerance)
                is_duplicate = False
                for existing_point in intersection_points:
                    if (existing_point - intersection).length() < 1e-8:
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
    spatial subdivision and skew line transversal.

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
        return []  # No common parent object, cannot form a triple point
    
    # Create a spatial subdivision box for more efficient computation
    box = Box()
    
    # Set box bounds to the intersection of the bounds of both polylines
    # First, compute the bounds of intersection1
    if len(intersection1.points) > 0:
        min1 = Vector3D(
            min(p.x for p in intersection1.points),
            min(p.y for p in intersection1.points),
            min(p.z for p in intersection1.points)
        )
        max1 = Vector3D(
            max(p.x for p in intersection1.points),
            max(p.y for p in intersection1.points),
            max(p.z for p in intersection1.points)
        )
        
        # Compute the bounds of intersection2
        min2 = Vector3D(
            min(p.x for p in intersection2.points),
            min(p.y for p in intersection2.points),
            min(p.z for p in intersection2.points)
        )
        max2 = Vector3D(
            max(p.x for p in intersection2.points),
            max(p.y for p in intersection2.points),
            max(p.z for p in intersection2.points)
        )
        
        # Set the box bounds to the intersection of the two bounds
        box.min.x = max(min1.x, min2.x)
        box.min.y = max(min1.y, min2.y)
        box.min.z = max(min1.z, min2.z)
        box.max.x = min(max1.x, max2.x)
        box.max.y = min(max1.y, max2.y)
        box.max.z = min(max1.z, max2.z)
        
        # Check if there's no overlap in the bounding boxes
        if (box.min.x > box.max.x or 
            box.min.y > box.max.y or 
            box.min.z > box.max.z):
            return []  # No overlap, cannot have triple points
    else:
        return []  # No points in one of the intersections
    
    # Gather segments in the overlap box
    for i in range(len(intersection1.points) - 1):
        p1 = intersection1.points[i]
        p2 = intersection1.points[i + 1]
        if box.seg_in_box(p1, p2):
            box.N1s.append((p1, p2))
    
    for i in range(len(intersection2.points) - 1):
        p1 = intersection2.points[i]
        p2 = intersection2.points[i + 1]
        if box.seg_in_box(p1, p2):
            box.N2s.append((p1, p2))
    
    # No segments in the overlap box
    if not box.N1s or not box.N2s:
        return []
    
    # List to store found triple points
    found_triple_points = []
    
    # Use recursive spatial subdivision to find triple points
    if box.too_much_seg():
        box.split_seg(found_triple_points, intersection1_idx, intersection2_idx)
    else:
        # Direct testing for small number of segments
        for seg1_idx, seg1 in enumerate(box.N1s):
            p1a, p1b = seg1[0], seg1[1]
            for seg2_idx, seg2 in enumerate(box.N2s):
                p2a, p2b = seg2[0], seg2[1]

                # Calculate distance and closest points between segments FIRST
                dist, closest1, closest2 = segment_segment_distance(p1a, p1b, p2a, p2b)

                # Check if distance is within tolerance
                if dist < tolerance:
                    # Calculate triple point as the midpoint
                    tp_point = (closest1 + closest2) * 0.5
                    # Just append the raw point coordinate to the list passed by reference
                    found_triple_points.append(tp_point)

                    # --- REMOVED duplicate check and TriplePoint object creation ---
                    # # Check for duplicates within the accumulating list
                    # is_duplicate = False
                    # for existing_tp in found_triple_points:
                    #     # This check is problematic here, should be done after collecting all points
                    #     # if (existing_tp.point - tp_point).length() < tolerance:
                    #     #     # Merge intersection IDs into the existing TP
                    #     #     existing_tp.add_intersection(i1)
                    #     #     existing_tp.add_intersection(i2)
                    #     #     is_duplicate = True
                    #     #     break
                    #
                    # if not is_duplicate:
                    #     # Create a new TriplePoint object
                    #     # This creation should happen after merging
                    #     # triple_point_obj = TriplePoint(tp_point)
                    #     # triple_point_obj.add_intersection(i1)
                    #     # triple_point_obj.add_intersection(i2)
                    #     # found_triple_points.append(triple_point_obj)
                    # --- END REMOVAL ---

    # Return list of potential coordinate points (Vector3D)
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
                # Calculate distance to the LINE containing the segment, not the clamped point
                # This helps find the correct segment topologically even if the TP is slightly off
                dist_to_line_sq = (tp.point - closest_on_seg).length()**2 # Simplified check: distance to clamped point

                # Find the segment topologically closest to the triple point
                if dist_to_line_sq < min_dist_to_segment:
                     min_dist_to_segment = dist_to_line_sq
                     best_segment_idx = k

            # Insert the point if a best segment was found
            if best_segment_idx != -1:
                 # Insert after the starting point of the best segment found
                 points.insert(best_segment_idx + 1, tp.point)
                 inserted = True
            else:
                 # This case should be rare if points list is not empty
                 print(f"Warning: Could not find any segment for triple point {tp.point} on intersection {intersection_idx}. Appending instead.")
                 points.append(tp.point) # Append as a fallback


def clean_identical_points(points_list: List[Vector3D], tolerance=1e-8) -> List[Vector3D]:
    """Removes duplicate points from a list, preserving order and special types."""
    if not points_list:
        return []
    
    cleaned_list = []
    for point_to_add in points_list:
        if not cleaned_list:
            cleaned_list.append(point_to_add)
            continue

        # Check against the last added point in the cleaned list
        if (point_to_add - cleaned_list[-1]).length() > tolerance:
            cleaned_list.append(point_to_add)
        else:
            # Points are identical or very close
            # Prioritize special types over DEFAULT or if types differ, log it (or define priority)
            if point_to_add.type != "DEFAULT" and cleaned_list[-1].type == "DEFAULT":
                cleaned_list[-1] = point_to_add # Replace default with special
            elif point_to_add.type != "DEFAULT" and cleaned_list[-1].type != "DEFAULT" and point_to_add.type != cleaned_list[-1].type:
                # Both are special but different. For now, keep the one already in cleaned_list.
                # logger.warning(f"CleanIdenticalPoints: Conflicting special types for merged points: {cleaned_list[-1].type} and {point_to_add.type}. Keeping first.")
                pass # Keep the existing special point in cleaned_list
            # If point_to_add.type is DEFAULT and cleaned_list[-1] is special, do nothing.
            # If both are DEFAULT, or both are the same special type, do nothing.
            
    return cleaned_list


def align_intersections_to_convex_hull(surface_idx: int, model):
    """
    Align intersection points to the convex hull of a surface.
    Modified to closely follow C++: inserts points into the hull and cleans it.
    
    Args:
        surface_idx: Index of the surface in model.surfaces
        model: MeshItModel instance (temporary model wrapper from GUI)
    """
    surface = model.surfaces[surface_idx]
    
    if not hasattr(surface, 'convex_hull') or not surface.convex_hull:
        # In the GUI context, convex_hull should be populated from dataset['hull_points']
        # For robustness, if it's missing, we might log and return or try to compute if a method existed.
        # print(f"Warning: Surface {surface_idx} ({getattr(surface, 'name', 'N/A')}) has no convex hull for alignment.")
        return
    
    # Loop over all intersections in the model
    for intersection_idx, intersection in enumerate(model.intersections):
        # Check if the current surface is involved in this intersection
        # In the temp_model, id1 and id2 are indices for surfaces or polylines list
        # We need to check if it's the current surface based on whether it's a surface or polyline
        is_surface1 = not model.is_polyline.get(intersection.id1, True) # Default to polyline if not in map
        is_surface2 = not model.is_polyline.get(intersection.id2, True)

        surface_is_id1 = is_surface1 and intersection.id1 == surface_idx
        surface_is_id2 = is_surface2 and intersection.id2 == surface_idx

        
        
        if not (surface_is_id1 or surface_is_id2):
            continue # This surface is not part of this intersection
            
        if len(intersection.points) < 1: # Need at least one point to align
            continue

        points_to_process = []
        if len(intersection.points) == 1:
            points_to_process = [(0, intersection.points[0])] # Index and point
        elif len(intersection.points) >= 2:
            points_to_process = [(0, intersection.points[0]), (-1, intersection.points[-1])] # Process first and last

        for point_index_in_intersection, intersection_point_obj in points_to_process:
            original_intersection_point = Vector3D(intersection_point_obj.x, intersection_point_obj.y, intersection_point_obj.z, intersection_point_obj.type)
            snapped_to_existing_special = False

            # 1. Check if intersection point is already close to a SPECIAL convex hull point
            for hull_pt_idx, hull_pt_obj in enumerate(surface.convex_hull):
                if hull_pt_obj.type != "DEFAULT": # It's a special point
                    if (original_intersection_point - hull_pt_obj).length() < 1e-8:
                        if point_index_in_intersection == 0:
                            intersection.points[0] = hull_pt_obj
                        else: # -1 for last point
                            intersection.points[-1] = hull_pt_obj
                        snapped_to_existing_special = True
                        break
            if snapped_to_existing_special:
                continue # Move to the next intersection point (or next intersection)

            # 2. If not snapped to special, check projection onto hull segments
            segment_idx = 0
            point_inserted_or_snapped_on_segment = False
            while segment_idx < len(surface.convex_hull) - 1: # Use while for dynamic length of hull list
                p1 = surface.convex_hull[segment_idx]
                p2 = surface.convex_hull[segment_idx + 1]
                
                closest_pt_on_segment = closest_point_on_segment(original_intersection_point, p1, p2)
                dist_to_segment_projection = (original_intersection_point - closest_pt_on_segment).length()

                new_hull_point = Vector3D(closest_pt_on_segment.x, closest_pt_on_segment.y, closest_pt_on_segment.z)
                # Add this logging statement:
                logger.info(f"Adding COMMON_INTERSECTION_CONVEXHULL_POINT to surface {surface_idx} at ({new_hull_point.x:.3f}, {new_hull_point.y:.3f}, {new_hull_point.z:.3f})")
                new_hull_point.type = "COMMON_INTERSECTION_CONVEXHULL_POINT"

                if dist_to_segment_projection < 1e-8: # Projection is on or very close to this segment
                    # Check if this closest_pt_on_segment is an existing hull vertex (p1 or p2 or any other)
                    snapped_to_existing_vertex_on_segment = False
                    for existing_hull_pt_idx, existing_hull_pt_obj in enumerate(surface.convex_hull):
                        if (closest_pt_on_segment - existing_hull_pt_obj).length() < 1e-8:
                           if point_index_in_intersection == 0:
                            intersection.points[0] = new_hull_point
                        else:
                            intersection.points[-1] = new_hull_point
                            snapped_to_existing_vertex_on_segment = True
                            break
                    
                    if not snapped_to_existing_vertex_on_segment:
                        # Not an existing vertex, so insert this new point into the hull
                        new_hull_point = Vector3D(closest_pt_on_segment.x, closest_pt_on_segment.y, closest_pt_on_segment.z, 
                                                point_type="COMMON_INTERSECTION_CONVEXHULL_POINT")
                        
                        # Update intersection point to new hull point
                        if point_index_in_intersection == 0:
                            intersection.points[0] = new_hull_point
                        else:
                            intersection.points[-1] = new_hull_point
                        
                        # Insert the new point into the hull
                        surface.convex_hull.insert(segment_idx + 1, new_hull_point)
                        # Point inserted, break from this segment loop for this endpoint (mimics C++)
                        # The while loop condition (len) will be updated in the next iteration.
                    
                    point_inserted_or_snapped_on_segment = True
                    break # Break from while segment_idx loop (found its place on a segment)
                
                segment_idx += 1
            # End of while loop for segments for this intersection_point_obj
        # End of loop for points_to_process (first/last of an intersection)
    # End of loop for all intersections for this surface

    # After all intersections involving this surface have been processed and potentially modified the hull:
    # Clean up the convex hull for this surface
        # After all intersections involving this surface have been processed and potentially modified the hull:
    # Clean up the convex hull for this surface
    # Replace the existing code in meshit/intersection_utils.py around line 1580-1600 with this:
    if hasattr(surface, 'convex_hull') and surface.convex_hull:
        surface.convex_hull = clean_identical_points(surface.convex_hull)
        
        # Replicate C++ RefineByLength for the convex hull
        # Get the surface's target edge length - use a default if not available
        target_length = getattr(surface, 'size', 20.0)
        
        # Refine the convex hull polygon into segments of target_length
        refined_hull = []
        
        # For each segment in the convex hull
        for i in range(len(surface.convex_hull)):
            p1 = surface.convex_hull[i]
            p2 = surface.convex_hull[(i + 1) % len(surface.convex_hull)]  # Wrap around for last segment
            
            # Always include the first point of the segment
            refined_hull.append(p1)
            
            # Calculate segment length
            segment_length = (p2 - p1).length()
            
            # Calculate how many points to add - C++ style with better spacing
            if segment_length > target_length * 1.2:  # Add 20% buffer to avoid tiny segments
                # Calculate number of segments (not points)
                num_segments = max(1, int(round(segment_length / target_length)))
                segment_vector = (p2 - p1) / num_segments
                
                # Add intermediate points at exact intervals
                for j in range(1, num_segments):
                    # Create point at exact position
                    new_point = Vector3D(
                        p1.x + segment_vector.x * j,
                        p1.y + segment_vector.y * j,
                        p1.z + segment_vector.z * j,
                        point_type="COMMON_INTERSECTION_CONVEXHULL_POINT"
                    )
                    refined_hull.append(new_point)
                    # Log the new point for debugging
                    logging.getLogger('meshit.intersection_utils').info(
                        f"Adding COMMON_INTERSECTION_CONVEXHULL_POINT to surface {surface_idx} at "
                        f"({new_point.x:.3f}, {new_point.y:.3f}, {new_point.z:.3f})"
                    )
        
        # Replace the convex hull with the refined version
        surface.convex_hull = refined_hull
        
        # Clean up again in case the refinement introduced any duplicate points
        surface.convex_hull = clean_identical_points(surface.convex_hull)

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


def cluster_points(points: List[Vector3D], tolerance: float) -> List[List[Vector3D]]:
    """Groups points that are within tolerance of each other."""
    clusters = []
    used = [False] * len(points)
    for i in range(len(points)):
        if used[i]:
            continue
        current_cluster = [points[i]]
        used[i] = True
        # Check subsequent points
        for j in range(i + 1, len(points)):
            if not used[j]:
                # Check distance to any point already in the current cluster
                is_close = False
                for cluster_point in current_cluster:
                     if (points[j] - cluster_point).length() < tolerance:
                          is_close = True
                          break
                if is_close:
                    current_cluster.append(points[j])
                    used[j] = True
        clusters.append(current_cluster)
    return clusters


def calculate_cluster_center(cluster: List[Vector3D]) -> Vector3D:
    """Calculates the average coordinate of points in a cluster."""
    if not cluster:
        return Vector3D() # Should not happen
    sum_vec = Vector3D()
    for p in cluster:
        sum_vec += p
    return sum_vec / len(cluster)


def run_intersection_workflow(model, progress_callback=None, tolerance=1e-5):
    """
    Run the complete intersection workflow, including:
    1. Surface-Surface intersections
    2. Polyline-Surface intersections (if polylines exist)
    3. Calculating and merging Triple Points
    4. Inserting Triple Points into intersection lines

    Args:
        model: MeshItModel instance
        progress_callback: Optional callback function for progress updates
        tolerance: Distance tolerance for finding intersections and merging points

    Returns:
        The updated model instance
    """
    def report_progress(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(message)

    report_progress(">Calculating Surface-Surface Intersections...")
    model.intersections.clear() # Start fresh
    n_surfaces = len(model.surfaces)
    # ... (Surface-Surface intersection calculation using executor) ...
    # --- Assume this part correctly populates model.intersections ---
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures_ss = []
        for s1 in range(n_surfaces - 1):
            for s2 in range(s1 + 1, n_surfaces):
                futures_ss.append(executor.submit(calculate_surface_surface_intersection, s1, s2, model))
        for future in concurrent.futures.as_completed(futures_ss):
            result = future.result()
            if result: model.intersections.append(result)
    report_progress(">...Surface-Surface finished")

    # --- Optional: Polyline-Surface Intersections ---
    if hasattr(model, 'model_polylines') and model.model_polylines:
        report_progress(">Calculating Polyline-Surface Intersections...")
        # ... (Polyline-Surface intersection calculation using executor) ...
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures_ps = []
            for p_idx in range(len(model.model_polylines)):
                for s_idx in range(n_surfaces):
                    futures_ps.append(executor.submit(calculate_polyline_surface_intersection, p_idx, s_idx, model))
            for future in concurrent.futures.as_completed(futures_ps):
                result = future.result()
                if result: model.intersections.append(result)
        report_progress(">...Polyline-Surface finished")

    # --- Calculate Triple Points --- 
    report_progress(">Calculating Intersection Triple Points...")
    model.triple_points.clear()
    potential_tp_coords = [] # Store raw coordinates first
    num_intersections = len(model.intersections)
    
    if num_intersections >= 2:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures_tp = []
            # Submit tasks for each intersection pair
            for i1 in range(num_intersections - 1):
                for i2 in range(i1 + 1, num_intersections):
                    # Pass the tolerance down
                    futures_tp.append(executor.submit(calculate_triple_points, i1, i2, model, tolerance))
            
            # Collect all potential points
            for future in concurrent.futures.as_completed(futures_tp):
                potential_tp_coords.extend(future.result()) # result is now a list of Vector3D

    report_progress(f">  Found {len(potential_tp_coords)} potential triple point candidates.")

    # --- Cluster and Merge Triple Points --- 
    if potential_tp_coords:
        report_progress(">  Clustering and merging triple points...")
        clusters = cluster_points(potential_tp_coords, tolerance)
        final_triple_points = []
        for cluster in clusters:
            if not cluster: continue
            # Calculate the center of the cluster
            center_point = calculate_cluster_center(cluster)
            center_point.type = "TRIPLE_POINT" # Set type
            
            # Create the final TriplePoint object
            final_tp_obj = TriplePoint(center_point)
            
            # Find which original intersections contributed to this cluster 
            # (Requires relating raw points back or re-checking proximity)
            # For simplicity now, we won't store intersection_ids accurately here.
            # A more robust implementation would track origins or re-calculate.
            # We will add dummy intersection IDs for now to maintain structure.
            # TODO: Implement accurate tracking/calculation of involved intersection IDs
            final_tp_obj.add_intersection(-1) # Placeholder ID
            final_tp_obj.add_intersection(-2) # Placeholder ID

            final_triple_points.append(final_tp_obj)
            
        model.triple_points = final_triple_points # Store the final merged points
        report_progress(f">  Resulted in {len(model.triple_points)} final triple points after merging.")
    else:
         report_progress(">  No potential triple points found to cluster.")

    # --- Insert Triple Points into Intersection Lines --- 
    # The insert_triple_points function might still be useful 
    # to snap the final averaged points exactly onto the lines.
    report_progress(">Inserting Triple Points into Intersection Lines...")
    insert_triple_points(model, tolerance)
    report_progress(">...Triple Points finished")

    # Final steps (align, constraints etc. - keep as is for now)
    # ...

    return model


def compute_angle_between_segments(p1, p2, p3):
    """
    Calculate the angle in degrees between two line segments p1-p2 and p2-p3
    
    Args:
        p1, p2, p3: Vector3D points forming two segments
        
    Returns:
        Angle in degrees
    """
    if p1 is None or p2 is None or p3 is None:
        return 0.0
        
    # Create vectors for the two segments
    v1 = Vector3D(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
    v2 = Vector3D(p3.x - p2.x, p3.y - p2.y, p3.z - p2.z)
    
    # Normalize the vectors
    len1 = v1.length()
    len2 = v2.length()
    
    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0
    
    v1 = v1 / len1
    v2 = v2 / len2
    
    # Calculate the dot product
    dot_product = v1.dot(v2)
    
    # Clamp dot product to [-1, 1] to avoid numerical issues
    dot_product = max(-1.0, min(1.0, dot_product))
    
    # Calculate angle in degrees
    angle_rad = math.acos(dot_product)
    angle_deg = angle_rad * 180.0 / math.pi
    
    return angle_deg


import math # Ensure math is imported
# Ensure logger is configured if you use logger.info, e.g.:
# import logging
# logger = logging.getLogger(__name__) # Or your specific logger

# ... (Vector3D, Intersection, compute_angle_between_segments, clean_identical_points - assumed to be present) ...

def refine_intersection_line_by_length(intersection, target_length, min_angle_deg=20.0, uniform_meshing=True):
    """
    Refines an intersection line by ensuring segment lengths are close to the target length.
    Identifies and marks special points based on internal geometric criteria.
    The min_angle_deg from UI is for eventual mesh quality, not directly for point classification here.
    
    Args:
        intersection: Intersection object to refine. Its .points attribute will be modified.
        target_length: Target segment length, determined by calling context.
        min_angle_deg: (Ignored for point classification here) UI hint for mesh cell quality.
        uniform_meshing: If True, strictly adheres to target_length. If False, uses ceil
                         to ensure segments are not longer than target_length.
    Returns:
        The list of refined points (also updates intersection.points directly).
    """
    if not intersection.points or len(intersection.points) < 2:
        return intersection.points if hasattr(intersection, 'points') else []

    # Internal thresholds for point classification on the line based on angle severity
    HC_ANGLE_THRESHOLD = 5.0  # High Curvature
    FE_ANGLE_THRESHOLD = 2.0  # Feature Edge
    SP_ANGLE_THRESHOLD = 0.2  # General Special Point (slight deviation from straight)
    
    logger.info("==================================================")
    logger.info(f"REFINE_LINE: TargetLen={target_length:.3f}, UI_MinAngle(ignored)={min_angle_deg:.1f}, Uniform={uniform_meshing}")
    logger.info(f"Original points: {len(intersection.points)}")
    
    refined_points = []
    original_points = intersection.points # Work with the original list for iteration

    # Add the first point, preserving its type if already set (e.g. COMMON_*, TRIPLE_POINT)
    # If not set, mark as START_POINT.
    p_start = original_points[0]
    if not hasattr(p_start, 'point_type') or p_start.point_type is None:
        p_start.point_type = "START_POINT"
    refined_points.append(p_start)

    for i in range(len(original_points) - 1):
        p1 = original_points[i]      # Start of the current original segment
        p2 = original_points[i+1]    # End of the current original segment

        # The actual segment to divide starts from the last point added to refined_points,
        # which is p1 (or a point very close to it if p1 was merged by clean_identical_points previously,
        # though clean_identical_points is at the end now).
        # For simplicity in this loop, we consider subdividing the original segment p1-p2.
        # The first point of this segment (p1) is already in refined_points (or handled as p_start).

        segment_vec = p2 - p1
        length = segment_vec.length()

        if length < 1e-7: # Effectively a zero-length segment in original list
            num_segments = 1
        elif target_length < 1e-7: # Avoid division by zero if target_length is tiny
             num_segments = 1
        elif uniform_meshing:
            num_segments = max(1, round(length / target_length))
        else: # Non-uniform: ensure segments are not longer than target_length
            num_segments = max(1, math.ceil(length / target_length))
        
        # Add intermediate points for the segment p1 to p2
        for j in range(1, num_segments): # Inserts num_segments - 1 points
            t = j / num_segments
            new_intermediate_point = p1 + segment_vec * t
            new_intermediate_point.point_type = None # Type to be determined in post-processing
            refined_points.append(new_intermediate_point)
        
        # Add the end point of the original segment (p2)
        # Preserve its type if already set (e.g., TRIPLE_POINT)
        if not hasattr(p2, 'point_type'): # Ensure attribute exists
            p2.point_type = None
        # If it's the very last point of the line, its type will be set to END_POINT later
        # if not already something more specific.
        refined_points.append(p2)

    # Clean identical points that might have resulted from adding original p1 and then p2 if num_segments was 1.
    # This pass also handles start/end points if the list is very short.
    if refined_points:
        refined_points = clean_identical_points(refined_points) # Use the helper

    # Ensure Start and End points are correctly typed if not overridden by something more specific
    if refined_points:
        if not refined_points[0].point_type or refined_points[0].point_type == "DEFAULT":
            refined_points[0].point_type = "START_POINT"
        if len(refined_points) > 1: # Only if there's more than one point
            # If last point's type is None, DEFAULT, or was START (in case of 2-pt line that became 1pt after clean)
            if not refined_points[-1].point_type or \
               refined_points[-1].point_type == "DEFAULT" or \
               (refined_points[-1].point_type == "START_POINT" and len(refined_points) == 1) :
                 refined_points[-1].point_type = "END_POINT"


    # --- Post-processing pass for angle-based classification ---
    if len(refined_points) > 2:
        # We iterate using indices on a temporary copy to avoid issues if points were ever merged (though clean handles it now)
        # The actual modification happens on `refined_points`'s point objects.
        points_to_check_angles = list(refined_points)
        for k_idx in range(1, len(points_to_check_angles) - 1):
            p_prev = points_to_check_angles[k_idx-1]
            curr_p = points_to_check_angles[k_idx] # This is the point object from refined_points
            p_next = points_to_check_angles[k_idx+1]
            
            # Only classify/re-classify if not an immutable type like START, END, TRIPLE, COMMON
            # Allow re-classification if it's None, DEFAULT, or a previous angle-based type.
            current_type = curr_p.point_type if hasattr(curr_p, 'point_type') else None
            is_immutable_type = current_type in ["START_POINT", "END_POINT", "TRIPLE_POINT", "COMMON_INTERSECTION_CONVEXHULL_POINT"]
            
            if not is_immutable_type:
                angle = compute_angle_between_segments(p_prev, curr_p, p_next)
                new_angle_type = None
                if angle >= HC_ANGLE_THRESHOLD: new_angle_type = "HIGH_CURVATURE_POINT"
                elif angle >= FE_ANGLE_THRESHOLD: new_angle_type = "FEATURE_EDGE_POINT"
                elif angle >= SP_ANGLE_THRESHOLD: new_angle_type = "SPECIAL_POINT"
                
                if new_angle_type:
                    # Logic to upgrade if already an angle type, or set if None/DEFAULT
                    angle_types_priority = ["SPECIAL_POINT", "FEATURE_EDGE_POINT", "HIGH_CURVATURE_POINT"]
                    current_priority = angle_types_priority.index(current_type) if current_type in angle_types_priority else -1
                    new_priority = angle_types_priority.index(new_angle_type)

                    if new_priority > current_priority:
                        curr_p.point_type = new_angle_type
                elif current_type in ["SPECIAL_POINT", "FEATURE_EDGE_POINT", "HIGH_CURVATURE_POINT"]: 
                    # Was an angle type, but angle no longer qualifies. Reset to None.
                    curr_p.point_type = None 


        # Ensure the last point is always marked as an END_POINT in its type
    # We need to preserve any existing special type but add END_POINT designation
    
    
    # Final pass of cleaning, in case post-processing angles created near-duplicates (unlikely but safe)
    if refined_points:
        intersection.points = clean_identical_points(refined_points)
    else: # Should not happen if original had points
        intersection.points = []
    if len(intersection.points) > 1:
        last_point = intersection.points[-1]
        current_type = last_point.point_type if hasattr(last_point, 'point_type') else None
        
        # If it already has a special type, append END_POINT to it
        if current_type and current_type not in ["END_POINT", "DEFAULT", None]:
            last_point.point_type = f"{current_type}_END_POINT"
        else:
            # If no special type or just DEFAULT, set to END_POINT
            last_point.point_type = "END_POINT"
            
    # Similarly, ensure first point is always marked as START_POINT
    if len(intersection.points) > 0:
        first_point = intersection.points[0]
        current_type = first_point.point_type if hasattr(first_point, 'point_type') else None
        
        # If it already has a special type, append START_POINT to it
        if current_type and current_type not in ["START_POINT", "DEFAULT", None]:
            first_point.point_type = f"{current_type}_START_POINT"
        else:
            # If no special type or just DEFAULT, set to START_POINT
            first_point.point_type = "START_POINT"
    # Log summary of point types from the final list
    _types_count = {}
    for p in intersection.points:
        ptype = p.point_type if hasattr(p, 'point_type') and p.point_type else "NONE_OR_DEFAULT"
        _types_count[ptype] = _types_count.get(ptype, 0) + 1
    
    logger.info(f"REFINEMENT COMPLETE: {len(intersection.points)} final points.")
    logger.info(f"  Final Types: {_types_count}")
    logger.info("==================================================")
    
    return intersection.points