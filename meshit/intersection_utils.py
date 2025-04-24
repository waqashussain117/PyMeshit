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
        self.type = "DEFAULT"  # Point type: DEFAULT, CORNER, INTERSECTION_POINT, TRIPLE_POINT, COMMON_INTERSECTION_CONVEXHULL_POINT
    
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
                    for seg1 in self.Box[b].N1s:
                        for seg2 in self.Box[b].N2s:
                            # Use the skew line transversal calculation
                            tp = calculate_skew_line_transversal(seg1[0], seg1[1], seg2[0], seg2[1])
                            if tp:
                                # Mark as triple point
                                tp.type = "TRIPLE_POINT"
                                
                                # Create triple point
                                triple_point = TriplePoint(tp)
                                triple_point.add_intersection(i1)
                                triple_point.add_intersection(i2)
                                triple_points.append(triple_point)


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
        for seg1 in box.N1s:
            for seg2 in box.N2s:
                tp = calculate_skew_line_transversal(seg1[0], seg1[1], seg2[0], seg2[1])
                if tp:
                    # Mark as triple point
                    tp.type = "TRIPLE_POINT"
                    
                    # Check for duplicates
                    is_duplicate = False
                    for existing_tp in found_triple_points:
                        if (existing_tp.point - tp).length() < tolerance:
                            existing_tp.add_intersection(intersection1_idx)
                            existing_tp.add_intersection(intersection2_idx)
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        triple_point = TriplePoint(tp)
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
    Following the C++ implementation more closely.
    
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
    
    # For each intersection involving this surface
    for intersection in model.intersections:
        if intersection.id1 == surface_idx or intersection.id2 == surface_idx:
            # Check first and last points of the intersection line
            if len(intersection.points) < 2:
                continue
            
            # Check the first point
            first_point = intersection.points[0]
            last_point = intersection.points[-1]
            
            # For the first point: Check if it's close to a special point in the convex hull
            close_to_special_first = False
            close_to_special_last = False
            
            for i, hull_point in enumerate(surface.convex_hull):
                if hull_point.type != "DEFAULT":  # It's a special point (corner, etc.)
                    # Check first point proximity
                    if (first_point - hull_point).length() < 1e-8:
                        # Merge first point to hull special point
                        intersection.points[0] = hull_point
                        close_to_special_first = True
                    
                    # Check last point proximity
                    if (last_point - hull_point).length() < 1e-8:
                        # Merge last point to hull special point
                        intersection.points[-1] = hull_point
                        close_to_special_last = True
            
            # If not close to a special point, check if it's close to a segment in the convex hull
            if not close_to_special_first:
                closest_on_hull_first = None
                min_dist_first = float('inf')
                
                for i in range(len(surface.convex_hull) - 1):
                    p1 = surface.convex_hull[i]
                    p2 = surface.convex_hull[i+1]
                    
                    # Find closest point on this segment
                    closest = closest_point_on_segment(first_point, p1, p2)
                    dist = (first_point - closest).length()
                    
                    if dist < min_dist_first:
                        min_dist_first = dist
                        closest_on_hull_first = closest
                
                # If close enough to a segment, snap to it
                if min_dist_first < 1e-8 and closest_on_hull_first:
                    # If the closest point is not directly on a vertex, add it to the hull as a special point
                    for hull_point in surface.convex_hull:
                        if (closest_on_hull_first - hull_point).length() < 1e-8:
                            intersection.points[0] = hull_point
                            break
                    else:
                        # Point not already in hull, add it with special type
                        closest_on_hull_first.type = "COMMON_INTERSECTION_CONVEXHULL_POINT"
                        # In real implementation, would add it to the hull here
                        # surface.convex_hull.append(closest_on_hull_first)
                        intersection.points[0] = closest_on_hull_first
            
            # Do the same for the last point if it wasn't close to a special hull point
            if not close_to_special_last:
                closest_on_hull_last = None
                min_dist_last = float('inf')
                
                for i in range(len(surface.convex_hull) - 1):
                    p1 = surface.convex_hull[i]
                    p2 = surface.convex_hull[i+1]
                    
                    # Find closest point on this segment
                    closest = closest_point_on_segment(last_point, p1, p2)
                    dist = (last_point - closest).length()
                    
                    if dist < min_dist_last:
                        min_dist_last = dist
                        closest_on_hull_last = closest
                
                # If close enough to a segment, snap to it
                if min_dist_last < 1e-8 and closest_on_hull_last:
                    # If the closest point is not directly on a vertex, add it to the hull as a special point
                    for hull_point in surface.convex_hull:
                        if (closest_on_hull_last - hull_point).length() < 1e-8:
                            intersection.points[-1] = hull_point
                            break
                    else:
                        # Point not already in hull, add it with special type
                        closest_on_hull_last.type = "COMMON_INTERSECTION_CONVEXHULL_POINT"
                        # In real implementation, would add it to the hull here
                        # surface.convex_hull.append(closest_on_hull_last)
                        intersection.points[-1] = closest_on_hull_last


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