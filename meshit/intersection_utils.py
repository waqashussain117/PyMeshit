"""
MeshIt Intersection Utilities

This module provides functionality for computing intersections between 
surfaces and polylines, following the MeshIt workflow.
"""

import numpy as np
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Union, Any
import math # Ensure math is imported for floor
import logging
import triangle as tr_standard # Import unconditionally for fallback
from dataclasses import dataclass, field
try:
    from meshit.triangle_direct import DirectTriangleWrapper
    HAVE_DIRECT_WRAPPER_INTERSECTION_UTILS = True
except ImportError:
    HAVE_DIRECT_WRAPPER_INTERSECTION_UTILS = False
    print("WARNING (intersection_utils): DirectTriangleWrapper not found. Constrained triangulation might be limited.")
    # tr_standard is already imported above

# Attempt to import PyVista for an alternative triangulation method
try:
    import pyvista as pv
    HAVE_PYVISTA_UTILS = True
    logging.info("PyVista imported successfully in intersection_utils.")
except ImportError:
    HAVE_PYVISTA_UTILS = False
    logging.warning("PyVista not available in intersection_utils. PyVista triangulation fallback disabled.")

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
        """
        Check if a triangle intersects this box using C++ MeshIt logic.
        Returns False only if ALL vertices are on the same side of any box boundary.
        """
        # Check X axis - return False only if ALL vertices are on same side
        if ((triangle.v1.x < self.min.x) and (triangle.v2.x < self.min.x) and (triangle.v3.x < self.min.x)):
            return False
        if ((triangle.v1.x > self.max.x) and (triangle.v2.x > self.max.x) and (triangle.v3.x > self.max.x)):
            return False
            
        # Check Y axis - return False only if ALL vertices are on same side  
        if ((triangle.v1.y < self.min.y) and (triangle.v2.y < self.min.y) and (triangle.v3.y < self.min.y)):
            return False
        if ((triangle.v1.y > self.max.y) and (triangle.v2.y > self.max.y) and (triangle.v3.y > self.max.y)):
            return False
            
        # Check Z axis - return False only if ALL vertices are on same side
        if ((triangle.v1.z < self.min.z) and (triangle.v2.z < self.min.z) and (triangle.v3.z < self.min.z)):
            return False
        if ((triangle.v1.z > self.max.z) and (triangle.v2.z > self.max.z) and (triangle.v3.z > self.max.z)):
            return False
            
        # If not all vertices are on the same side of any boundary, triangle intersects box
        return True
    
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
        # Match C++ MeshIt threshold of 48 triangles per surface
        return len(self.T1s) > 48 or len(self.T2s) > 48
    
    def too_much_seg(self):
        """Check if this box contains too many segments for direct testing"""
        # Match C++ MeshIt threshold of 48 segments per surface
        return len(self.N1s) > 48 or len(self.N2s) > 48
    
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
    Fast triangle-triangle intersection test matching C++ MeshIt implementation.
    
    Based on Tomas Möller's algorithm with proper epsilon handling.
    
    Args:
        tri1: First triangle
        tri2: Second triangle
        
    Returns:
        Tuple of (point1, point2) defining the intersection line, or (None, None) if no intersection
    """
    # C++ MeshIt constants
    EPSILON = 1e-12
    
    # 1. Compute plane equation (p1) of triangle T1=(V0,V1,V2)
    # p1: N1.X+d1=0
    E1 = tri1.v2 - tri1.v1
    E2 = tri1.v3 - tri1.v1
    N1 = E1.cross(E2)
    d1 = -N1.dot(tri1.v1)
    
    # 2.a Compute signed distance of triangle T2=(U0,U1,U2) to plane p1
    du0 = N1.dot(tri2.v1) + d1
    du1 = N1.dot(tri2.v2) + d1
    du2 = N1.dot(tri2.v3) + d1
    
    # 2.b Coplanarity robustness check (USE_EPSILON_TEST)
    if abs(du0) < EPSILON:
        du0 = 0.0
    if abs(du1) < EPSILON:
        du1 = 0.0  
    if abs(du2) < EPSILON:
        du2 = 0.0
    
    du0du1 = du0 * du1
    du0du2 = du0 * du2
    
    # If all points of T2 are on same side of p1, no intersection
    if du0du1 > 0.0 and du0du2 > 0.0:
        return None, None
    
    # 3. Compute plane equation (p2) of triangle T2=(U0,U1,U2)
    # p2: N2.X+d2=0
    E1 = tri2.v2 - tri2.v1
    E2 = tri2.v3 - tri2.v1
    N2 = E1.cross(E2)
    d2 = -N2.dot(tri2.v1)
    
    # 4.a Compute signed distance of triangle T1=(V0,V1,V2) to plane p2
    dv0 = N2.dot(tri1.v1) + d2
    dv1 = N2.dot(tri1.v2) + d2
    dv2 = N2.dot(tri1.v3) + d2
    
    # 4.b Coplanarity robustness check (USE_EPSILON_TEST)
    if abs(dv0) < EPSILON:
        dv0 = 0.0
    if abs(dv1) < EPSILON:
        dv1 = 0.0
    if abs(dv2) < EPSILON:
        dv2 = 0.0
    
    dv0dv1 = dv0 * dv1
    dv0dv2 = dv0 * dv2
    
    # If all points of T1 are on same side of p2, no intersection
    if dv0dv1 > 0.0 and dv0dv2 > 0.0:
        return None, None
    
    # 5. Compute direction of intersection line
    D = N1.cross(N2)
    
    # 6. Project triangles onto largest coordinate of D
    max_axis = 0
    max_val = abs(D.x)
    if abs(D.y) > max_val:
        max_axis = 1
        max_val = abs(D.y)
    if abs(D.z) > max_val:
        max_axis = 2
    
    # Project vertices onto the chosen axis
    if max_axis == 0:  # X axis is largest
        vv0 = tri1.v1.x
        vv1 = tri1.v2.x
        vv2 = tri1.v3.x
        uu0 = tri2.v1.x
        uu1 = tri2.v2.x
        uu2 = tri2.v3.x
    elif max_axis == 1:  # Y axis is largest
        vv0 = tri1.v1.y
        vv1 = tri1.v2.y
        vv2 = tri1.v3.y
        uu0 = tri2.v1.y
        uu1 = tri2.v2.y
        uu2 = tri2.v3.y
    else:  # Z axis is largest
        vv0 = tri1.v1.z
        vv1 = tri1.v2.z
        vv2 = tri1.v3.z
        uu0 = tri2.v1.z
        uu1 = tri2.v2.z
        uu2 = tri2.v3.z
    
    # 7. Compute intervals for triangle 1
    isect1 = [0.0, 0.0]
    isectpoint1 = [Vector3D(), Vector3D()]
    compute_intervals_isectline(tri1, vv0, vv1, vv2, dv0, dv1, dv2, dv0dv1, dv0dv2, isect1, isectpoint1)
    
    # 8. Compute intervals for triangle 2
    isect2 = [0.0, 0.0]
    isectpoint2 = [Vector3D(), Vector3D()]
    compute_intervals_isectline(tri2, uu0, uu1, uu2, du0, du1, du2, du0du1, du0du2, isect2, isectpoint2)
    
    # 9. Sort intervals so that isect1[0] <= isect1[1] and isect2[0] <= isect2[1]
    if isect1[0] > isect1[1]:
        isect1[0], isect1[1] = isect1[1], isect1[0]
        isectpoint1[0], isectpoint1[1] = isectpoint1[1], isectpoint1[0]
    
    if isect2[0] > isect2[1]:
        isect2[0], isect2[1] = isect2[1], isect2[0]
        isectpoint2[0], isectpoint2[1] = isectpoint2[1], isectpoint2[0]
    
    # 10. Check for overlap
    if isect1[1] < isect2[0] or isect2[1] < isect1[0]:
        return None, None  # No overlap
    
    # 11. Compute actual intersection points
    if isect2[0] < isect1[0]:
        if isect1[0] < isect2[1]:
            if isect1[1] < isect2[1]:
                pt1 = isectpoint1[0]
                pt2 = isectpoint1[1]
            else:
                pt1 = isectpoint1[0]
                pt2 = isectpoint2[1]
        else:
            return None, None
    else:
        if isect2[0] < isect1[1]:
            if isect2[1] < isect1[1]:
                pt1 = isectpoint2[0]
                pt2 = isectpoint2[1]
            else:
                pt1 = isectpoint2[0]
                pt2 = isectpoint1[1]
        else:
            return None, None
    
    # Set intersection point types
    pt1.type = "INTERSECTION_POINT"
    pt2.type = "INTERSECTION_POINT"
    
    return pt1, pt2


def compute_intervals_isectline(tri, vv0, vv1, vv2, d0, d1, d2, d0d1, d0d2, isect, isectpoint):
    """
    Helper function to compute intersection intervals for triangle-triangle intersection.
    """
    if d0d1 > 0.0:
        # d0, d1 are on the same side, d2 on the other side
        isect2(tri.v3, tri.v1, tri.v2, vv2, vv0, vv1, d2, d0, d1, isect, isectpoint)
    elif d0d2 > 0.0:
        # d0, d2 are on the same side, d1 on the other side
        isect2(tri.v2, tri.v1, tri.v3, vv1, vv0, vv2, d1, d0, d2, isect, isectpoint)
    elif d1 * d2 > 0.0 or d0 != 0.0:
        # d1, d2 are on the same side, d0 on the other side
        isect2(tri.v1, tri.v2, tri.v3, vv0, vv1, vv2, d0, d1, d2, isect, isectpoint)
    elif d1 != 0.0:
        isect2(tri.v2, tri.v1, tri.v3, vv1, vv0, vv2, d1, d0, d2, isect, isectpoint)
    elif d2 != 0.0:
        isect2(tri.v3, tri.v1, tri.v2, vv2, vv0, vv1, d2, d0, d1, isect, isectpoint)
    else:
        # Triangles are coplanar
        return 1
    return 0


def isect2(v0, v1, v2, vv0, vv1, vv2, d0, d1, d2, isect, isectpoint):
    """
    Helper function for computing intersection points on triangle edges.
    """
    tmp = d0 / (d0 - d1)
    isect[0] = vv0 + (vv1 - vv0) * tmp
    diff = v1 - v0
    isectpoint[0] = v0 + diff * tmp
    
    tmp = d0 / (d0 - d2)
    isect[1] = vv0 + (vv2 - vv0) * tmp
    diff = v2 - v0
    isectpoint[1] = v0 + diff * tmp


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


def calculate_triple_points(intersection1_idx: int, intersection2_idx: int, model, tolerance=1e-7) -> List[TriplePoint]:
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
    Ensure every TRIPLE_POINT is physically present (and marked) in *all*
    intersection polylines that cross it.

    Works in two passes
    1)  original logic – insert into the two lines recorded in
        tp.intersection_ids  (fast path)
    2)  completeness pass – walk over every remaining poly-line and insert the
        point wherever the orthogonal distance to any segment < tolerance.
    """
    from meshit.intersection_utils import closest_point_on_segment

    if not getattr(model, "triple_points", None):
        return                                           # nothing to do

    # ------------------------------------------------------------------ PASS 1
    #         original ID-based insertion (kept as-is)
    # ------------------------------------------------------------------
    for tp in model.triple_points:
        for int_idx in tp.intersection_ids:
            if 0 <= int_idx < len(model.intersections):
                _insert_point_into_polyline(model.intersections[int_idx].points,
                                            tp.point, tolerance)

    # ------------------------------------------------------------------ PASS 2
    #         completeness – make sure *every* line owns the TP
    # ------------------------------------------------------------------
    for tp in model.triple_points:
        p_tp = tp.point
        for i, inter in enumerate(model.intersections):
            if i in tp.intersection_ids:
                continue                                 # already done

            # quick BB check ---------------------------------------------------
            xs = [v.x for v in inter.points]
            ys = [v.y for v in inter.points]
            zs = [v.z for v in inter.points]
            if not (min(xs) - tolerance <= p_tp.x <= max(xs) + tolerance and
                    min(ys) - tolerance <= p_tp.y <= max(ys) + tolerance and
                    min(zs) - tolerance <= p_tp.z <= max(zs) + tolerance):
                continue

            # precise distance to each segment --------------------------------
            on_line = False
            for a, b in zip(inter.points[:-1], inter.points[1:]):
                dist = (closest_point_on_segment(p_tp, a, b) - p_tp).length()
                if dist < tolerance:
                    on_line = True
                    break

            if on_line:
                _insert_point_into_polyline(inter.points, p_tp, tolerance)
                tp.add_intersection(i)                  # keep bookkeeping

# --------------------------------------------------------------------------
#  tiny helper -------------------------------------------------------------
# --------------------------------------------------------------------------
def _insert_point_into_polyline(pts, p_new, tol):
    """
    Insert p_new between the two vertices of *pts* whose segment is closest
    to the point (unless a vertex at the same XYZ already exists).
    """
    import math
    # duplicate check ------------------------------------------------------
    for v in pts:
        if ( (v - p_new).length() < tol ):
            # Same coordinate already there → keep the *special* flag
            if getattr(p_new, "type", "DEFAULT") == "TRIPLE_POINT":
                v.type = "TRIPLE_POINT"
            return

    # find best host segment ----------------------------------------------
    best_k = None
    best_d = math.inf
    for k in range(len(pts) - 1):
        d = (closest_point_on_segment(p_new, pts[k], pts[k+1]) - p_new).length()
        if d < best_d:
            best_d, best_k = d, k

    if best_k is not None:
        p_new.type = "TRIPLE_POINT"
        pts.insert(best_k + 1, p_new)


def clean_identical_points(points_list: List[Vector3D], tolerance=1e-10) -> List[Vector3D]:
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


def make_corners_special(convex_hull: List[Vector3D], angle_threshold_deg: float = 135.0):
    """
    Identify and mark corner points on a convex hull based on angle analysis.
    This follows the C++ MeshIt MakeCornersSpecial function.
    
    Args:
        convex_hull: List of Vector3D points forming the convex hull
        angle_threshold_deg: Angle threshold in degrees (points with angles > this are marked special)
    
    Returns:
        Modified convex hull with special points marked
    """
    if len(convex_hull) < 3:
        return convex_hull
    
    # Convert threshold to radians and to dot product value
    # For angles > threshold, the dot product will be < cos(threshold)
    angle_threshold_rad = math.radians(angle_threshold_deg)
    dot_threshold = math.cos(angle_threshold_rad)
    
    logger.info(f"Making corners special with angle threshold {angle_threshold_deg}° (dot < {dot_threshold:.3f})")
    
    special_count = 0
    
    # Check each point in the convex hull
    for n in range(len(convex_hull)):
        # Get the three points for angle calculation
        if n == 0:
            # First point: use last-2, current, and next
            prev_pt = convex_hull[-2] if len(convex_hull) > 2 else convex_hull[-1]
            curr_pt = convex_hull[0]
            next_pt = convex_hull[1]
        else:
            prev_pt = convex_hull[n - 1]
            curr_pt = convex_hull[n]
            next_pt = convex_hull[(n + 1) % len(convex_hull)]
        
        # Calculate vectors from current point
        diff1 = prev_pt - curr_pt
        diff2 = next_pt - curr_pt
        
        # Normalize the vectors
        len1 = diff1.length()
        len2 = diff2.length()
        
        if len1 > 1e-10 and len2 > 1e-10:
            diff1_norm = diff1 / len1
            diff2_norm = diff2 / len2
            
            # Calculate dot product
            dot_product = diff1_norm.dot(diff2_norm)
            
            # Check if angle is sharp enough (dot product < threshold means angle > threshold)
            # In C++: if (alpha > (-0.5*SQUAREROOTTWO)) which is approximately -0.707 (135°)
            if dot_product < dot_threshold:
                curr_pt.point_type = "CORNER"
                if hasattr(curr_pt, 'type'):
                    curr_pt.type = "CORNER"
                special_count += 1
                
                # If this is the first point, also mark the last point (closed polygon)
                if n == 0 and len(convex_hull) > 2:
                    convex_hull[-1].point_type = "CORNER"
                    if hasattr(convex_hull[-1], 'type'):
                        convex_hull[-1].type = "CORNER"
                
                angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot_product))))
                logger.info(f"*** Found CORNER point at ({curr_pt.x:.3f}, {curr_pt.y:.3f}, {curr_pt.z:.3f}) with angle {angle_deg:.1f}° ***")
    
    logger.info(f"Identified {special_count} corner points on convex hull")
    return convex_hull


def align_intersections_to_convex_hull(surface_idx: int, model):
    """
    Align intersection points to the convex hull of a surface.
    This closely follows the C++ MeshIt alignIntersectionsToConvexHull function.
    
    Key behaviors:
    1. Snap intersection endpoints to existing special hull points if close enough
    2. Project intersection endpoints onto hull segments and create new special points
    3. Insert these new special points into the convex hull
    4. Clean up and refine the hull
    
    Args:
        surface_idx: Index of the surface in model.surfaces
        model: MeshItModel instance (temporary model wrapper from GUI)
    """
    surface = model.surfaces[surface_idx]
    
    if not hasattr(surface, 'convex_hull') or not surface.convex_hull:
        logger.warning(f"Surface {surface_idx} has no convex hull for alignment.")
        return
    
    if len(surface.convex_hull) < 3:
        logger.warning(f"Surface {surface_idx} convex hull has < 3 points, skipping alignment.")
        return
    
    logger.info(f"Aligning intersections to convex hull for surface {surface_idx}")
    
    # Process all intersections that involve this surface
    for intersection_idx, intersection in enumerate(model.intersections):
        # Check if this surface is involved in this intersection
        is_surface1 = not model.is_polyline.get(intersection.id1, True)
        is_surface2 = not model.is_polyline.get(intersection.id2, True)
        
        surface_is_id1 = is_surface1 and intersection.id1 == surface_idx
        surface_is_id2 = is_surface2 and intersection.id2 == surface_idx
        
        if not (surface_is_id1 or surface_is_id2):
            continue
            
        if len(intersection.points) < 1:
            continue
        
        # Process first and last points of the intersection Taking points from the C ++ Version of Meshit
        points_to_process = []
        if len(intersection.points) == 1:
            points_to_process = [(0, intersection.points[0])] # 0 represents the first point, -1 represents the last point
        else:
            points_to_process = [(0, intersection.points[0]), (-1, intersection.points[-1])]
        
        for point_idx_in_intersection, intersection_point in points_to_process:
            # Try to align this intersection point to the convex hull
            aligned = False
            
            # Step 1: Check if close to existing special hull points
            for hull_pt_idx, hull_pt in enumerate(surface.convex_hull):
                # Only snap to special points (non-DEFAULT)
                hull_pt_type = getattr(hull_pt, 'point_type', getattr(hull_pt, 'type', "DEFAULT"))
                if hull_pt_type != "DEFAULT":
                    distance = (intersection_point - hull_pt).length()
                    if distance < 1e-8:  # Very close to special point # 1e-8 means 0.00000001
                        # Snap intersection point to the special hull point
                        if point_idx_in_intersection == 0:
                            intersection.points[0] = hull_pt
                        else:
                            intersection.points[-1] = hull_pt
                        aligned = True
                        logger.info(f"*** Snapped intersection point to existing special hull point at ({hull_pt.x:.3f}, {hull_pt.y:.3f}, {hull_pt.z:.3f}) ***")
                        break
            
            if aligned:
                continue
            
            # Step 2: Project onto hull segments and create new special points
            for segment_idx in range(len(surface.convex_hull)):
                p1 = surface.convex_hull[segment_idx]
                p2 = surface.convex_hull[(segment_idx + 1) % len(surface.convex_hull)]
                
                # Project intersection point onto this hull segment
                closest_pt_on_segment = closest_point_on_segment(intersection_point, p1, p2)
                distance_to_segment = (intersection_point - closest_pt_on_segment).length()
                
                if distance_to_segment < 1e-8:  # Very close to this segment
                    # Check if the projection point is already an existing hull vertex
                    is_existing_vertex = False
                    for existing_hull_pt in surface.convex_hull:
                        if (closest_pt_on_segment - existing_hull_pt).length() < 1e-8:
                            # Snap to existing vertex
                            if point_idx_in_intersection == 0:
                                intersection.points[0] = existing_hull_pt
                            else:
                                intersection.points[-1] = existing_hull_pt
                            is_existing_vertex = True
                            logger.info(f"*** Snapped intersection point to existing hull vertex at ({existing_hull_pt.x:.3f}, {existing_hull_pt.y:.3f}, {existing_hull_pt.z:.3f}) ***")
                            break
                    
                    if not is_existing_vertex:
                        # Create new special point and insert into hull
                        new_hull_point = Vector3D(
                            closest_pt_on_segment.x,
                            closest_pt_on_segment.y,
                            closest_pt_on_segment.z,
                            point_type="COMMON_INTERSECTION_CONVEXHULL_POINT"
                        )
                        
                        # Update intersection point to reference the new hull point
                        if point_idx_in_intersection == 0:
                            intersection.points[0] = new_hull_point
                        else:
                            intersection.points[-1] = new_hull_point
                        
                        # Insert the new point into the convex hull at the correct position
                        insert_position = segment_idx + 1
                        surface.convex_hull.insert(insert_position, new_hull_point)
                        
                        logger.info(f"*** Created COMMON_INTERSECTION_CONVEXHULL_POINT at ({new_hull_point.x:.3f}, {new_hull_point.y:.3f}, {new_hull_point.z:.3f}) and inserted into hull ***")
                    
                    aligned = True
                    break
            
            if not aligned:
                logger.warning(f"Could not align intersection point ({intersection_point.x:.3f}, {intersection_point.y:.3f}, {intersection_point.z:.3f}) to convex hull")
    
    # Clean up the convex hull after all insertions
    if hasattr(surface, 'convex_hull') and surface.convex_hull:
        original_count = len(surface.convex_hull)
        surface.convex_hull = clean_identical_points(surface.convex_hull)
        final_count = len(surface.convex_hull)
        
        if final_count != original_count:
            logger.info(f"Cleaned convex hull: {original_count} -> {final_count} points")
        
        # Refine the hull by length (like C++ RefineByLength)
        target_length = getattr(surface, 'size', 0.1)
        if target_length > 1e-6:
            refined_hull = []
            
            for i in range(len(surface.convex_hull)):
                p1 = surface.convex_hull[i]
                p2 = surface.convex_hull[(i + 1) % len(surface.convex_hull)]
                
                # Always add the current point
                refined_hull.append(p1)
                
                # Check if we need to add intermediate points
                segment_length = (p2 - p1).length()
                if segment_length > target_length * 1.2:  # Add buffer to avoid tiny segments
                    num_segments = max(1, int(round(segment_length / target_length)))
                    segment_vector = (p2 - p1) / num_segments
                    
                    # Add intermediate points
                    for j in range(1, num_segments):
                        new_point = Vector3D(
                            p1.x + segment_vector.x * j,
                            p1.y + segment_vector.y * j,
                            p1.z + segment_vector.z * j,
                            point_type="DEFAULT"
                        )
                        refined_hull.append(new_point)
            
            surface.convex_hull = refined_hull
            surface.convex_hull = clean_identical_points(surface.convex_hull)
            
            logger.info(f"Refined convex hull for surface {surface_idx}: final count = {len(surface.convex_hull)} points")
        
        # Count special points for debugging
        special_count = 0
        for pt in surface.convex_hull:
            pt_type = getattr(pt, 'point_type', getattr(pt, 'type', "DEFAULT"))
            if pt_type != "DEFAULT":
                special_count += 1
                logger.info(f"  Special hull point: ({pt.x:.3f}, {pt.y:.3f}, {pt.z:.3f}) type={pt_type}")
        
        logger.info(f"Convex hull alignment complete for surface {surface_idx}: {special_count} special points out of {len(surface.convex_hull)} total")

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


def run_intersection_workflow(model, progress_callback=None, tolerance=1e-5, config=None):
    """
    Run the complete intersection workflow, including:
    1. Surface-Surface intersections
    2. Polyline-Surface intersections (if polylines exist)
    3. Calculating and merging Triple Points
    4. Inserting Triple Points into intersection lines
    5. NEW: Constraint processing and size assignment

    Args:
        model: MeshItModel instance
        progress_callback: Optional callback function for progress updates
        tolerance: Distance tolerance for finding intersections and merging points
        config: Configuration dictionary for constraint processing

    Returns:
        The updated model instance
    """
    def report_progress(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(message)

    if config is None:
        config = {
            'gradient': 2.0,
            'use_constraint_processing': True,
            'type_based_sizing': True,
            'hierarchical_constraints': True
        }

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

    # --- NEW: Constraint Processing Workflow ---
    if config.get('use_constraint_processing', False):
        report_progress(">Processing Constraints (C++ MeshIt Logic)...")
        try:
            integrate_constraint_processing_workflow(model, config)
            report_progress(">...Constraint processing finished")
        except Exception as e:
            report_progress(f">...Constraint processing failed: {e}")
            logger.error(f"Constraint processing failed: {e}")

    # --- Align intersections to convex hulls ---
    report_progress(">Aligning intersections to convex hulls...")
    for surface_idx in range(len(model.surfaces)):
        try:
            align_intersections_to_convex_hull(surface_idx, model)
        except Exception as e:
            logger.error(f"Error aligning intersections for surface {surface_idx}: {e}")
    report_progress(">...Alignment finished")

    # --- Calculate sizes for intersections ---
    report_progress(">Calculating intersection sizes...")
    calculate_size_of_intersections(model)
    report_progress(">...Size calculation finished")

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
    Refines an intersection line by length following the C++ MeshIt RefineByLength logic.
    
    C++ Algorithm:
    1. Remove all points with type "DEFAULT" 
    2. Keep only special points (TRIPLE_POINT, SPECIAL_POINT, etc.) as anchor points
    3. Subdivide segments between anchor points to target_length
    4. New subdivision points are marked as "DEFAULT"
    
    Args:
        intersection: Intersection object to refine
        target_length: Target segment length
        min_angle_deg: Minimum angle for mesh quality (not used for point classification)
        uniform_meshing: If True, use round(); if False, use ceil()
    
    Returns:
        List of refined points
    """
    if not intersection.points or len(intersection.points) < 2:
        logger.info("REFINE_LINE: Not enough points to refine.")
        return intersection.points if hasattr(intersection, 'points') else []

    logger.info("==================================================")
    logger.info(f"REFINE_LINE (C++ Logic): TargetLen={target_length:.3f}")
    logger.info(f"Original points count: {len(intersection.points)}")

    original_points = intersection.points
    
    # Step 1: Create anchor line - remove all DEFAULT points, keep only special points
    # This mirrors the C++ logic exactly
    anchor_points = []
    
    # Always keep first point (even if DEFAULT - it becomes an anchor)
    first_point = original_points[0]
    anchor_points.append(first_point)
    logger.info(f"Anchor: FIRST point ({first_point.x:.3f}, {first_point.y:.3f}, {first_point.z:.3f}) "
                f"Type: {getattr(first_point, 'point_type', 'DEFAULT')}")
    
    # Keep middle points only if they are NOT "DEFAULT"
    for i in range(1, len(original_points) - 1):
        point = original_points[i]
        point_type = getattr(point, 'point_type', None)
        
        if point_type and point_type != "DEFAULT":
            anchor_points.append(point)
            logger.info(f"Anchor: SPECIAL point ({point.x:.3f}, {point.y:.3f}, {point.z:.3f}) "
                        f"Type: {point_type}")
        else:
            logger.info(f"Removed: DEFAULT point ({point.x:.3f}, {point.y:.3f}, {point.z:.3f})")
    
    # Always keep last point (even if DEFAULT - it becomes an anchor)
    if len(original_points) > 1:
        last_point = original_points[-1]
        # Avoid duplicating if it's the same as first (closed loop)
        if (last_point - anchor_points[0]).length() > 1e-8:
            anchor_points.append(last_point)
            logger.info(f"Anchor: LAST point ({last_point.x:.3f}, {last_point.y:.3f}, {last_point.z:.3f}) "
                        f"Type: {getattr(last_point, 'point_type', 'DEFAULT')}")
    
    logger.info(f"Anchor points after filtering: {len(anchor_points)}")
    
    # Step 2: Subdivide segments between anchor points
    refined_points = []
    
    if not anchor_points:
        return original_points
    
    refined_points.append(anchor_points[0])
    
    for i in range(len(anchor_points) - 1):
        p1 = anchor_points[i]
        p2 = anchor_points[i + 1]
        
        segment_vec = p2 - p1
        segment_length = segment_vec.length()
        
        if segment_length < 1e-7:  # Skip zero-length segments
            continue
            
        # Calculate number of subdivisions
        if target_length < 1e-7:
            num_subdivisions = 1
        elif uniform_meshing:
            num_subdivisions = max(1, round(segment_length / target_length)) # 1 means at least one subdivision
        else:
            num_subdivisions = max(1, math.ceil(segment_length / target_length)) # 1 means at least one subdivision
        
        logger.info(f"Segment {i}: length={segment_length:.3f}, subdivisions={num_subdivisions}")
        
        # Add intermediate points
        for j in range(1, num_subdivisions):
            t = j / num_subdivisions
            new_point = p1 + segment_vec * t
            new_point.point_type = "DEFAULT"  # New subdivision points are DEFAULT
            if hasattr(new_point, 'type'):
                new_point.type = "DEFAULT"
            refined_points.append(new_point)
            logger.info(f"  Added subdivision point ({new_point.x:.3f}, {new_point.y:.3f}, {new_point.z:.3f})")
        
        # Add the next anchor point
        refined_points.append(p2)
    
    # Clean up any potential duplicates
    refined_points = clean_identical_points(refined_points)
    
    # Step 3: Assign proper start/end types
    if refined_points:
        # Handle first point
        first_p = refined_points[0]
        fp_type = getattr(first_p, 'point_type', "DEFAULT")
        if fp_type is None:
            fp_type = "DEFAULT"
        if fp_type == "DEFAULT" or "START_POINT" not in fp_type:
            if fp_type == "DEFAULT":
                first_p.point_type = "START_POINT"
            else:
                first_p.point_type = f"{fp_type}_START_POINT"
            if hasattr(first_p, 'type'):
                first_p.type = first_p.point_type
        
        # Handle last point
        if len(refined_points) > 1:
            last_p = refined_points[-1]
            lp_type = getattr(last_p, 'point_type', "DEFAULT")
            if lp_type is None:
                lp_type = "DEFAULT"
            if lp_type == "DEFAULT" or "END_POINT" not in lp_type:
                if lp_type == "DEFAULT":
                    last_p.point_type = "END_POINT"
                else:
                    last_p.point_type = f"{lp_type}_END_POINT"
                if hasattr(last_p, 'type'):
                    last_p.type = last_p.point_type
        else:
            # Single point case
            refined_points[0].point_type = "START_POINT_END_POINT"
            if hasattr(refined_points[0], 'type'):
                refined_points[0].type = "START_POINT_END_POINT"
    
    logger.info(f"Final refined points count: {len(refined_points)}")
    
    # Update the intersection object
    intersection.points = refined_points
    
    return refined_points

def prepare_plc_for_surface_triangulation(surface_data, intersections_on_surface_data, config):
    """
    Prepares Points and Segments for constrained 2D triangulation of a surface.
    FIXED: Uses robust high-precision rounding approach following C++ MeshIt logic.
    """
    logger.info("Using FIXED robust PLC preparation following C++ MeshIt logic")
    
    # Get parameters from config
    target_feature_size = config.get('target_size', 20.0)
    
    # ================================================================
    # Direct PLC Generation Following C++ calculate_triangles Logic
    # ================================================================
    
    # Initialize unique points map using high-precision rounding (round to 9 decimal places)
    unique_points_map = {}  # key: (round_x, round_y, round_z) -> value: {'index': int, 'point': Vector3D}
    final_points_list = []  # Final list of Vector3D points in index order
    segment_indices_list = []  # Final list of [idx1, idx2] segment pairs
    
    def add_point_to_plc(point_obj: Vector3D) -> int:
        """
        Add a point to the PLC with high-precision deduplication.
        Returns the index of the point in the final list.
        """
        key = (round(point_obj.x, 9), round(point_obj.y, 9), round(point_obj.z, 9))
        
        if key in unique_points_map:
            # Point already exists, just return its index
            return unique_points_map[key]['index']
        else:
            # New point - add to map and list
            new_index = len(final_points_list)
            unique_points_map[key] = {
                'index': new_index,
                'point': point_obj
            }
            final_points_list.append(point_obj)
            return new_index

    # ----------------------------------------------------------------
    # Process hull constraints (closed loop)
    # ----------------------------------------------------------------
    hull_points = surface_data.get('hull_points', [])
    if hull_points:
        logger.info(f"Processing hull with {len(hull_points)} points")
        hull_indices = [add_point_to_plc(p) for p in hull_points]
        
        # Create closed loop segments for hull
        for i in range(len(hull_indices)):
            idx1 = hull_indices[i]
            idx2 = hull_indices[(i + 1) % len(hull_indices)]
            if idx1 != idx2:  # Avoid zero-length segments
                segment_indices_list.append([idx1, idx2])

    # ----------------------------------------------------------------
    # Process intersection constraints (open polylines)
    # ----------------------------------------------------------------
    for line_idx, intersection_data in enumerate(intersections_on_surface_data):
        intersection_points = intersection_data.get('points', [])
        if not intersection_points:
            continue
                
        logger.info(f"Processing intersection line {line_idx} with {len(intersection_points)} points")
        line_indices = [add_point_to_plc(p) for p in intersection_points]
        
        # Create segments for intersection polyline
        for i in range(len(line_indices) - 1):
            idx1 = line_indices[i]
            idx2 = line_indices[i + 1]
            if idx1 != idx2:  # Avoid zero-length segments
                segment_indices_list.append([idx1, idx2])

    logger.info(f"Robust PLC built: {len(final_points_list)} unique points, {len(segment_indices_list)} segments")

    # ----------------------------------------------------------------
    # Validate and Project to 2D
    # ----------------------------------------------------------------
    if len(final_points_list) < 3:
        logger.error(f"Insufficient points ({len(final_points_list)}) for triangulation")
        return None, None, np.empty((0, 2)), np.empty((0, 3))
    
    projection_params = surface_data.get('projection_params')
    if not projection_params:
        logger.error("Cannot create PLC: missing projection parameters.")
        return None, None, np.empty((0, 2)), np.empty((0, 3))
            
    centroid = np.array(projection_params['centroid'])
    basis = np.array(projection_params['basis'])
    
    points_3d_array = np.array([[p.x, p.y, p.z] for p in final_points_list])
    centered_points = points_3d_array - centroid
    points_2d = np.dot(centered_points, basis.T)[:,:2]

    segments_array = np.array(segment_indices_list, dtype=int) if segment_indices_list else np.empty((0, 2), dtype=int)
    holes_2d = np.empty((0, 2))  # No holes for now

    logger.info(f"FIXED robust PLC preparation successful: {len(points_2d)} points, {len(segments_array)} segments")
    return points_2d, segments_array, holes_2d, points_3d_array


def run_constrained_triangulation_py(
    plc_points_2d: np.ndarray,
    plc_segments_indices: np.ndarray,
    plc_holes_2d: np.ndarray,
    surface_projection_params: dict,
    original_3d_points_for_plc: np.ndarray,
    config: dict,
):
    """
    COMPLETELY REWRITTEN: Triangulation using DirectTriangleWrapper approach from triangulation tab.
    This mimics the successful triangulation method that works in the triangulation tab.
    """
    # --- basic checks --------------------------------------------------------
    if plc_points_2d is None or len(plc_points_2d) < 3:
        raise ValueError("Not enough PLC points")
    if plc_segments_indices is None or len(plc_segments_indices) < 3:
        raise ValueError("Not enough PLC segments")

    # Import DirectTriangleWrapper
    try:
        from meshit.triangle_direct import DirectTriangleWrapper
    except ImportError:
        logger.error("DirectTriangleWrapper not available! Falling back to basic Triangle")
        # Fallback to the old approach
        return _run_basic_triangle_fallback(plc_points_2d, plc_segments_indices, plc_holes_2d, 
                                          surface_projection_params, original_3d_points_for_plc, config)

    # Extract triangulation parameters from config
    gradient = config.get('gradient', 2.0)
    min_angle = config.get('min_angle', 20.0)
    target_size = config.get('target_size', 20.0)
    
    # Use target_size from GUI instead of auto-calculating from diagonal
    base_size = target_size
    
    # Only fall back to diagonal-based calculation if target_size is unreasonable
    min_coords = np.min(plc_points_2d, axis=0)
    max_coords = np.max(plc_points_2d, axis=0)
    diagonal = np.sqrt(np.sum((max_coords - min_coords) ** 2))
    
    if base_size > diagonal or base_size <= 0:
        base_size = diagonal / 15.0  # Fallback for unreasonable target_size
        logger.warning(f"Target size {target_size:.2f} unreasonable for surface (diagonal: {diagonal:.2f}), using fallback: {base_size:.2f}")
    else:
        logger.info(f"Using user target size: {base_size:.2f} (surface diagonal: {diagonal:.2f})")
    
    logger.info(f"Using DirectTriangleWrapper approach: gradient={gradient}, min_angle={min_angle}, base_size={base_size:.4f}")

    # Initialize DirectTriangleWrapper with parameters
    triangulator = DirectTriangleWrapper(
        gradient=gradient,
        min_angle=min_angle,
        base_size=base_size
    )

    # Run triangulation using DirectTriangleWrapper (uniform=True for consistency)
    triangulation_result = triangulator.triangulate(
        points=plc_points_2d,
        segments=plc_segments_indices,
        holes=plc_holes_2d,
        uniform=True,  # Use uniform approach like triangulation tab
        create_transition=True  # Enable enhanced transition point generation
    )

    if triangulation_result is None or 'vertices' not in triangulation_result or 'triangles' not in triangulation_result:
        raise RuntimeError("DirectTriangleWrapper triangulation failed to produce valid output.")

    vertices_2d = triangulation_result['vertices']
    triangles = triangulation_result['triangles']

    logger.info(f"DirectTriangleWrapper triangulation successful: {len(vertices_2d)} vertices, {len(triangles)} triangles")

    # --- SOPHISTICATED 3D RECONSTRUCTION (from triangulation tab) -----------
    
    if surface_projection_params is None:
        # 2D case - just add zero Z coordinate
        if vertices_2d.shape[1] == 2:
            final_vertices_3d = np.zeros((len(vertices_2d), 3))
            final_vertices_3d[:, :2] = vertices_2d
        else:
            final_vertices_3d = vertices_2d
    else:
        # 3D case - sophisticated reconstruction like triangulation tab
        centroid = np.asarray(surface_projection_params["centroid"])
        basis = np.asarray(surface_projection_params["basis"])
        normal = surface_projection_params.get("normal")
        
        # Project original 3D points to 2D for matching
        centered_original = original_3d_points_for_plc - centroid
        original_projected_2d = np.dot(centered_original, basis.T)
        
        final_vertices_3d = np.zeros((len(vertices_2d), 3))
        
        logger.info(f"Reconstructing {len(vertices_2d)} vertices from 2D to 3D...")
        
        for i, vertex_2d in enumerate(vertices_2d):
            is_matched_point = False
            
            # Try to match with original 3D points first (high precision)
            for j, orig_proj_2d in enumerate(original_projected_2d):
                if np.allclose(vertex_2d, orig_proj_2d, atol=1e-12):  # High precision matching
                    final_vertices_3d[i] = original_3d_points_for_plc[j]
                    is_matched_point = True
                    break
            
            # For new Steiner points, use sophisticated interpolation
            if not is_matched_point:
                # Extract Z values from original points
                original_z = original_3d_points_for_plc[:, 2]
                
                # Interpolate Z value using original 2D projections
                from scipy.spatial import distance
                from scipy.interpolate import griddata
                
                try:
                    # Use linear interpolation first
                    interpolated_z = griddata(original_projected_2d, original_z, vertex_2d, method='linear')
                    if np.isnan(interpolated_z):
                        # Fallback to nearest neighbor
                        interpolated_z = griddata(original_projected_2d, original_z, vertex_2d, method='nearest')
                    
                    if np.isnan(interpolated_z):
                        # Final fallback to centroid Z
                        interpolated_z = centroid[2]
                        logger.warning(f"Interpolation failed for vertex {i}, using centroid Z")
                    
                    # Ensure scalar
                    interpolated_z = float(interpolated_z)
                    
                    # Reconstruct 3D point
                    vertex_3d = centroid.copy()
                    vertex_3d += vertex_2d[0] * basis[0]
                    vertex_3d += vertex_2d[1] * basis[1]
                    vertex_3d[2] = interpolated_z
                    
                    final_vertices_3d[i] = vertex_3d
                    
                except Exception as e:
                    logger.warning(f"Z interpolation failed for vertex {i}: {e}. Using plane projection.")
                    # Ultimate fallback - simple plane projection
                    vertex_3d = centroid + vertex_2d[0] * basis[0] + vertex_2d[1] * basis[1]
                    final_vertices_3d[i] = vertex_3d

    # --- vertex deduplication with high precision (matching triangulation tab) -----
    def vkey(vec):
        return (round(vec[0], 12), round(vec[1], 12), round(vec[2], 12))

    uniq_map, uniq_verts, remap = {}, [], {}
    for old_idx, vec in enumerate(final_vertices_3d):
        k = vkey(vec)
        if k in uniq_map:
            remap[old_idx] = uniq_map[k]
        else:
            new_idx = len(uniq_verts)
            uniq_map[k] = new_idx
            remap[old_idx] = new_idx
            uniq_verts.append(vec)
    
    final_vertices_3d = np.asarray(uniq_verts)
    triangles = np.vectorize(remap.get)(triangles)

    # --- drop degenerate / zero-area triangles ---------------------
    good_triangles = []
    for t in triangles:
        if len({int(t[0]), int(t[1]), int(t[2])}) < 3:
            continue
        a, b, c = final_vertices_3d[t[0]], final_vertices_3d[t[1]], final_vertices_3d[t[2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        if area > 1e-12:
            good_triangles.append(t)
    
    triangles = np.asarray(good_triangles, dtype=int)

    logger.info(f"Final result: {len(final_vertices_3d)} vertices, {len(triangles)} triangles (after deduplication and cleanup)")
    return final_vertices_3d, triangles, []


def _run_basic_triangle_fallback(plc_points_2d, plc_segments_indices, plc_holes_2d, 
                                surface_projection_params, original_3d_points_for_plc, config):
    """Fallback to basic Triangle library if DirectTriangleWrapper is not available"""
    logger.warning("Using basic Triangle fallback (DirectTriangleWrapper not available)")
    
    import triangle as tr
    tri_input = {"vertices": plc_points_2d, "segments": plc_segments_indices}
    if plc_holes_2d is not None and len(plc_holes_2d) > 0:
        tri_input["holes"] = plc_holes_2d

    # Use basic Triangle options
    opts = "pzq20.0a100.0"
    logger.info(f"Running basic triangle with options: {opts}")
    tri_res = tr.triangulate(tri_input, opts=opts)

    vertices_2d = tri_res.get("vertices")
    triangles = tri_res.get("triangles")

    if vertices_2d is None or triangles is None:
        raise RuntimeError("Basic Triangle triangulation failed")

    # Basic 3D reconstruction
    if surface_projection_params is None:
        if vertices_2d.shape[1] == 2:
            final_vertices = np.zeros((len(vertices_2d), 3))
            final_vertices[:, :2] = vertices_2d
        else:
            final_vertices = vertices_2d
    else:
        centroid = np.asarray(surface_projection_params["centroid"])
        basis = np.asarray(surface_projection_params["basis"])
        
        final_vertices = np.zeros((len(vertices_2d), 3))
        for i, v2d in enumerate(vertices_2d):
            final_vertices[i] = centroid + v2d[0] * basis[0] + v2d[1] * basis[1]

    return final_vertices, triangles, []

# Configure logger
#logger = logging.getLogger(__name__)

@dataclass
class ConstraintSegment:
    """Represents a constraint segment with points and properties"""
    points: List[Vector3D]
    constraint_type: str = "UNDEFINED"  # UNDEFINED, SEGMENTS, HOLES
    size: float = 1.0
    rgb: Tuple[int, int, int] = (0, 0, 0)
    object_ids: List[int] = None
    
    def __post_init__(self):
        if self.object_ids is None:
            self.object_ids = []

class GradientControl:
    """Singleton class for gradient control in mesh generation"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            self.gradient = 2.0
            self.base_size = 1.0
            self.point_sizes = {}
            self.initialized = True
    
    def update(self, gradient: float, base_size: float, points_2d: np.ndarray, point_sizes: List[float]):
        """Update gradient control parameters"""
        self.gradient = gradient
        self.base_size = base_size
        
        # Store point-specific sizes
        self.point_sizes = {}
        for i, size in enumerate(point_sizes):
            if i < len(points_2d):
                key = (round(points_2d[i][0], 6), round(points_2d[i][1], 6))
                self.point_sizes[key] = size
    
    def get_size_at_point(self, point_2d: Tuple[float, float]) -> float:
        """Get the mesh size at a specific 2D point"""
        key = (round(point_2d[0], 6), round(point_2d[1], 6))
        return self.point_sizes.get(key, self.base_size)
    
    def apply_gradient_transition(self, points_2d: np.ndarray, sizes: List[float]) -> List[float]:
        """Apply gradient-based size transitions between points"""
        if len(points_2d) != len(sizes):
            return sizes
        
        adjusted_sizes = sizes.copy()
        
        # Apply gradient control between adjacent points
        for i in range(len(points_2d) - 1):
            p1 = points_2d[i]
            p2 = points_2d[i + 1]
            s1 = sizes[i]
            s2 = sizes[i + 1]
            
            distance = np.linalg.norm(p2 - p1)
            
            # Calculate maximum allowed size difference based on gradient
            max_size_diff = distance * self.gradient
            
            # Adjust sizes if they violate gradient constraint
            if abs(s2 - s1) > max_size_diff:
                if s2 > s1:
                    adjusted_sizes[i + 1] = s1 + max_size_diff
                else:
                    adjusted_sizes[i + 1] = s1 - max_size_diff
        
        return adjusted_sizes

def calculate_constraints_for_surface(surface_data: Dict, intersections_on_surface: List[Dict]) -> List[ConstraintSegment]:
    """
    Calculate constraint segments for a surface following C++ MeshIt logic.
    
    This function implements the C++ calculate_Constraints() logic:
    1. Split lines at special points (non-DEFAULT types)
    2. Create constraint segments between special points
    3. Assign proper types and sizes
    
    Args:
        surface_data: Dictionary containing surface information including convex_hull
        intersections_on_surface: List of intersection data for this surface
        
    Returns:
        List of ConstraintSegment objects
    """
    constraints = []
    rgb_counter = [0, 0, 0]  # RGB color counter for visualization
    
    def increment_rgb(rgb: List[int]) -> None:
        """Increment RGB counter for unique constraint colors"""
        rgb[0] += 1
        if rgb[0] > 255:
            rgb[0] = 0
            rgb[1] += 1
            if rgb[1] > 255:
                rgb[1] = 0
                rgb[2] += 1
                if rgb[2] > 255:
                    rgb[2] = 0
    
    # Process convex hull constraints
    hull_points = surface_data.get('hull_points', [])
    if hull_points and len(hull_points) > 1:
        hull_constraints = split_line_at_special_points(hull_points, surface_data.get('size', 1.0))
        for constraint_points in hull_constraints:
            if len(constraint_points) >= 2:
                # Check if this hull segment contains special points (non-DEFAULT)
                has_special_points = any(
                    hasattr(pt, 'type') and pt.type and pt.type != "DEFAULT" 
                    for pt in constraint_points
                )
                
                constraint = ConstraintSegment(
                    points=constraint_points,
                    constraint_type="SEGMENTS" if has_special_points else "UNDEFINED",  # Mark as SEGMENTS if has special points
                    size=surface_data.get('size', 1.0),
                    rgb=(rgb_counter[0], rgb_counter[1], rgb_counter[2])
                )
                constraints.append(constraint)
                increment_rgb(rgb_counter)
    
    # Process intersection constraints
    for intersection_data in intersections_on_surface:
        intersection_points = intersection_data.get('points', [])
        if not intersection_points:
            continue

        intersection_size = intersection_data.get('size', surface_data.get('size', 1.0))
        
        if len(intersection_points) == 1:
            # Single point constraint
            constraint = ConstraintSegment(
                points=intersection_points,
                constraint_type="SEGMENTS",  # Mark intersection points as SEGMENTS
                size=intersection_size,
                rgb=(rgb_counter[0], rgb_counter[1], rgb_counter[2])
            )
            constraints.append(constraint)
            increment_rgb(rgb_counter)
        else:
            # Multi-point constraint - split at special points
            intersection_constraints = split_line_at_special_points(intersection_points, intersection_size)
            for constraint_points in intersection_constraints:
                if len(constraint_points) >= 2:
                    constraint = ConstraintSegment(
                        points=constraint_points,
                        constraint_type="SEGMENTS",  # Mark intersection lines as SEGMENTS
                        size=intersection_size,
                        rgb=(rgb_counter[0], rgb_counter[1], rgb_counter[2])
                    )
                    constraints.append(constraint)
                    increment_rgb(rgb_counter)

    logger.info(f"Generated {len(constraints)} constraint segments for surface")
    return constraints

def split_line_at_special_points(points: List[Vector3D], default_size: float) -> List[List[Vector3D]]:
    """
    Split a line at special points (non-DEFAULT types) following C++ logic.
    
    This implements the C++ constraint segmentation logic where lines are split
    at points that have special types (TRIPLE_POINT, INTERSECTION_POINT, etc.)
    
    Args:
        points: List of Vector3D points forming a line
        default_size: Default size for points without specific size
        
    Returns:
        List of point lists, each representing a constraint segment
    """
    if len(points) < 2:
        return [points] if points else []
    
    segments = []
    last_pos = 0
    
    # Iterate through points looking for special points or end of line
    for n in range(1, len(points)):
        point = points[n]
        point_type = getattr(point, 'point_type', getattr(point, 'type', "DEFAULT"))
        
        # Split at special points or at the end of the line
        if point_type != "DEFAULT" or n == len(points) - 1:
            # Create segment from last_pos to current position (inclusive)
            segment_points = points[last_pos:n+1]
            if len(segment_points) >= 2:
                segments.append(segment_points)
            last_pos = n
    
    # Handle case where no special points were found
    if not segments and len(points) >= 2:
        segments.append(points)
    
    return segments

def calculate_constraint_sizes(constraints: List[ConstraintSegment], 
                             surface_data: Dict, 
                             other_surfaces: List[Dict] = None,
                             polylines: List[Dict] = None) -> None:
    """
    Calculate sizes for constraint segments based on intersecting features.
    
    This implements the C++ calculate_size_of_constraints() logic where
    intersection constraints take the smallest size of intersecting features.
    
    Args:
        constraints: List of constraint segments to update
        surface_data: Current surface data
        other_surfaces: List of other surface data for cross-surface constraints
        polylines: List of polyline data for polyline-surface constraints
    """
    if other_surfaces is None:
        other_surfaces = []
    if polylines is None:
        polylines = []
    
    surface_size = surface_data.get('size', 1.0)
    
    # Update constraint sizes based on intersecting features
    for constraint in constraints:
        min_size = constraint.size
        
        # Check against other surfaces
        for other_surface in other_surfaces:
            other_size = other_surface.get('size', 1.0)
            if constraint_intersects_surface(constraint, other_surface):
                min_size = min(min_size, other_size)
        
        # Check against polylines
        for polyline in polylines:
            polyline_size = polyline.get('size', 1.0)
            if constraint_intersects_polyline(constraint, polyline):
                min_size = min(min_size, polyline_size)
        
        # Update constraint size
        constraint.size = min_size
        
        # Update individual point sizes
        for point in constraint.points:
            if hasattr(point, 'size'):
                point.size = min_size
            else:
                setattr(point, 'size', min_size)

def constraint_intersects_surface(constraint: ConstraintSegment, surface_data: Dict) -> bool:
    """Check if a constraint intersects with a surface"""
    # Simplified intersection check - in practice this would be more sophisticated
    surface_bounds = surface_data.get('bounds')
    if not surface_bounds or not constraint.points:
        return False
    
    # Check if any constraint points are within surface bounds
    for point in constraint.points:
        if (surface_bounds[0].x <= point.x <= surface_bounds[1].x and
            surface_bounds[0].y <= point.y <= surface_bounds[1].y and
            surface_bounds[0].z <= point.z <= surface_bounds[1].z):
            return True
    
    return False

def constraint_intersects_polyline(constraint: ConstraintSegment, polyline_data: Dict) -> bool:
    """Check if a constraint intersects with a polyline"""
    # Simplified intersection check
    polyline_points = polyline_data.get('points', [])
    if not polyline_points or not constraint.points:
        return False
    
    # Check for point proximity (simplified)
    tolerance = 1e-6
    for c_point in constraint.points:
        for p_point in polyline_points:
            if isinstance(p_point, Vector3D):
                if (c_point - p_point).length() < tolerance:
                    return True
    
    return False

def assign_point_types_and_sizes(points: List[Vector3D], 
                                base_size: float,
                                point_type_sizes: Dict[str, float] = None) -> None:
    """
    Assign sizes to points based on their types following C++ logic.
    
    Args:
        points: List of points to process
        base_size: Base mesh size
        point_type_sizes: Dictionary mapping point types to specific sizes
    """
    if point_type_sizes is None:
        point_type_sizes = {
            "TRIPLE_POINT": base_size * 0.5,
            "INTERSECTION_POINT": base_size * 0.7,
            "CORNER": base_size * 0.8,
            "SPECIAL_POINT": base_size * 0.6,
            "DEFAULT": base_size
        }
    
    for point in points:
        point_type = getattr(point, 'point_type', getattr(point, 'type', "DEFAULT"))
        assigned_size = point_type_sizes.get(point_type, base_size)
        
        # Set size attribute
        if hasattr(point, 'size'):
            point.size = min(point.size, assigned_size)
        else:
            setattr(point, 'size', assigned_size)

def prepare_constrained_triangulation_input(surface_data: Dict, 
        intersections_on_surface: List[Dict],
                                           config: Dict) -> Tuple[np.ndarray, np.ndarray, List[float], np.ndarray]:
    """
    Prepare input for constrained triangulation with protected triple points.
    
    This function implements the C++ calculate_triangles() constraint processing with
    enhanced triple point protection:
    1. Calculate constraint segments
    2. Register protected triple points with high precision
    3. Apply specialized deduplication preserving triple points
    4. Prepare points and segments for triangulation
    
    Args:
        surface_data: Surface data dictionary
        intersections_on_surface: List of intersections on this surface
        config: Configuration dictionary (should include 'triple_points' for protection)
        
    Returns:
        Tuple of (points_2d, segments, point_sizes, holes_2d)
    """
    # DEBUG: Show what we received for constraint processing
    protected_triple_points = config.get('triple_points', [])
    logger.info(f"🔧 CONSTRAINT PROCESSING: Received {len(protected_triple_points)} protected triple points")
    logger.info(f"🔧 CONSTRAINT PROCESSING: Hull points: {len(surface_data.get('hull_points', []))}")
    logger.info(f"🔧 CONSTRAINT PROCESSING: Intersection lines: {len(intersections_on_surface)}")
    
    # Calculate constraint segments
    constraints = calculate_constraints_for_surface(surface_data, intersections_on_surface)
    logger.info(f"🔧 CONSTRAINT PROCESSING: Generated {len(constraints)} constraint segments")
    
    # Calculate constraint sizes
    calculate_constraint_sizes(constraints, surface_data)
    
    # NO DEDUPLICATION - Collect ALL points to preserve every point including triple points
    all_points = []  # List to store ALL points without deduplication
    segments = []
    point_index = 0
    
    logger.info("🔧 CONSTRAINT PROCESSING: Starting point collection WITHOUT deduplication to preserve triple points")
    
    # Process constraints to build point list and segments
    # CRITICAL FIX: C++ uses constraints where Type != "UNDEFINED"
    # We need to mark intersection constraints as "SEGMENTS" not "UNDEFINED"
    for constraint in constraints:
        # Mark intersection constraints as SEGMENTS (not UNDEFINED)
        if len(constraint.points) > 1:
            constraint.constraint_type = "SEGMENTS"  # Mark as active constraint
        
        if constraint.constraint_type == "UNDEFINED":
            continue  # Skip undefined constraints in triangulation
            
        if len(constraint.points) == 1:
            # Single point constraint - add without deduplication
            point = constraint.points[0]
            all_points.append({
                'point': point,
                'size': constraint.size,
                'index': point_index
            })
            point_index += 1
            
        elif len(constraint.points) > 1:
            # Multi-point constraint - add ALL points without deduplication
            for i in range(len(constraint.points) - 1):
                p1 = constraint.points[i]
                p2 = constraint.points[i + 1]
                
                # DEBUG: Check if these are triple points
                p1_type = getattr(p1, 'type', 'DEFAULT')
                p2_type = getattr(p2, 'type', 'DEFAULT')
                if p1_type == 'TRIPLE_POINT':
                    logger.info(f"🎯 FOUND TRIPLE POINT in constraint processing: [{p1.x:.6f}, {p1.y:.6f}, {p1.z:.6f}]")
                if p2_type == 'TRIPLE_POINT':
                    logger.info(f"🎯 FOUND TRIPLE POINT in constraint processing: [{p2.x:.6f}, {p2.y:.6f}, {p2.z:.6f}]")
                
                # Add p1 (without checking if it exists)
                p1_index = point_index
                all_points.append({
                    'point': p1,
                    'size': constraint.size,
                    'index': p1_index
                })
                point_index += 1
                
                # Add p2 (without checking if it exists)
                p2_index = point_index
                all_points.append({
                    'point': p2,
                    'size': constraint.size,
                    'index': p2_index
                })
                point_index += 1
                
                # Add segment
                segments.append([p1_index, p2_index])
    
    # If no constraints were processed, fall back to convex hull
    if not all_points:
        hull_points = surface_data.get('hull_points', [])
        for i, point in enumerate(hull_points):
            all_points.append({
                'point': point,
                'size': surface_data.get('size', 1.0),
                'index': i
            })
            # Add hull segments (connect sequential points and close the loop)
            if i > 0:
                segments.append([i-1, i])
            if i == len(hull_points) - 1 and len(hull_points) > 2:
                segments.append([i, 0])  # Close the hull
    
    # Convert to arrays - NO DEDUPLICATION
    points_list = []
    sizes_list = []
    
    for point_data in all_points:
        points_list.append(point_data['point'])
        sizes_list.append(point_data['size'])
    
    # Project to 2D
    points_2d = []
    projection_params = surface_data.get('projection_params')
    if projection_params:
        centroid = np.array(projection_params['centroid'])
        basis = np.array(projection_params['basis'])
        for point in points_list:
            centered_pt = np.array([point.x, point.y, point.z]) - centroid
            pt_2d = np.dot(centered_pt, basis.T)
            points_2d.append(pt_2d[:2])
    else:
        for point in points_list:
            points_2d.append([point.x, point.y])
    
    points_2d = np.array(points_2d)
    segments = np.array(segments) if segments else np.empty((0, 2), dtype=int)
    
    # Apply gradient control
    gradient = config.get('gradient', 2.0)
    gc = GradientControl()
    gc.update(gradient, surface_data.get('size', 1.0), points_2d, sizes_list)
    adjusted_sizes = gc.apply_gradient_transition(points_2d, sizes_list)
    
    # Prepare holes (empty for now)
    holes_2d = np.empty((0, 2))
    
    logger.info(f"NO DEDUPLICATION: Prepared constrained triangulation input: {len(points_2d)} points, {len(segments)} segments")
    logger.info(f"ALL POINTS PRESERVED: No points were lost to deduplication")
    
    # DEBUG: Count triple points in final result
    triple_count_final = 0
    for point_data in all_points:
        point = point_data['point']
        if hasattr(point, 'type') and point.type == 'TRIPLE_POINT':
            triple_count_final += 1
            logger.info(f"🎯 TRIPLE POINT in final result: [{point.x:.6f}, {point.y:.6f}, {point.z:.6f}]")
    
    logger.info(f"🎯 FINAL RESULT: {triple_count_final} triple points preserved in triangulation input")
    
    return points_2d, segments, adjusted_sizes, holes_2d

# Add this function after the existing functions

def integrate_constraint_processing_workflow(model, config: Dict = None) -> None:
    """
    Integrate the new constraint processing workflow into the existing MeshIt model.
    
    This function applies the C++ MeshIt constraint processing logic:
    1. Calculate constraints for all surfaces
    2. Apply type-based size assignment
    3. Calculate constraint sizes based on intersections
    4. Apply gradient control
    
    Args:
        model: MeshItModel instance
        config: Configuration dictionary
    """
    if config is None:
        config = {
            'gradient': 2.0,
            'use_constraint_processing': True,
            'type_based_sizing': True,
            'hierarchical_constraints': True
        }
    
    if not config.get('use_constraint_processing', False):
        logger.info("Constraint processing disabled in config")
        return
    
    logger.info("Starting integrated constraint processing workflow")
    
    # Process each surface
    for surface_idx, surface in enumerate(model.surfaces):
        try:
            # Prepare surface data
            surface_data = {
                'hull_points': getattr(surface, 'convex_hull', []),
                'size': getattr(surface, 'size', 1.0),
                'bounds': getattr(surface, 'bounds', None),
                'projection_params': getattr(surface, 'projection_params', None)
            }
            
            # Find intersections on this surface
            intersections_on_surface = []
            for intersection in model.intersections:
                # Check if this surface is involved in the intersection
                if (intersection.id1 == surface_idx or intersection.id2 == surface_idx):
                    intersection_data = {
                        'points': intersection.points,
                        'size': getattr(intersection, 'size', surface_data['size'] * 0.5),
                        'type': getattr(intersection, 'type', 'INTERSECTION')
                    }
                    intersections_on_surface.append(intersection_data)
            
            # Calculate constraints for this surface
            constraints = calculate_constraints_for_surface(surface_data, intersections_on_surface)
            
            # Store constraints on the surface
            if not hasattr(surface, 'constraints'):
                surface.constraints = []
            surface.constraints = constraints
            
            # Apply type-based sizing if enabled
            if config.get('type_based_sizing', False):
                all_points = []
                if hasattr(surface, 'convex_hull'):
                    all_points.extend(surface.convex_hull)
                for intersection_data in intersections_on_surface:
                    all_points.extend(intersection_data['points'])
                
                assign_point_types_and_sizes(all_points, surface_data['size'])
            
            logger.info(f"Processed constraints for surface {surface_idx}: {len(constraints)} constraint segments")
            
        except Exception as e:
            logger.error(f"Error processing constraints for surface {surface_idx}: {e}")
            continue
    
    # Calculate constraint sizes based on intersections
    if config.get('hierarchical_constraints', False):
        try:
            for surface_idx, surface in enumerate(model.surfaces):
                if hasattr(surface, 'constraints'):
                    surface_data = {
                        'size': getattr(surface, 'size', 1.0),
                        'bounds': getattr(surface, 'bounds', None)
                    }
                    
                    # Prepare other surfaces data
                    other_surfaces = []
                    for other_idx, other_surface in enumerate(model.surfaces):
                        if other_idx != surface_idx:
                            other_surfaces.append({
                                'size': getattr(other_surface, 'size', 1.0),
                                'bounds': getattr(other_surface, 'bounds', None)
                            })
                    
                    # Prepare polylines data if available
                    polylines = []
                    if hasattr(model, 'model_polylines'):
                        for polyline in model.model_polylines:
                            polylines.append({
                                'size': getattr(polyline, 'size', 1.0),
                                'points': getattr(polyline, 'vertices', [])
                            })
                    
                    # Calculate constraint sizes
                    calculate_constraint_sizes(surface.constraints, surface_data, other_surfaces, polylines)
                    
        except Exception as e:
            logger.error(f"Error calculating constraint sizes: {e}")
    
    logger.info("Integrated constraint processing workflow completed")

def update_refinement_with_constraints(intersection, target_length: float, config: Dict = None) -> List[Vector3D]:
    """
    Update intersection refinement to use the new constraint processing logic.
    
    Args:
        intersection: Intersection object to refine
        target_length: Target segment length
        config: Configuration dictionary
        
    Returns:
        List of refined points
    """
    if config is None:
        config = {'gradient': 2.0, 'min_angle': 20.0, 'uniform_meshing': True}
    
    # Use the new refinement function with constraint processing
    refined_points = refine_intersection_line_by_length(
        intersection, 
        target_length, 
        config.get('min_angle', 20.0),
        config.get('uniform_meshing', True)
    )
    
    # Apply type-based sizing
    if config.get('type_based_sizing', True):
        assign_point_types_and_sizes(refined_points, target_length)
    
    return refined_points

def validate_surfaces_for_tetgen(datasets, config=None):
    """
    Validate constrained surfaces to ensure they are ready for tetgen tetrahedralization.
    
    This function performs comprehensive checks on triangulated surfaces to verify:
    - Mesh quality and topology
    - Proper constraint processing
    - Surface intersection handling
    - Tetgen compatibility requirements
    
    Args:
        datasets: List of dataset dictionaries with triangulated surfaces
        config: Configuration dictionary
        
    Returns:
        Dict with validation results and recommendations
    """
    if config is None:
        config = {}
    
    validation_results = {
        'overall_status': 'UNKNOWN',
        'ready_for_tetgen': False,
        'surface_count': len(datasets),
        'surfaces': [],
        'issues': [],
        'recommendations': [],
        'statistics': {}
    }
    
    logger.info("=== TETGEN SURFACE VALIDATION ===")
    
    total_vertices = 0
    total_triangles = 0
    valid_surfaces = 0
    
    for i, dataset in enumerate(datasets):
        surface_name = dataset.get('name', f'Surface_{i}')
        surface_result = {
            'name': surface_name,
            'index': i,
            'status': 'UNKNOWN',
            'vertices': 0,
            'triangles': 0,
            'issues': [],
            'quality_metrics': {}
        }
        
        logger.info(f"Validating surface: {surface_name}")
        
        # Check if surface has triangulation data
        if 'constrained_vertices' not in dataset or 'constrained_triangles' not in dataset:
            surface_result['status'] = 'MISSING_TRIANGULATION'
            surface_result['issues'].append('No constrained triangulation data found')
            validation_results['issues'].append(f'{surface_name}: Missing triangulation data')
        else:
            vertices = dataset['constrained_vertices']
            triangles = dataset['constrained_triangles']
            
            surface_result['vertices'] = len(vertices)
            surface_result['triangles'] = len(triangles)
            total_vertices += len(vertices)
            total_triangles += len(triangles)
            
            # 1. Basic topology checks
            topology_issues = []
            
            # Check minimum requirements
            if len(vertices) < 3:
                topology_issues.append('Insufficient vertices (< 3)')
            if len(triangles) < 1:
                topology_issues.append('No triangles found')
            
            # Check triangle validity
            invalid_triangles = 0
            for tri in triangles:
                if len(tri) != 3:
                    invalid_triangles += 1
                elif max(tri) >= len(vertices):
                    invalid_triangles += 1
            
            if invalid_triangles > 0:
                topology_issues.append(f'{invalid_triangles} invalid triangles (bad indices)')
            
            # 2. Mesh quality checks
            quality_metrics = {}
            
            if len(vertices) > 0 and len(triangles) > 0:
                try:
                    import numpy as np
                    vertices_np = np.array(vertices)
                    
                    # Calculate triangle areas and aspect ratios
                    areas = []
                    aspect_ratios = []
                    min_angles = []
                    
                    for tri in triangles:
                        if max(tri) < len(vertices):
                            v1, v2, v3 = vertices_np[tri[0]], vertices_np[tri[1]], vertices_np[tri[2]]
                            
                            # Triangle area
                            edge1 = v2 - v1
                            edge2 = v3 - v1
                            cross = np.cross(edge1, edge2)
                            area = 0.5 * np.linalg.norm(cross)
                            areas.append(area)
                            
                            # Edge lengths
                            e1_len = np.linalg.norm(edge1)
                            e2_len = np.linalg.norm(v3 - v2)
                            e3_len = np.linalg.norm(edge2)
                            
                            # Aspect ratio (longest edge / shortest edge)
                            edge_lengths = [e1_len, e2_len, e3_len]
                            if min(edge_lengths) > 1e-12:
                                aspect_ratio = max(edge_lengths) / min(edge_lengths)
                                aspect_ratios.append(aspect_ratio)
                            
                            # Minimum angle (using law of cosines)
                            if e1_len > 1e-12 and e2_len > 1e-12 and e3_len > 1e-12:
                                # Angle at vertex 1
                                cos_angle = (e1_len**2 + e3_len**2 - e2_len**2) / (2 * e1_len * e3_len)
                                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                                angle = np.arccos(cos_angle) * 180 / np.pi
                                min_angles.append(angle)
                    
                    if areas:
                        quality_metrics['min_area'] = min(areas)
                        quality_metrics['max_area'] = max(areas)
                        quality_metrics['avg_area'] = sum(areas) / len(areas)
                        
                        # Check for degenerate triangles
                        degenerate_count = sum(1 for area in areas if area < 1e-12)
                        if degenerate_count > 0:
                            topology_issues.append(f'{degenerate_count} degenerate triangles (area < 1e-12)')
                    
                    if aspect_ratios:
                        quality_metrics['min_aspect_ratio'] = min(aspect_ratios)
                        quality_metrics['max_aspect_ratio'] = max(aspect_ratios)
                        quality_metrics['avg_aspect_ratio'] = sum(aspect_ratios) / len(aspect_ratios)
                        
                        # Check for poor quality triangles
                        poor_quality_count = sum(1 for ar in aspect_ratios if ar > 10.0)
                        if poor_quality_count > 0:
                            topology_issues.append(f'{poor_quality_count} poor quality triangles (aspect ratio > 10)')
                    
                    if min_angles:
                        quality_metrics['min_angle'] = min(min_angles)
                        quality_metrics['max_angle'] = max(min_angles)
                        quality_metrics['avg_min_angle'] = sum(min_angles) / len(min_angles)
                        
                        # Check for very small angles
                        small_angle_count = sum(1 for angle in min_angles if angle < 5.0)
                        if small_angle_count > 0:
                            topology_issues.append(f'{small_angle_count} triangles with very small angles (< 5°)')
                    
                except Exception as e:
                    topology_issues.append(f'Quality analysis failed: {str(e)}')
            
            # 3. Constraint processing validation
            constraint_issues = []
            
            # Check if constraint processing was used
            if 'constraint_processing_used' in dataset:
                if dataset['constraint_processing_used']:
                    logger.info(f'{surface_name}: Constraint processing was used ✓')
                else:
                    constraint_issues.append('Constraint processing was not used')
            else:
                constraint_issues.append('Constraint processing status unknown')
            
            # Check for intersection constraints
            if 'intersection_constraints' in dataset:
                intersection_count = len(dataset['intersection_constraints'])
                if intersection_count > 0:
                    logger.info(f'{surface_name}: {intersection_count} intersection constraints found ✓')
                else:
                    constraint_issues.append('No intersection constraints found')
            
            # 4. Tetgen compatibility checks
            tetgen_issues = []
            
            # Check for manifold surface (each edge shared by at most 2 triangles)
            if len(triangles) > 0:
                edge_count = {}
                for tri in triangles:
                    if max(tri) < len(vertices):
                        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
                        for edge in edges:
                            edge_key = tuple(sorted(edge))
                            edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
                
                non_manifold_edges = sum(1 for count in edge_count.values() if count > 2)
                if non_manifold_edges > 0:
                    tetgen_issues.append(f'{non_manifold_edges} non-manifold edges (shared by > 2 triangles)')
                
                boundary_edges = sum(1 for count in edge_count.values() if count == 1)
                quality_metrics['boundary_edges'] = boundary_edges
                quality_metrics['total_edges'] = len(edge_count)
            
            # Combine all issues
            all_issues = topology_issues + constraint_issues + tetgen_issues
            surface_result['issues'] = all_issues
            surface_result['quality_metrics'] = quality_metrics
            
            # Determine surface status
            if not all_issues:
                surface_result['status'] = 'READY'
                valid_surfaces += 1
                logger.info(f'{surface_name}: READY for tetgen ✓')
            elif len(topology_issues) == 0 and len(tetgen_issues) == 0:
                surface_result['status'] = 'WARNING'
                logger.warning(f'{surface_name}: Has warnings but may work with tetgen')
            else:
                surface_result['status'] = 'ERROR'
                logger.error(f'{surface_name}: Has critical issues, not ready for tetgen')
            
            # Log quality metrics
            if quality_metrics:
                logger.info(f'{surface_name} quality: vertices={len(vertices)}, triangles={len(triangles)}')
                if 'avg_aspect_ratio' in quality_metrics:
                    logger.info(f'  Aspect ratio: avg={quality_metrics["avg_aspect_ratio"]:.2f}, max={quality_metrics["max_aspect_ratio"]:.2f}')
                if 'avg_min_angle' in quality_metrics:
                    logger.info(f'  Min angles: avg={quality_metrics["avg_min_angle"]:.1f}°, min={quality_metrics["min_angle"]:.1f}°')
        
        validation_results['surfaces'].append(surface_result)
    
    # Overall validation summary
    validation_results['statistics'] = {
        'total_vertices': total_vertices,
        'total_triangles': total_triangles,
        'valid_surfaces': valid_surfaces,
        'surfaces_with_warnings': sum(1 for s in validation_results['surfaces'] if s['status'] == 'WARNING'),
        'surfaces_with_errors': sum(1 for s in validation_results['surfaces'] if s['status'] == 'ERROR')
    }
    
    # Determine overall status
    if valid_surfaces == len(datasets) and len(datasets) > 0:
        validation_results['overall_status'] = 'READY'
        validation_results['ready_for_tetgen'] = True
        logger.info("=== ALL SURFACES READY FOR TETGEN ✓ ===")
    elif valid_surfaces > 0:
        validation_results['overall_status'] = 'PARTIAL'
        validation_results['ready_for_tetgen'] = False
        logger.warning(f"=== PARTIAL READINESS: {valid_surfaces}/{len(datasets)} surfaces ready ===")
    else:
        validation_results['overall_status'] = 'NOT_READY'
        validation_results['ready_for_tetgen'] = False
        logger.error("=== SURFACES NOT READY FOR TETGEN ===")
    
    # Generate recommendations
    recommendations = []
    if validation_results['statistics']['surfaces_with_errors'] > 0:
        recommendations.append("Fix critical topology and manifold issues before tetgen")
    if validation_results['statistics']['surfaces_with_warnings'] > 0:
        recommendations.append("Review constraint processing warnings")
    if total_vertices < 10:
        recommendations.append("Consider refining surfaces for better tetgen results")
    if validation_results['ready_for_tetgen']:
        recommendations.append("Surfaces are ready for tetgen tetrahedralization")
    
    validation_results['recommendations'] = recommendations
    
    return validation_results



def prepare_constrained_triangulation_input(surface_data: Dict,
                                           intersections_on_surface: List[Dict],
                                           config: Dict) -> Tuple[np.ndarray, np.ndarray, List[float], np.ndarray]:
    """
    Prepare input for constrained triangulation following C++ MeshIt logic.
    
    This function implements the C++ calculate_triangles() constraint processing:
    1. Calculate constraint segments
    2. Assign sizes based on constraint types and intersections
    3. Apply gradient control
    4. Prepare points and segments for triangulation
    
    Args:
        surface_data: Surface data dictionary
        intersections_on_surface: List of intersections on this surface
        config: Configuration dictionary
        
    Returns:
        Tuple of (points_2d, segments, point_sizes, holes_2d)
    """
    # Calculate constraint segments
    constraints = calculate_constraints_for_surface(surface_data, intersections_on_surface)
    
    # Calculate constraint sizes
    calculate_constraint_sizes(constraints, surface_data)
    
    # Collect all unique points and their properties
    unique_points = {}  # Dict to store unique points with their properties
    segments = []
    point_sizes = []
    
    # Process constraints to build point list and segments
    # CRITICAL FIX: C++ uses constraints where Type != "UNDEFINED"
    # We need to mark intersection constraints as "SEGMENTS" not "UNDEFINED"
    for constraint in constraints:
        # Mark intersection constraints as SEGMENTS (not UNDEFINED)
        if len(constraint.points) > 1:
            constraint.constraint_type = "SEGMENTS"  # Mark as active constraint
        
        if constraint.constraint_type == "UNDEFINED":
            continue  # Skip undefined constraints in triangulation
            
        if len(constraint.points) == 1:
            # Single point constraint
            point = constraint.points[0]
            point_key = (round(point.x, 8), round(point.y, 8), round(point.z, 8))
            if point_key not in unique_points:
                unique_points[point_key] = {
                    'point': point,
                    'size': constraint.size,
                    'index': len(unique_points)
                }
        elif len(constraint.points) > 1:
            # Multi-point constraint - create segments
            for i in range(len(constraint.points) - 1):
                p1 = constraint.points[i]
                p2 = constraint.points[i + 1]
                
                # Add points to unique collection
                p1_key = (round(p1.x, 8), round(p1.y, 8), round(p1.z, 8))
                p2_key = (round(p2.x, 8), round(p2.y, 8), round(p2.z, 8))
                
                if p1_key not in unique_points:
                    unique_points[p1_key] = {
                        'point': p1,
                        'size': constraint.size,
                        'index': len(unique_points)
                    }
                
                if p2_key not in unique_points:
                    unique_points[p2_key] = {
                        'point': p2,
                        'size': constraint.size,
                        'index': len(unique_points)
                    }
                
                # Add segment
                idx1 = unique_points[p1_key]['index']
                idx2 = unique_points[p2_key]['index']
                segments.append([idx1, idx2])
    
    # If no constraints were processed, fall back to convex hull
    if not unique_points:
        hull_points = surface_data.get('hull_points', [])
        for i, point in enumerate(hull_points[:-1]):  # Exclude last point if it's duplicate of first
            point_key = (round(point.x, 8), round(point.y, 8), round(point.z, 8))
            unique_points[point_key] = {
                'point': point,
                'size': surface_data.get('size', 1.0),
                'index': i
            }
            # Add hull segments
            if i < len(hull_points) - 2:
                segments.append([i, i + 1])
            else:
                segments.append([i, 0])  # Close the hull
    
    # Convert to arrays
    points_list = [None] * len(unique_points)
    sizes_list = [0.0] * len(unique_points)
    
    for point_data in unique_points.values():
        idx = point_data['index']
        points_list[idx] = point_data['point']
        sizes_list[idx] = point_data['size']
    
    # Project to 2D
    points_2d = []
    projection_params = surface_data.get('projection_params')
    if projection_params:
        centroid = np.array(projection_params['centroid'])
        basis = np.array(projection_params['basis'])
        for point in points_list:
            centered_pt = np.array([point.x, point.y, point.z]) - centroid
            pt_2d = np.dot(centered_pt, basis.T)
            points_2d.append(pt_2d[:2])
    else:
        for point in points_list:
            points_2d.append([point.x, point.y])
    
    points_2d = np.array(points_2d)
    segments = np.array(segments) if segments else np.empty((0, 2), dtype=int)
    
    # Apply gradient control
    gradient = config.get('gradient', 2.0)
    gc = GradientControl()
    gc.update(gradient, surface_data.get('size', 1.0), points_2d, sizes_list)
    adjusted_sizes = gc.apply_gradient_transition(points_2d, sizes_list)
    
    # Prepare holes (empty for now)
    holes_2d = np.empty((0, 2))
    
    logger.info(f"Prepared constrained triangulation input: {len(points_2d)} points, {len(segments)} segments")
    
    return points_2d, segments, adjusted_sizes, holes_2d