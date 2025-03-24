#!/usr/bin/env python
"""
Test for surface-surface intersection calculation in MeshIt.

This script demonstrates the complete workflow:
1. Generate test points
2. Calculate convex hull
3. Perform triangulation with feature points
4. Calculate surface-surface intersections
5. Align intersections to convex hulls
6. Calculate constraints
7. Perform final refined triangulation
"""

import sys
import numpy as np
from pathlib import Path
import pyvista as pv
from scipy.spatial import cKDTree
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import os
import time
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import triangle as tr
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import math
import random
import traceback  # Import traceback for error diagnostics
from typing import List, Dict, Tuple, Optional, Union, Any

try:
    import pyvista as pv
except ImportError:
    print("PyVista not found. Installing required packages for visualization...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyvista"])
    import pyvista as pv

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from meshit.triangle_wrapper import TriangleWrapper
from meshit.triangle_mesh import create_hull_segments
from meshit.intersection_utils import Vector3D as MeshItVector3D
from meshit.intersection_utils import calculate_surface_surface_intersection, calculate_polyline_surface_intersection
from meshit.extensions import triangulate_with_special_points

class Vector3D:
    """Simple 3D vector class."""
    
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
    
    def __rmul__(self, scalar):
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
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalized(self):
        length = self.length()
        if length < 1e-10:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / length, self.y / length, self.z / length)
    
    def __repr__(self):
        return f"Vector3D({self.x}, {self.y}, {self.z})"
    
    @staticmethod
    def to_meshit_vector(v):
        """Convert Vector3D to MeshItVector3D"""
        return MeshItVector3D(v.x, v.y, v.z)
    
    @staticmethod
    def from_meshit_vector(mv):
        """Convert MeshItVector3D to Vector3D"""
        return Vector3D(mv.x, mv.y, mv.z)

class Intersection:
    """Represents an intersection between two objects (surfaces or polylines)"""
    def __init__(self, id1: int, id2: int, is_polyline_mesh: bool = False):
        self.id1 = id1
        self.id2 = id2
        self.is_polyline_mesh = is_polyline_mesh
        self.points = []  # List of Vector3D points
        self.size = 1.0  # Default size
    
    def add_point(self, point):
        """Add intersection point"""
        self.points.append(point)

class TriplePoint:
    """Represents a triple point where three or more intersections meet"""
    def __init__(self, point):
        self.point = point
        self.intersections = []
    
    def add_intersection(self, intersection):
        """Associate intersection with this triple point"""
        if intersection not in self.intersections:
            self.intersections.append(intersection)

class Surface:
    """Simple surface class for testing."""
    def __init__(self, name, vertices=None, triangles=None):
        self.name = name
        self.vertices = vertices or []  # List of Vector3D
        self.triangles = triangles or []  # List of triangle indices
        self.size = 1.0  # Default size
        self.bounds = [Vector3D(), Vector3D()]  # min, max
        self.convex_hull = []  # List of Vector3D for convex hull points
        self.model = None  # Reference to the model this surface belongs to
        
    def calculate_bounds(self):
        """Calculate bounding box."""
        if not self.vertices:
            return
            
        # Initialize bounds with first vertex
        self.bounds[0] = Vector3D(self.vertices[0].x, self.vertices[0].y, self.vertices[0].z)
        self.bounds[1] = Vector3D(self.vertices[0].x, self.vertices[0].y, self.vertices[0].z)
        
        # Update bounds with remaining vertices
        for v in self.vertices:
            self.bounds[0].x = min(self.bounds[0].x, v.x)
            self.bounds[0].y = min(self.bounds[0].y, v.y)
            self.bounds[0].z = min(self.bounds[0].z, v.z)
            self.bounds[1].x = max(self.bounds[1].x, v.x)
            self.bounds[1].y = max(self.bounds[1].y, v.y)
            self.bounds[1].z = max(self.bounds[1].z, v.z)
            
    def enhanced_calculate_convex_hull(self):
        """
        Calculate convex hull for the surface with the best projection plane.
        This maintains backward compatibility with MeshIt's convex hull structure.
        """
        if not self.vertices:
            return
            
        # Get vertex positions as numpy array
        points = np.array([[v.x, v.y, v.z] for v in self.vertices])
        
        # Find the principal components to get the best projection plane
        mean = np.mean(points, axis=0)
        points_centered = points - mean
        
        # Use SVD to find the best projection plane
        u, s, vh = np.linalg.svd(points_centered, full_matrices=False)
        
        # The last row of vh is the normal of the best-fit plane
        normal = vh[-1]
        
        # Ensure the normal has length 1
        normal = normal / np.linalg.norm(normal)
        
        print(f"Calculated best projection plane with normal: {normal}")
        
        # Construct a rotation that takes the normal to (0,0,1)
        # Find the rotation axis and angle
        axis = np.cross(normal, [0, 0, 1])
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            # Normal is already aligned with z-axis
            rotation_matrix = np.eye(3)
        else:
            # Normalize the rotation axis
            axis = axis / axis_norm
            # Calculate the rotation angle
            angle = np.arccos(np.dot(normal, [0, 0, 1]))
            # Create the rotation matrix using Rodrigues' rotation formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # Project the points onto the x-y plane
        points_rotated = points_centered @ rotation_matrix.T
        points_2d = points_rotated[:, :2]
        
        # Calculate the 2D convex hull
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points_2d)
            hull_points_2d = points_2d[hull.vertices]
            
            # Map the 2D hull points back to 3D
            hull_points_3d = hull_points_2d.copy()
            hull_points_3d = np.column_stack((hull_points_3d, np.zeros(len(hull_points_3d))))
            hull_points_3d = hull_points_3d @ rotation_matrix + mean
            
            # Store the convex hull points as Vector3D objects
            self.convex_hull = [Vector3D(p[0], p[1], p[2]) for p in hull_points_3d]
            
            # For backward compatibility, create a Ns attribute on the convex_hull list
            # This is a workaround to support both old and new visualization code
            class HullWrapper:
                pass
            hull_wrapper = HullWrapper()
            hull_wrapper.Ns = self.convex_hull
            self.convex_hull = hull_wrapper
            
            print(f"Created convex hull with {len(self.convex_hull.Ns)} points")
            
            # Return the rotation matrix and mean for future reference
            return rotation_matrix, mean, normal
            
        except Exception as e:
            print(f"Error calculating convex hull: {str(e)}")
            traceback.print_exc()
            
            # Create a simple rectangular hull as fallback
            min_x, max_x = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])
            min_y, max_y = np.min(points_2d[:, 1]), np.max(points_2d[:, 1])
            
            hull_points_2d = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ])
            
            # Map the 2D hull points back to 3D
            hull_points_3d = hull_points_2d.copy()
            hull_points_3d = np.column_stack((hull_points_3d, np.zeros(len(hull_points_3d))))
            hull_points_3d = hull_points_3d @ rotation_matrix + mean
            
            # Store the convex hull points as Vector3D objects
            hull_points = [Vector3D(p[0], p[1], p[2]) for p in hull_points_3d]
            
            # For backward compatibility
            class HullWrapper:
                pass
            hull_wrapper = HullWrapper()
            hull_wrapper.Ns = hull_points
            self.convex_hull = hull_wrapper
            
            print(f"Created fallback convex hull with {len(self.convex_hull.Ns)} points")
            
            # Return the rotation matrix and mean for future reference
            return rotation_matrix, mean, normal
            
    def align_intersections_to_convex_hull(self):
        """
        Align intersection points to the convex hull exactly as in MeshIt C++.
        
        This function projects intersection points onto the convex hull surface,
        ensuring that intersections properly terminate at the boundary.
        """
        if not self.convex_hull:
            self.enhanced_calculate_convex_hull()
            
        if len(self.convex_hull) < 3:
            print("Warning: Convex hull has less than 3 points, cannot align intersections")
            return self.convex_hull
        
        # Create triangles from convex hull points (similar to MeshIt's C++ implementation)
        hull_points = np.array([[v.x, v.y, v.z] for v in self.convex_hull])
        
        # Find best-fit plane normal for the convex hull
        centroid = np.mean(hull_points, axis=0)
        hull_centered = hull_points - centroid
        u, s, vh = np.linalg.svd(hull_centered, full_matrices=False)
        normal = vh[2, :]  # Third singular vector is normal to best-fitting plane
        normal = normal / np.linalg.norm(normal)
        
        # Create rotation matrix to align hull with XY plane (for triangulation)
        z_axis = np.array([0, 0, 1])
        axis = np.cross(normal, z_axis)
        
        # Handle case where hull is already aligned with XY plane
        if np.linalg.norm(axis) < 1e-12:
            rotation_matrix = np.eye(3)
        else:
            axis = axis / np.linalg.norm(axis)
            cos_theta = np.dot(z_axis, normal)
            sin_theta = np.sqrt(1 - cos_theta**2)
            C = 1 - cos_theta
            
            # Build rotation matrix (similar to MeshIt's rotation)
            r1 = np.array([axis[0]*axis[0]*C + cos_theta, 
                           axis[0]*axis[1]*C - axis[2]*sin_theta, 
                           axis[0]*axis[2]*C + axis[1]*sin_theta])
            
            r2 = np.array([axis[1]*axis[0]*C + axis[2]*sin_theta, 
                           axis[1]*axis[1]*C + cos_theta, 
                           axis[1]*axis[2]*C - axis[0]*sin_theta])
            
            r3 = np.array([axis[2]*axis[0]*C - axis[1]*sin_theta, 
                           axis[2]*axis[1]*C + axis[0]*sin_theta, 
                           axis[2]*axis[2]*C + cos_theta])
            
            rotation_matrix = np.vstack((r1, r2, r3))
        
        # Rotate hull points to align with XY plane
        rotated_hull_points = np.zeros_like(hull_points)
        for i, point in enumerate(hull_points):
            rotated_hull_points[i] = np.dot(rotation_matrix, point - centroid) + centroid
        
        # Extract the 2D points (x, y) for triangulation
        hull_2d_points = rotated_hull_points[:, :2]
        
        # Use Delaunay triangulation on the 2D hull points
        hull_triangles = []
        try:
            tri = Delaunay(hull_2d_points)
            # Convert to our format
            hull_triangles = tri.simplices
        except Exception as e:
            print(f"Error in Delaunay triangulation of hull: {e}")
            # Fallback to a simple fan triangulation
            for i in range(1, len(self.convex_hull) - 1):
                hull_triangles.append([0, i, i+1])
        
        print(f"Created {len(hull_triangles)} triangles from convex hull for projection")
        
        # Find all model intersections that involve this surface
        if not hasattr(self, 'model') or not self.model:
            print("Warning: Surface is not part of a model, skipping intersection alignment")
            return self.convex_hull
            
        surface_idx = None
        for idx, surf in enumerate(self.model.surfaces):
            if surf == self:
                surface_idx = idx
                break
                
        if surface_idx is None:
            print("Warning: Could not find surface in model, skipping intersection alignment")
            return self.convex_hull
        
        # Now project intersection points onto the hull
        for intersection in self.model.intersections:
            if intersection.id1 == surface_idx or intersection.id2 == surface_idx:
                # For each point in the intersection
                for i, point in enumerate(intersection.points):
                    # First check if point is already very close to a hull point
                    for idx, hull_point in enumerate(self.convex_hull):
                        dist_sq = (point.x - hull_point.x)**2 + \
                                  (point.y - hull_point.y)**2 + \
                                  (point.z - hull_point.z)**2
                        if dist_sq < 1e-12:
                            # Already at hull point, just assign exactly
                            intersection.points[i] = hull_point
                            print(f"Aligned intersection point to existing hull point {idx}")
                            break
                    else:  # No break occurred, point not matched to hull
                        # Convert to numpy for projection
                        pt = np.array([point.x, point.y, point.z])
                        
                        # Project point onto each hull triangle and find closest
                        min_dist = float('inf')
                        closest_projection = None
                        
                        for tri in hull_triangles:
                            v1 = hull_points[tri[0]]
                            v2 = hull_points[tri[1]]
                            v3 = hull_points[tri[2]]
                            
                            # Calculate triangle normal
                            tri_normal = np.cross(v2 - v1, v3 - v1)
                            tri_normal = tri_normal / np.linalg.norm(tri_normal)
                            
                            # Calculate plane equation
                            d = -np.dot(tri_normal, v1)
                            
                            # Project point onto plane
                            t = -(np.dot(tri_normal, pt) + d) / np.dot(tri_normal, tri_normal)
                            projected = pt + t * tri_normal
                            
                            # Check if projection is inside triangle
                            area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
                            area1 = 0.5 * np.linalg.norm(np.cross(v2 - projected, v3 - projected))
                            area2 = 0.5 * np.linalg.norm(np.cross(v3 - projected, v1 - projected))
                            area3 = 0.5 * np.linalg.norm(np.cross(v1 - projected, v2 - projected))
                            
                            # Barycentric check for inside/outside
                            if abs(area - (area1 + area2 + area3)) < 1e-6:
                                # Projection is inside triangle
                                dist = np.linalg.norm(projected - pt)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_projection = projected
                        
                        if closest_projection is not None:
                            # Update intersection point
                            new_point = Vector3D(
                                closest_projection[0], 
                                closest_projection[1], 
                                closest_projection[2]
                            )
                            intersection.points[i] = new_point
                            print(f"Projected intersection point to hull triangle, distance: {min_dist}")
        
        return self.convex_hull
        
    def calculate_constraints(self):
        """Calculate constraints for triangulation."""
        if not self.convex_hull:
            self.enhanced_calculate_convex_hull()
        
        # Create constraints from convex hull edges
        constraints = []
        for i in range(len(self.convex_hull) - 1):
            constraints.append([i, i + 1])
        
        return constraints
        
    def triangulate(self, gradient=1.0, base_size=None):
        """Triangulate the surface using triangulate_surface function.
        
        Args:
            gradient: Gradient control parameter (default: 1.0)
            base_size: Base size for triangulation (if None, calculated from bounding box)
            
        Returns:
            The updated surface with triangulation
        """
        print(f"Triangulating surface {self.name} with gradient={gradient}, base_size={base_size}")
        try:
            # Ensure we have vertices to triangulate
            if not self.vertices or len(self.vertices) < 3:
                print(f"Not enough vertices to triangulate surface {self.name}")
                return self
                
            # Calculate convex hull if not already done
            if not hasattr(self, 'convex_hull') or not self.convex_hull:
                print(f"Calculating convex hull for surface {self.name}")
                self.enhanced_calculate_convex_hull()
                
            result = triangulate_surface(self, base_size=base_size, gradient=gradient)
            print(f"Triangulation complete: Created {len(self.triangles)} triangles for surface {self.name}")
            return result
        except Exception as e:
            print(f"Error during surface triangulation: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return self even if triangulation fails
            return self

class Polyline:
    """Simple polyline class for faults and other linear features."""
    def __init__(self, name, vertices=None, segments=None):
        self.name = name
        self.vertices = vertices or []  # List of Vector3D
        self.segments = segments or []  # List of segment indices [start, end]
        self.size = 1.0
        self.type = "Fault"  # Default type
        self.triangles = []  # List of triangle indices
        self.convex_hull = []  # List of Vector3D for convex hull points
        self.bounds = [Vector3D(), Vector3D()]  # min, max bounds
        
    def calculate_segments(self, use_fine_segmentation=False):
        """Calculate segments from vertices."""
        if len(self.vertices) < 2:
            return
            
        self.segments = []
        for i in range(len(self.vertices) - 1):
            self.segments.append([i, i + 1])
            
    def calculate_bounds(self):
        """Calculate bounding box."""
        if not self.vertices:
            return
            
        # Initialize bounds with first vertex
        self.bounds[0] = Vector3D(self.vertices[0].x, self.vertices[0].y, self.vertices[0].z)
        self.bounds[1] = Vector3D(self.vertices[0].x, self.vertices[0].y, self.vertices[0].z)
        
        # Update bounds with remaining vertices
        for v in self.vertices:
            self.bounds[0].x = min(self.bounds[0].x, v.x)
            self.bounds[0].y = min(self.bounds[0].y, v.y)
            self.bounds[0].z = min(self.bounds[0].z, v.z)
            self.bounds[1].x = max(self.bounds[1].x, v.x)
            self.bounds[1].y = max(self.bounds[1].y, v.y)
            self.bounds[1].z = max(self.bounds[1].z, v.z)
            
    def enhanced_calculate_convex_hull(self):
        """Calculate the convex hull of the polyline using the custom triangle wrapper for proper alignment."""
        if not self.vertices:
            print("No vertices to calculate convex hull for polyline")
            return []
            
        # Convert vertices to numpy array for calculation
        points = np.array([[v.x, v.y, v.z] for v in self.vertices])
        print(f"Calculating convex hull for polyline with {len(points)} points")
        
        # Calculate the principal components using SVD for robust plane fitting
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # Use SVD to find the best fitting plane
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        
        # The singular vectors represent the principal directions
        # The last right singular vector is the normal to the best plane
        normal = vh[2, :]
        # The first two right singular vectors define the plane
        v1 = vh[0, :]
        v2 = vh[1, :]
        
        print(f"Calculated best projection plane for polyline with normal: {normal}")
        
        # Project points onto the plane defined by v1 and v2
        points_2d = np.zeros((len(points), 2))
        for i, point in enumerate(centered):
            points_2d[i, 0] = np.dot(point, v1)
            points_2d[i, 1] = np.dot(point, v2)
        
        # Calculate 2D convex hull using our triangle mesh utility
        try:
            # Need at least 3 points for convex hull
            if len(points_2d) < 3:
                raise ValueError("Need at least 3 points for convex hull")
            
            # Use our create_hull_segments utility from triangle_mesh
            hull_vertices, segments = create_hull_segments(points_2d)
            hull_points_2d = points_2d[hull_vertices]
            
            # Convert hull points back to 3D
            self.convex_hull = []
            for i in range(len(hull_points_2d)):
                # Project point back to 3D using the plane basis vectors
                point_3d = centroid + hull_points_2d[i, 0] * v1 + hull_points_2d[i, 1] * v2
                self.convex_hull.append(Vector3D(point_3d[0], point_3d[1], point_3d[2]))
                
            print(f"Created polyline convex hull with {len(self.convex_hull)} points")
        except Exception as e:
            print(f"Error calculating polyline convex hull: {e}")
            print("Falling back to rectangular hull")
            
            # Find the bounding box of the 2D points
            min_x = np.min(points_2d[:, 0])
            max_x = np.max(points_2d[:, 0])
            min_y = np.min(points_2d[:, 1])
            max_y = np.max(points_2d[:, 1])
            
            # Create rectangular hull points in 2D
            rect_points_2d = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ])
            
            # Convert rectangular hull points back to 3D
            self.convex_hull = []
            for p in rect_points_2d:
                point_3d = centroid + p[0]*v1 + p[1]*v2
                self.convex_hull.append(Vector3D(point_3d[0], point_3d[1], point_3d[2]))
        
        # Store the projection data for later use in triangulation
        self._projection_data = {
            'centroid': centroid,
            'normal': normal,
            'v1': v1,
            'v2': v2
        }
        
        return self.convex_hull
            
    def triangulate(self, gradient=1.0, base_size=None):
        """Triangulate the fault using our custom TriangleWrapper."""
        if not self.vertices:
            print("No vertices to triangulate fault")
            return self
            
        # First calculate convex hull if not already done
        if not hasattr(self, 'convex_hull') or not self.convex_hull:
            print("No convex hull found, calculating now")
            self.enhanced_calculate_convex_hull()
            
        # Calculate segments if not already calculated
        if not hasattr(self, 'segments') or not self.segments:
            print("No segments found, calculating now")
            self.calculate_segments(use_fine_segmentation=False)
        
        # Convert vertices to numpy array for calculation
        points = np.array([[v.x, v.y, v.z] for v in self.vertices])
        print(f"Triangulating fault with {len(points)} vertices")
        
        # Use projection data from convex hull calculation if available
        if hasattr(self, '_projection_data') and self._projection_data:
            print("Using stored projection data from convex hull")
            centroid = self._projection_data['centroid']
            normal = self._projection_data['normal']
            v1 = self._projection_data['v1']
            v2 = self._projection_data['v2']
        else:
            # Calculate principal directions using SVD for robust plane fitting
            print("Calculating new projection plane")
            centroid = np.mean(points, axis=0)
            centered = points - centroid
            
            # Use SVD to find the best fitting plane
            u, s, vh = np.linalg.svd(centered, full_matrices=False)
            
            # The singular vectors represent the principal directions
            # The last right singular vector is the normal to the best plane
            normal = vh[2, :]
            # The first two right singular vectors define the plane
            v1 = vh[0, :]
            v2 = vh[1, :]
            
            # Store projection data for future use
            self._projection_data = {
                'centroid': centroid,
                'normal': normal,
                'v1': v1,
                'v2': v2
            }
        
        # Project points onto the plane defined by v1 and v2
        points_2d = np.zeros((len(points), 2))
        for i, point in enumerate(points - centroid):
            points_2d[i, 0] = np.dot(point, v1)
            points_2d[i, 1] = np.dot(point, v2)
        
        print(f"Projected fault points to 2D plane: {len(points_2d)} points")
        
        # Apply jitter to avoid precision issues
        jitter = 1e-10
        jittered_points = points_2d + np.random.uniform(-jitter, jitter, points_2d.shape)
        
        # Calculate appropriate base_size if not provided
        if base_size is None:
            # Calculate bounding box diagonal as reference
            min_x = np.min(jittered_points[:, 0])
            max_x = np.max(jittered_points[:, 0])
            min_y = np.min(jittered_points[:, 1])
            max_y = np.max(jittered_points[:, 1])
            diagonal = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
            base_size = diagonal / 15.0
            print(f"Automatically calculated base_size: {base_size:.6f}")
            
        # Ensure base_size is not too small
        min_acceptable_size = 1e-3
        if base_size < min_acceptable_size:
            print(f"Warning: base_size {base_size} is too small, increasing to {min_acceptable_size}")
            base_size = min_acceptable_size
        
        # Calculate boundary segments - try convex hull first, fallback to rectangle if it fails
        try:
            # Use create_hull_segments from triangle_mesh
            hull_vertices, segments = create_hull_segments(jittered_points)
            print(f"Created convex hull segments with {len(segments)} segments")
        except Exception as e:
            print(f"Error in hull segments calculation: {e}")
            print("Creating rectangular segments instead")
            
            # Create rectangular bounds
            min_x = np.min(jittered_points[:, 0])
            max_x = np.max(jittered_points[:, 0])
            min_y = np.min(jittered_points[:, 1])
            max_y = np.max(jittered_points[:, 1])
            
            # Add a small margin
            margin = (max_x - min_x) * 0.05
            min_x -= margin
            max_x += margin
            min_y -= margin
            max_y += margin
            
            # Create rectangular segments
            segments = np.array([
                [0, 1], [1, 2], [2, 3], [3, 0]
            ])
            
            # Create rectangular vertices
            rect_points = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ])
            
            # Add these to the beginning of the points array
            orig_points_2d = jittered_points.copy()
            jittered_points = np.vstack([rect_points, orig_points_2d])
            hull_vertices = np.array([0, 1, 2, 3])
            
            print(f"Created rectangular bounds with {len(segments)} segments")
        
        # Create triangle wrapper with more relaxed parameters
        wrapper = TriangleWrapper(gradient=gradient, min_angle=10.0, base_size=base_size)
        
        # Set fault segments as feature points for better triangulation
        feature_points = []
        feature_sizes = []
        
        # Add convex hull points as features
        hull_point_size = base_size * 0.5
        if len(hull_vertices) > 0:
            for idx in hull_vertices:
                feature_points.append(jittered_points[idx])
                feature_sizes.append(hull_point_size)
                
        # Add midpoints of hull segments if available
        if len(segments) > 0:
            for seg in segments:
                p1_idx = seg[0]
                p2_idx = seg[1]
                if p1_idx < len(jittered_points) and p2_idx < len(jittered_points):
                    midpoint = (jittered_points[p1_idx] + jittered_points[p2_idx]) / 2.0
                    feature_points.append(midpoint)
                    feature_sizes.append(hull_point_size * 0.8)  # Slightly smaller size for segments
        
        # If we have feature points, set them
        if feature_points:
            wrapper.set_feature_points(np.array(feature_points), np.array(feature_sizes))
            print(f"Added {len(feature_points)} feature points for refinement")
        
        # Try triangulation with increasingly relaxed parameters if needed
        try:
            print(f"Running triangulation with gradient={gradient}")
            result = wrapper.triangulate(jittered_points, segments)
            print(f"Triangulation complete with {len(result['triangles'])} triangles")
        except Exception as e:
            print(f"Error with TriangleWrapper: {e}")
            print("Trying direct triangulation with Triangle library...")
            
            try:
                # Simplest option that preserves boundary
                tri_opts = "p"  # Just preserve boundary, no quality constraints
                tri_data = {
                    'vertices': jittered_points,
                    'segments': segments
                }
                result = tr.triangulate(tri_data, tri_opts)
                print(f"Basic triangulation complete with {len(result['triangles'])} triangles")
                
                # Add refinement points to improve quality
                refinement_points = []
                
                # Include feature points if available
                if feature_points:
                    refinement_points.extend(feature_points)
                
                # Add interior points
                interior_points = []
                for i in range(len(jittered_points)):
                    if i not in hull_vertices:
                        interior_points.append(jittered_points[i])
                
                if interior_points:
                    # Add centroid
                    centroid_interior = np.mean(interior_points, axis=0)
                    refinement_points.append(centroid_interior)
                    
                    # Add some random interior points for better triangulation
                    if len(interior_points) > 3:
                        for _ in range(min(10, len(interior_points))):
                            idx1 = np.random.randint(0, len(interior_points))
                            idx2 = np.random.randint(0, len(interior_points))
                            if idx1 != idx2:
                                avg_point = (interior_points[idx1] + interior_points[idx2]) / 2.0
                                refinement_points.append(avg_point)
                
                # Add refinement points to vertices if any
                if refinement_points:
                    print(f"Adding {len(refinement_points)} refinement points")
                    all_vertices = np.vstack((jittered_points, refinement_points))
                    
                    # Re-triangulate with refinement points
                    tri_data = {
                        'vertices': all_vertices,
                        'segments': segments
                    }
                    result = tr.triangulate(tri_data, tri_opts)
                    print(f"Refined triangulation complete with {len(result['triangles'])} triangles")
                
            except Exception as e2:
                print(f"Basic triangulation also failed: {e2}")
                print("Falling back to simple fan triangulation")
                
                # Create a very simple fan triangulation from hull to centroid
                all_points = jittered_points
                
                # Calculate centroid
                centroid_point = np.mean(all_points, axis=0)
                centroid_idx = len(all_points)
                
                # Add centroid to points
                all_points = np.vstack((all_points, [centroid_point]))
                
                # Create triangles connecting boundary segments to centroid
                triangles = []
                for i in range(len(segments)):
                    triangles.append([segments[i][0], segments[i][1], centroid_idx])
                    
                result = {
                    'vertices': all_points,
                    'triangles': np.array(triangles)
                }
                print(f"Created simple fan triangulation with {len(triangles)} triangles")
            
        # Convert triangulated points back to 3D and update
        vertices_3d = []
        
        # Original vertex count for indexing
        orig_vertex_count = len(self.vertices)
        
        # Map output vertices to 3D
        for i, v in enumerate(result['vertices']):
            # Project back to 3D
            point_3d = centroid + v[0] * v1 + v[1] * v2
            vertices_3d.append(Vector3D(point_3d[0], point_3d[1], point_3d[2]))
        
        # Update vertices, replacing original ones
        self.vertices = vertices_3d
        
        # Create triangles with correct indices
        self.triangles = []
        for tri in result['triangles']:
            self.triangles.append([tri[0], tri[1], tri[2]])
        
        print(f"Created {len(self.triangles)} triangles for fault")
        return self

class SimpleModel:
    """Simple model class for testing."""
    
    def __init__(self):
        self.surfaces = []
        self.faults = []  # Add missing faults attribute
        self.intersections = []
        self.triple_points = []
        self.model_polylines = []  # Store faults and other polylines
        self.fault = None  # Store the main fault separately
    
    def add_surface(self, surface):
        """Add a surface to the model."""
        self.surfaces.append(surface)
        return len(self.surfaces) - 1
        
    def add_fault(self, fault):
        """Add a fault (polyline) to the model."""
        self.model_polylines.append(fault)
        self.fault = fault  # Store reference to the main fault
        # Ensure fault is added to faults list
        self.faults.append(fault)
        return len(self.faults) - 1
    
    def calculate_size_of_intersections(self, surface1_idx, surface2_idx):
        """Calculate size parameters for intersections."""
        if surface1_idx >= len(self.surfaces) or surface2_idx >= len(self.surfaces):
            raise ValueError("Invalid surface indices")
        
        surface1 = self.surfaces[surface1_idx]
        surface2 = self.surfaces[surface2_idx]
        
        if not surface1.vertices or not surface1.triangles or not surface2.vertices or not surface2.triangles:
            return 0
        
        intersection_points = []
        
        # Direct surface-surface intersection - check triangles against each other
        for tri1_idx, tri1 in enumerate(surface1.triangles):
            v1_1 = surface1.vertices[tri1[0]]
            v1_2 = surface1.vertices[tri1[1]]
            v1_3 = surface1.vertices[tri1[2]]
            
            for tri2_idx, tri2 in enumerate(surface2.triangles):
                v2_1 = surface2.vertices[tri2[0]]
                v2_2 = surface2.vertices[tri2[1]]
                v2_3 = surface2.vertices[tri2[2]]
                
                # Check triangle edges vs other triangle
                # Check each edge of triangle 1 against triangle 2
                edges1 = [
                    (v1_1, v1_2),
                    (v1_2, v1_3),
                    (v1_3, v1_1)
                ]
                
                for start, end in edges1:
                    intersection = self.segment_triangle_intersection(
                        start, end, v2_1, v2_2, v2_3
                    )
                    if intersection:
                        intersection_points.append(intersection)
                
                # Check each edge of triangle 2 against triangle 1
                edges2 = [
                    (v2_1, v2_2),
                    (v2_2, v2_3),
                    (v2_3, v2_1)
                ]
                
                for start, end in edges2:
                    intersection = self.segment_triangle_intersection(
                        start, end, v1_1, v1_2, v1_3
                    )
                    if intersection:
                        intersection_points.append(intersection)
        
        if not intersection_points:
            return 0
        
        # Calculate average size of involved surfaces
        size1 = getattr(surface1, 'size', 1.0)
        size2 = getattr(surface2, 'size', 1.0)
        intersection_size = min(size1, size2) * 0.5
        
        # Store intersection points for visualization
        if len(intersection_points) > 0:
            # Create a polyline from intersection points
            from collections import defaultdict
            
            print(f"Found {len(intersection_points)} intersection points")
            
            # Create a polyline to represent the intersection
            intersection_polyline = Polyline(f"Intersection_{surface1_idx}_{surface2_idx}")
            
            # Add unique points to the polyline
            unique_points = []
            for point in intersection_points:
                # Check if this point is already in the list (within tolerance)
                is_duplicate = False
                for existing in unique_points:
                    dx = point.x - existing.x
                    dy = point.y - existing.y
                    dz = point.z - existing.z
                    dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                    if dist < 1e-5:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_points.append(point)
            
            # Add points to polyline
            for point in unique_points:
                intersection_polyline.vertices.append(point)
                
            # Add polyline to model
            self.model_polylines.append(intersection_polyline)
        
        return intersection_size
    
    def segment_triangle_intersection(self, start, end, v1, v2, v3):
        """Calculate intersection point between a line segment and a triangle."""
        # Convert to numpy arrays for calculation
        p1 = np.array([start.x, start.y, start.z])
        p2 = np.array([end.x, end.y, end.z])
        triangle = np.array([
            [v1.x, v1.y, v1.z],
            [v2.x, v2.y, v2.z],
            [v3.x, v3.y, v3.z]
        ])
        
        # Calculate triangle normal
        edge1 = triangle[1] - triangle[0]
        edge2 = triangle[2] - triangle[0]
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)
        
        # Calculate intersection with triangle plane
        segment = p2 - p1
        d = np.dot(normal, triangle[0])
        
        if abs(np.dot(normal, segment)) < 1e-10:
            return None  # Segment is parallel to triangle
            
        t = (d - np.dot(normal, p1)) / np.dot(normal, segment)
        
        if t < 0 or t > 1:
            return None  # Intersection point not on segment
            
        # Calculate intersection point
        point = p1 + t * segment
        
        # Check if point is inside triangle
        if self.point_in_triangle(point, triangle):
            return Vector3D(point[0], point[1], point[2])
        
        return None
    
    def point_in_triangle(self, point, triangle):
        """Check if a point lies within a triangle using barycentric coordinates."""
        v0 = triangle[1] - triangle[0]
        v1 = triangle[2] - triangle[0]
        v2 = point - triangle[0]
        
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            return False
            
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        return (v >= 0 and w >= 0 and u >= 0 and v + w <= 1)

class SurfaceViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Surface Intersection Viewer")
        self.model = SimpleModel()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Surface 1 frame
        self.surface1_frame = ttk.LabelFrame(self.main_frame, text="Surface 1", padding="5")
        self.surface1_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Button(self.surface1_frame, text="Load Vertices", 
                   command=lambda: self.load_vertices(1)).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(self.surface1_frame, text="Load Triangles",
                   command=lambda: self.load_triangles(1)).grid(row=0, column=1, padx=5, pady=2)
        
        # Surface 2 frame
        self.surface2_frame = ttk.LabelFrame(self.main_frame, text="Surface 2", padding="5")
        self.surface2_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Button(self.surface2_frame, text="Load Vertices",
                   command=lambda: self.load_vertices(2)).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(self.surface2_frame, text="Load Triangles",
                   command=lambda: self.load_triangles(2)).grid(row=0, column=1, padx=5, pady=2)
        
        # Add fault/polyline frame
        self.fault_frame = ttk.LabelFrame(self.main_frame, text="Fault/Polyline", padding="5")
        self.fault_frame.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Button(self.fault_frame, text="Load Fault Points",
                   command=self.load_fault).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(self.fault_frame, text="Calculate Intersections",
                   command=self.calculate_fault_intersections).grid(row=0, column=1, padx=5, pady=2)
        
        # Visualization options frame
        self.viz_frame = ttk.LabelFrame(self.main_frame, text="Visualization Options", padding="5")
        self.viz_frame.grid(row=3, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.show_points_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.viz_frame, text="Show Points", 
                       variable=self.show_points_var).grid(row=0, column=0, padx=5)
        
        self.show_triangles_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.viz_frame, text="Show Triangles",
                       variable=self.show_triangles_var).grid(row=0, column=1, padx=5)
        
        self.show_faces_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.viz_frame, text="Show Faces",
                       variable=self.show_faces_var).grid(row=0, column=2, padx=5)
        
        self.show_edges_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.viz_frame, text="Show Edges",
                       variable=self.show_edges_var).grid(row=0, column=3, padx=5)
        
        self.show_intersections_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.viz_frame, text="Show Intersections",
                       variable=self.show_intersections_var).grid(row=0, column=4, padx=5)
        
        self.show_fault_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.viz_frame, text="Show Fault",
                       variable=self.show_fault_var).grid(row=0, column=5, padx=5)
        
        self.show_triangulation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.viz_frame, text="Show Triangulation",
                       variable=self.show_triangulation_var).grid(row=0, column=6, padx=5)
        
        # Triangulation parameters frame
        self.triangulation_frame = ttk.LabelFrame(self.main_frame, text="Triangulation Parameters", padding="5")
        self.triangulation_frame.grid(row=4, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(self.triangulation_frame, text="Base Size:").grid(row=0, column=0, padx=5, pady=2)
        self.base_size_var = tk.DoubleVar(value=0.1)  # Default base size
        self.base_size_entry = ttk.Entry(self.triangulation_frame, textvariable=self.base_size_var, width=8)
        self.base_size_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(self.triangulation_frame, text="(Smaller values = denser triangulation)").grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Button(self.triangulation_frame, text="Triangulate Surfaces", 
                  command=self.triangulate_surfaces).grid(row=0, column=3, padx=5, pady=2)
        
        # Add export format selection
        self.export_frame = ttk.LabelFrame(self.main_frame, text="Export Options", padding="5")
        self.export_frame.grid(row=5, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.export_format = tk.StringVar(value='csv')
        ttk.Label(self.export_frame, text="Export Format:").grid(row=0, column=0, padx=5)
        ttk.Radiobutton(self.export_frame, text="CSV", variable=self.export_format, 
                       value='csv').grid(row=0, column=1, padx=5)
        ttk.Radiobutton(self.export_frame, text="TXT", variable=self.export_format, 
                       value='txt').grid(row=0, column=2, padx=5)
        ttk.Radiobutton(self.export_frame, text="DAT", variable=self.export_format, 
                       value='dat').grid(row=0, column=3, padx=5)
        
        # Add export buttons
        ttk.Button(self.export_frame, text="Export Surface 1",
                   command=lambda: self.export_surface(0)).grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        ttk.Button(self.export_frame, text="Export Surface 2",
                   command=lambda: self.export_surface(1)).grid(row=1, column=2, columnspan=2, padx=5, pady=5)
        
        # Action buttons
        self.action_frame = ttk.Frame(self.main_frame)
        self.action_frame.grid(row=6, column=0, pady=10)
        
        ttk.Button(self.action_frame, text="Visualize Current",
                   command=self.visualize).grid(row=0, column=0, padx=5)
        ttk.Button(self.action_frame, text="Visualize All Steps",
                   command=self.visualize_all_steps).grid(row=0, column=1, padx=5)
        ttk.Button(self.action_frame, text="Calculate Intersection",
                   command=self.calculate_intersection).grid(row=0, column=2, padx=5)
        ttk.Button(self.action_frame, text="Clear All",
                   command=self.clear_all).grid(row=0, column=3, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var)
        self.status_bar.grid(row=7, column=0, pady=5)
        self.status_var.set("Ready")
        
        # Initialize empty surfaces
        self.surfaces = [Surface("Surface 1"), Surface("Surface 2")]
        self.model.surfaces = self.surfaces
        
        # Initialize empty fault
        self.fault = Polyline("Fault 1")
        self.model.model_polylines = [self.fault]
    
    def load_file_data(self, filename):
        """Load data from various file formats with support for complex separators."""
        try:
            # First try reading raw lines to handle complex formats
            points = []
            with open(filename, 'r') as f:
                for line in f:
                    # Remove any whitespace and skip empty lines
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Replace semicolons with spaces
                    line = line.replace(';', ' ')
                    # Replace commas with spaces
                    line = line.replace(',', ' ')
                    # Replace multiple spaces with single space
                    line = ' '.join(line.split())
                    
                    # Split on space
                    values = line.split(' ')
                    coords = []
                    
                    for val in values:
                        # Clean the value - remove any non-numeric chars except - and +
                        clean_val = ''.join(c for c in val if c.isdigit() or c in '.-+')
                        if clean_val:
                            try:
                                coords.append(float(clean_val))
                            except ValueError:
                                continue
                    
                    # If we have at least 2 coordinates, add them
                    if len(coords) >= 2:
                        # If only 2 coordinates provided, add z=0
                        if len(coords) == 2:
                            coords.append(0.0)
                        # Take only the first 3 coordinates if more are provided
                        points.append(coords[:3])
            
            if not points:
                # If manual parsing failed, try pandas with different separators
                separators = [',', '\t', ' ', ';']
                data = None
                
                for sep in separators:
                    try:
                        data = pd.read_csv(filename, sep=sep, header=None)
                        if data.shape[1] >= 2 and len(data) > 0:
                            # Convert to points format
                            for _, row in data.iterrows():
                                coords = [float(x) for x in row[:3]]
                                if len(coords) == 2:
                                    coords.append(0.0)
                                points.append(coords[:3])
                            break
                    except:
                        continue
            
            if not points:
                raise ValueError("No valid points found in file")
            
            # Convert points to DataFrame
            data = pd.DataFrame(points, columns=['x', 'y', 'z'])
            return data
        
        except Exception as e:
            raise Exception(f"Error loading file {filename}: {str(e)}")
    
    def load_vertices(self, surface_idx):
        """Load vertices from a file."""
        filename = filedialog.askopenfilename(
            title=f"Load vertices for Surface {surface_idx}",
            filetypes=[
                ("All supported files", "*.csv;*.txt;*.dat"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("DAT files", "*.dat"),
                ("All files", "*.*")
            ]
        )
        if not filename:
            return
            
        try:
            # Load data using the new function
            data = self.load_file_data(filename)
            
            # Clear existing vertices
            self.surfaces[surface_idx-1].vertices = []
            
            # Convert to Vector3D objects
            for _, row in data.iterrows():
                # Skip index column if present
                x = float(row.get('x', row.iloc[0]))
                y = float(row.get('y', row.iloc[1]))
                z = float(row.get('z', row.iloc[2]))
                self.surfaces[surface_idx-1].vertices.append(Vector3D(x, y, z))
            
            self.status_var.set(f"Loaded {len(self.surfaces[surface_idx-1].vertices)} vertices for Surface {surface_idx}")
            
            # Visualize the loaded points
            visualize_step(
                self.model,
                f"Surface {surface_idx} Vertices",
                1,
                show_intersections=False,
                show_convex_hull=False,
                show_points=True,
                show_triangles=False,
                show_fault=True,
                show_faces=self.show_faces_var.get(),
                show_edges=self.show_edges_var.get()
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load vertices: {str(e)}")
    
    def load_triangles(self, surface_idx):
        """Load triangle indices from a file."""
        filename = filedialog.askopenfilename(
            title=f"Load triangles for Surface {surface_idx}",
            filetypes=[
                ("All supported files", "*.csv;*.txt;*.dat"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("DAT files", "*.dat"),
                ("All files", "*.*")
            ]
        )
        if not filename:
            return
            
        try:
            # Load data using the new function
            data = self.load_file_data(filename)
            
            # Clear existing triangles
            self.surfaces[surface_idx-1].triangles = []
            
            # Convert to triangle indices
            for _, row in data.iterrows():
                # Skip index column if present
                v1 = int(row.get('v1', row.iloc[0]))
                v2 = int(row.get('v2', row.iloc[1]))
                v3 = int(row.get('v3', row.iloc[2]))
                self.surfaces[surface_idx-1].triangles.append([v1, v2, v3])
            
            self.status_var.set(f"Loaded {len(self.surfaces[surface_idx-1].triangles)} triangles for Surface {surface_idx}")
            
            # Visualize the loaded triangles
            visualize_step(
                self.model,
                f"Surface {surface_idx} Triangles",
                1,
                show_intersections=False,
                show_convex_hull=False,
                show_points=False,
                show_triangles=True,
                show_fault=True,
                show_faces=self.show_faces_var.get(),
                show_edges=self.show_edges_var.get()
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load triangles: {str(e)}")
    
    def load_fault(self):
        """Load fault points from a file."""
        filename = filedialog.askopenfilename(
            title="Load Fault Points",
            filetypes=[
                ("All supported files", "*.csv;*.txt;*.dat"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("DAT files", "*.dat"),
                ("All files", "*.*")
            ]
        )
        if not filename:
            return
            
        try:
            # Load data
            data = self.load_file_data(filename)
            
            # Clear existing vertices
            self.fault.vertices = []
            
            # Convert to Vector3D objects
            for _, row in data.iterrows():
                x = float(row.get('x', row.iloc[0]))
                y = float(row.get('y', row.iloc[1]))
                z = float(row.get('z', row.iloc[2]))
                self.fault.vertices.append(Vector3D(x, y, z))
            
            # Calculate segments
            self.fault.calculate_segments()
            
            self.status_var.set(f"Loaded {len(self.fault.vertices)} fault points")
            
            # Visualize the loaded fault
            visualize_step(
                self.model,
                "Fault Points",
                1,
                show_intersections=False,
                show_convex_hull=False,
                show_points=True,
                show_triangles=False,
                show_fault=True,
                show_faces=self.show_faces_var.get(),
                show_edges=self.show_edges_var.get()
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load fault points: {str(e)}")
    
    def calculate_fault_intersections(self):
        """Calculate intersections between fault and surfaces."""
        if not self.fault.vertices:
            messagebox.showerror("Error", "No fault data loaded")
            return
            
        try:
            # Calculate intersections with each surface
            new_intersections = []
            for i, surface in enumerate(self.surfaces):
                if surface.vertices and surface.triangles:
                    intersection = self.model.calculate_polyline_surface_intersection(0, i)
                    if intersection:
                        new_intersections.append(intersection)
            
            # Update model's intersections
            self.model.intersections.extend(new_intersections)
            
            if new_intersections:
                self.status_var.set(f"Found {len(new_intersections)} fault intersections")
                self.visualize()
            else:
                self.status_var.set("No fault intersections found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate fault intersections: {str(e)}")
    
    def visualize(self):
        """Visualize the current surfaces and fault."""
        try:
            # Check if we have any data to visualize
            has_data = any(s.vertices for s in self.surfaces) or (self.fault and self.fault.vertices)
            
            if not has_data:
                messagebox.showinfo("No Data", "No surfaces or fault data loaded. Please load data first.")
                return
                
            # Log which surfaces have data
            print("Visualization status:")
            for i, surface in enumerate(self.surfaces):
                print(f"  Surface {i+1}: {len(surface.vertices)} vertices, {len(surface.triangles)} triangles")
            
            if self.fault:
                print(f"  Fault: {len(self.fault.vertices)} vertices, {len(self.fault.segments)} segments")
            
            print(f"  Intersections: {len(self.model.intersections)}")
            print(f"  Triple points: {len(self.model.triple_points)}")
            
            # Try to visualize
            visualize_step(
                self.model,
                "Surface and Fault Visualization",
                1,
                show_intersections=self.show_intersections_var.get(),
                show_convex_hull=False,
                show_points=self.show_points_var.get(),
                show_triangles=self.show_triangles_var.get(),
                show_fault=self.show_fault_var.get(),
                show_triangulation=self.show_triangulation_var.get(),
                show_faces=self.show_faces_var.get(),
                show_edges=self.show_edges_var.get()
            )
            self.status_var.set("Visualization complete")
            
        except Exception as e:
            error_msg = str(e)
            self.status_var.set(f"Visualization error: {error_msg[:50]}...")
            print(f"Visualization error: {error_msg}")
            messagebox.showerror("Error", f"Failed to visualize: {error_msg}")
    
    def calculate_intersection(self):
        """Calculate intersection between surfaces."""
        if not all(s.vertices for s in self.surfaces):
            messagebox.showerror("Error", "Both surfaces must have vertices loaded")
            return
            
        try:
            intersection = calculate_surface_surface_intersection(0, 1, self.model)
            if intersection:
                self.model.intersections = [intersection]
                self.status_var.set(f"Found intersection with {len(intersection.points)} points")
                self.visualize()
            else:
                self.status_var.set("No intersection found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate intersection: {str(e)}")
    
    def clear_all(self):
        """Clear all loaded data."""
        self.surfaces = [Surface("Surface 1"), Surface("Surface 2")]
        self.model.surfaces = self.surfaces
        self.fault = Polyline("Fault 1")
        self.model.model_polylines = [self.fault]
        self.model.intersections = []
        self.status_var.set("All data cleared")
    
    def export_surface(self, surface_idx):
        """Export surface data in the selected format."""
        if not self.surfaces[surface_idx].vertices:
            messagebox.showerror("Error", f"Surface {surface_idx + 1} has no data to export")
            return
            
        try:
            # Get export directory
            directory = filedialog.askdirectory(
                title=f"Choose export directory for Surface {surface_idx + 1}"
            )
            if not directory:
                return
                
            # Create base filename
            base = os.path.join(directory, f"surface{surface_idx + 1}")
            
            # Export using the new function
            self.export_surface_to_files(
                self.surfaces[surface_idx],
                base,
                self.export_format.get()
            )
            
            self.status_var.set(f"Exported Surface {surface_idx + 1} data successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export surface: {str(e)}")

    def visualize_all_steps(self, base_size=None, gradient=1.0):
        """Visualize the different steps of the algorithm"""
        # Step 1: Original vertices
        visualize_step(self.model, "Step 1: Initial Vertices", 1, 
                      show_convex_hull=False, show_triangles=False)
        
        # Step 2: Convex Hull
        # Calculate convex hull for each surface if not already calculated
        for surface in self.model.surfaces:
            if not hasattr(surface, "convex_hull") or surface.convex_hull is None:
                surface.enhanced_calculate_convex_hull()
                
        visualize_step(self.model, "Step 2: Convex Hull Calculation", 2, 
                      show_triangles=False)
        
        # Step 3: Coarse segmentation
        if self.model.fault:
            # Calculate segments for fault polyline if not already calculated
            if not hasattr(self.model.fault, "segments") or not self.model.fault.segments:
                self.model.fault.calculate_segments(use_fine_segmentation=False)
                
            # Calculate intersection with surfaces if not already calculated
            if not hasattr(self.model, "intersections") or not self.model.intersections:
                for i, surface in enumerate(self.model.surfaces):
                    self.model.calculate_polyline_surface_intersection(0, i)
                    
        visualize_step(self.model, "Step 3: Coarse Segmentation", 3, 
                      show_triangles=False)
        
        # Step 4: Coarse triangulation
        self.triangulate_surfaces(base_size=base_size, gradient=gradient)
        
        visualize_step(self.model, "Step 4: Coarse Triangulation", 4)

    def triangulate_surfaces(self, base_size=None, gradient=1.0):
        """
        Triangulate all surfaces and faults in the model.
        
        Args:
            base_size: Edge length used for refinement (RefineByLength parameter in MeshIt).
                       If None, will be calculated based on model size.
            gradient: Controls triangle size growth (lower = more uniform triangles)
        """
        # Clear any existing triangulation
        for surface in self.surfaces + self.faults:
            surface.triangles = []
        
        # Calculate a reasonable base_size if not specified
        if base_size is None:
            # Get model bounding box
            all_vertices = []
            for surface in self.surfaces + self.faults:
                all_vertices.extend([[v.x, v.y, v.z] for v in surface.vertices])
        
            # No vertices, can't calculate
            if not all_vertices:
                print("No vertices found in model, can't calculate base_size")
                return
            
            # Calculate bounding box
            all_vertices = np.array(all_vertices)
            min_coords = np.min(all_vertices, axis=0)
            max_coords = np.max(all_vertices, axis=0)
        
            # Calculate diagonal length
            diagonal = np.linalg.norm(max_coords - min_coords)
        
            # Set base_size to 1/20 of diagonal (MeshIt typically uses 1/15 to 1/20)
            base_size = diagonal / 20.0
            print(f"Auto-calculated refinement length (base_size): {base_size}")
        else:
            print(f"Using user-specified refinement length (base_size): {base_size}")
        
        # Triangulate each surface
        for i, surface in enumerate(self.surfaces + self.faults):
            print(f"Triangulating {'surface' if i < len(self.surfaces) else 'fault'} {i % len(self.surfaces) + 1}...")
            
            try:
                # Visualize initial points
                self.visualize_step(f"Initial points of {'surface' if i < len(self.surfaces) else 'fault'} {i % len(self.surfaces) + 1}")
                
                # Calculate convex hull
                hull = ConvexHull(np.array([[v.x, v.y, v.z] for v in surface.vertices]))
                
                # Triangulate the surface using MeshIt's approach
                triangulate_surface(surface, base_size=base_size, gradient=gradient)
                
                # Visualize after triangulation
                self.visualize_step(f"Triangulated {'surface' if i < len(self.surfaces) else 'fault'} {i % len(self.surfaces) + 1}")
                
            except Exception as e:
                print(f"Error triangulating {'surface' if i < len(self.surfaces) else 'fault'} {i % len(self.surfaces) + 1}: {str(e)}")
        
        # Calculate intersections for visualization
        self.intersections = []
        print("Calculating intersections for visualization...")
        try:
            # Add all surface-surface intersections
            for i in range(len(self.surfaces)):
                for j in range(i+1, len(self.surfaces)):
                    # Skip if either surface has no triangles
                    if not self.surfaces[i].triangles or not self.surfaces[j].triangles:
                        continue
                    
                    intersections = self.surfaces[i].intersect(self.surfaces[j])
                    if intersections:
                        self.intersections.extend(intersections)
                        
            # Add all surface-fault intersections
            for i in range(len(self.surfaces)):
                for j in range(len(self.faults)):
                    # Skip if either surface has no triangles
                    if not self.surfaces[i].triangles or not self.faults[j].triangles:
                        continue
                    
                    intersections = self.surfaces[i].intersect(self.faults[j])
                    if intersections:
                        self.intersections.extend(intersections)
        except Exception as e:
            print(f"Error calculating intersections: {str(e)}")

def triangulate_surface(points_or_surface, base_size=None, gradient=1.0):
    """
    Triangulate a surface using MeshIt's approach with precise refinement control.
    
    Args:
        points_or_surface: Either a Surface object or a list of 3D points
        base_size: Base size for the triangulation (if None, calculated from bounding box)
        gradient: Gradient control parameter (1.0 is uniform, higher values allow more variation)
        
    Returns:
        If input is Surface: The updated Surface object
        If input is points: Dict with 'vertices' and 'triangles'
    """
    # Check if input is Surface object
    is_surface_obj = False
    if hasattr(points_or_surface, '__class__') and points_or_surface.__class__.__name__ == 'Surface':
        is_surface_obj = True
        vertices_3d = np.array([[v.x, v.y, v.z] for v in points_or_surface.vertices])
        surface_name = points_or_surface.name if hasattr(points_or_surface, 'name') else "Unknown"
        print(f"Triangulating surface {surface_name}...")
        print(f"Triangulating surface {surface_name} with gradient={gradient}, base_size={base_size}")
    else:
        # Convert input points to numpy array
        vertices_3d = np.array(points_or_surface)
        print(f"Triangulating point cloud with gradient={gradient}, base_size={base_size}")
    
    # Need at least 3 points to triangulate
    if len(vertices_3d) < 3:
        raise ValueError("Need at least 3 points to triangulate a surface")
    
    # Calculate surface normal for projection
    normal = np.zeros(3)
    
    # Find the surface normal by computing cross products of triangles and averaging
    triangles_sample = min(len(vertices_3d) - 2, 10)  # Use up to 10 triangles
    for i in range(triangles_sample):
        # Use points that are far apart for better normal calculation
        idx1 = 0
        idx2 = (i * 13 + 1) % len(vertices_3d)  # Use prime number to avoid patterns
        idx3 = (i * 17 + 2) % len(vertices_3d)
        
        p1 = vertices_3d[idx1]
        p2 = vertices_3d[idx2]
        p3 = vertices_3d[idx3]
        
        v1 = p2 - p1
        v2 = p3 - p1
        
        # Cross product
        cross = np.cross(v1, v2)
        len_cross = np.linalg.norm(cross)
        
        # Only use if triangle is not degenerate
        if len_cross > 1e-8:
            normal += cross / len_cross

    # Normalize the normal vector
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-8:
        print("Warning: Could not compute valid normal for surface. Points may be collinear. Using Z-axis.")
        normal = np.array([0, 0, 1])  # Default to Z-axis as fallback
    else:
        normal = normal / normal_len
    
    print(f"Calculated best projection plane with normal: {normal}")
    
    # Make sure normal has a consistent direction (match MeshIt behavior)
    if abs(normal[0]) > abs(normal[1]) and abs(normal[0]) > abs(normal[2]):
        # X is dominant
        if normal[0] < 0:
            normal = -normal
    elif abs(normal[1]) > abs(normal[2]):
        # Y is dominant
        if normal[1] < 0:
            normal = -normal
    else:
        # Z is dominant
        if normal[2] < 0:
            normal = -normal
    
    # Create a robust coordinate system for the projection
    # Choose a reference vector that's not parallel to the normal
    if abs(normal[2]) < 0.9:  # If normal is not too close to Z axis
        ref = np.array([0, 0, 1])
    else:  # If normal is close to Z axis, use X axis
        ref = np.array([1, 0, 0])
    
    # First basis vector (perpendicular to normal)
    basis1 = np.cross(normal, ref)
    basis1_len = np.linalg.norm(basis1)
    
    # Handle edge case where normal is parallel to ref
    if basis1_len < 1e-8:
        # Try a different reference vector
        ref = np.array([0, 1, 0])
        basis1 = np.cross(normal, ref)
        basis1_len = np.linalg.norm(basis1)
        
        if basis1_len < 1e-8:
            # This should almost never happen, but just in case
            ref = np.array([1, 0, 0])
            basis1 = np.cross(normal, ref)
            basis1_len = np.linalg.norm(basis1)
    
    basis1 = basis1 / basis1_len
    
    # Second basis vector (perpendicular to both normal and basis1)
    basis2 = np.cross(normal, basis1)
    basis2 = basis2 / np.linalg.norm(basis2)

    # Project points to 2D
    vertices_2d = np.zeros((len(vertices_3d), 2))
    for i, p in enumerate(vertices_3d):
        # Project point onto the plane
        vertices_2d[i, 0] = np.dot(p, basis1)
        vertices_2d[i, 1] = np.dot(p, basis2)
    
    # Calculate the convex hull with robust error handling
    hull_points = None
    hull_points_3d = None
    hull_indices = None
    
    try:
        hull = ConvexHull(vertices_2d)
        hull_indices = hull.vertices
        hull_points = vertices_2d[hull_indices]
        hull_points_3d = vertices_3d[hull_indices]
        print(f"Created convex hull with {len(hull_points)} points")
    except Exception as e:
        print(f"Error computing convex hull: {e}")
        print("Falling back to simpler hull computation")
        
        # Create a simpler convex hull: just use min/max bounds
        x_min, y_min = np.min(vertices_2d, axis=0)
        x_max, y_max = np.max(vertices_2d, axis=0)
        
        # Create a rectangular hull
        hull_points = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])
        
        # Map back to 3D by finding closest points
        hull_indices = []
        for p in hull_points:
            distances = np.sum((vertices_2d - p)**2, axis=1)
            closest_idx = np.argmin(distances)
            hull_indices.append(closest_idx)
        
        hull_points = vertices_2d[hull_indices]
        hull_points_3d = vertices_3d[hull_indices]
        print(f"Created fallback hull with {len(hull_points)} points")
    
    # Ensure hull is not empty
    if hull_points is None or len(hull_points) < 3:
        raise ValueError("Failed to compute valid convex hull")
    
    # If base_size is not specified, calculate from bounding box
    if base_size is None:
        # Calculate average hull edge length for base_size
        hull_edges = []
        for i in range(len(hull_points)):
            next_i = (i + 1) % len(hull_points)
            edge_len = np.linalg.norm(hull_points[i] - hull_points[next_i])
            hull_edges.append(edge_len)
        
        base_size = sum(hull_edges) / len(hull_edges) / 5.0
        print(f"Calculated base_size: {base_size}")
    
    # --- MeshIt-style hull refinement ---
    # Refine the convex hull by adding intermediate points along the hull edges
    # This will ensure the triangulation respects the convex hull boundaries
    refined_hull_points = []
    refined_hull_indices = []
    
    for i in range(len(hull_points)):
        next_i = (i + 1) % len(hull_points)
        p1 = hull_points[i]
        p2 = hull_points[next_i]
        
        # Add the current point
        refined_hull_points.append(p1)
        refined_hull_indices.append(len(refined_hull_points) - 1)
        
        # Determine how many intermediate points to add based on edge length
        edge_len = np.linalg.norm(p2 - p1)
        num_points = max(1, int(edge_len / (base_size * 0.5)))
        
        # Add intermediate points along the edge
        for j in range(1, num_points):
            t = j / num_points
            intermediate_point = p1 * (1 - t) + p2 * t
            refined_hull_points.append(intermediate_point)
            refined_hull_indices.append(len(refined_hull_points) - 1)
    
    refined_hull_points = np.array(refined_hull_points)
    
    # Create boundary segments connecting consecutive refined hull points
    boundary_segments = []
    for i in range(len(refined_hull_indices)):
        next_i = (i + 1) % len(refined_hull_indices)
        boundary_segments.append([i, next_i])
    
    boundary_segments = np.array(boundary_segments)
    
    # Add refined hull points to the set of all 2D points
    all_points_2d = refined_hull_points.copy()
    
    # --- Generate interior points ---
    # Calculate grid dimensions based on bounding box and base_size
    x_min, y_min = np.min(hull_points, axis=0)
    x_max, y_max = np.max(hull_points, axis=0)
    width = x_max - x_min
    height = y_max - y_min
    
    # Calculate grid resolution based on aspect ratio
    aspect_ratio = width / height if height > 0 else 1.0
    
    if aspect_ratio > 3 or aspect_ratio < 0.3:
        # High aspect ratio - adjust grid
        if aspect_ratio > 3:  
            x_res = max(10, int(width / base_size))
            y_res = max(10, int(height / (base_size * 0.3)))
        else:  
            x_res = max(10, int(width / (base_size * 0.3)))
            y_res = max(10, int(height / base_size))
    else:
        # Normal aspect ratio
        x_res = max(10, int(width / base_size))
        y_res = max(10, int(height / base_size))
    
    print(f"Grid resolution: {x_res}x{y_res}, base_size={base_size}")
    
    # Add interior points using grid
    x_coords = np.linspace(x_min + base_size * 0.1, x_max - base_size * 0.1, x_res)
    y_coords = np.linspace(y_min + base_size * 0.1, y_max - base_size * 0.1, y_res)
    
    # Add small jitter to avoid grid artifacts
    jitter_scale = base_size * 0.05
    x_jitter = np.random.uniform(-jitter_scale, jitter_scale, size=len(x_coords))
    y_jitter = np.random.uniform(-jitter_scale, jitter_scale, size=len(y_coords))
    
    interior_points = []
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            # Add jitter
            jittered_point = np.array([x + x_jitter[i], y + y_jitter[j]])
            
            # Check if point is inside convex hull
            if is_point_in_polygon(jittered_point, hull_points):
                interior_points.append(jittered_point)
    
    print(f"Generated {len(interior_points)} interior points")
    
    # Combine with the refined hull points
    if interior_points:
        all_interior_points = np.array(interior_points)
        all_points_2d = np.vstack([all_points_2d, all_interior_points])
    
    # --- Setup Triangle with MeshIt options ---
    try:
        from triangle import triangulate
        
        # Add feature points for triangle refinement
        feature_points = refined_hull_points  # Hull points are important features
        feature_sizes = np.ones(len(feature_points)) * base_size * 0.5  # Smaller size at boundary
        
        # Prepare Triangle options - use 'pzYYu' with additional area constraint
        area_max = base_size * base_size * 0.5
        triangle_opts = f'pzYYu'  # Exact MeshIt options
        print(f"Using Triangle library for triangulation")
        print(f"Using MeshIt's exact triangle options: '{triangle_opts}'")
        
        # Setup Triangle input
        tri_input = {
            'vertices': all_points_2d,
            'segments': boundary_segments
        }
        
        # Run initial triangulation
        result = triangulate(tri_input, triangle_opts)
        initial_triangles = len(result['triangles'])
        print(f"Initial triangulation: {initial_triangles} triangles")
        
        # --- Refinement iterations ---
        # Implement MeshIt-like refinement process
        max_iterations = 8  # MeshIt typically uses 8-10 iterations
        sq_meshsize = base_size * base_size
        sq_grad = gradient * gradient
        
        for iteration in range(max_iterations):
            # Find triangles that need refinement
            unsuitable_triangles = []
            for i, tri in enumerate(result['triangles']):
                tri_verts = result['vertices'][tri]
                
                # Calculate triangle properties
                # Edge vectors
                dxoa = tri_verts[0][0] - tri_verts[2][0]
                dyoa = tri_verts[0][1] - tri_verts[2][1]
                dxda = tri_verts[1][0] - tri_verts[2][0]
                dyda = tri_verts[1][1] - tri_verts[2][1]
                dxod = tri_verts[0][0] - tri_verts[1][0]
                dyod = tri_verts[0][1] - tri_verts[1][1]
                
                # Square of edge lengths
                oalen = dxoa * dxoa + dyoa * dyoa
                dalen = dxda * dxda + dyda * dyda
                odlen = dxod * dxod + dyod * dyod
                
                # Get max edge length
                max_sq_len = max(oalen, dalen, odlen)
                
                # Check if triangle is too large
                if max_sq_len > sq_meshsize:
                    unsuitable_triangles.append(i)
                    continue
                
                # Calculate centroid for feature point checks
                cx = (tri_verts[0][0] + tri_verts[1][0] + tri_verts[2][0]) / 3.0
                cy = (tri_verts[0][1] + tri_verts[1][1] + tri_verts[2][1]) / 3.0
                centroid = np.array([cx, cy])
                
                # Check against feature points
                for j, feat_point in enumerate(feature_points):
                    sq_refinesize = feature_sizes[j] * feature_sizes[j]
                    sq_dist = np.sum((centroid - feat_point)**2)
                    
                    # Apply MeshIt's refinement criteria
                    if sq_dist < sq_grad * (sq_meshsize - sq_refinesize):
                        target_sq_size = sq_dist / sq_grad + sq_refinesize
                        if max_sq_len > target_sq_size:
                            unsuitable_triangles.append(i)
                            break
            
            if not unsuitable_triangles:
                print(f"Refinement complete after {iteration} iterations")
                break
            
            print(f"Refinement iteration {iteration+1}: {len(unsuitable_triangles)} triangles need refinement")
            
            # Generate refinement points
            refinement_points = []
            for idx in unsuitable_triangles:
                tri = result['triangles'][idx]
                tri_verts = result['vertices'][tri]
                
                # Calculate circumcenter (MeshIt's strategy)
                a = tri_verts[0]
                b = tri_verts[1]
                c = tri_verts[2]
                
                ab = a - b
                ac = a - c
                abNorm = np.dot(ab, ab)
                acNorm = np.dot(ac, ac)
                d = 2 * (ab[0] * ac[1] - ab[1] * ac[0])
                
                if abs(d) < 1e-10:
                    # Degenerate case - use centroid
                    refinement_points.append(np.mean(tri_verts, axis=0))
                    continue
                
                # Calculate circumcenter
                ux = (ac[1] * abNorm - ab[1] * acNorm) / d
                uy = (ab[0] * acNorm - ac[0] * abNorm) / d
                circumcenter = a + np.array([ux, uy])
                
                refinement_points.append(circumcenter)
            
            print(f"Adding {len(refinement_points)} refinement points")
            
            # Add refinement points and re-triangulate
            new_vertices = np.vstack((result['vertices'], refinement_points))
            
            tri_input = {
                'vertices': new_vertices,
                'segments': boundary_segments
            }
            
            result = triangulate(tri_input, triangle_opts)
            print(f"After iteration {iteration+1}: {len(result['triangles'])} triangles")
        
        # Final triangulation stats
        print(f"Triangulation complete:")
        print(f"  Initial triangles: {initial_triangles}")
        print(f"  Final triangles: {len(result['triangles'])}")
        print(f"  Increase: {len(result['triangles']) - initial_triangles} triangles " +
              f"({(len(result['triangles']) - initial_triangles) / initial_triangles * 100:.1f}%)")
        
        # Get final triangulation result
        tri_points_2d = result['vertices']
        tri_triangles = result['triangles']
        
        # Project triangulated points back to 3D
        tri_points_3d = np.zeros((len(tri_points_2d), 3))
        
        # Project hull points back to their original 3D positions
        for i in range(len(refined_hull_points)):
            if i < len(hull_points):
                idx = hull_indices[i % len(hull_indices)]
                tri_points_3d[i] = vertices_3d[idx]
            else:
                # For intermediate hull points, interpolate 3D positions
                # Find which segment this point belongs to
                for j in range(len(hull_points)):
                    next_j = (j + 1) % len(hull_points)
                    p1 = hull_points[j]
                    p2 = hull_points[next_j]
                    p = refined_hull_points[i]
                    
                    # Check if p is on line segment p1-p2
                    v1 = p2 - p1
                    v2 = p - p1
                    len_v1 = np.linalg.norm(v1)
                    
                    if len_v1 < 1e-8:
                        continue
                    
                    # Projection of v2 onto v1
                    t = np.dot(v2, v1) / len_v1**2
                    
                    if 0 <= t <= 1 and np.linalg.norm(p1 + t * v1 - p) < 1e-8:
                        # Point is on this segment, interpolate 3D position
                        p1_3d = vertices_3d[hull_indices[j]]
                        p2_3d = vertices_3d[hull_indices[next_j]]
                        tri_points_3d[i] = p1_3d * (1 - t) + p2_3d * t
                        break
                
                # If we couldn't find the segment, use plane projection
                if np.all(tri_points_3d[i] == 0):
                    p_2d = refined_hull_points[i]
                    p_3d = p_2d[0] * basis1 + p_2d[1] * basis2
                    # Position at the right height
                    mean_point_3d = np.mean(hull_points_3d, axis=0)
                    point_to_mean = mean_point_3d - p_3d
                    dist_to_plane = np.dot(point_to_mean, normal)
                    tri_points_3d[i] = p_3d + normal * dist_to_plane
        
        # Project interior points to 3D
        for i in range(len(refined_hull_points), len(tri_points_2d)):
            p_2d = tri_points_2d[i]
            p_3d = p_2d[0] * basis1 + p_2d[1] * basis2
            
            # Position at the right height (average of surrounding points)
            mean_point_3d = np.mean(hull_points_3d, axis=0)
            point_to_mean = mean_point_3d - p_3d
            dist_to_plane = np.dot(point_to_mean, normal)
            tri_points_3d[i] = p_3d + normal * dist_to_plane
        
        # Return results based on input type
        if is_surface_obj:
            # Update the surface object
            points_or_surface.vertices = []
            for p in tri_points_3d:
                points_or_surface.vertices.append(Vector3D(p[0], p[1], p[2]))
            
            points_or_surface.triangles = []
            for t in tri_triangles:
                points_or_surface.triangles.append([int(t[0]), int(t[1]), int(t[2])])
            
            print(f"Surface {points_or_surface.name} now has {len(points_or_surface.vertices)} vertices and {len(points_or_surface.triangles)} triangles")
            
            # Verify triangle indices are valid
            max_index = max(max(tri) for tri in points_or_surface.triangles)
            if max_index >= len(points_or_surface.vertices):
                print(f"WARNING: Invalid triangle indices detected (max={max_index}, vertices={len(points_or_surface.vertices)})")
                # Filter only valid triangles
                valid_triangles = []
                for tri in points_or_surface.triangles:
                    if all(idx < len(points_or_surface.vertices) for idx in tri):
                        valid_triangles.append(tri)
                points_or_surface.triangles = valid_triangles
                print(f"After filtering: {len(points_or_surface.triangles)} valid triangles")
            
            return points_or_surface
        else:
            # Return as dictionary
            return {
                'vertices': tri_points_3d,
                'triangles': tri_triangles
            }
    
    except Exception as e:
        print(f"Error during triangulation: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty result or fall back to simple triangulation
        if is_surface_obj:
            return points_or_surface
        else:
            return {
                'vertices': vertices_3d,
                'triangles': []
            }

def is_point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using the ray casting algorithm.
    
    Args:
        point: The point to check [x, y]
        polygon: Array of polygon vertices [(x1,y1), (x2,y2), ...]
    
    Returns:
        bool: True if the point is inside the polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n):
        p2x, p2y = polygon[i]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def clean_close_points(points, threshold=1e-7):
    """Remove points that are too close together to avoid precision issues in triangulation.
    
    Args:
        points: Array of 3D points
        threshold: Distance threshold for considering points as duplicates
        
    Returns:
        cleaned_points: Array of points with duplicates removed
        indices_map: Array mapping from new indices to original indices
    """
    if len(points) <= 3:
        # For small inputs, just create identity mapping
        return points, np.arange(len(points))
        
    keep_indices = []  # Will store indices of points to keep
    
    # Check every point
    for i in range(len(points)):
        # Assume we'll keep this point until proven otherwise
        keep_it = True
        
        # Compare with previously kept points
        for j in keep_indices:
            dist = np.sqrt(np.sum((points[i] - points[j])**2))
            if dist < threshold:
                # Too close to an existing point
                keep_it = False
                break
                
        if keep_it:
            keep_indices.append(i)
    
    # Create mapping from new indices to original indices
    indices_map = np.array(keep_indices)
    
    # Create cleaned array of points
    cleaned_points = points[keep_indices]
    
    print(f"Cleaned points: {len(points)} -> {len(cleaned_points)}")
    
    return cleaned_points, indices_map

def visualize_step(model, step_description=None, show_intersections=False, show_hull=False, 
                  show_points=True, show_triangles=True, show_fault=False,
                  show_faces=True, show_edges=True):
    """
    Visualize a step in the triangulation process using PyVista.
    
    Args:
        model: The SimpleModel containing surfaces to visualize
        step_description: Description to show in the plot title
        show_intersections: Whether to show intersections between surfaces
        show_hull: Whether to show convex hulls for surfaces
        show_points: Whether to show vertices for surfaces
        show_triangles: Whether to show triangulated faces for surfaces
        show_fault: Whether to show fault polylines
        show_faces: Whether to show filled triangle faces (False for wireframe only)
        show_edges: Whether to show edges of triangles
    """
    try:
        import pyvista as pv
        import numpy as np
        
        # Create a plotter
        plotter = pv.Plotter(window_size=[1200, 800])
        plotter.set_background('white')
        
        # Keep track of meshes to check if visualization has content
        has_visualizable_content = False
        
        # Process each surface
        for i, surface in enumerate(model.surfaces):
            if not surface.vertices:
                continue
                
            # Define unique colors for each surface
            colors = [
                (0.1, 0.5, 0.8),  # Blue
                (0.8, 0.3, 0.1),  # Orange
                (0.1, 0.7, 0.3),  # Green
                (0.7, 0.1, 0.7),  # Purple
            ]
            color = colors[i % len(colors)]
            
            # Point cloud for vertices
            if show_points and surface.vertices:
                points = np.array([[v.x, v.y, v.z] for v in surface.vertices])
                point_cloud = pv.PolyData(points)
                plotter.add_points(point_cloud, color=color, point_size=5, 
                                  render_points_as_spheres=True,
                                  label=f"{surface.name} Points")
                has_visualizable_content = True
            
            # Convex hull (if available and requested)
            if show_hull and hasattr(surface, 'convex_hull') and surface.convex_hull:
                hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
                # Ensure hull is closed by repeating the first point at the end
                if len(hull_points) > 0 and not np.array_equal(hull_points[0], hull_points[-1]):
                    hull_points = np.vstack([hull_points, hull_points[0:1]])
                
                # Create line segments for the hull edges
                if len(hull_points) >= 2:
                    # Create a line for each edge of the hull
                    for i in range(len(hull_points) - 1):
                        line = pv.Line(hull_points[i], hull_points[i+1])
                        tube = line.tube(radius=0.05)
                        plotter.add_mesh(tube, color=(0.9, 0.1, 0.1), label=f"Hull Edge {i}")
                    
                    # Also add hull vertices as distinct points for emphasis
                    if len(hull_points) > 0:
                        hull_point_cloud = pv.PolyData(hull_points)
                        plotter.add_points(hull_point_cloud, color=(0.9, 0.1, 0.1), 
                                         point_size=8, render_points_as_spheres=True)
                    
                    has_visualizable_content = True
            
            # Triangulated surface
            if show_triangles and surface.triangles and len(surface.triangles) > 0:
                # Create vertices array for triangulated surface
                vertices = np.array([[v.x, v.y, v.z] for v in surface.vertices])
                
                # Create faces array with structure [n, id1, id2, id3, ...]
                faces = []
                for tri in surface.triangles:
                    # Check if triangle indices are valid
                    if all(idx < len(vertices) for idx in tri):
                        faces.append(3)  # Number of vertices in face
                        faces.extend(tri)  # Append the three vertex indices
                
                if len(faces) > 0:
                    # Create mesh
                    surf_mesh = pv.PolyData(vertices, faces)
                    
                    # Add mesh with appropriate display options
                    plotter.add_mesh(surf_mesh, color=color, opacity=0.7 if show_faces else 0.0,
                                   show_edges=show_edges, line_width=1.0, edge_color='black',
                                   label=f"{surface.name} Mesh")
                    has_visualizable_content = True
        
        # Show polylines/faults if requested
        if show_fault and hasattr(model, 'model_polylines'):
            for i, polyline in enumerate(model.model_polylines):
                if not polyline.vertices or len(polyline.vertices) < 2:
                    continue
                
                # Convert polyline vertices to NumPy array
                points = np.array([[v.x, v.y, v.z] for v in polyline.vertices])
                
                # If we have segments, use them to create line segments
                if hasattr(polyline, 'segments') and polyline.segments:
                    for segment in polyline.segments:
                        if len(segment) >= 2:
                            p1 = points[segment[0]]
                            p2 = points[segment[1]]
                            line = pv.Line(p1, p2)
                            tube = line.tube(radius=0.08)
                            plotter.add_mesh(tube, color=(0.9, 0.1, 0.9), 
                                          label=f"{polyline.name} Segment")
                            has_visualizable_content = True
                # Otherwise, just connect the points in order
                elif len(points) >= 2:
                    # Create a line
                    for i in range(len(points) - 1):
                        line = pv.Line(points[i], points[i+1])
                        tube = line.tube(radius=0.08)
                        plotter.add_mesh(tube, color=(0.9, 0.1, 0.9), 
                                      label=f"{polyline.name} Line")
                    has_visualizable_content = True
        
        # Show intersections if requested
        if show_intersections and hasattr(model, 'model_polylines'):
            for i, polyline in enumerate(model.model_polylines):
                if "Intersection" not in polyline.name:
                    continue
                    
                # Get intersection points
                if polyline.vertices and len(polyline.vertices) > 0:
                    points = np.array([[v.x, v.y, v.z] for v in polyline.vertices])
                    point_cloud = pv.PolyData(points)
                    plotter.add_points(point_cloud, color=(1.0, 0.0, 0.0), 
                                     point_size=10, render_points_as_spheres=True,
                                     label=f"{polyline.name}")
                    has_visualizable_content = True
        
        # If we have content to visualize, set title and show
        if has_visualizable_content:
            title = step_description if step_description else "MeshIt Visualization"
            plotter.add_title(title, font_size=16)
            plotter.show_bounds(grid=True, location='outer')
            plotter.add_axes()
            plotter.show()
        else:
            print("No content to visualize in this step")
    
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def sort_intersection_points(points):
    """Sort intersection points to form a coherent line."""
    if len(points) <= 2:
        return points
        
    # Start with the first point
    sorted_points = [points[0]]
    remaining_points = list(points[1:])
    
    # Repeatedly find the closest point to the last added point
    while remaining_points:
        last_point = sorted_points[-1]
        
        # Find the closest remaining point
        closest_idx = 0
        closest_dist = float('inf')
        
        for i, point in enumerate(remaining_points):
            dist = np.sum((last_point - point)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        
        # Add the closest point and remove it from the remaining points
        sorted_points.append(remaining_points[closest_idx])
        remaining_points.pop(closest_idx)
    
    return np.array(sorted_points)

def triangulate_polyline(polyline, gradient=1.0, base_size=None):
    """Triangulate a fault polyline properly.
    This is a wrapper for the Polyline.triangulate method to ensure consistent API.
    """
    try:
        # First, make sure the polyline has a convex hull
        if not hasattr(polyline, 'convex_hull') or not polyline.convex_hull:
            print("Calculating convex hull for polyline before triangulation")
            polyline.enhanced_calculate_convex_hull()
            
        print(f"Triangulating polyline with gradient={gradient} and base_size={base_size}")
        return polyline.triangulate(gradient=gradient, base_size=base_size)
    except TypeError as e:
        # If the error is about base_size being an unexpected argument
        if "unexpected keyword argument 'base_size'" in str(e):
            # Try without base_size
            print("Warning: base_size not supported in Polyline.triangulate, using default value")
            return polyline.triangulate(gradient=gradient)
        else:
            # Re-raise other TypeError errors
            print(f"TypeError during polyline triangulation: {e}")
            import traceback
            traceback.print_exc()
            raise
    except Exception as e:
        print(f"Error in triangulate_polyline: {e}")
        import traceback
        traceback.print_exc()
        # Make sure polyline.triangles exists
        if not hasattr(polyline, 'triangles'):
            polyline.triangles = []
        return polyline

def visualize_model(model, title="Model Visualization", show_faces=True, show_edges=True):
    """Visualize the model with PyVista."""
    return visualize_step(model, title, "Final", True, True, True, True, True, True, show_faces, show_edges)

def export_surface_to_files(surface, base_filename, format='csv'):
    """
    Export surface vertices and triangles to files in specified format.
    
    Args:
        surface: Surface object to export
        base_filename: Base filename without extension
        format: File format ('csv', 'txt', or 'dat')
    """
    # Determine file extension and separator
    if format == 'csv':
        ext = '.csv'
        sep = ','
        header = True
    else:  # txt or dat
        ext = '.' + format
        sep = ' '
        header = False
    
    # Export vertices
    vertices_file = f"{base_filename}_vertices{ext}"
    vertices_data = pd.DataFrame(
        [[v.x, v.y, v.z] for v in surface.vertices],
        columns=['x', 'y', 'z']
    )
    vertices_data.to_csv(vertices_file, sep=sep, index=False, header=header)
    print(f"Exported vertices to {vertices_file}")
    
    # Export triangles if available
    if surface.triangles:
        triangles_file = f"{base_filename}_triangles{ext}"
        triangles_data = pd.DataFrame(
            surface.triangles,
            columns=['v1', 'v2', 'v3']
        )
        triangles_data.to_csv(triangles_file, sep=sep, index=False, header=header)
        print(f"Exported triangles to {triangles_file}")

def calculate_triple_points(intersection1, intersection2):
    """Calculate triple points between two intersections."""
    
    triple_points = []
    
    # Check if both intersections have points
    if not hasattr(intersection1, 'points') or not intersection1.points:
        return []
    if not hasattr(intersection2, 'points') or not intersection2.points:
        return []
    
    # Find points that are close to each other
    for p1 in intersection1.points:
        for p2 in intersection2.points:
            # Calculate distance between points
            dx = p1.x - p2.x
            dy = p1.y - p2.y
            dz = p1.z - p2.z
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # If points are close enough, create a triple point
            if dist < 1e-5:
                # Create triple point at midpoint
                mid_x = (p1.x + p2.x) / 2
                mid_y = (p1.y + p2.y) / 2
                mid_z = (p1.z + p2.z) / 2
                triple_point = TriplePoint(Vector3D(mid_x, mid_y, mid_z))
                triple_point.add_intersection(intersection1)
                triple_point.add_intersection(intersection2)
                triple_points.append(triple_point)
    
    print(f"Found {len(triple_points)} triple points")
    return triple_points

def main():
    # Create a model for testing
    model = SimpleModel()
    
    # Create two overlapping surfaces with nicely spaced points for triangulation
    
    # Surface 1: Base grid with small random displacement for better visualization
    grid_size = 15
    z_base1 = 0.5  # Slightly elevated for visibility
    x_range = np.linspace(-5, 5, grid_size)
    y_range = np.linspace(-5, 5, grid_size)
    
    # Generate base points with small noise
    surface1_points = []
    np.random.seed(42)  # For reproducibility
    for x in x_range:
        for y in y_range:
            # Add small noise to avoid perfectly rectangular grid
            noise_x = np.random.uniform(-0.1, 0.1)
            noise_y = np.random.uniform(-0.1, 0.1)
            noise_z = np.random.uniform(-0.05, 0.05)
            surface1_points.append(Vector3D(x + noise_x, y + noise_y, z_base1 + noise_z))
    
    # Add perimeter points for better boundary definition
    for x in np.linspace(-5.5, 5.5, 20):
        surface1_points.append(Vector3D(x, -5.5, z_base1))
        surface1_points.append(Vector3D(x, 5.5, z_base1))
    
    for y in np.linspace(-5.5, 5.5, 20):
        surface1_points.append(Vector3D(-5.5, y, z_base1))
        surface1_points.append(Vector3D(5.5, y, z_base1))
    
    # Surface 2: Tilted and partially overlapping with Surface 1
    grid_size = 15
    x_range = np.linspace(-3, 7, grid_size)
    y_range = np.linspace(-3, 7, grid_size)
    
    # Define a tilted plane
    plane_normal = np.array([0.2, 0.1, 0.97])  # Slightly tilted normal
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    plane_point = np.array([2, 2, 0])  # Center point of the plane
    
    # Generate points on the tilted plane
    surface2_points = []
    np.random.seed(43)  # Different seed for variety
    for x in x_range:
        for y in y_range:
            # Calculate z from plane equation
            z = plane_point[2] + (plane_normal[0] * (plane_point[0] - x) + 
                                  plane_normal[1] * (plane_point[1] - y)) / plane_normal[2]
            
            # Add small noise
            noise_x = np.random.uniform(-0.1, 0.1)
            noise_y = np.random.uniform(-0.1, 0.1)
            noise_z = np.random.uniform(-0.05, 0.05)
            surface2_points.append(Vector3D(x + noise_x, y + noise_y, z + noise_z))
    
    # Add perimeter points for better boundary definition
    points_on_perimeter = 20
    for t in np.linspace(0, 2*np.pi, points_on_perimeter, endpoint=False):
        x = 5 * np.cos(t) + 2  # Center at (2, 2)
        y = 5 * np.sin(t) + 2
        z = plane_point[2] + (plane_normal[0] * (plane_point[0] - x) + 
                              plane_normal[1] * (plane_point[1] - y)) / plane_normal[2]
        surface2_points.append(Vector3D(x, y, z))
    
    # Create and add surfaces to the model
    surface1 = Surface("Surface 1", surface1_points, [])
    surface2 = Surface("Surface 2", surface2_points, [])
    
    # Add surfaces to model
    model.add_surface(surface1)
    model.add_surface(surface2)
    
    # -- MeshIt Workflow --
    # Triangulate surfaces with larger base_size and gradient for better visualization
    base_size = 1.0  # Increased from 0.3 to 1.0
    gradient = 2.0   # Increased from 1.2 to 2.0
    
    print(f"Triangulating surfaces with base_size={base_size}, gradient={gradient}")
    for surface in model.surfaces:
        print(f"Triangulating {surface.name}...")
        triangulate_surface(surface, base_size=base_size, gradient=gradient)
    
    # Visualize just the initial points and final triangulation
    
    # Initial points visualization
    visualize_step(model, step_description="Initial points", 
                  show_points=True, 
                  show_triangles=False,
                  show_hull=False,
                  show_intersections=False)
    
    # Triangulation visualization
    visualize_step(model, step_description="Surface triangulation", 
                  show_points=False,  # Hide points to see triangles better
                  show_triangles=True,
                  show_hull=False,
                  show_intersections=False)
    
    # Full model visualization with edges only
    visualize_step(model, step_description="Complete model", 
                  show_points=False, 
                  show_triangles=True,
                  show_hull=False,
                  show_intersections=False,
                  show_faces=False,  # Show edges only for better visibility
                  show_edges=True)

    return model

if __name__ == "__main__":
    main() 