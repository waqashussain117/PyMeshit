"""
Custom wrapper for the Triangle library to support refinement callbacks.

This module extends the functionality of the Python Triangle library
to support custom refinement functions similar to the C++ triunsuitable
function in the original MeshIt implementation.
"""

import numpy as np
import triangle as tr
from typing import Callable, Dict, List, Tuple, Union, Optional
import logging
from scipy.spatial import Delaunay

class TriangleWrapper:
    """
    Custom wrapper for the Triangle library to support MeshIt-style refinement.
    
    This class works around the limitation that the Python Triangle library
    doesn't directly support custom refinement callbacks by implementing
    an iterative refinement process that mimics the behavior of the C++ Triangle
    library with the triunsuitable function.
    """
    
    def __init__(self, gradient: float = 2.0, min_angle: float = 20.0, 
                 base_size: Optional[float] = None):
        """
        Initialize the TriangleWrapper with refinement parameters.
        
        Args:
            gradient: Gradient control parameter (default: 2.0, matches MeshIt core)
            min_angle: Minimum angle for triangle quality (default: 20.0)
            base_size: Base size for triangles (if None, calculated from input)
        """
        self.gradient = gradient
        self.min_angle = min_angle
        self.base_size = base_size
        self.sq_meshsize = None  # Will be set during triangulation
        self.feature_points = None
        self.feature_sizes = None
        self.initial_triangles = 0
        self.logger = logging.getLogger("TriangleWrapper")
        
    def set_feature_points(self, points: np.ndarray, sizes: np.ndarray):
        """Set feature points exactly as in MeshIt's GradientControl"""
        self.feature_points = points
        self.feature_sizes = sizes
        # Ensure squared values are pre-calculated
        self.sq_feature_sizes = sizes * sizes
        
    def calculate_sizing_function(self, point: np.ndarray) -> float:
        """
        Calculate the target mesh size at a given point based on feature points.
        
        Implements the exact sizing function used in MeshIt C++:
        1. Start with base size
        2. For each feature point, calculate influence based on distance
        3. Take the minimum size from all influences
        
        Args:
            point: The 2D point to evaluate (x,y)
            
        Returns:
            The target size (edge length) at this location
        """
        # If no feature points, return base size
        if self.feature_points is None or len(self.feature_points) == 0:
            return self.base_size
        
        # Start with base size
        size = self.base_size
        
        # For each feature point, calculate influence
        sq_grad = self.gradient * self.gradient
        sq_meshsize = self.base_size * self.base_size
        
        for i, feat_point in enumerate(self.feature_points):
            # Feature size
            feat_size = self.feature_sizes[i]
            sq_refinesize = feat_size * feat_size
            
            # Distance to feature point
            sq_dist = np.sum((point - feat_point)**2)
            
            # Calculate influence region exactly as in triunsuitable
            influence_radius_sq = sq_grad * (sq_meshsize - sq_refinesize)
            
            # Check if point is within influence radius
            if sq_dist < influence_radius_sq:
                # Calculate target size exactly as in triunsuitable
                target_size = np.sqrt(sq_dist / sq_grad + sq_refinesize)
                
                # Take minimum size from all influences
                size = min(size, target_size)
        
        return size
        
    def _is_triangle_suitable(self, vertices: np.ndarray) -> bool:
        """
        Direct port of C++ triunsuitable function from core.cpp
        
        This matches the exact logic in src/core.cpp:236-282.
        """
        # Calculate edge vectors exactly as in C++
        dxoa = vertices[0][0] - vertices[2][0]
        dyoa = vertices[0][1] - vertices[2][1]
        dxda = vertices[1][0] - vertices[2][0]
        dyda = vertices[1][1] - vertices[2][1]
        dxod = vertices[0][0] - vertices[1][0]
        dyod = vertices[0][1] - vertices[1][1]

        # Find squares of lengths exactly as in C++
        oalen = dxoa * dxoa + dyoa * dyoa
        dalen = dxda * dxda + dyda * dyda
        odlen = dxod * dxod + dyod * dyod

        # Find the maximum edge length squared (not mean length as in previous implementation)
        max_sq_len = max(oalen, dalen, odlen)
        
        # First check against mesh size - reject if any edge is too long
        if max_sq_len > self.sq_meshsize:
            return False

        # Calculate the centroid
        ONETHIRD = 1.0/3.0
        cx = (vertices[0][0] + vertices[1][0] + vertices[2][0]) * ONETHIRD
        cy = (vertices[0][1] + vertices[1][1] + vertices[2][1]) * ONETHIRD

        # Check against feature points exactly as in C++
        if self.feature_points is not None and len(self.feature_points) > 0:
            sq_grad = self.gradient * self.gradient

            for i, feat_point in enumerate(self.feature_points):
                sq_refinesize = self.feature_sizes[i] * self.feature_sizes[i]
                sq_dist = (cx - feat_point[0])**2 + (cy - feat_point[1])**2

                # Check if point is within influence radius
                if sq_dist < sq_grad * (self.sq_meshsize - sq_refinesize):
                    # Calculate target size at this location
                    target_sq_size = sq_dist / sq_grad + sq_refinesize
                    
                    # If max edge length exceeds target size, triangle is unsuitable
                    if max_sq_len > target_sq_size:
                        return False

        return True
    
    def generate_refinement_points(self, vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """
        Generate refinement points for unsuitable triangles.
        
        This method follows the exact approach used in MeshIt:
        1. Identify unsuitable triangles based on edge length and feature points
        2. For each unsuitable triangle, calculate its circumcenter
        3. If circumcenter is outside the triangle, use off-center point instead
        
        Returns:
            Array of refinement points
        """
        refinement_points = []

        for tri_idx, tri in enumerate(triangles):
            tri_verts = vertices[tri]
            
            # Skip triangles that already meet quality criteria
            if self._is_triangle_suitable(tri_verts):
                continue

            # Use MeshIt's exact circumcenter calculation
            a = tri_verts[0]
            b = tri_verts[1]
            c = tri_verts[2]

            # Vector AB and AC
            ab = a - b
            ac = a - c
            
            # Calculate determinant
            D = 2 * (ab[0] * ac[1] - ab[1] * ac[0])
            
            # Handle degenerate case
            if abs(D) < 1e-10:
                # Use triangle centroid instead
                centroid = (a + b + c) / 3.0
                refinement_points.append(centroid)
                continue
                
            # Calculate squared norms
            abSq = np.dot(ab, ab)
            acSq = np.dot(ac, ac)
            
            # Calculate circumcenter
            ux = (ac[1] * abSq - ab[1] * acSq) / D
            uy = (ab[0] * acSq - ac[0] * abSq) / D
            
            # Calculate circumcenter coordinates
            circumcenter = a + np.array([ux, uy])
            
            # Check if circumcenter is inside triangle
            # Use barycentric coordinates test
            area = 0.5 * abs(D)
            alpha = ((b[1] - c[1]) * (circumcenter[0] - c[0]) + (c[0] - b[0]) * (circumcenter[1] - c[1])) / (2 * area)
            beta = ((c[1] - a[1]) * (circumcenter[0] - c[0]) + (a[0] - c[0]) * (circumcenter[1] - c[1])) / (2 * area)
            gamma = 1 - alpha - beta
            
            # If circumcenter is inside, use it directly
            if alpha >= 0 and beta >= 0 and gamma >= 0:
                refinement_points.append(circumcenter)
            else:
                # If outside, use an off-center point between centroid and circumcenter
                # This is what MeshIt does to ensure better mesh quality
                centroid = (a + b + c) / 3.0
                # Start at 70% toward circumcenter from centroid
                off_center = centroid + 0.7 * (circumcenter - centroid)
                refinement_points.append(off_center)

        # Remove any duplicate points that might have been generated
        if len(refinement_points) > 0:
            refinement_array = np.array(refinement_points)
            # Use a small tolerance to identify duplicates
            cleaned_points, _ = self._clean_vertices(refinement_array, 1e-8)
            return cleaned_points
        else:
            return np.empty((0, 2))
    
    def _direct_dense_refinement(self, vertices: np.ndarray, 
                                triangles: np.ndarray) -> np.ndarray:
        """
        Create extra refinement points directly around feature points.
        
        This function creates a pattern of points around feature points to ensure
        quality refinement in these areas from the start.
        
        Args:
            vertices: Current vertices
            triangles: Current triangles
            
        Returns:
            Array of additional refinement points
        """
        if self.feature_points is None or len(self.feature_points) == 0:
            return np.empty((0, 2))
            
        extra_points = []
        
        # For each feature point
        for i, feature in enumerate(self.feature_points):
            # Get size associated with this feature
            size = self.feature_sizes[i]
            
            # Scale number of points based on gradient
            # Higher gradients need more points for sharper transitions
            # Increasing significantly to match MeshIt's higher density
            base_points = 16  # Doubled from original implementation
            num_points = int(base_points * (1.0 + 0.5 * self.gradient))
            
            # Create more rings with dynamically adjusted sizing
            num_rings = 4  # Increased from 2 to 4 rings
            
            for ring in range(1, num_rings + 1):
                # More gradual spacing for smaller inner rings
                if ring == 1:
                    ring_factor = 0.4
                elif ring == 2:
                    ring_factor = 0.8
                else:
                    ring_factor = 0.7 * ring
                
                # Adjust number of points for outer rings (more points on outer rings)
                ring_points = num_points
                if ring > 2:
                    ring_points = int(ring_points * 1.25)
                
                for j in range(ring_points):
                    angle = 2.0 * np.pi * j / ring_points
                    
                    # Distance from feature - controlled by feature size and gradient
                    # For higher gradients, create points closer to feature
                    # For MeshIt compatibility, adjust the distance calculation
                    distance = size * ring_factor * (1.0 + 0.2/self.gradient)
                    
                    # Generate point on circle
                    dx = distance * np.cos(angle)
                    dy = distance * np.sin(angle)
                    
                    new_point = feature + np.array([dx, dy])
                    extra_points.append(new_point)
            
            # Add some jittered points in the innermost region for better transitions
            # This helps match MeshIt's smoother refinement pattern
            inner_points = int(8 * self.gradient)
            for _ in range(inner_points):
                # Random angle
                angle = 2.0 * np.pi * np.random.random()
                # Random distance - close to feature
                distance = size * 0.2 * (1.0 + np.random.random() * 0.5)
                
                dx = distance * np.cos(angle)
                dy = distance * np.sin(angle)
                
                new_point = feature + np.array([dx, dy])
                extra_points.append(new_point)
        
        return np.array(extra_points) if extra_points else np.empty((0, 2))
                
    def triangulate(self, vertices: np.ndarray, segments: Optional[np.ndarray] = None, 
                    holes: Optional[np.ndarray] = None, max_iterations: int = 8) -> Dict:
        """
        Triangulate points with MeshIt's exact parameters.
        
        Args:
            vertices: Vertex coordinates (N, 2)
            segments: Segment indices for constraining the triangulation (optional)
            holes: Coordinates of holes in the triangulation (optional)
            max_iterations: Maximum number of refinement iterations
            
        Returns:
            Dictionary with triangulation results (vertices, triangles)
        """
        # Clean up input vertices first to avoid precision issues
        if len(vertices) > 3:
            # Remove duplicate or very close vertices
            clean_vertices, index_map = self._clean_vertices(vertices, 1e-10)
            
            # Update segments to use the new indices if needed
            if segments is not None and len(clean_vertices) < len(vertices):
                new_segments = []
                for seg in segments:
                    # Map old indices to new ones, skip if any endpoint was removed
                    if seg[0] in index_map and seg[1] in index_map:
                        new_segments.append([index_map[seg[0]], index_map[seg[1]]])
                
                if len(new_segments) > 0:
                    segments = np.array(new_segments)
                    print(f"Remapped {len(segments)} segments after cleaning {len(vertices) - len(clean_vertices)} close vertices")
                else:
                    print("Warning: All segments were invalid after cleaning vertices")
                    segments = None
            
            vertices = clean_vertices
            
        # Verify input to avoid degenerate cases
        if len(vertices) < 3:
            raise ValueError("Need at least 3 points for triangulation")
            
        # Verify that vertices are not collinear
        if self._are_points_collinear(vertices):
            raise ValueError("Points are collinear, cannot triangulate")
            
        # Calculate base size if not set
        if self.base_size is None:
            diagonal = self._calculate_diagonal(vertices)
            self.base_size = diagonal / 15.0  # MeshIt's exact scaling
        
        # Set squared mesh size as in C++
        self.sq_meshsize = self.base_size * self.base_size
        
        # Calculate area constraint based on squared mesh size
        area_constraint = self.sq_meshsize * 0.5
        
        # Adjust min angle based on gradient exactly as in MeshIt
        if self.gradient > 1.0:
            self.min_angle = max(20.0 - (self.gradient - 1.0) * 5.0, 10.0)
        
        # FIXED: Use relaxed triangle options to allow intersection point merging
        # p = Use PSLG (segments)
        # z = Number triangles from zero
        # Y = Prohibit Steiner points on the boundary (single Y, not YY)
        # q = Quality mesh generation with minimum angle
        # a = Maximum area constraint
        # Note: Removed second Y to allow interior Steiner points for proper intersection handling
        triangle_opts = f'pzYq{self.min_angle}a{area_constraint}'
        
        print(f"Using MeshIt's exact triangle options: '{triangle_opts}'")
        
        # Try to triangulate with these options
        try:
            # Prepare input for Triangle
            tri_data = {'vertices': vertices}
            
            if segments is not None:
                tri_data['segments'] = segments
                
            if holes is not None:
                tri_data['holes'] = holes
                
            result = tr.triangulate(tri_data, triangle_opts)
            
            # Check for triangles and standardize the result
            triangle_key = None
            for key in ['triangles', 'triangulation', 'elements']:
                if key in result:
                    triangle_key = key
                    break
                    
            if triangle_key:
                if triangle_key != 'triangles':
                    result['triangles'] = result[triangle_key]
                    
                self.initial_triangles = len(result['triangles'])
                print(f"Initial triangulation: {self.initial_triangles} triangles")
            else:
                raise ValueError("No triangles in result")
                
        except Exception as e:
            print(f"Error with exact triangle options '{triangle_opts}': {str(e)}")
            print("Falling back to relaxed quality constraints")
            
            # Try with more standard options if the exact MeshIt options fail
            fallback_opts = f'pzq{self.min_angle}a{area_constraint}'
            
            try:
                # Try again with fallback options
                tri_data = {'vertices': vertices}
                
                if segments is not None:
                    tri_data['segments'] = segments
                    
                if holes is not None:
                    tri_data['holes'] = holes
                    
                result = tr.triangulate(tri_data, fallback_opts)
                
                # Check if triangles exist in the result
                triangle_key = None
                for key in ['triangles', 'triangulation', 'elements']:
                    if key in result:
                        triangle_key = key
                        break
                        
                if triangle_key:
                    if triangle_key != 'triangles':
                        result['triangles'] = result[triangle_key]
                        
                    self.initial_triangles = len(result['triangles'])
                    print(f"Initial triangulation with fallback options: {self.initial_triangles} triangles")
                else:
                    raise ValueError("No triangles in result with fallback options")
                    
            except Exception as e2:
                print(f"Error with fallback options: {str(e2)}")
                print("Falling back to minimal constraints as last resort")
                
                # Last resort: try with minimal constraints
                last_resort_opts = 'p' if segments is not None else ''
                
                try:
                    # Try one last time
                    tri_data = {'vertices': vertices}
                    
                    if segments is not None:
                        tri_data['segments'] = segments
                        
                    if holes is not None:
                        tri_data['holes'] = holes
                        
                    result = tr.triangulate(tri_data, last_resort_opts)
                    
                    # Check if triangles exist in the result
                    triangle_key = None
                    for key in ['triangles', 'triangulation', 'elements']:
                        if key in result:
                            triangle_key = key
                            break
                            
                    if triangle_key:
                        if triangle_key != 'triangles':
                            result['triangles'] = result[triangle_key]
                            
                        self.initial_triangles = len(result['triangles'])
                        print(f"Initial triangulation with minimal constraints: {self.initial_triangles} triangles")
                    else:
                        # If all else fails, try Delaunay triangulation directly
                        print("Triangle library failed with all options, falling back to Delaunay")
                        del_tri = Delaunay(vertices)
                        result = {
                            'vertices': vertices,
                            'triangles': del_tri.simplices
                        }
                        self.initial_triangles = len(result['triangles'])
                        print(f"Delaunay created {self.initial_triangles} triangles")
                    
                except Exception as e3:
                    # If all else fails, raise a clear error
                    raise ValueError(f"Failed to triangulate after all attempts: {str(e)}")
        
        # --- Iterative refinement ---
        for iteration in range(max_iterations):
            # Check if triangulation needs refinement
            unsuitable_triangles = []
            for i, tri in enumerate(result['triangles']):
                tri_vertices = result['vertices'][tri]
                if not self._is_triangle_suitable(tri_vertices):
                    unsuitable_triangles.append(i)
            
            # If no unsuitable triangles, we're done
            if not unsuitable_triangles:
                print(f"Refinement complete after {iteration} iterations - all triangles meet quality criteria")
                break
                
            print(f"Refinement iteration {iteration+1}: {len(unsuitable_triangles)} triangles need refinement")
            
            # If we've already done several iterations and aren't making progress, stop
            if iteration >= 4 and len(unsuitable_triangles) > 0.7 * len(result['triangles']):
                print("Refinement progress is slow - accepting current mesh quality")
                break
                
            # Generate refinement points
            refinement_points = self.generate_refinement_points(
                result['vertices'], 
                result['triangles'][unsuitable_triangles]
            )
            
            if len(refinement_points) == 0:
                print("No refinement points generated - stopping refinement")
                break
                
            print(f"Adding {len(refinement_points)} refinement points")
            
            # Add refinement points to vertices
            new_vertices = np.vstack((result['vertices'], refinement_points))
            
            # Re-triangulate with the updated vertices
            try:
                tri_data = {'vertices': new_vertices}
                
                if segments is not None:
                    tri_data['segments'] = segments
                    
                if holes is not None:
                    tri_data['holes'] = holes
                    
                result = tr.triangulate(tri_data, triangle_opts)
                
                # Check if triangles exist in the result
                triangle_key = None
                for key in ['triangles', 'triangulation', 'elements']:
                    if key in result:
                        triangle_key = key
                        break
                        
                if triangle_key:
                    if triangle_key != 'triangles':
                        result['triangles'] = result[triangle_key]
                else:
                    raise ValueError(f"No triangles in iteration {iteration+1}")
                
                print(f"After iteration {iteration+1}: {len(result['triangles'])} triangles")
                
            except Exception as e:
                print(f"Error in refinement iteration {iteration+1}: {str(e)}")
                print("Stopping refinement due to error")
                break
        
        # Report final statistics
        print(f"Triangulation complete:")
        print(f"  Initial triangles: {self.initial_triangles}")
        print(f"  Final triangles: {len(result['triangles'])}")
        print(f"  Increase: {len(result['triangles']) - self.initial_triangles} triangles " +
              f"({(len(result['triangles']) - self.initial_triangles) / self.initial_triangles * 100:.1f}%)")
        
        return result
        
    def _clean_vertices(self, vertices: np.ndarray, threshold: float = 1e-10) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Remove duplicate or very close vertices that could cause precision issues.
        
        Args:
            vertices: Input vertices array
            threshold: Threshold distance for considering points as duplicates
            
        Returns:
            Tuple of (cleaned vertices array, mapping from old to new indices)
        """
        if len(vertices) <= 3:
            # Not enough vertices to clean
            return vertices, {i: i for i in range(len(vertices))}
            
        keep_indices = [0]  # Always keep the first point
        index_map = {}
        index_map[0] = 0
        
        for i in range(1, len(vertices)):
            # Check if this point is too close to any point we're keeping
            too_close = False
            for j in keep_indices:
                dist = np.sqrt(np.sum((vertices[i] - vertices[j])**2))
                if dist < threshold:
                    too_close = True
                    # No need to map - we'll skip this point
                    break
                    
            if not too_close:
                keep_indices.append(i)
                index_map[i] = len(keep_indices) - 1
        
        # Create cleaned array
        cleaned_vertices = vertices[keep_indices]
        
        if len(cleaned_vertices) < len(vertices):
            print(f"Cleaned {len(vertices) - len(cleaned_vertices)} points that were too close together")
            
        return cleaned_vertices, index_map
    
    def _are_points_collinear(self, points: np.ndarray, threshold: float = 1e-12) -> bool:
        """
        Check if points are collinear (all lie on a straight line).
        
        Args:
            points: Points to check
            threshold: Threshold for considering points as collinear
            
        Returns:
            True if points are collinear, False otherwise
        """
        if len(points) < 3:
            # Less than 3 points are always collinear
            return True
            
        # Take first point as reference
        p0 = points[0]
        
        # And second point to define a direction
        p1 = points[1]
        
        # Direction vector
        v = p1 - p0
        v_norm = np.linalg.norm(v)
        
        if v_norm < threshold:
            # First two points are identical, check if all points are the same
            for i in range(2, len(points)):
                if np.linalg.norm(points[i] - p0) > threshold:
                    return False
            return True
            
        # Normalize direction vector
        v = v / v_norm
        
        # Check if all points lie on the line defined by p0 and v
        for i in range(2, len(points)):
            # Vector from p0 to current point
            w = points[i] - p0
            
            # Project w onto v
            proj = np.dot(w, v) * v
            
            # Distance from point to line
            dist = np.linalg.norm(w - proj)
            
            if dist > threshold:
                # Found a point that's not on the line
                return False
                
        # All points lie on the line
        return True

    def _calculate_diagonal(self, vertices: np.ndarray) -> float:
        """Calculate bounding box diagonal exactly as in MeshIt"""
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        return np.sqrt(np.sum((max_coords - min_coords) ** 2))