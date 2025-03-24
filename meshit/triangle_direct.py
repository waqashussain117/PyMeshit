"""
DirectTriangleWrapper: Improved implementation of Triangle refinement with C callback.

This module provides a direct wrapper for the Triangle library using the
C++ extension module for the triunsuitable callback.
"""

import numpy as np
import triangle as tr
import logging
from typing import Dict, List, Optional, Tuple, Union
from matplotlib.path import Path

# Import the C++ extension module
from . import triangle_callback

class DirectTriangleWrapper:
    """
    Direct wrapper for Triangle with C++ callback for gradient-based refinement.
    
    This class provides a more efficient implementation of the Triangle refinement
    by using a C++ extension to handle the triunsuitable callback directly.
    """
    
    def __init__(self, gradient: float = 2.0, min_angle: float = 20.0, 
                base_size: Optional[float] = None):
        """
        Initialize the DirectTriangleWrapper with refinement parameters.
        
        Args:
            gradient: Gradient control parameter (default: 2.0)
            min_angle: Minimum angle for triangle quality (default: 20.0)
            base_size: Base size for triangles (calculated from input if None)
        """
        self.gradient = float(gradient)
        self.min_angle = float(min_angle)
        self.base_size = base_size
        self.feature_points = None
        self.feature_sizes = None
        self.logger = logging.getLogger("DirectTriangleWrapper")
        self.triangle_opts = None  # Will store custom triangle options if set
        
    def set_feature_points(self, points: np.ndarray, sizes: np.ndarray):
        """
        Set feature points and their associated sizes.
        
        These points influence the mesh density with a smooth transition
        based on the gradient parameter.
        
        Args:
            points: Array of feature points (N, 2)
            sizes: Array of sizes for each feature point (N,)
        """
        self.feature_points = np.asarray(points, dtype=np.float64)
        self.feature_sizes = np.asarray(sizes, dtype=np.float64)
        
    def _create_boundary_feature_points(self, hull_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create dense feature points along the convex hull boundary.
        
        This ensures a much more uniform transition from boundary to interior.
        
        Args:
            hull_points: Points on the convex hull
            
        Returns:
            Tuple of (boundary feature points, boundary feature sizes)
        """
        if len(hull_points) < 3:
            return np.empty((0, 2)), np.empty(0)
            
        # Create evenly spaced points along each hull edge
        # for a more uniform boundary transition
        boundary_points = []
        boundary_sizes = []
        
        # Size for boundary points - smaller than the base size
        boundary_size = self.base_size * 0.2
        
        # For each hull edge, create intermediate points
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]
            
            # Edge vector
            edge = p2 - p1
            edge_length = np.linalg.norm(edge)
            
            if edge_length < 1e-8:
                continue
                
            # Normalize
            edge = edge / edge_length
            
            # Number of divisions depends on edge length
            num_divisions = max(2, int(edge_length / (boundary_size * 0.5)))
            
            # Create points along the edge
            for j in range(1, num_divisions):
                t = j / num_divisions
                point = p1 + t * (p2 - p1)
                boundary_points.append(point)
                boundary_sizes.append(boundary_size)
        
        return np.array(boundary_points), np.array(boundary_sizes)
        
    def _create_offset_feature_points(self, hull_points: np.ndarray, 
                                     num_layers: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature points offset inward from the hull boundary.
        
        Args:
            hull_points: Points on the convex hull
            num_layers: Number of inward offset layers to create
            
        Returns:
            Tuple of (offset feature points, offset feature sizes)
        """
        if len(hull_points) < 3:
            return np.empty((0, 2)), np.empty(0)
            
        # Calculate centroid
        centroid = np.mean(hull_points, axis=0)
        
        # Create offset points
        offset_points = []
        offset_sizes = []
        
        # For each hull point
        for hull_pt in hull_points:
            # Vector from centroid to hull point
            vec = hull_pt - centroid
            dist = np.linalg.norm(vec)
            
            if dist < 1e-8:
                continue
                
            # Normalize
            vec = vec / dist
            
            # Create offset points along the ray from hull to centroid
            for i in range(1, num_layers + 1):
                # Offset distance increases with each layer
                offset_dist = i * self.base_size * 0.75
                
                # Make sure we don't go past the centroid
                if offset_dist >= dist:
                    offset_dist = dist * 0.8
                    
                # Create offset point
                offset_pt = hull_pt - vec * offset_dist
                
                # Size increases as we move inward
                size_factor = 0.25 + 0.5 * (i / num_layers)
                offset_size = self.base_size * size_factor
                
                offset_points.append(offset_pt)
                offset_sizes.append(offset_size)
        
        return np.array(offset_points), np.array(offset_sizes)
    
    def _create_transition_feature_points(self, points: np.ndarray, hull_points: np.ndarray, 
                                         segments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create more uniform transition feature points between hull and interior.
        Modified to create a more evenly distributed mesh like in MeshIt.
        
        Args:
            points: All input points
            hull_points: Points on the convex hull
            segments: Segment indices forming the boundary
            
        Returns:
            Tuple of (transition feature points, transition feature sizes)
        """
        # Calculate centroid of all points
        centroid = np.mean(points, axis=0)
        
        # Calculate bounding box diagonal for scaling
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        diagonal = np.sqrt(np.sum((max_coords - min_coords) ** 2))
        
        # For more uniform meshes, use a more consistent base size across the domain
        uniform_size = self.base_size * 0.8
        
        # Create boundary feature points with more consistent sizing
        boundary_points = []
        boundary_sizes = []
        
        # Use a more uniform size for boundary points
        boundary_size = uniform_size
        
        # For each hull edge, create evenly spaced points
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]
            
            edge_length = np.linalg.norm(p2 - p1)
            
            if edge_length < 1e-8:
                continue
                
            # More evenly distributed divisions
            num_divisions = max(2, int(edge_length / (boundary_size * 1.2)))
            
            for j in range(1, num_divisions):
                t = j / num_divisions
                point = p1 + t * (p2 - p1)
                boundary_points.append(point)
                # Use more consistent sizing
                boundary_sizes.append(boundary_size)
        
        # Create grid-based interior points for more uniform distribution
        # This is closer to MeshIt's approach in the second image
        interior_points = []
        interior_sizes = []
        
        # Find a suitable grid spacing based on the domain size
        # This creates a more uniform mesh
        grid_spacing = uniform_size * 1.5
        
        # Create a bounding box with some margin
        x_min, y_min = np.min(hull_points, axis=0) + grid_spacing * 0.5
        x_max, y_max = np.max(hull_points, axis=0) - grid_spacing * 0.5
        
        # Generate grid points
        x_range = np.arange(x_min, x_max + grid_spacing, grid_spacing)
        y_range = np.arange(y_min, y_max + grid_spacing, grid_spacing)
        
        # Create a path object for point-in-polygon test
        hull_path = Path(hull_points)
        
        # Add points on a regular grid inside the hull
        for x in x_range:
            for y in y_range:
                point = np.array([x, y])
                if hull_path.contains_point(point):
                    interior_points.append(point)
                    
                    # Use more consistent sizing for uniform triangulation
                    # Only slight variation to avoid grid artifacts
                    variation = np.random.uniform(0.9, 1.1)
                    interior_sizes.append(uniform_size * variation)
        
        # Add some jittered grid points to break up the regularity
        # This helps achieve a more natural but still uniform distribution
        num_jittered = int(len(interior_points) * 0.3)
        for _ in range(num_jittered):
            # Pick a random existing interior point
            if interior_points:
                idx = np.random.randint(0, len(interior_points))
                base_point = interior_points[idx]
                
                # Add jitter (within 30% of grid spacing)
                jitter = (np.random.random(2) - 0.5) * grid_spacing * 0.6
                jittered_point = base_point + jitter
                
                # Check if still inside hull
                if hull_path.contains_point(jittered_point):
                    interior_points.append(jittered_point)
                    interior_sizes.append(uniform_size * np.random.uniform(0.9, 1.1))
        
        # Combine points
        if boundary_points and interior_points:
            all_trans_points = np.vstack((np.array(boundary_points), np.array(interior_points)))
            all_trans_sizes = np.concatenate((np.array(boundary_sizes), np.array(interior_sizes)))
        elif boundary_points:
            all_trans_points = np.array(boundary_points)
            all_trans_sizes = np.array(boundary_sizes)
        elif interior_points:
            all_trans_points = np.array(interior_points)
            all_trans_sizes = np.array(interior_sizes)
        else:
            all_trans_points = np.empty((0, 2))
            all_trans_sizes = np.empty(0)
            
        return all_trans_points, all_trans_sizes
        
    def triangulate(self, points: np.ndarray, segments: Optional[np.ndarray] = None,
                   holes: Optional[np.ndarray] = None, create_feature_points: bool = False,
                   create_transition: bool = False, uniform: bool = True) -> Dict:
        """
        Triangulate points using Triangle with simplified uniform approach by default.
        
        Args:
            points: Input points (N, 2)
            segments: Optional segment indices (M, 2) for constraining triangulation
            holes: Optional hole points (P, 2)
            create_feature_points: Whether to create feature points (default: False for uniform mesh)
            create_transition: Whether to create transition feature points (default: False for uniform mesh)
            uniform: Whether to use uniform mesh generation (default: True)
            
        Returns:
            Dictionary with triangulation results (vertices, triangles)
        """
        # Ensure inputs are numpy arrays
        points = np.asarray(points, dtype=np.float64)
        
        if segments is not None:
            segments = np.asarray(segments, dtype=np.int32)
        
        if holes is not None:
            holes = np.asarray(holes, dtype=np.float64)
            
        # Calculate base_size if not provided
        if self.base_size is None:
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            diagonal = np.sqrt(np.sum((max_coords - min_coords) ** 2))
            self.base_size = diagonal / 15.0  # MeshIt's scaling
            
        # Set up Triangle input
        tri_input = {'vertices': points}
        
        if segments is not None and len(segments) > 0:
            tri_input['segments'] = segments
            
        if holes is not None and len(holes) > 0:
            tri_input['holes'] = holes

        # For uniform triangulation (default and simpler approach)
        if uniform:
            # Simple approach: Use quality and area constraints for uniform meshes
            effective_min_angle = 25.0  # Higher angle for better quality
            area_constraint = self.base_size * self.base_size * 0.5
            
            # Simple Triangle options for uniform meshes - no feature points needed
            # p = PSLG (use segments)
            # z = Number vertices from zero
            # q = Quality constraint (higher min angle)
            # a = Area constraint for uniform size
            # Y = Prohibits Steiner points on the boundary
            tri_options = f'pzYq{effective_min_angle}a{area_constraint}'
            self.logger.info(f"Using simple uniform Triangle options: '{tri_options}'")
            
            # Feature points are only used if explicitly requested
            if create_feature_points and self.feature_points is not None and len(self.feature_points) > 0:
                # Use feature points with uniform sizing
                combined_feature_points = self.feature_points
                combined_feature_sizes = np.ones(len(self.feature_points)) * self.base_size
                
                # Initialize C++ callback
                self.logger.info(f"Using C++ callback with uniform sizing for {len(combined_feature_points)} feature points")
                triangle_callback.initialize_gradient_control(
                    1.0,  # Use gradient=1.0 for uniform sizing
                    self.base_size * self.base_size,
                    combined_feature_points,
                    combined_feature_sizes
                )
                # Add 'u' option to use the callback
                tri_options += 'u'
        
        # For gradient-based refinement (more complex, non-default approach)
        else:
            # Original behavior with gradient-based refinement
            effective_gradient = self.gradient
            
            # Adjust min angle based on gradient
            effective_min_angle = self.min_angle
            if self.gradient > 1.0:
                effective_min_angle = max(20.0 - (self.gradient - 1.0) * 5.0, 10.0)
                
            # Area constraint
            area_constraint = self.base_size * self.base_size * 0.5
            
            # Base triangle options
            tri_options = f'pzYYq{effective_min_angle}a{area_constraint}'
            
            # Generate feature points only if requested
            if create_feature_points:
                all_feature_points = []
                all_feature_sizes = []
                
                # Add user-provided feature points
                if self.feature_points is not None and len(self.feature_points) > 0:
                    all_feature_points.append(self.feature_points)
                    all_feature_sizes.append(self.feature_sizes)
                
                # Add hull points as features if segments provided
                if segments is not None and len(segments) > 0 and create_feature_points:
                    hull_indices = np.unique(segments.flatten())
                    hull_points = points[hull_indices]
                    
                    hull_feature_points = hull_points
                    hull_feature_sizes = np.ones(len(hull_points)) * self.base_size * 0.15
                    
                    all_feature_points.append(hull_feature_points)
                    all_feature_sizes.append(hull_feature_sizes)
                    
                    # Generate transition feature points if requested
                    if create_transition:
                        trans_points, trans_sizes = self._create_transition_feature_points(
                            points, hull_points, segments)
                        
                        if len(trans_points) > 0:
                            all_feature_points.append(trans_points)
                            all_feature_sizes.append(trans_sizes)
                
                # Combine all feature points
                if all_feature_points:
                    combined_feature_points = np.vstack(all_feature_points)
                    combined_feature_sizes = np.concatenate(all_feature_sizes)
                    
                    # Initialize C++ callback with all feature points
                    if len(combined_feature_points) > 0:
                        self.logger.info(f"Using C++ callback with {len(combined_feature_points)} feature points")
                        triangle_callback.initialize_gradient_control(
                            effective_gradient,
                            self.base_size * self.base_size,
                            combined_feature_points,
                            combined_feature_sizes
                        )
                        # Add 'u' option to use callback
                        tri_options += 'u'
        
        # Allow overriding triangle options
        if hasattr(self, 'triangle_opts'):
            tri_options = self.triangle_opts
            self.logger.info(f"Using custom Triangle options: '{tri_options}'")
        
        try:
            # Run Triangle with options
            result = tr.triangulate(tri_input, tri_options)
            
            # Check if triangulation was successful
            if 'triangles' in result and len(result['triangles']) > 0:
                self.logger.info(f"Triangulation complete: {len(result['triangles'])} triangles")
                return result
            else:
                self.logger.error("Triangulation failed to produce triangles")
                # Fall back to standard Triangle without callback
                fallback_options = tri_options.replace('u', '')
                self.logger.info(f"Falling back to standard Triangle: '{fallback_options}'")
                return tr.triangulate(tri_input, fallback_options)
        except Exception as e:
            self.logger.error(f"Error during triangulation: {str(e)}")
            # Fall back to standard Triangle without callback
            fallback_options = tri_options.replace('u', '')
            self.logger.info(f"Falling back to standard Triangle due to error: '{fallback_options}'")
            return tr.triangulate(tri_input, fallback_options)
    
    def _create_uniform_grid_points(self, hull_points: np.ndarray, 
                                  spacing: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a uniform grid of points inside the hull for better uniform meshing.
        
        Args:
            hull_points: Points defining the convex hull
            spacing: Spacing between grid points (defaults to base_size if None)
            
        Returns:
            Tuple of (grid points, grid point sizes)
        """
        if len(hull_points) < 3:
            return np.empty((0, 2)), np.empty(0)
            
        if spacing is None:
            spacing = self.base_size * 1.5
            
        # Create a path object for point-in-polygon test
        try:
            # Use matplotlib's Path for point-in-polygon test
            hull_path = Path(hull_points)
            
            # Find a suitable grid spacing based on the domain size
            # Create a bounding box with some margin
            x_min, y_min = np.min(hull_points, axis=0) + spacing * 0.5
            x_max, y_max = np.max(hull_points, axis=0) - spacing * 0.5
            
            # Generate grid points
            x_range = np.arange(x_min, x_max + spacing, spacing)
            y_range = np.arange(y_min, y_max + spacing, spacing)
            
            # Create grid points
            grid_points = []
            grid_sizes = []
            
            # Add points on a regular grid inside the hull
            for x in x_range:
                for y in y_range:
                    point = np.array([x, y])
                    if hull_path.contains_point(point):
                        grid_points.append(point)
                        grid_sizes.append(self.base_size)
            
            # Add some jittered grid points to break up the regularity
            # This helps achieve a more natural but still uniform distribution
            num_jittered = int(len(grid_points) * 0.2)
            for _ in range(num_jittered):
                # Pick a random existing grid point
                if grid_points:
                    idx = np.random.randint(0, len(grid_points))
                    base_point = grid_points[idx]
                    
                    # Add jitter (within 30% of grid spacing)
                    jitter = (np.random.random(2) - 0.5) * spacing * 0.3
                    jittered_point = base_point + jitter
                    
                    # Check if still inside hull
                    if hull_path.contains_point(jittered_point):
                        grid_points.append(jittered_point)
                        grid_sizes.append(self.base_size)
            
            return np.array(grid_points), np.array(grid_sizes)
        except Exception as e:
            self.logger.error(f"Error creating uniform grid: {str(e)}")
            return np.empty((0, 2)), np.empty(0)
            
    def __del__(self):
        """Clean up C++ resources when the wrapper is destroyed."""
        # No explicit cleanup needed, the Python-C++ binding handles this 

    def set_triangle_options(self, options: str):
        """
        Set custom options for the Triangle library.
        
        This allows direct control over Triangle's behavior by passing specific
        options string that will override the default options generated.
        
        Args:
            options: String with Triangle options (e.g., 'pzq30a40')
        """
        self.triangle_opts = options
        self.logger.info(f"Setting custom Triangle options: {options}") 