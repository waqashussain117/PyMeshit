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
        Create more uniform transition feature points between hull and interior
        with enhanced distribution for high-quality meshes similar to C++ MeshIt.
        
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
        domain_width = max_coords[0] - min_coords[0]
        domain_height = max_coords[1] - min_coords[1]
        diagonal = np.sqrt(domain_width**2 + domain_height**2)
        
        # Calculate target element size based on domain size
        # C++ MeshIt likely uses similar scaling
        uniform_size = self.base_size * 0.85  # Base size is already set from outside
        
        # --- Boundary feature points with consistent spacing ---
        boundary_points = []
        boundary_sizes = []
        
        # Higher density for boundary points (smaller than uniform_size)
        boundary_size = uniform_size * 0.7
        
        # Create evenly distributed points along the boundary
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]
            
            edge_length = np.linalg.norm(p2 - p1)
            if edge_length < 1e-8:
                continue
                
            # Calculate number of divisions based on edge length
            num_divisions = max(3, int(edge_length / (boundary_size * 0.8)))
            
            # Create points along the edge
            for j in range(1, num_divisions):
                t = j / num_divisions
                point = p1 + t * (p2 - p1)
                boundary_points.append(point)
                # Use consistent sizing for boundary
                boundary_sizes.append(boundary_size)
        
        # --- Interior grid with controlled perturbation (C++ MeshIt style) ---
        interior_points = []
        interior_sizes = []
        
        # Calculate the optimal grid spacing based on base_size
        grid_spacing = uniform_size * 1.3  # Slightly larger than element size
        
        # Determine the safe zone for interior points (inset from hull to avoid bad elements)
        hull_bb_min = np.min(hull_points, axis=0)
        hull_bb_max = np.max(hull_points, axis=0)
        
        # Add a small inset to avoid points too close to boundary
        inset = grid_spacing * 0.25
        x_min = hull_bb_min[0] + inset
        y_min = hull_bb_min[1] + inset
        x_max = hull_bb_max[0] - inset
        y_max = hull_bb_max[1] - inset
        
        # Generate grid with slight offset to avoid alignment with boundary
        offset_factor = 0.1
        x_offset = grid_spacing * offset_factor
        y_offset = grid_spacing * offset_factor
        
        # Create grid ranges with offset
        x_range = np.arange(x_min + x_offset, x_max, grid_spacing)
        y_range = np.arange(y_min + y_offset, y_max, grid_spacing)
        
        # Create a path object for point-in-polygon test
        hull_path = Path(hull_points)
        
        # Generate grid points with MeshIt-style sinusoidal perturbation
        base_grid_points = []
        
        # First create a base grid
        for x in x_range:
            for y in y_range:
                point = np.array([x, y])
                if hull_path.contains_point(point):
                    base_grid_points.append(point)
        
        # Apply sinusoidal perturbation to create C++ MeshIt-like distribution
        grid_points = []
        
        # Parameters for sinusoidal perturbation
        freq_factor_x = 2.0 * np.pi / domain_width
        freq_factor_y = 2.0 * np.pi / domain_height
        amp_factor = grid_spacing * 0.2  # Amplitude of perturbation (20% of grid spacing)
        
        for point in base_grid_points:
            # Distance from centroid (normalized)
            dx = point[0] - centroid[0]
            dy = point[1] - centroid[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            # Calculate angle from centroid
            angle = np.arctan2(dy, dx)
            
            # Calculate perturbation factors using multiple frequencies for natural distribution
            # This creates the sinusoidal pattern visible in C++ MeshIt
            perturb_x = amp_factor * np.sin(freq_factor_x * point[0] + freq_factor_y * point[1] * 0.5)
            perturb_y = amp_factor * np.sin(freq_factor_y * point[1] + freq_factor_x * point[0] * 0.5)
            
            # Add small radial component to make elements grow gradually from boundary
            radial_factor = 0.15
            boundary_dist = min_distance_to_boundary(point, hull_points)
            scaling = min(1.0, boundary_dist / (grid_spacing * 2))
            radial_perturb = radial_factor * scaling * grid_spacing
            
            perturb_x += radial_perturb * np.cos(angle)
            perturb_y += radial_perturb * np.sin(angle)
            
            # Apply perturbation to create a more natural distribution
            perturbed_point = np.array([point[0] + perturb_x, point[1] + perturb_y])
            
            # Verify perturbed point is still inside hull
            if hull_path.contains_point(perturbed_point):
                grid_points.append(perturbed_point)
                
                # Size variation based on distance from boundary
                size_factor = 0.9 + 0.2 * (scaling ** 0.5)  # Smoother transition
                interior_sizes.append(uniform_size * size_factor)
        
        # --- Add additional points for smoother transitions ---
        # This is similar to how C++ MeshIt creates more uniform transitions
        transition_points = []
        transition_sizes = []
        
        # Add radial rays from centroid to help with size transitions
        num_rays = 12
        ray_angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
        
        # Find typical distance from centroid to boundary
        distances_to_centroid = [np.linalg.norm(p - centroid) for p in hull_points]
        avg_radius = np.mean(distances_to_centroid) * 0.8
        
        # Create rays from centroid (not all the way to boundary)
        for angle in ray_angles:
            # Create points along ray with increasing spacing
            num_points = 5
            for i in range(1, num_points):
                # Gradually increasing spacing from centroid
                t = (i / num_points) ** 1.2  # Non-linear spacing for better transitions
                ray_length = avg_radius * t
                
                # Calculate point on ray
                ray_point = np.array([
                    centroid[0] + ray_length * np.cos(angle),
                    centroid[1] + ray_length * np.sin(angle)
                ])
                
                # Check if point is inside hull
                if hull_path.contains_point(ray_point):
                    # Size increases with distance from centroid
                    size_factor = 0.8 + 0.4 * t
                    
                    # Add slightly randomized point to break regularity
                    jitter = (np.random.random(2) - 0.5) * grid_spacing * 0.1
                    jittered_ray_point = ray_point + jitter
                    
                    if hull_path.contains_point(jittered_ray_point):
                        transition_points.append(jittered_ray_point)
                        transition_sizes.append(uniform_size * size_factor)
        
        # --- Combine all points ---
        all_feature_points = []
        all_feature_sizes = []
        
        # Add boundary points (always include these)
        if boundary_points:
            all_feature_points.extend(boundary_points)
            all_feature_sizes.extend(boundary_sizes)
        
        # Add interior grid points (main distribution)
        if grid_points:
            all_feature_points.extend(grid_points)
            all_feature_sizes.extend(interior_sizes)
            
        # Add transition points (improves smoothness)
        if transition_points:
            all_feature_points.extend(transition_points)
            all_feature_sizes.extend(transition_sizes)
            
        # Convert to numpy arrays
        if all_feature_points:
            all_trans_points = np.array(all_feature_points)
            all_trans_sizes = np.array(all_feature_sizes)
        else:
            all_trans_points = np.empty((0, 2))
            all_trans_sizes = np.empty(0)
            
        self.logger.info(f"Created {len(all_trans_points)} enhanced transition points for more uniform mesh")
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
            # Simple approach: Use quality and area constraints derived from GUI/init params
            effective_min_angle = self.min_angle # *** USE self.min_angle FROM GUI ***
            area_constraint = self.base_size * self.base_size * 0.5 # Use calculated base_size
            
            # Triangle options: pzqa (respect GUI angle and density/base_size)
            # We avoid 'Y' to allow boundary refinement if needed for quality.
            tri_options = f'pzq{effective_min_angle:.1f}a{area_constraint:.8f}'
            self.logger.info(f"Using options derived from GUI for uniform request: '{tri_options}'")
            
            # NOTE: We are intentionally NOT using the C++ callback path for uniform=True,
            # as the goal is consistent triangulation based on angle/area constraints.
            # Feature points are effectively ignored in this mode for simplicity.
        
        # For gradient-based refinement (more complex, non-default approach)
        else:
            # Original behavior with gradient-based refinement
            effective_gradient = self.gradient
            
            # Adjust min angle based on gradient
            effective_min_angle = self.min_angle
            if self.gradient > 1.0:
                effective_min_angle = max(20.0 - (self.gradient - 1.0) * 5.0, 10.0)
                
            # Area constraint (we will not use it in tri_options directly for this test)
            # area_constraint = self.base_size * self.base_size * 0.5 
            
            # Base triangle options - MODIFIED: Removed area_constraint 'a' for robustness
            # Still includes q (quality), c (enclose convex hull), C (consistency), D (Conforming Delaunay)
            tri_options = f'pzq{effective_min_angle:.1f}cCD'
            self.logger.info(f"Using robust options (no area constraint) for constrained triangulation: '{tri_options}'")
            
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
        if hasattr(self, 'triangle_opts') and self.triangle_opts:
            tri_options = self.triangle_opts
            self.logger.info(f"Using custom Triangle options: '{tri_options}'")
        
        # --- Add detailed logging before the call --- 
        self.logger.debug(f"Calling tr.triangulate with options: '{tri_options}'")
        self.logger.debug(f"Input vertices shape: {tri_input.get('vertices', np.empty(0)).shape}")
        self.logger.debug(f"Input segments shape: {tri_input.get('segments', np.empty(0)).shape}")
        self.logger.debug(f"Input holes shape: {tri_input.get('holes', np.empty(0)).shape}")
        # Log first few vertices/segments if they exist
        if 'vertices' in tri_input and len(tri_input['vertices']) > 0:
            self.logger.debug(f"First 3 vertices:\n{tri_input['vertices'][:3]}")
        if 'segments' in tri_input and len(tri_input['segments']) > 0:
            self.logger.debug(f"First 3 segments:\n{tri_input['segments'][:3]}")
        # --- End detailed logging --- 
        
        try:
            # Run Triangle with options
            result = tr.triangulate(tri_input, tri_options)
            
            # Check if triangulation was successful
            if 'triangles' in result and len(result['triangles']) > 0:
                self.logger.info(f"Triangulation complete: {len(result['triangles'])} triangles")
                return result
            else:
                self.logger.error("Triangulation failed to produce triangles")
                # Fall back to standard Triangle without callback and with exact arithmetic
                base_fallback_options = tri_options.replace('u', '') # Remove size callback option
                robust_fallback_options = base_fallback_options.replace('Y', '') # Remove no-exact-arithmetic option
                
                if robust_fallback_options == base_fallback_options: # Y was not there
                    self.logger.info(f"Falling back to standard Triangle (no 'u', Y was not present): '{robust_fallback_options}'")
                else:
                    self.logger.info(f"Falling back to standard Triangle with exact arithmetic (no 'u', removed 'Y'): '{robust_fallback_options}'")
                
                # It's possible the error is unrelated to 'u' or 'Y', so try with robust_fallback_options
                try:
                    return tr.triangulate(tri_input, robust_fallback_options)
                except Exception as e_fallback:
                    self.logger.error(f"Robust fallback triangulation also failed: {str(e_fallback)}")
                    # As a last resort, if the original options didn't have 'u' but had 'Y', try removing 'Y' from original
                    if 'u' not in tri_options and 'Y' in tri_options:
                        ultimate_fallback_options = tri_options.replace('Y', '')
                        self.logger.info(f"Attempting ultimate fallback (original opts, removed 'Y'): '{ultimate_fallback_options}'")
                        try:
                            return tr.triangulate(tri_input, ultimate_fallback_options)
                        except Exception as e_ultimate:
                            self.logger.error(f"Ultimate fallback triangulation also failed: {str(e_ultimate)}")
                            raise e_ultimate # Re-raise the last error if all fallbacks fail
                    raise e_fallback # Re-raise the fallback error
        except Exception as e:
            self.logger.error(f"Error during triangulation: {str(e)}")
            # Fall back to standard Triangle without callback and with exact arithmetic
            base_fallback_options = tri_options.replace('u', '') # Remove size callback option
            robust_fallback_options = base_fallback_options.replace('Y', '') # Remove no-exact-arithmetic option
            
            if robust_fallback_options == base_fallback_options: # Y was not there
                self.logger.info(f"Falling back to standard Triangle (no 'u', Y was not present): '{robust_fallback_options}'")
            else:
                self.logger.info(f"Falling back to standard Triangle with exact arithmetic (no 'u', removed 'Y'): '{robust_fallback_options}'")
            
            # It's possible the error is unrelated to 'u' or 'Y', so try with robust_fallback_options
            try:
                return tr.triangulate(tri_input, robust_fallback_options)
            except Exception as e_fallback:
                self.logger.error(f"Robust fallback triangulation also failed: {str(e_fallback)}")
                # As a last resort, if the original options didn't have 'u' but had 'Y', try removing 'Y' from original
                if 'u' not in tri_options and 'Y' in tri_options:
                    ultimate_fallback_options = tri_options.replace('Y', '')
                    self.logger.info(f"Attempting ultimate fallback (original opts, removed 'Y'): '{ultimate_fallback_options}'")
                    try:
                        return tr.triangulate(tri_input, ultimate_fallback_options)
                    except Exception as e_ultimate:
                        self.logger.error(f"Ultimate fallback triangulation also failed: {str(e_ultimate)}")
                        raise e_ultimate # Re-raise the last error if all fallbacks fail
                raise e_fallback # Re-raise the fallback error
    
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

# Helper function for calculating minimum distance to boundary
def min_distance_to_boundary(point, hull_points):
    """Calculate minimum distance from a point to the hull boundary."""
    min_dist = float('inf')
    
    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]
        
        # Line segment vector
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-8:
            continue
            
        # Normalize line vector
        line_vec = line_vec / line_len
        
        # Vector from line start to point
        point_vec = point - p1
        
        # Project point onto line
        proj_len = np.dot(point_vec, line_vec)
        
        # Calculate closest point on line segment
        if proj_len < 0:
            closest = p1  # Before start of segment
        elif proj_len > line_len:
            closest = p2  # After end of segment
        else:
            # On the segment
            closest = p1 + line_vec * proj_len
            
        # Calculate distance to closest point
        dist = np.linalg.norm(point - closest)
        min_dist = min(min_dist, dist)
        
    return min_dist 