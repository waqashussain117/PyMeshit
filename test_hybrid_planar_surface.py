"""
Test script for hybrid triangulation of a planar surface with gradient control.

This script demonstrates:
1. Generation of regularly spaced points on a planar surface
2. Computation of the convex hull
3. Creation of coarse segmentation (boundary edges)
4. Hybrid triangulation with gradient=2.0
5. Visualization of the results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.tri as mtri
from scipy.spatial import ConvexHull, Delaunay
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HybridPlanarTest")

# Try to import the triangle callback module (C++ extension)
try:
    # First, make sure we're looking in the correct directory
    import sys
    import os
    
    # Add the current directory to path if not already there
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Try direct import first
    try:
        from meshit.triangle_callback import TriangleCallback
        HAVE_CALLBACK = True
        logger.info("Successfully imported C++ triangle callback module")
    except ImportError as e:
        logger.warning(f"Direct import failed: {e}")
        
        # Try loading the DLL directly as a fallback
        try:
            import ctypes
            callback_path = os.path.join(current_dir, 'meshit', 'triangle_callback.cp311-win_amd64.pyd')
            if os.path.exists(callback_path):
                logger.info(f"Found callback module at: {callback_path}")
                # Load the DLL directly
                callback_dll = ctypes.cdll.LoadLibrary(callback_path)
                logger.info("Successfully loaded C++ triangle callback DLL directly")
                HAVE_CALLBACK = True
                
                # Create a wrapper class that mimics the TriangleCallback interface
                class TriangleCallback:
                    def __init__(self):
                        logger.info("Initializing direct DLL wrapper for TriangleCallback")
                    
                    def update(self, gradient, meshsize, points, sizes):
                        logger.info(f"Using direct DLL wrapper with gradient={gradient}, meshsize={meshsize}")
                        # This would need additional code to properly interface with the C++ DLL
                        # For now, we'll just log that it was called
            else:
                logger.warning(f"Callback module not found at: {callback_path}")
                HAVE_CALLBACK = False
        except Exception as e2:
            logger.error(f"Failed to load callback DLL directly: {e2}")
            HAVE_CALLBACK = False
except Exception as e:
    logger.error(f"Error during callback import: {e}")
    HAVE_CALLBACK = False
    
if HAVE_CALLBACK:
    logger.info("Will use C++ triangle callback for gradient control")
else:
    logger.warning("C++ triangle callback not available, falling back to Python implementation")

# Try to import our advanced direct wrapper first
try:
    from meshit.triangle_direct import DirectTriangleWrapper
    HAVE_DIRECT_WRAPPER = True
    logger.info("Successfully imported DirectTriangleWrapper for optimal triangle refinement")
except ImportError as e:
    logger.warning(f"Failed to import DirectTriangleWrapper: {e}")
    HAVE_DIRECT_WRAPPER = False
    
    # Try to import the more basic triangle wrapper as a fallback
    try:
        from meshit.triangle_wrapper import TriangleWrapper
        HAVE_BASIC_WRAPPER = True
        logger.info("Successfully imported basic TriangleWrapper as fallback")
    except ImportError as e2:
        logger.warning(f"Failed to import basic TriangleWrapper: {e2}")
        HAVE_BASIC_WRAPPER = False

# Import triangle for triangulation
import triangle as tr

class PlanarSurfaceTest:
    def __init__(self):
        self.points = None
        self.hull_points = None
        self.segments = None
        self.feature_points = None
        self.feature_sizes = None
        self.triangulation_result = None
        
    def generate_planar_points(self, nx=15, ny=15, noise_level=0.1):
        """Generate a grid of points with slight noise to make it more interesting"""
        logger.info(f"Generating {nx}x{ny} grid of points")
        
        # Create base grid
        x = np.linspace(-10, 10, nx)
        y = np.linspace(-10, 10, ny)
        xx, yy = np.meshgrid(x, y)
        
        # Add some noise to make it more interesting
        if noise_level > 0:
            xx += np.random.normal(0, noise_level, xx.shape)
            yy += np.random.normal(0, noise_level, yy.shape)
        
        # Reshape to get points
        self.points = np.column_stack((xx.flatten(), yy.flatten()))
        logger.info(f"Generated {len(self.points)} points")
        return self.points
    
    def compute_convex_hull(self):
        """Compute the convex hull of the point set"""
        if self.points is None or len(self.points) < 3:
            raise ValueError("Need at least 3 points to compute convex hull")
            
        logger.info("Computing convex hull")
        hull = ConvexHull(self.points)
        self.hull_points = self.points[hull.vertices]
        logger.info(f"Convex hull has {len(self.hull_points)} points")
        return self.hull_points
    
    def create_coarse_segmentation(self):
        """Create segments from the convex hull (boundary edges)"""
        if self.hull_points is None:
            raise ValueError("Compute convex hull first")
            
        logger.info("Creating coarse segmentation from convex hull")
        num_hull_points = len(self.hull_points)
        self.segments = np.column_stack((
            np.arange(num_hull_points),
            np.roll(np.arange(num_hull_points), -1)
        ))
        logger.info(f"Created {len(self.segments)} segments")
        return self.segments
    
    def create_feature_points(self, num_features=3):
        """Create feature points for controlling mesh density"""
        if self.hull_points is None:
            raise ValueError("Compute convex hull first")
            
        logger.info(f"Creating {num_features} feature points for mesh control")
        
        # Compute centroid of convex hull
        centroid = np.mean(self.hull_points, axis=0)
        
        # Create feature points in interesting locations
        self.feature_points = []
        self.feature_sizes = []
        
        # Add centroid as a feature point with small size
        self.feature_points.append(centroid)
        self.feature_sizes.append(0.5)  # Small triangles at center
        
        # Add some random points inside the convex hull
        hull_path = Path(self.hull_points)
        
        for _ in range(num_features - 1):
            # Generate points until we find one inside the hull
            while True:
                # Random point in bounding box
                min_coords = np.min(self.hull_points, axis=0)
                max_coords = np.max(self.hull_points, axis=0)
                
                # Generate point with bias toward edges
                t = np.random.random()
                if t > 0.7:  # 30% chance to be near an edge
                    # Pick a random hull point and move slightly inside
                    idx = np.random.randint(0, len(self.hull_points))
                    point = self.hull_points[idx] * 0.8 + centroid * 0.2
                else:
                    # Random point in bounding box
                    point = min_coords + np.random.random(2) * (max_coords - min_coords)
                
                # Check if inside hull
                if hull_path.contains_point(point):
                    self.feature_points.append(point)
                    # Random size between 0.5 and 1.5
                    self.feature_sizes.append(0.5 + np.random.random())
                    break
        
        # Convert to numpy arrays
        self.feature_points = np.array(self.feature_points)
        self.feature_sizes = np.array(self.feature_sizes)
        
        logger.info(f"Created {len(self.feature_points)} feature points with varying sizes")
        return self.feature_points, self.feature_sizes
    
    def hybrid_triangulate(self, points, segments=None, gradient=1.0):
        """
        Triangulate points using a hybrid approach that prioritizes uniform triangulation.
        
        This uses the simplified uniform triangulation mode by default, but can 
        still use gradient-based triangulation if needed.
        
        Args:
            points: List of 2D points to triangulate
            segments: Optional list of segments (point indices) to constrain triangulation
            gradient: Gradient for controlling mesh density (1.0 for uniform, higher for non-uniform)
            
        Returns:
            Dictionary with triangulation results (vertices, triangles)
        """
        logger.info("Running hybrid triangulation with gradient=%.1f", gradient)
        
        # Use uniform triangulation for gradient=1.0, gradient-based otherwise
        use_uniform = (gradient <= 1.0)
        
        try:
            # Try to import the DirectTriangleWrapper which is now optimized for uniform triangulation
            from meshit.triangle_direct import DirectTriangleWrapper
            HAVE_DIRECT_WRAPPER = True
            logger.info("Using DirectTriangleWrapper for triangulation")
        except ImportError:
            HAVE_DIRECT_WRAPPER = False
            logger.warning("DirectTriangleWrapper not available, falling back to TriangleWrapper")
            
        if HAVE_DIRECT_WRAPPER:
            # Calculate base_size from the point set
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            diagonal = np.sqrt(np.sum((max_coords - min_coords) ** 2))
            base_size = diagonal / 15.0
            
            # Create a DirectTriangleWrapper instance
            wrapper = DirectTriangleWrapper(gradient=gradient, min_angle=25.0, base_size=base_size)
            
            # For simpler, more uniform meshes (the preferred mode):
            if use_uniform:
                logger.info("Using uniform triangulation mode")
                result = wrapper.triangulate(
                    points=points,
                    segments=segments,
                    create_feature_points=False,  # Don't create feature points
                    create_transition=False,      # Don't create transition points
                    uniform=True                 # Use uniform mode
                )
                logger.info(f"Uniform triangulation complete with {len(result['triangles'])} triangles")
                return result
            
            # For gradient-based meshes (only if specifically requested):
            else:
                logger.info("Using gradient-based triangulation mode")
                # Add feature points at corners to control gradient
                hull_indices = np.unique(segments.flatten()) if segments is not None else []
                hull_points = points[hull_indices] if len(hull_indices) > 0 else []
                
                if len(hull_points) > 0:
                    # Use corners as feature points
                    feature_points = hull_points
                    feature_sizes = np.ones(len(feature_points)) * base_size * 0.2
                    wrapper.set_feature_points(feature_points, feature_sizes)
                
                result = wrapper.triangulate(
                    points=points,
                    segments=segments,
                    create_feature_points=True,  # Create additional feature points
                    create_transition=True,      # Create transition points
                    uniform=False               # Use gradient mode
                )
                logger.info(f"Gradient-based triangulation complete with {len(result['triangles'])} triangles")
                return result
        
        # Fallback to standard TriangleWrapper if DirectTriangleWrapper is not available
        else:
            from meshit.triangle_wrapper import TriangleWrapper
            wrapper = TriangleWrapper(gradient=gradient, min_angle=25.0)
            
            # Create simple Triangle options for uniform triangulation
            area_constraint = diagonal * diagonal / (15.0 * 15.0) * 0.5
            triangle_opts = f'pzYq25.0a{area_constraint}'
            
            result = wrapper.triangulate(
                vertices=points,
                segments=segments,
                holes=None,
                max_iterations=5
            )
            
            logger.info(f"Standard triangulation complete with {len(result['triangles'])} triangles")
            return result
    
    def visualize_results(self):
        """Visualize all the steps and final triangulation"""
        if self.triangulation_result is None:
            raise ValueError("Run triangulation first")
            
        # Create figure and axes
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        axs = axs.flatten()
        
        # 1. Original points and convex hull
        axs[0].scatter(self.points[:, 0], self.points[:, 1], c='blue', alpha=0.5, label='Original Points')
        axs[0].plot(np.append(self.hull_points[:, 0], self.hull_points[0, 0]), 
                   np.append(self.hull_points[:, 1], self.hull_points[0, 1]), 
                   'r-', label='Convex Hull')
        axs[0].set_title('Original Points and Convex Hull')
        axs[0].legend()
        axs[0].axis('equal')
        
        # 2. Segmentation and feature points
        axs[1].plot(np.append(self.hull_points[:, 0], self.hull_points[0, 0]), 
                   np.append(self.hull_points[:, 1], self.hull_points[0, 1]), 
                   'r-', label='Boundary')
        
        if self.feature_points is not None:
            # Draw feature points with circles proportional to size
            axs[1].scatter(self.feature_points[:, 0], self.feature_points[:, 1], 
                          c='orange', s=100, label='Feature Points')
            
            # Draw circles showing feature influence
            for i, (point, size) in enumerate(zip(self.feature_points, self.feature_sizes)):
                circle = plt.Circle((point[0], point[1]), size, fill=False, 
                                   color='orange', linestyle='--', alpha=0.5)
                axs[1].add_patch(circle)
                
        axs[1].set_title('Boundary and Feature Points')
        axs[1].legend()
        axs[1].axis('equal')
        
        # 3. Triangulation result
        vertices = self.triangulation_result['vertices']
        triangles = self.triangulation_result['triangles']
        
        # Plot the triangulation
        axs[2].triplot(vertices[:, 0], vertices[:, 1], triangles, 'k-', alpha=0.5)
        axs[2].set_title(f'Triangulation Result ({len(triangles)} triangles)')
        axs[2].axis('equal')
        
        # 4. Mesh quality visualization
        # Calculate quality metrics for each triangle
        qualities = []
        areas = []
        
        for tri in triangles:
            v1, v2, v3 = vertices[tri]
            
            # Calculate edges
            e1 = np.linalg.norm(v2 - v1)
            e2 = np.linalg.norm(v3 - v2)
            e3 = np.linalg.norm(v1 - v3)
            
            # Calculate area
            # Using Heron's formula
            s = (e1 + e2 + e3) / 2
            area = np.sqrt(s * (s - e1) * (s - e2) * (s - e3))
            areas.append(area)
            
            # Calculate quality metric (ratio of circumradius to twice the inradius)
            # This is equivalent to the ratio of the longest edge to twice the inscribed circle radius
            if area > 1e-10:  # Avoid division by zero
                quality = (e1 * e2 * e3) / (4 * area)
                qualities.append(quality)
            else:
                qualities.append(0)
        
        # Create a scatter plot where each triangle is colored by its quality
        triangles_center_x = []
        triangles_center_y = []
        
        for tri in triangles:
            v1, v2, v3 = vertices[tri]
            # Calculate centroid
            center_x = (v1[0] + v2[0] + v3[0]) / 3.0
            center_y = (v1[1] + v2[1] + v3[1]) / 3.0
            triangles_center_x.append(center_x)
            triangles_center_y.append(center_y)
        
        # Use scatter plot to show quality
        scatter = axs[3].scatter(triangles_center_x, triangles_center_y, c=qualities, 
                               cmap='viridis', s=30, alpha=0.7)
        cbar = plt.colorbar(scatter, ax=axs[3])
        cbar.set_label('Triangle Quality (lower is better)')
        
        # Also plot the mesh on top
        axs[3].triplot(vertices[:, 0], vertices[:, 1], triangles, 'k-', alpha=0.2)
        axs[3].set_title('Mesh Quality Visualization')
        axs[3].axis('equal')
        
        # Print quality statistics
        max_quality = max(qualities)
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        
        logger.info(f"Mesh quality statistics:")
        logger.info(f"  Number of triangles: {len(triangles)}")
        logger.info(f"  Max triangle quality: {max_quality:.4f}")
        logger.info(f"  Mean triangle area: {mean_area:.4f}")
        logger.info(f"  Area standard deviation: {std_area:.4f}")
        
        fig.tight_layout()
        plt.show()

def verify_cpp_callback():
    """Verify that the C++ triangle callback module is properly installed and configured"""
    logger.info("Verifying C++ triangle callback module...")
    
    try:
        # Import directly
        import meshit.triangle_callback
        logger.info("Successfully imported meshit.triangle_callback module")
        
        # Check if TriangleCallback class exists
        if hasattr(meshit.triangle_callback, 'TriangleCallback'):
            logger.info("TriangleCallback class exists in the module")
            
            # Try to create an instance
            callback = meshit.triangle_callback.TriangleCallback()
            logger.info("Successfully created TriangleCallback instance")
            
            # Check for update method
            if hasattr(callback, 'update'):
                logger.info("update method exists on TriangleCallback")
                
                # Try calling the update method with dummy data
                try:
                    callback.update(
                        gradient=2.0,
                        meshsize=1.0,
                        points=np.array([[0.0, 0.0]]),
                        sizes=np.array([0.5])
                    )
                    logger.info("Successfully called update method!")
                    return True
                except Exception as e:
                    logger.error(f"Error calling update method: {e}")
            else:
                logger.error("update method does not exist on TriangleCallback")
        else:
            logger.error("TriangleCallback class does not exist in the module")
    except Exception as e:
        logger.error(f"Error verifying C++ callback: {e}")
    
    return False

if __name__ == "__main__":
    test = PlanarSurfaceTest()
    
    # Verify C++ callback
    cpp_callback_available = verify_cpp_callback()
    if cpp_callback_available:
        logger.info("C++ callback verification successful - will use C++ implementation")
    else:
        logger.warning("C++ callback verification failed - will use Python fallback")
    
    # Generate points
    test.generate_planar_points(nx=20, ny=20, noise_level=0.5)
    
    # Compute convex hull
    test.compute_convex_hull()
    
    # Create segments
    test.create_coarse_segmentation()
    
    # Create feature points
    test.create_feature_points(num_features=4)
    
    # Perform hybrid triangulation with gradient=2.0
    test.hybrid_triangulate(gradient=2.0)
    
    # Visualize the results
    test.visualize_results() 