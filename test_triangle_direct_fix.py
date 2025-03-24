"""
A direct test for Triangle triangulation with our C++ callback.

This script tests the Triangle library with our C++ callback directly,
using a simpler approach to understand the triangulation issues.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
from scipy.spatial import ConvexHull
import logging
from meshit.triangle_direct import DirectTriangleWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TriangleTest")

def create_test_points():
    """Create a test point set"""
    np.random.seed(42)  # For reproducibility
    
    # Generate boundary points (a flower-like shape)
    theta = np.linspace(0, 2*np.pi, 30)
    r = 10 + 2 * np.sin(5*theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    boundary_points = np.column_stack((x, y))
    
    # Generate random interior points
    num_interior = 50
    interior_points = []
    
    for _ in range(num_interior):
        # Random point within the approximate boundary
        angle = np.random.random() * 2 * np.pi
        radius = np.random.random() * 8  # Smaller than boundary radius
        interior_points.append([
            radius * np.cos(angle), 
            radius * np.sin(angle)
        ])
    
    interior_points = np.array(interior_points)
    
    # Combine boundary and interior points
    all_points = np.vstack((boundary_points, interior_points))
    
    return all_points

def compute_convex_hull(points):
    """Compute the convex hull of a point set"""
    if len(points) < 3:
        return points
    
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    return hull_points, hull.vertices

def test_triangulation_with_direct_callbacks():
    """Test triangulation with direct callbacks"""
    logger.info("Starting triangulation test...")
    
    # Create test points
    points = create_test_points()
    logger.info(f"Created {len(points)} test points")
    
    # Plot original points
    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1], c='b')
    plt.axis('equal')
    plt.title('Original Points')
    plt.savefig('direct_test_points.png')
    logger.info("Saved original points plot")
    
    # Compute convex hull
    hull_points, hull_indices = compute_convex_hull(points)
    
    # Plot convex hull
    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1], c='b')
    
    # Plot hull as a loop
    hull_plus_first = np.vstack((hull_points, hull_points[0:1]))
    plt.plot(hull_plus_first[:, 0], hull_plus_first[:, 1], 'r-', linewidth=2)
    
    plt.axis('equal')
    plt.title('Points with Convex Hull')
    plt.savefig('direct_test_hull.png')
    logger.info(f"Saved convex hull plot with {len(hull_points)} hull points")
    
    # =============== METHOD 1: STANDARD TRIANGLE LIBRARY ===============
    # Create segment pairs using hull vertices
    segments = []
    for i in range(len(hull_indices)):
        segments.append([hull_indices[i], hull_indices[(i+1) % len(hull_indices)]])
    segments = np.array(segments)
    
    # Prepare input for Triangle
    tri_input = {
        'vertices': points,
        'segments': segments
    }
    
    # First try standard Triangle with quality constraints
    for gradient in [0.5, 1.0, 2.0]:
        # Adjust quality options based on gradient
        min_angle = 20.0
        if gradient > 1.0:
            min_angle = max(20.0 - (gradient - 1.0) * 5.0, 10.0)
            
        # Triangle options: p=PSLG, q=quality, a=area constraint
        area_constraint = 1.0  # Adjust based on input size
        tri_opts = f'pq{min_angle}a{area_constraint}'
        
        logger.info(f"Running standard Triangle with options: {tri_opts}")
        
        try:
            # Run standard triangulation
            result = tr.triangulate(tri_input, tri_opts)
            
            if 'triangles' in result:
                num_triangles = len(result['triangles'])
                logger.info(f"Standard Triangle created {num_triangles} triangles")
                
                # Plot triangulation
                plt.figure(figsize=(12, 10))
                plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
                            result['triangles'], 'b-', lw=0.5)
                plt.scatter(points[:, 0], points[:, 1], c='r', s=10)
                plt.axis('equal')
                plt.title(f'Standard Triangle (g={gradient}, {num_triangles} triangles)')
                plt.savefig(f'direct_standard_tri_g{gradient}.png')
                
                # Also save triangle data for comparison
                np.save(f'standard_vertices_g{gradient}.npy', result['vertices'])
                np.save(f'standard_triangles_g{gradient}.npy', result['triangles'])
            else:
                logger.warning("Standard triangulation failed to produce triangles")
        except Exception as e:
            logger.error(f"Error in standard triangulation: {e}")
    
    # =============== METHOD 2: DIRECT TRIANGLE WRAPPER ===============
    for gradient in [0.5, 1.0, 2.0]:
        try:
            logger.info(f"Testing DirectTriangleWrapper with gradient {gradient}")
            
            # Initialize our wrapper
            wrapper = DirectTriangleWrapper(gradient=gradient)
            
            # Set feature points - use hull points with small size and centroid with larger size
            feature_points = []
            feature_sizes = []
            
            # Add hull points as features with small size
            for pt in hull_points:
                feature_points.append(pt)
                feature_sizes.append(0.5)  # Small size at boundary
                
            # Add centroid with larger size
            centroid = np.mean(points, axis=0)
            feature_points.append(centroid)
            feature_sizes.append(2.0)  # Larger size at center
            
            feature_points = np.array(feature_points)
            feature_sizes = np.array(feature_sizes)
            
            # Set feature points for refinement
            wrapper.set_feature_points(feature_points, feature_sizes)
            logger.info(f"Using {len(feature_points)} feature points")
            
            # We can't pass segment indices directly to our wrapper, as it doesn't handle them correctly yet
            # Instead, let's use the hull points array and segment indices relative to that
            
            # Create new array with only hull points first, then interior points
            ordered_points = np.vstack((hull_points, points))
            
            # Create segments relative to hull points
            hull_segments = np.array([[i, (i+1) % len(hull_points)] for i in range(len(hull_points))])
            
            logger.info(f"Running direct triangulation with {len(ordered_points)} points, {len(hull_segments)} segments")
            
            # Call our wrapper
            result = wrapper.triangulate(ordered_points, hull_segments)
            
            # Extract results
            vertices = result['vertices']
            triangles = result['triangles']
            
            logger.info(f"DirectTriangleWrapper created {len(triangles)} triangles, {len(vertices)} vertices")
            
            if len(triangles) > 0:
                # Plot direct triangulation
                plt.figure(figsize=(12, 10))
                plt.triplot(vertices[:, 0], vertices[:, 1], triangles, 'b-', lw=0.5)
                plt.scatter(points[:, 0], points[:, 1], c='r', s=10)
                
                # Draw feature points specially
                plt.scatter(feature_points[:, 0], feature_points[:, 1], c='g', s=30, marker='*')
                
                plt.axis('equal')
                plt.title(f'Direct Triangle Wrapper (g={gradient}, {len(triangles)} triangles)')
                plt.savefig(f'direct_wrapper_tri_g{gradient}.png')
                
                # Save triangle data
                np.save(f'direct_vertices_g{gradient}.npy', vertices)
                np.save(f'direct_triangles_g{gradient}.npy', triangles)
            else:
                logger.warning("Direct triangulation failed to produce triangles")
            
        except Exception as e:
            logger.error(f"Error in direct triangulation: {e}")
    
    # =============== METHOD 3: HYBRID APPROACH ===============
    # Use standard Triangle for initial triangulation, then apply our refinement
    for gradient in [0.5, 1.0, 2.0]:
        try:
            logger.info(f"Testing hybrid approach with gradient {gradient}")
            
            # First get standard triangulation (reuse from above)
            min_angle = 20.0
            if gradient > 1.0:
                min_angle = max(20.0 - (gradient - 1.0) * 5.0, 10.0)
                
            tri_opts = f'pq{min_angle}a'
            standard_result = tr.triangulate(tri_input, tri_opts)
            
            if 'triangles' in standard_result:
                std_vertices = standard_result['vertices']
                std_triangles = standard_result['triangles']
                
                # Now refine using our wrapper
                wrapper = DirectTriangleWrapper(gradient=gradient)
                
                # Set feature points - use hull points and some interior points
                feature_points = []
                feature_sizes = []
                
                # Add hull points as features
                for pt in hull_points:
                    feature_points.append(pt)
                    feature_sizes.append(0.5)  # Small size at boundary
                    
                # Add a few interior points determined by existing triangulation
                # Use centers of existing triangles that are close to centroid
                centroid = np.mean(points, axis=0)
                
                for tri in std_triangles:
                    v1, v2, v3 = std_vertices[tri]
                    tri_center = (v1 + v2 + v3) / 3.0
                    
                    # Distance to centroid
                    dist = np.linalg.norm(tri_center - centroid)
                    
                    # If near center, add as feature
                    if dist < 5.0:  # Adjust threshold as needed
                        feature_points.append(tri_center)
                        feature_sizes.append(1.0 + dist/5.0)  # Size increases with distance
                
                feature_points = np.array(feature_points)
                feature_sizes = np.array(feature_sizes)
                
                # Set feature points for refinement
                wrapper.set_feature_points(feature_points, feature_sizes)
                
                # Define quality based on gradient
                quality_opts = f'q{min_angle}a'
                
                # Call refinement method to improve existing triangulation
                # Direct refinement isn't implemented yet in our wrapper, so we'll simulate it
                
                # For demonstration, let's use the standard Triangle result with plotting
                logger.info(f"Hybrid approach would refine {len(std_triangles)} triangles with {len(feature_points)} features")
                
                # Plot hybrid result (currently just standard result with feature points)
                plt.figure(figsize=(12, 10))
                plt.triplot(std_vertices[:, 0], std_vertices[:, 1], std_triangles, 'b-', lw=0.5)
                plt.scatter(points[:, 0], points[:, 1], c='r', s=10)
                
                # Draw feature points specially
                plt.scatter(feature_points[:, 0], feature_points[:, 1], c='g', s=30, marker='*')
                
                plt.axis('equal')
                plt.title(f'Hybrid Approach (g={gradient}, {len(std_triangles)} triangles)')
                plt.savefig(f'hybrid_tri_g{gradient}.png')
                
            else:
                logger.warning("Hybrid approach failed - standard triangulation failed")
            
        except Exception as e:
            logger.error(f"Error in hybrid approach: {e}")
    
    plt.show()
    
if __name__ == "__main__":
    test_triangulation_with_direct_callbacks() 