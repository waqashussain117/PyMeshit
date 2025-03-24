"""
Hybrid triangulation test for MeshIt.

This script demonstrates a hybrid approach that combines the standard Triangle 
library with our C++ triunsuitable callback for gradient-based refinement.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
from scipy.spatial import ConvexHull
import logging
from meshit.triangle_direct import DirectTriangleWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HybridTriangulationTest")

def create_test_points():
    """Create a test point set with a flower-like boundary"""
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
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    return hull_points, hull.vertices

def create_smooth_transition_points(hull_points, centroid, num_points=40):
    """
    Create smooth transition points between hull and interior
    to guide mesh refinement for better uniformity.
    """
    transition_points = []
    
    # For each hull point
    for i in range(len(hull_points)):
        # Current hull point and next hull point (for creating transitions along edges)
        p1 = hull_points[i]
        p2 = hull_points[(i+1) % len(hull_points)]
        
        # Create points along edge (from p1 to p2)
        num_edge_points = 3  # Number of intermediate points on each edge
        for j in range(1, num_edge_points):
            t = j / (num_edge_points + 1)
            edge_pt = p1 + t * (p2 - p1)
            transition_points.append(edge_pt)
        
        # Create interior points along ray from hull to centroid
        ray = centroid - p1
        ray_length = np.linalg.norm(ray)
        if ray_length < 1e-10:
            continue
            
        ray = ray / ray_length  # Normalize
        
        # Create points at different depths along ray
        depths = [0.2, 0.4, 0.6, 0.8]  # Relative distance from hull to centroid
        for depth in depths:
            interior_pt = p1 + depth * ray_length * ray
            transition_points.append(interior_pt)
    
    # Add some distributed interior points
    num_interior = num_points - len(transition_points)
    if num_interior > 0:
        # Use circular pattern
        angles = np.linspace(0, 2*np.pi, num_interior, endpoint=False)
        # Multiple radii for more uniform coverage
        radii = [0.3, 0.5, 0.7]
        
        # Find maximum distance from centroid to hull
        max_dist = 0
        for p in hull_points:
            dist = np.linalg.norm(p - centroid)
            max_dist = max(max_dist, dist)
        
        # Generate points in concentric circles
        for radius_factor in radii:
            for angle in angles:
                x = centroid[0] + radius_factor * max_dist * 0.5 * np.cos(angle)
                y = centroid[1] + radius_factor * max_dist * 0.5 * np.sin(angle)
                transition_points.append(np.array([x, y]))
    
    return np.array(transition_points)

def triangulate_with_hybrid_approach(points, gradient=2.0):
    """
    Hybrid triangulation approach:
    1. Use standard Triangle to get initial triangulation with convex hull boundary
    2. Use our C++ callback for gradient-based refinement
    """
    logger.info(f"Starting hybrid triangulation with gradient={gradient}...")
    
    # Plot original points
    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1], c='b', marker='o')
    plt.axis('equal')
    plt.grid(True)
    plt.title('Input Points')
    plt.savefig(f'hybrid_input_g{gradient}.png')
    
    # Step 1: Compute convex hull
    hull_points, hull_indices = compute_convex_hull(points)
    
    # Plot convex hull
    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1], c='b', marker='o')
    
    # Plot hull as a loop
    hull_plus_first = np.vstack((hull_points, hull_points[0:1]))
    plt.plot(hull_plus_first[:, 0], hull_plus_first[:, 1], 'r-', linewidth=2)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title('Points with Convex Hull')
    plt.savefig(f'hybrid_hull_g{gradient}.png')
    
    # Step 2: Create segment pairs from hull vertices
    segments = []
    for i in range(len(hull_indices)):
        segments.append([hull_indices[i], hull_indices[(i+1) % len(hull_indices)]])
    segments = np.array(segments)
    
    # Step 3: Setup input for Triangle
    tri_input = {
        'vertices': points,
        'segments': segments
    }
    
    # Step 4: Calculate base mesh size from bounding box diagonal
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    diagonal = np.sqrt(np.sum((max_coords - min_coords) ** 2))
    base_size = diagonal / 15.0  # MeshIt's scaling
    
    # Step 5: Adjust quality options based on gradient
    min_angle = 20.0
    if gradient > 1.0:
        min_angle = max(20.0 - (gradient - 1.0) * 5.0, 10.0)
    
    # Setup area constraint - this is important for proper triangle sizing
    area_constraint = base_size * base_size * 0.5
    
    # Initial triangulation with standard Triangle
    triangle_opts = f'pq{min_angle}a{area_constraint}'
    logger.info(f"Performing initial triangulation with options: {triangle_opts}")
    
    try:
        # Run standard triangulation
        result = tr.triangulate(tri_input, triangle_opts)
        
        if 'triangles' in result:
            num_triangles = len(result['triangles'])
            logger.info(f"Initial triangulation created {num_triangles} triangles")
            
            # Plot initial triangulation
            plt.figure(figsize=(12, 10))
            plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
                        result['triangles'], 'b-', lw=0.5)
            plt.scatter(points[:, 0], points[:, 1], c='r', s=10)
            plt.axis('equal')
            plt.grid(True)
            plt.title(f'Initial Triangulation (g={gradient}, {num_triangles} triangles)')
            plt.savefig(f'hybrid_initial_g{gradient}.png')
        else:
            logger.error("Initial triangulation failed to produce triangles")
            return None
            
    except Exception as e:
        logger.error(f"Error in initial triangulation: {e}")
        return None
    
    # Step 6: Setup the DirectTriangleWrapper with feature points
    logger.info("Setting up DirectTriangleWrapper with feature points...")
    wrapper = DirectTriangleWrapper(gradient=gradient, base_size=base_size)
    
    # The DirectTriangleWrapper will create its own transition points
    # automatically, so we only need to add strategic feature points
    
    feature_points = []
    feature_sizes = []
    
    # Calculate centroid for reference
    centroid = np.mean(points, axis=0)
    
    # Add hull points with smaller size
    for pt in hull_points:
        feature_points.append(pt)
        feature_sizes.append(base_size * 0.2)  # Small size at boundary
    
    # Add centroid with larger size
    feature_points.append(centroid)
    feature_sizes.append(base_size * 0.7)  # Larger size at center
    
    # Create and add transition points
    transition_points = create_smooth_transition_points(hull_points, centroid)
    
    # Plot transition points for visualization
    plt.figure(figsize=(12, 10))
    plt.scatter(points[:, 0], points[:, 1], c='b', s=10, alpha=0.5)
    plt.scatter(hull_points[:, 0], hull_points[:, 1], c='r', s=30, marker='o')
    plt.scatter(transition_points[:, 0], transition_points[:, 1], c='g', s=20, marker='x')
    plt.scatter([centroid[0]], [centroid[1]], c='m', s=50, marker='*')
    
    # Plot hull as a loop
    hull_plus_first = np.vstack((hull_points, hull_points[0:1]))
    plt.plot(hull_plus_first[:, 0], hull_plus_first[:, 1], 'r-', linewidth=1)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title('Feature Points and Transition Points')
    plt.savefig(f'hybrid_features_g{gradient}.png')
    
    # Add transition points with calculated sizes
    for pt in transition_points:
        # Calculate distance to centroid and hull
        dist_to_centroid = np.linalg.norm(pt - centroid)
        
        # Find closest hull point
        min_dist_to_hull = float('inf')
        for hull_pt in hull_points:
            dist = np.linalg.norm(pt - hull_pt)
            min_dist_to_hull = min(min_dist_to_hull, dist)
        
        # Find max distance from centroid to hull for normalization
        max_dist = 0
        for hull_pt in hull_points:
            dist = np.linalg.norm(hull_pt - centroid)
            max_dist = max(max_dist, dist)
            
        if max_dist < 1e-10:
            continue
            
        # Normalize distance to centroid
        relative_dist = dist_to_centroid / max_dist
        
        # Size increases as we get closer to centroid
        size_factor = 0.2 + 0.6 * (1.0 - relative_dist)
        size = base_size * size_factor
        
        feature_points.append(pt)
        feature_sizes.append(size)
        
    # Convert to numpy arrays
    feature_points = np.array(feature_points)
    feature_sizes = np.array(feature_sizes)
    
    # Setup wrapper with feature points 
    wrapper.set_feature_points(feature_points, feature_sizes)
    
    # Step 7: Apply gradient-based refinement with our C++ callback
    logger.info("Applying gradient-based refinement...")
    
    # Use the DirectTriangleWrapper for complete refinement
    refined_result = wrapper.triangulate(points, segments)
    
    if 'triangles' in refined_result and len(refined_result['triangles']) > 0:
        num_triangles = len(refined_result['triangles'])
        logger.info(f"Refined triangulation created {num_triangles} triangles")
        
        # Plot refined triangulation
        plt.figure(figsize=(12, 10))
        plt.triplot(refined_result['vertices'][:, 0], refined_result['vertices'][:, 1], 
                    refined_result['triangles'], 'b-', lw=0.5)
        
        # Plot original points
        plt.scatter(points[:, 0], points[:, 1], c='r', s=10, alpha=0.5)
        
        # Plot feature points
        plt.scatter(feature_points[:, 0], feature_points[:, 1], 
                    c='g', s=10, marker='*', alpha=0.5, label='Feature Points')
        
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title(f'Refined Triangulation (g={gradient}, {num_triangles} triangles)')
        plt.savefig(f'hybrid_refined_g{gradient}.png')
        
        return refined_result
    else:
        logger.warning("Refined triangulation failed - returning initial triangulation")
        return result
    
def test_triangulation_with_various_gradients():
    """Test triangulation with various gradient values"""
    # Create test points
    points = create_test_points()
    logger.info(f"Created {len(points)} test points")
    
    # Test with various gradient values
    for gradient in [0.5, 1.0, 2.0]:
        result = triangulate_with_hybrid_approach(points, gradient)
        
        if result is not None:
            logger.info(f"Gradient {gradient}: Created {len(result['triangles'])} triangles with {len(result['vertices'])} vertices")
        else:
            logger.error(f"Gradient {gradient}: Triangulation failed")
    
if __name__ == "__main__":
    test_triangulation_with_various_gradients()
    plt.show() 