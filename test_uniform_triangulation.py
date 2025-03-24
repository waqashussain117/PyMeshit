"""
Test script for uniform triangulation of a planar surface.

This script demonstrates the uniform triangulation approach using the modified
DirectTriangleWrapper implementation to produce more evenly distributed triangles,
similar to MeshIt's results.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UniformMeshTest")

# Ensure current directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try to import the direct triangle wrapper
try:
    from meshit.triangle_direct import DirectTriangleWrapper
    HAVE_DIRECT_WRAPPER = True
    logger.info("Successfully imported DirectTriangleWrapper")
except ImportError as e:
    logger.error(f"Failed to import DirectTriangleWrapper: {e}")
    HAVE_DIRECT_WRAPPER = False
    sys.exit(1)

def generate_square_points(size=20.0, num_points=400, noise_level=0.2):
    """Generate points in a square with some noise."""
    # Create a grid of points
    points_per_side = int(np.sqrt(num_points))
    x = np.linspace(-size/2, size/2, points_per_side)
    y = np.linspace(-size/2, size/2, points_per_side)
    xx, yy = np.meshgrid(x, y)
    
    # Add some noise to make it more interesting
    if noise_level > 0:
        xx += np.random.normal(0, noise_level * size / points_per_side, xx.shape)
        yy += np.random.normal(0, noise_level * size / points_per_side, yy.shape)
    
    # Reshape to get points
    points = np.column_stack((xx.flatten(), yy.flatten()))
    
    return points

def create_square_hull(size=20.0, num_sides=4):
    """Create a square hull for the domain."""
    # Generate square vertices
    theta = np.linspace(0, 2*np.pi, num_sides, endpoint=False)
    x = size/2 * np.cos(theta + np.pi/4)
    y = size/2 * np.sin(theta + np.pi/4)
    hull_points = np.column_stack((x, y))
    
    # Create segments - consecutive vertices form segments
    segments = np.column_stack((
        np.arange(num_sides),
        np.roll(np.arange(num_sides), -1)
    ))
    
    return hull_points, segments

def create_feature_points(hull_points, num_features=3):
    """Create feature points for controlling mesh density."""
    # Compute centroid of hull
    centroid = np.mean(hull_points, axis=0)
    
    # Create feature points in interesting locations
    feature_points = []
    feature_sizes = []
    
    # Add centroid as a feature point 
    feature_points.append(centroid)
    feature_sizes.append(1.0)  # Medium-sized triangles at center
    
    # Define a boundary size from hull dimensions
    min_x, min_y = np.min(hull_points, axis=0)
    max_x, max_y = np.max(hull_points, axis=0)
    domain_size = max(max_x - min_x, max_y - min_y)
    base_size = domain_size / 15.0
    
    # Add additional feature points
    for i in range(num_features - 1):
        # Position - use golden ratio to distribute points
        angle = 2.0 * np.pi * i * 0.618033988749895
        radius = domain_size * 0.3
        
        point = centroid + radius * np.array([np.cos(angle), np.sin(angle)])
        feature_points.append(point)
        
        # Use consistent sizes for uniform triangulation
        feature_sizes.append(base_size)
    
    return np.array(feature_points), np.array(feature_sizes)

def run_uniform_triangulation():
    """Run the uniform triangulation test."""
    logger.info("Starting uniform triangulation test")
    
    # Generate domain and points
    size = 20.0
    hull_points, segments = create_square_hull(size)
    interior_points = generate_square_points(size, num_points=400, noise_level=0.2)
    
    # Create feature points
    feature_points, feature_sizes = create_feature_points(hull_points, num_features=4)
    
    # Create a DirectTriangleWrapper instance
    wrapper = DirectTriangleWrapper(
        gradient=1.0,  # Use minimal gradient for uniform meshes
        min_angle=25.0,  # Higher minimum angle for better quality
        base_size=size/15.0  # Base size scaled to domain
    )
    
    # Set feature points
    wrapper.set_feature_points(feature_points, feature_sizes)
    
    # Combine hull points and interior points
    all_points = np.vstack((hull_points, interior_points))
    
    # Run triangulation with uniform mesh option enabled
    logger.info("Running triangulation with uniform mesh option")
    result = wrapper.triangulate(
        points=all_points,
        segments=segments,
        create_feature_points=True,
        create_transition=True,
        uniform=True  # Enable uniform mesh generation
    )
    
    # For comparison, also run with non-uniform option
    logger.info("Running triangulation with gradient-based refinement for comparison")
    # Create a DirectTriangleWrapper with higher gradient
    gradient_wrapper = DirectTriangleWrapper(
        gradient=2.0,  # Higher gradient for more variation
        min_angle=20.0,
        base_size=size/15.0
    )
    gradient_wrapper.set_feature_points(feature_points, feature_sizes)
    
    # Run with gradient-based refinement
    gradient_result = gradient_wrapper.triangulate(
        points=all_points,
        segments=segments,
        create_feature_points=True,
        create_transition=True,
        uniform=False  # Disable uniform mesh generation
    )
    
    # Visualize results
    visualize_results(
        hull_points=hull_points,
        interior_points=interior_points,
        feature_points=feature_points,
        feature_sizes=feature_sizes,
        uniform_result=result,
        gradient_result=gradient_result
    )
    
    logger.info("Uniform triangulation test completed")
    
def visualize_results(hull_points, interior_points, feature_points, feature_sizes,
                     uniform_result, gradient_result):
    """Visualize the triangulation results for comparison."""
    # Create figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Input points and features
    ax = axes[0, 0]
    ax.scatter(interior_points[:, 0], interior_points[:, 1], 
              c='blue', alpha=0.5, label='Interior Points')
    ax.plot(np.append(hull_points[:, 0], hull_points[0, 0]),
           np.append(hull_points[:, 1], hull_points[0, 1]),
           'r-', linewidth=2, label='Boundary')
    
    # Draw feature points with their influence zones
    ax.scatter(feature_points[:, 0], feature_points[:, 1],
              c='orange', s=100, label='Feature Points')
    for point, size in zip(feature_points, feature_sizes):
        circle = plt.Circle((point[0], point[1]), size,
                           fill=False, color='orange', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
    
    ax.set_title('Input Points and Features')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 2. Uniform triangulation result
    ax = axes[0, 1]
    vertices = uniform_result['vertices']
    triangles = uniform_result['triangles']
    
    ax.triplot(vertices[:, 0], vertices[:, 1], triangles, 'b-', alpha=0.5)
    ax.set_title(f'Uniform Triangulation ({len(triangles)} triangles)')
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 3. Gradient-based triangulation result
    ax = axes[1, 0]
    vertices = gradient_result['vertices']
    triangles = gradient_result['triangles']
    
    ax.triplot(vertices[:, 0], vertices[:, 1], triangles, 'g-', alpha=0.5)
    ax.set_title(f'Gradient-based Triangulation ({len(triangles)} triangles)')
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 4. Comparison of triangle quality
    ax = axes[1, 1]
    
    # Calculate quality metrics for uniform triangulation
    uniform_vertices = uniform_result['vertices']
    uniform_triangles = uniform_result['triangles']
    uniform_qualities = []
    
    for tri in uniform_triangles:
        v1, v2, v3 = uniform_vertices[tri]
        e1 = np.linalg.norm(v2 - v1)
        e2 = np.linalg.norm(v3 - v2)
        e3 = np.linalg.norm(v1 - v3)
        s = (e1 + e2 + e3) / 2
        area = np.sqrt(s * (s - e1) * (s - e2) * (s - e3))
        if area > 1e-10:
            quality = (e1 * e2 * e3) / (4 * area)
            uniform_qualities.append(quality)
    
    # Calculate quality metrics for gradient-based triangulation
    gradient_vertices = gradient_result['vertices']
    gradient_triangles = gradient_result['triangles']
    gradient_qualities = []
    
    for tri in gradient_triangles:
        v1, v2, v3 = gradient_vertices[tri]
        e1 = np.linalg.norm(v2 - v1)
        e2 = np.linalg.norm(v3 - v2)
        e3 = np.linalg.norm(v1 - v3)
        s = (e1 + e2 + e3) / 2
        area = np.sqrt(s * (s - e1) * (s - e2) * (s - e3))
        if area > 1e-10:
            quality = (e1 * e2 * e3) / (4 * area)
            gradient_qualities.append(quality)
    
    # Plot histogram of quality metrics
    ax.hist(uniform_qualities, bins=50, alpha=0.5, color='blue', label='Uniform')
    ax.hist(gradient_qualities, bins=50, alpha=0.5, color='green', label='Gradient-based')
    ax.set_xlabel('Triangle Quality (lower is better)')
    ax.set_ylabel('Count')
    ax.set_title('Triangle Quality Comparison')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Print statistics
    uniform_mean = np.mean(uniform_qualities)
    uniform_std = np.std(uniform_qualities)
    gradient_mean = np.mean(gradient_qualities)
    gradient_std = np.std(gradient_qualities)
    
    logger.info(f"Uniform triangulation statistics:")
    logger.info(f"  Triangles: {len(uniform_triangles)}")
    logger.info(f"  Mean quality: {uniform_mean:.4f}")
    logger.info(f"  Quality std dev: {uniform_std:.4f}")
    
    logger.info(f"Gradient-based triangulation statistics:")
    logger.info(f"  Triangles: {len(gradient_triangles)}")
    logger.info(f"  Mean quality: {gradient_mean:.4f}")
    logger.info(f"  Quality std dev: {gradient_std:.4f}")
    
    fig.tight_layout()
    plt.savefig('triangulation_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_uniform_triangulation() 