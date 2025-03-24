"""
Test script for MeshIt-style triangulation.

This script demonstrates a triangulation approach that closely matches the 
original MeshIt implementation, with uniform triangle sizes across the mesh.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MeshIt-Style")

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

def create_polygon_boundary(num_points=20, radius=10.0, noise=0.0):
    """Create a polygon boundary with some randomness."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # Add some noise to the radius
    if noise > 0:
        radii = radius * (1 + noise * (np.random.random(num_points) - 0.5))
    else:
        radii = np.ones(num_points) * radius
    
    # Create points
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    points = np.column_stack((x, y))
    
    # Create segments
    segments = np.column_stack((
        np.arange(num_points),
        np.roll(np.arange(num_points), -1)
    ))
    
    return points, segments

def create_random_points(num_points=100, bounds=(-10, 10)):
    """Create random points within bounds."""
    min_val, max_val = bounds
    x = np.random.uniform(min_val, max_val, num_points)
    y = np.random.uniform(min_val, max_val, num_points)
    return np.column_stack((x, y))

def create_feature_points(num_points=3, radius=5.0):
    """Create feature points for controlling mesh density."""
    # Create feature points at various positions
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # Randomize the radius a bit
    radii = radius * (0.5 + 0.5 * np.random.random(num_points))
    
    # Create points
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    feature_points = np.column_stack((x, y))
    
    # Add a center point
    feature_points = np.vstack((feature_points, np.array([0, 0])))
    
    # Use consistent sizing for all feature points in MeshIt-style
    feature_sizes = np.ones(len(feature_points))
    
    return feature_points, feature_sizes

def meshit_triangulation():
    """Run MeshIt-style triangulation."""
    logger.info("Starting MeshIt-style triangulation")
    
    # Create boundary
    hull_points, segments = create_polygon_boundary(num_points=20, radius=10.0, noise=0.1)
    
    # Create interior points
    interior_points = create_random_points(num_points=200, bounds=(-9, 9))
    
    # Create feature points
    feature_points, feature_sizes = create_feature_points(num_points=4, radius=5.0)
    
    # Set base size based on domain
    domain_size = np.max(hull_points) - np.min(hull_points)
    base_size = domain_size / 15.0
    
    # Scale feature sizes by base size
    feature_sizes *= base_size
    
    # Create a DirectTriangleWrapper instance configured like MeshIt
    wrapper = DirectTriangleWrapper(
        gradient=1.0,  # Critical: Use 1.0 for uniform meshes like in MeshIt
        min_angle=25.0,  # Higher minimum angle for better quality
        base_size=base_size  # Base size scaled to domain
    )
    
    # Set feature points with uniform sizing
    wrapper.set_feature_points(feature_points, feature_sizes)
    
    # Combine all points
    all_points = np.vstack((hull_points, interior_points))
    
    # Run triangulation with uniform mesh option (MeshIt-style)
    logger.info("Running MeshIt-style uniform triangulation")
    result = wrapper.triangulate(
        points=all_points,
        segments=segments,
        create_feature_points=True,
        create_transition=True,
        uniform=True  # Critical: enable uniform mesh generation
    )
    
    # For comparison, also run with higher gradient
    logger.info("Running gradient-based triangulation for comparison")
    gradient_wrapper = DirectTriangleWrapper(
        gradient=2.0,  # Higher gradient causes density variation
        min_angle=20.0,
        base_size=base_size
    )
    gradient_wrapper.set_feature_points(feature_points, feature_sizes)
    
    gradient_result = gradient_wrapper.triangulate(
        points=all_points,
        segments=segments,
        create_feature_points=True,
        create_transition=True,
        uniform=False  # Disable uniform mode for gradient effect
    )
    
    # Visualize results
    visualize_results(
        hull_points=hull_points, 
        interior_points=interior_points,
        feature_points=feature_points, 
        feature_sizes=feature_sizes,
        uniform_result=result,
        gradient_result=gradient_result,
        base_size=base_size
    )
    
    logger.info("MeshIt-style triangulation completed")
    return result

def visualize_results(hull_points, interior_points, feature_points, feature_sizes,
                     uniform_result, gradient_result, base_size):
    """Visualize triangulation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Input points and hull
    ax = axes[0, 0]
    ax.scatter(interior_points[:, 0], interior_points[:, 1], c='blue', alpha=0.5, label='Interior Points')
    ax.plot(np.append(hull_points[:, 0], hull_points[0, 0]),
           np.append(hull_points[:, 1], hull_points[0, 1]),
           'r-', linewidth=2, label='Convex Hull')
    
    # Draw feature points
    ax.scatter(feature_points[:, 0], feature_points[:, 1], c='orange', s=100, label='Feature Points')
    
    ax.set_title('Original Points and Convex Hull')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 2. Boundary and feature points
    ax = axes[0, 1]
    ax.plot(np.append(hull_points[:, 0], hull_points[0, 0]),
           np.append(hull_points[:, 1], hull_points[0, 1]),
           'r-', linewidth=2, label='Boundary')
    
    # Draw feature points with influence zones
    ax.scatter(feature_points[:, 0], feature_points[:, 1], c='orange', s=100, label='Feature Points')
    for point, size in zip(feature_points, feature_sizes):
        circle = plt.Circle((point[0], point[1]), size, fill=False, color='orange', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
    
    ax.set_title('Boundary and Feature Points')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 3. Triangulation result
    ax = axes[1, 0]
    vertices = uniform_result['vertices']
    triangles = uniform_result['triangles']
    
    ax.triplot(vertices[:, 0], vertices[:, 1], triangles, 'k-', alpha=0.7)
    ax.set_title(f'Triangulation Result ({len(triangles)} triangles)')
    ax.set_aspect('equal')
    ax.grid(False)
    
    # 4. Mesh quality visualization
    ax = axes[1, 1]
    
    # Calculate triangle qualities and colors
    qualities = np.zeros(len(triangles))
    for i, tri in enumerate(triangles):
        v1, v2, v3 = vertices[tri]
        e1 = np.linalg.norm(v2 - v1)
        e2 = np.linalg.norm(v3 - v2)
        e3 = np.linalg.norm(v1 - v3)
        
        # Triangle area using Heron's formula
        s = (e1 + e2 + e3) / 2
        area = np.sqrt(s * (s - e1) * (s - e2) * (s - e3))
        
        # One quality metric: ratio of mean edge to ideal edge length
        ideal_edge = np.sqrt(base_size * base_size * 2)
        mean_edge = (e1 + e2 + e3) / 3
        qualities[i] = mean_edge / ideal_edge
    
    # Create a colorful mesh visualization
    triang = Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    tpc = ax.tripcolor(triang, qualities, cmap='viridis')
    fig.colorbar(tpc, ax=ax, label='Quality (mean edge / ideal edge)')
    
    ax.set_title('Mesh Quality Visualization')
    ax.set_aspect('equal')
    ax.grid(False)
    
    # Calculate quality statistics
    mean_quality = np.mean(qualities)
    std_quality = np.std(qualities)
    
    logger.info(f"Mesh statistics:")
    logger.info(f"  Number of triangles: {len(triangles)}")
    logger.info(f"  Mean quality: {mean_quality:.4f}")
    logger.info(f"  Quality standard deviation: {std_quality:.4f}")
    
    plt.tight_layout()
    plt.savefig('meshit_triangulation.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    meshit_triangulation() 