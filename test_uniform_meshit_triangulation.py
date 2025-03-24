"""
Test script for simple uniform MeshIt-style triangulation.

This script demonstrates a much simpler, uniform triangulation approach
that matches the original MeshIt premeshjob style without density variation.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Simple-Uniform-MeshIt")

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

def simple_uniform_triangulation():
    """Run simple uniform triangulation without feature points."""
    logger.info("Starting simple uniform triangulation (MeshIt premeshjob style)")
    
    # Create boundary with more points for smoother outline
    hull_points, segments = create_polygon_boundary(num_points=30, radius=10.0, noise=0.05)
    
    # Calculate base size based on domain size
    min_coords = np.min(hull_points, axis=0)
    max_coords = np.max(hull_points, axis=0)
    diagonal = np.sqrt(np.sum((max_coords - min_coords) ** 2))
    base_size = diagonal / 15.0  # Standard scaling factor
    logger.info(f"Using base size: {base_size}")
    
    # Create a DirectTriangleWrapper with simplified settings
    wrapper = DirectTriangleWrapper(
        gradient=1.0,  # Key: Use 1.0 for uniform meshes
        min_angle=25.0,  # Higher angle for better quality
        base_size=base_size
    )
    
    # Don't set any feature points - this is key for uniform mesh
    
    # Run triangulation with simplified options
    logger.info("Running simple uniform triangulation")
    
    # Create triangle options directly (no feature points, just quality and area constraints)
    # This creates a much more uniform mesh like in MeshIt premeshjob
    area_constraint = base_size * base_size * 0.5
    triangle_opts = f'pzYq25.0a{area_constraint}'
    wrapper.triangle_opts = triangle_opts
    
    result = wrapper.triangulate(
        points=hull_points,  # Just use the hull points
        segments=segments,
        create_feature_points=False,  # Key: Don't create feature points
        create_transition=False,     # Key: Don't create transition points
        uniform=True
    )
    
    # Visualize the result
    visualize_result(hull_points, result, base_size)
    
    logger.info("Simple uniform triangulation completed")
    return result

def visualize_result(hull_points, result, base_size):
    """Visualize the triangulation result."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw the triangulation
    vertices = result['vertices']
    triangles = result['triangles']
    
    ax.triplot(vertices[:, 0], vertices[:, 1], triangles, 'k-', alpha=0.7)
    
    # Draw the hull boundary
    ax.plot(np.append(hull_points[:, 0], hull_points[0, 0]),
           np.append(hull_points[:, 1], hull_points[0, 1]),
           'r-', linewidth=2, label='Boundary')
           
    ax.set_title(f'Simple Uniform Triangulation ({len(triangles)} triangles)\nMeshIt premeshjob style')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    
    # Calculate triangle quality metrics
    logger.info(f"Calculating quality metrics for {len(triangles)} triangles")
    
    edge_lengths = []
    areas = []
    
    for tri in triangles:
        v1, v2, v3 = vertices[tri]
        e1 = np.linalg.norm(v2 - v1)
        e2 = np.linalg.norm(v3 - v2)
        e3 = np.linalg.norm(v1 - v3)
        
        edge_lengths.extend([e1, e2, e3])
        
        # Triangle area using Heron's formula
        s = (e1 + e2 + e3) / 2
        area = np.sqrt(s * (s - e1) * (s - e2) * (s - e3))
        areas.append(area)
    
    # Calculate statistics
    mean_edge = np.mean(edge_lengths)
    std_edge = np.std(edge_lengths)
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    
    # Log statistics
    logger.info(f"Triangulation statistics:")
    logger.info(f"  Number of triangles: {len(triangles)}")
    logger.info(f"  Mean edge length: {mean_edge:.4f} (target: ~{base_size:.4f})")
    logger.info(f"  Edge length std dev: {std_edge:.4f}")
    logger.info(f"  Mean area: {mean_area:.4f}")
    logger.info(f"  Area std dev: {std_area:.4f}")
    logger.info(f"  Edge uniformity (std/mean): {std_edge/mean_edge:.4f}")
    
    plt.tight_layout()
    plt.savefig('simple_uniform_triangulation.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    simple_uniform_triangulation() 