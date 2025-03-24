"""
Simple example script for uniform MeshIt-style triangulation.

This script demonstrates the simplified uniform triangulation approach 
that matches the original MeshIt premeshjob style.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Simple-Example")

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the DirectTriangleWrapper
from meshit.triangle_direct import DirectTriangleWrapper

def create_input_data(num_boundary_points=30, radius=10.0):
    """Create a simple circular boundary with segments."""
    # Create boundary points
    angles = np.linspace(0, 2 * np.pi, num_boundary_points, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    points = np.column_stack((x, y))
    
    # Create segments (connect consecutive points)
    segments = np.column_stack((
        np.arange(num_boundary_points),
        np.roll(np.arange(num_boundary_points), -1)
    ))
    
    return points, segments

def run_simple_triangulation():
    """Run simple uniform triangulation using the DirectTriangleWrapper."""
    logger.info("Starting simple uniform triangulation example")
    
    # Create input data (circular boundary)
    boundary_points, segments = create_input_data(num_boundary_points=30, radius=10.0)
    
    # Calculate base size from domain
    min_coords = np.min(boundary_points, axis=0)
    max_coords = np.max(boundary_points, axis=0)
    diagonal = np.sqrt(np.sum((max_coords - min_coords) ** 2))
    base_size = diagonal / 15.0
    
    logger.info(f"Domain size: {diagonal:.2f}, base size: {base_size:.2f}")
    
    # Create the DirectTriangleWrapper with uniform settings
    wrapper = DirectTriangleWrapper(
        gradient=1.0,        # Use 1.0 for uniform triangulation
        min_angle=25.0,      # Higher angle for better quality
        base_size=base_size  # Base size scaled to domain
    )
    
    # Perform triangulation without feature points
    # This matches MeshIt's premeshjob triangulation style
    result = wrapper.triangulate(
        points=boundary_points,  # Just use boundary points
        segments=segments,       # Boundary segments
        create_feature_points=False,  # No feature points needed
        create_transition=False,      # No transition points
        uniform=True                 # Use uniform mode
    )
    
    # Visualize the result
    visualize_result(boundary_points, result, base_size)
    
    logger.info("Simple uniform triangulation completed")
    return result

def visualize_result(boundary_points, result, base_size):
    """Visualize the triangulation result."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get triangulation data
    vertices = result['vertices']
    triangles = result['triangles']
    
    # Plot triangulation
    ax.triplot(vertices[:, 0], vertices[:, 1], triangles, 'k-', alpha=0.5)
    
    # Plot boundary
    ax.plot(np.append(boundary_points[:, 0], boundary_points[0, 0]),
           np.append(boundary_points[:, 1], boundary_points[0, 1]),
           'r-', linewidth=2, label='Boundary')
    
    # Set title and labels
    ax.set_title(f'Simple Uniform Triangulation\n({len(triangles)} triangles)', 
                fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    
    # Calculate mesh statistics
    edge_lengths = []
    for tri in triangles:
        v1, v2, v3 = vertices[tri]
        edge_lengths.extend([
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v3 - v2),
            np.linalg.norm(v1 - v3)
        ])
    
    mean_edge = np.mean(edge_lengths)
    std_edge = np.std(edge_lengths)
    
    logger.info(f"Mesh statistics:")
    logger.info(f"  Number of triangles: {len(triangles)}")
    logger.info(f"  Mean edge length: {mean_edge:.4f}")
    logger.info(f"  Edge length std dev: {std_edge:.4f}")
    logger.info(f"  Edge uniformity (std/mean): {std_edge/mean_edge:.4f}")
    
    # Show plot
    plt.tight_layout()
    plt.savefig('simple_uniform_triangulation.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_simple_triangulation() 