"""
Basic comparison script for uniform vs. gradient triangulation.

This script creates just two triangulation examples - a uniform one and one
with a gradient - to quickly show the difference between them.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from meshit.triangle_direct import DirectTriangleWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TriangulationComparison")

# Ensure results directory exists
RESULTS_DIR = "triangulation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_square_grid(size=10, spacing=1.0):
    """Create a regular grid of points within a square."""
    # Calculate number of points in each direction
    n_points = int(2 * size / spacing) + 1
    
    # Create regular grid
    x = np.linspace(-size, size, n_points)
    y = np.linspace(-size, size, n_points)
    xx, yy = np.meshgrid(x, y)
    
    # Flatten to get points array
    points = np.column_stack((xx.flatten(), yy.flatten()))
    
    return points

def create_square_boundary(size=10, points_per_edge=10):
    """Create points along the boundary of a square."""
    # Create points along each edge
    top_edge = np.column_stack((
        np.linspace(-size, size, points_per_edge),
        np.ones(points_per_edge) * size
    ))
    
    right_edge = np.column_stack((
        np.ones(points_per_edge) * size,
        np.linspace(size, -size, points_per_edge)
    ))
    
    bottom_edge = np.column_stack((
        np.linspace(size, -size, points_per_edge),
        np.ones(points_per_edge) * -size
    ))
    
    left_edge = np.column_stack((
        np.ones(points_per_edge) * -size,
        np.linspace(-size, size, points_per_edge)
    ))
    
    # Combine all edges
    boundary_points = np.vstack((top_edge, right_edge, bottom_edge, left_edge))
    
    return boundary_points

def compute_square_segments(boundary_points):
    """Create segment pairs connecting boundary points in order."""
    n_points = len(boundary_points)
    segments = []
    
    for i in range(n_points):
        segments.append([i, (i + 1) % n_points])
    
    return np.array(segments)

def run_basic_comparison():
    """Compare uniform triangulation with a gradient-based one."""
    # Common parameters
    grid_spacing = 3.0  # Larger spacing for fewer points and faster processing
    points_per_edge = 8  # Fewer boundary points for faster processing
    
    # Create input points
    boundary_points = create_square_boundary(size=10, points_per_edge=points_per_edge)
    interior_points = create_square_grid(size=9, spacing=grid_spacing)
    all_points = np.vstack((boundary_points, interior_points))
    
    # Create boundary segments
    segments = compute_square_segments(boundary_points)
    boundary_indices = np.arange(len(boundary_points))
    segment_pairs = []
    for i in range(len(segments)):
        segment_pairs.append([boundary_indices[segments[i][0]], 
                            boundary_indices[segments[i][1]]])
    segments = np.array(segment_pairs)
    
    # 1. Uniform triangulation
    logger.info("Creating uniform triangulation...")
    base_size = grid_spacing * 1.5
    uniform_wrapper = DirectTriangleWrapper(gradient=0.0, base_size=base_size)
    
    # Use quality triangle meshing without refinement
    uniform_wrapper.triangle_opts = "pzYYq30"
    
    # Triangulate
    uniform_result = uniform_wrapper.triangulate(
        all_points, segments,
        create_feature_points=False,
        create_transition=False
    )
    
    if uniform_result is not None and 'triangles' in uniform_result:
        # Plot uniform triangulation
        plt.figure(figsize=(12, 12))
        plt.triplot(uniform_result['vertices'][:, 0], uniform_result['vertices'][:, 1],
                  uniform_result['triangles'], 'b-', lw=0.5)
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'g-', lw=1.5)
        plt.scatter(all_points[:, 0], all_points[:, 1], c='r', s=5, alpha=0.3)
        plt.axis('equal')
        plt.grid(True)
        plt.title(f'Uniform Triangulation ({len(uniform_result["triangles"])} triangles)')
        plt.savefig(os.path.join(RESULTS_DIR, 'basic_uniform.png'), dpi=300)
        plt.close()
        
        logger.info(f"Created {len(uniform_result['triangles'])} triangles with uniform density")
    
    # 2. Gradient-based triangulation
    logger.info("Creating gradient-based triangulation...")
    gradient_wrapper = DirectTriangleWrapper(gradient=1.0, base_size=base_size)
    
    # Add a central feature point with small size to create refinement
    central_point = np.array([[0.0, 0.0]])
    central_size = np.array([base_size * 0.1])  # Small size = high density
    gradient_wrapper.set_feature_points(central_point, central_size)
    
    # Triangulate
    gradient_result = gradient_wrapper.triangulate(
        all_points, segments,
        create_feature_points=True,
        create_transition=True
    )
    
    if gradient_result is not None and 'triangles' in gradient_result:
        # Plot gradient triangulation
        plt.figure(figsize=(12, 12))
        plt.triplot(gradient_result['vertices'][:, 0], gradient_result['vertices'][:, 1],
                  gradient_result['triangles'], 'b-', lw=0.5)
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'g-', lw=1.5)
        plt.scatter(all_points[:, 0], all_points[:, 1], c='r', s=5, alpha=0.3)
        plt.scatter(central_point[:, 0], central_point[:, 1], c='m', s=50, marker='*')
        plt.axis('equal')
        plt.grid(True)
        plt.title(f'Gradient Triangulation (g=1.0, {len(gradient_result["triangles"])} triangles)')
        plt.savefig(os.path.join(RESULTS_DIR, 'basic_gradient.png'), dpi=300)
        plt.close()
        
        logger.info(f"Created {len(gradient_result['triangles'])} triangles with gradient density")
    
    logger.info("Basic comparison completed. Results saved to triangulation_results folder.")

if __name__ == "__main__":
    run_basic_comparison()
    plt.close('all')  # Close all figures 