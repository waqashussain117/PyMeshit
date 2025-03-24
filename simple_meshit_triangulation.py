"""
Simple MeshIt triangulation without refinement or feature points.

This script demonstrates how to use the MeshIt code to triangulate a square domain
with the simplest possible configuration - no refinement, no feature points,
just plain triangulation using Triangle library.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimpleMeshIt")

# Make sure output directory exists
RESULTS_DIR = "triangulation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_square_boundary(size=10.0, num_points=80):
    """
    Create points along the boundary of a square.
    
    Args:
        size: Half-width of the square domain
        num_points: Total number of points on the boundary
        
    Returns:
        Array of boundary point coordinates
    """
    # Calculate points per side (ensuring we don't double-count corners)
    points_per_side = num_points // 4
    
    # Create points for each side
    t = np.linspace(0, 1, points_per_side, endpoint=False)
    
    # Top side (left to right)
    top_x = -size + 2 * size * t
    top_y = np.ones_like(top_x) * size
    
    # Right side (top to bottom)
    right_y = size - 2 * size * t
    right_x = np.ones_like(right_y) * size
    
    # Bottom side (right to left)
    bottom_x = size - 2 * size * t
    bottom_y = np.ones_like(bottom_x) * (-size)
    
    # Left side (bottom to top)
    left_y = -size + 2 * size * t
    left_x = np.ones_like(left_y) * (-size)
    
    # Combine all sides
    x = np.concatenate([top_x, right_x, bottom_x, left_x])
    y = np.concatenate([top_y, right_y, bottom_y, left_y])
    
    return np.column_stack([x, y])

def create_interior_points(size=10.0, spacing=1.0, jitter=0.1):
    """
    Create interior points for triangulation with optional jitter.
    
    Args:
        size: Half-width of the square domain
        spacing: Spacing between points
        jitter: Amount of random displacement to apply (0 = grid)
        
    Returns:
        Array of interior point coordinates
    """
    # Create a grid of points
    range_min = -size + spacing
    range_max = size - spacing
    x = np.arange(range_min, range_max + spacing/2, spacing)
    y = np.arange(range_min, range_max + spacing/2, spacing)
    xx, yy = np.meshgrid(x, y)
    
    # Apply jitter if requested
    if jitter > 0:
        xx += np.random.uniform(-jitter*spacing/2, jitter*spacing/2, xx.shape)
        yy += np.random.uniform(-jitter*spacing/2, jitter*spacing/2, yy.shape)
    
    # Convert to array of points
    return np.column_stack([xx.flatten(), yy.flatten()])

def create_boundary_segments(num_boundary_points):
    """Create segments connecting consecutive boundary points."""
    return np.column_stack([
        np.arange(num_boundary_points),
        np.roll(np.arange(num_boundary_points), -1)
    ])

def simple_triangle_triangulation(vertices, segments):
    """
    Perform simple triangulation using Triangle library.
    
    This uses only the basic 'p' option to preserve the input segments,
    without any quality constraints or refinement.
    
    Args:
        vertices: Input vertices
        segments: Input segments
        
    Returns:
        Dict with triangulation results
    """
    # Prepare input for Triangle
    triangle_data = {
        'vertices': vertices,
        'segments': segments
    }
    
    # Triangulate with only 'p' option (preserve input, no quality constraints)
    logger.info("Performing basic triangulation with 'p' option (no quality constraints)")
    result = tr.triangulate(triangle_data, 'p')
    
    logger.info(f"Created {len(result['triangles'])} triangles from {len(result['vertices'])} vertices")
    return result

def basic_quality_triangulation(vertices, segments, min_angle=20.0):
    """
    Perform triangulation with basic quality constraints.
    
    Args:
        vertices: Input vertices
        segments: Input segments
        min_angle: Minimum angle constraint
        
    Returns:
        Dict with triangulation results
    """
    # Prepare input for Triangle
    triangle_data = {
        'vertices': vertices,
        'segments': segments
    }
    
    # Triangulate with quality constraint
    options = f'pq{min_angle}'
    logger.info(f"Performing quality triangulation with options: {options}")
    result = tr.triangulate(triangle_data, options)
    
    logger.info(f"Created {len(result['triangles'])} triangles from {len(result['vertices'])} vertices")
    return result

def plot_triangulation(result, boundary_points, filename, title):
    """
    Plot and save triangulation result.
    
    Args:
        result: Triangle result dict
        boundary_points: Original boundary points
        filename: Output filename
        title: Plot title
    """
    plt.figure(figsize=(12, 12))
    
    # Plot triangulation
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
               result['triangles'], 'b-', lw=0.5)
    
    # Draw boundary
    plt.plot(np.append(boundary_points[:, 0], boundary_points[0, 0]),
            np.append(boundary_points[:, 1], boundary_points[0, 1]), 
            'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"{title} ({len(result['triangles'])} triangles)")
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()

def run_meshit_basic_triangulation():
    """Run basic MeshIt-style triangulation without refinement."""
    # Domain parameters
    size = 10.0
    
    # Create boundary points
    boundary_points = create_square_boundary(size, num_points=60)
    num_boundary_points = len(boundary_points)
    logger.info(f"Created {num_boundary_points} boundary points")
    
    # Create boundary segments
    segments = create_boundary_segments(num_boundary_points)
    
    # Create interior points with slight jitter
    interior_points = create_interior_points(size, spacing=1.0, jitter=0.1)
    logger.info(f"Created {len(interior_points)} interior points")
    
    # Combine boundary and interior points
    all_points = np.vstack([boundary_points, interior_points])
    
    # Plot input
    plt.figure(figsize=(12, 12))
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], 
               c='r', s=20, label='Boundary Points')
    plt.scatter(interior_points[:, 0], interior_points[:, 1], 
               c='b', s=10, label='Interior Points')
    
    # Draw boundary
    plt.plot(np.append(boundary_points[:, 0], boundary_points[0, 0]),
            np.append(boundary_points[:, 1], boundary_points[0, 1]), 
            'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title("Input Points for Simple MeshIt Triangulation")
    plt.savefig(os.path.join(RESULTS_DIR, "meshit_simple_input.png"), dpi=300)
    plt.close()
    
    # Perform basic triangulation (no quality constraints)
    basic_result = simple_triangle_triangulation(all_points, segments)
    plot_triangulation(basic_result, boundary_points, 
                      "meshit_simple_basic.png", "Simple MeshIt Triangulation (No Quality)")
    
    # Perform quality triangulation (min angle = 20 degrees)
    quality_result = basic_quality_triangulation(all_points, segments, min_angle=20.0)
    plot_triangulation(quality_result, boundary_points, 
                      "meshit_simple_quality.png", "Simple MeshIt Triangulation (Quality)")
    
    # Perform quality triangulation (min angle = 30 degrees)
    high_quality_result = basic_quality_triangulation(all_points, segments, min_angle=30.0)
    plot_triangulation(high_quality_result, boundary_points, 
                      "meshit_simple_high_quality.png", "Simple MeshIt Triangulation (High Quality)")
    
    return basic_result, quality_result, high_quality_result

if __name__ == "__main__":
    logger.info("Running simple MeshIt triangulation")
    run_meshit_basic_triangulation()
    logger.info("Triangulation complete") 