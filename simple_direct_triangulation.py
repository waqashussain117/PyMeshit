"""
Direct triangulation using the Triangle library with quality constraints.

This script demonstrates simple triangulation with triangle quality constraints
similar to MeshIt's original approach.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import os
import logging

# Set up logging and output directory
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DirectTriangulation")
RESULTS_DIR = "triangulation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_grid_points(n=15, domain_size=10, jitter=0.0):
    """Generate a grid of points in the XY plane with optional jitter."""
    x = np.linspace(-domain_size, domain_size, n)
    y = np.linspace(-domain_size, domain_size, n)
    xx, yy = np.meshgrid(x, y)
    
    # Add jitter if requested
    if jitter > 0:
        xx = xx + np.random.uniform(-jitter, jitter, xx.shape)
        yy = yy + np.random.uniform(-jitter, jitter, yy.shape)
    
    # Combine coordinates
    points = np.column_stack((xx.flatten(), yy.flatten()))
    return points

def generate_boundary_points(n_points=40, domain_size=10):
    """Generate points along the boundary of a square."""
    # Create points along each edge of a square
    points_per_side = n_points // 4
    
    # Top edge
    top_x = np.linspace(-domain_size, domain_size, points_per_side)
    top_y = np.ones(points_per_side) * domain_size
    top_points = np.column_stack((top_x, top_y))
    
    # Right edge
    right_y = np.linspace(domain_size, -domain_size, points_per_side)
    right_x = np.ones(points_per_side) * domain_size
    right_points = np.column_stack((right_x, right_y))
    
    # Bottom edge
    bottom_x = np.linspace(domain_size, -domain_size, points_per_side)
    bottom_y = np.ones(points_per_side) * -domain_size
    bottom_points = np.column_stack((bottom_x, bottom_y))
    
    # Left edge
    left_y = np.linspace(-domain_size, domain_size, points_per_side)
    left_x = np.ones(points_per_side) * -domain_size
    left_points = np.column_stack((left_x, left_y))
    
    # Combine all edges
    boundary_points = np.vstack((top_points, right_points, bottom_points, left_points))
    
    return boundary_points

def compute_segments(boundary_points):
    """Create segments connecting boundary points in order."""
    n_points = len(boundary_points)
    segments = []
    
    for i in range(n_points):
        segments.append([i, (i + 1) % n_points])
    
    return np.array(segments)

def triangulate_direct(min_angle=30, max_area=4.0):
    """
    Perform direct triangulation with quality constraints.
    
    Args:
        min_angle: Minimum angle constraint (quality)
        max_area: Maximum area constraint
    """
    # Parameters
    domain_size = 10
    n_boundary = 60  # More boundary points for smoother edges
    interior_density = 15
    jitter = 0.1  # Small jitter for better triangulation
    
    # Generate interior points
    interior_points = generate_grid_points(interior_density, domain_size, jitter)
    logger.info(f"Generated {len(interior_points)} interior grid points")
    
    # Generate boundary points
    boundary_points = generate_boundary_points(n_boundary, domain_size)
    logger.info(f"Generated {len(boundary_points)} boundary points")
    
    # Create segments for the boundary
    segments = compute_segments(boundary_points)
    
    # Combine all points - boundary points must come first for segment indices to be correct
    all_points = np.vstack((boundary_points, interior_points))
    
    # Prepare options string for Triangle
    options = 'pq'  # p for PSLG (constrained Delaunay triangulation)
    
    if min_angle is not None:
        options += f'{min_angle}'  # Minimum angle constraint
    
    if max_area is not None:
        options += f'a{max_area}'  # Maximum area constraint
    
    logger.info(f"Triangulating with options: {options}")
    
    # Create input for Triangle
    tri_input = {
        'vertices': all_points,
        'segments': segments
    }
    
    # Triangulate
    result = tr.triangulate(tri_input, options)
    
    logger.info(f"Created {len(result['triangles'])} triangles")
    
    # Save input points visualization
    plt.figure(figsize=(12, 12))
    plt.scatter(interior_points[:, 0], interior_points[:, 1], c='r', s=3, label='Interior Points')
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5, label='Boundary')
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='b', s=10, label='Boundary Points')
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f"Input Points: {len(interior_points)} interior + {len(boundary_points)} boundary")
    plt.savefig(os.path.join(RESULTS_DIR, "direct_standard_input.png"), dpi=300)
    plt.close()
    
    # Plot triangulation
    plt.figure(figsize=(12, 12))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
              result['triangles'], 'b-', lw=0.5)
    
    # Show boundary
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    
    # Build title based on constraints
    title_parts = []
    if min_angle is not None:
        title_parts.append(f"Min Angle={min_angle}°")
    if max_area is not None:
        title_parts.append(f"Max Area={max_area}")
    
    constraints = " & ".join(title_parts) if title_parts else "No Constraints"
    plt.title(f"Standard Triangle ({constraints}, {len(result['triangles'])} triangles)")
    
    plt.savefig(os.path.join(RESULTS_DIR, "direct_standard_triangulation.png"), dpi=300)
    plt.close()
    
    # Create a closer view of the triangulation near the boundary
    plt.figure(figsize=(12, 12))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
              result['triangles'], 'b-', lw=0.5)
    
    # Show boundary
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    # Zoom to top-right corner
    plt.xlim(5, 10)
    plt.ylim(5, 10)
    
    plt.grid(True)
    plt.title(f"Zoomed View - Standard Triangle ({len(result['triangles'])} triangles)")
    
    plt.savefig(os.path.join(RESULTS_DIR, "direct_standard_triangulation_zoomed.png"), dpi=300)
    plt.close()
    
    return result, boundary_points, all_points
    
if __name__ == "__main__":
    triangulate_direct(min_angle=30, max_area=4.0)
    plt.close('all')  # Close all figures 