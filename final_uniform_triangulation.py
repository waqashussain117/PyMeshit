"""
Final uniform triangulation with high-quality mesh generation.

This script creates perfectly uniform triangulation with consistent
triangle sizes throughout the domain, without irregular patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import os
import logging

# Set up logging and output directory
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UniformTriangulation")
RESULTS_DIR = "triangulation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_uniform_grid(domain_size=10, spacing=1.0, jitter=0.0):
    """
    Create a uniform grid of points with precise spacing and optional jitter.
    
    Args:
        domain_size: Size of the square domain
        spacing: Spacing between points
        jitter: Amount of random displacement to add
        
    Returns:
        Array of points
    """
    # Calculate the number of points in each dimension
    n = int(2 * domain_size / spacing) + 1
    
    # Create a perfectly uniform grid
    x = np.linspace(-domain_size, domain_size, n)
    y = np.linspace(-domain_size, domain_size, n)
    xx, yy = np.meshgrid(x, y)
    
    # Add small jitter if requested (helps create better triangulation)
    if jitter > 0:
        xx = xx + np.random.uniform(-jitter * spacing, jitter * spacing, xx.shape)
        yy = yy + np.random.uniform(-jitter * spacing, jitter * spacing, yy.shape)
    
    # Filter points to keep only those inside the domain
    points = np.column_stack((xx.flatten(), yy.flatten()))
    
    # Create a mask to keep only interior points (not on the exact boundary)
    mask = np.ones(len(points), dtype=bool)
    boundary_tolerance = 1e-10
    
    for i, p in enumerate(points):
        if (abs(abs(p[0]) - domain_size) < boundary_tolerance or 
            abs(abs(p[1]) - domain_size) < boundary_tolerance):
            mask[i] = False
    
    return points[mask]

def create_boundary_points(domain_size=10, points_per_side=15):
    """
    Create evenly spaced points along the boundary.
    
    Args:
        domain_size: Size of the square domain
        points_per_side: Number of points per side
        
    Returns:
        Array of boundary points
    """
    # Create evenly spaced points on each edge
    t = np.linspace(0, 1, points_per_side, endpoint=False)
    
    # Top edge (left to right)
    top_x = -domain_size + 2 * domain_size * t
    top_y = np.ones(points_per_side) * domain_size
    top_points = np.column_stack((top_x, top_y))
    
    # Right edge (top to bottom)
    right_y = domain_size - 2 * domain_size * t
    right_x = np.ones(points_per_side) * domain_size
    right_points = np.column_stack((right_x, right_y))
    
    # Bottom edge (right to left)
    bottom_x = domain_size - 2 * domain_size * t
    bottom_y = np.ones(points_per_side) * -domain_size
    bottom_points = np.column_stack((bottom_x, bottom_y))
    
    # Left edge (bottom to top)
    left_y = -domain_size + 2 * domain_size * t
    left_x = np.ones(points_per_side) * -domain_size
    left_points = np.column_stack((left_x, left_y))
    
    # Combine all edges
    boundary_points = np.vstack((top_points, right_points, bottom_points, left_points))
    
    return boundary_points

def create_segments(boundary_points):
    """
    Create segments connecting boundary points in sequence.
    
    Args:
        boundary_points: Boundary point coordinates
        
    Returns:
        Array of segment indices
    """
    n_points = len(boundary_points)
    segments = []
    
    for i in range(n_points):
        segments.append([i, (i + 1) % n_points])
    
    return np.array(segments)

def triangulate_uniform(domain_size=10, min_angle=30, max_area=None):
    """
    Create a uniform triangulation with consistent triangle sizes.
    
    Args:
        domain_size: Size of the square domain
        min_angle: Minimum angle for quality constraint
        max_area: Maximum triangle area (if None, calculated automatically)
        
    Returns:
        Triangulation result
    """
    # Create a uniform spacing for the grid
    # We'll use a density that creates nicely sized triangles
    interior_spacing = domain_size / 15
    
    # Calculate boundary point spacing to match interior
    points_per_side = int(2 * domain_size / interior_spacing) + 1
    
    # Create boundary points
    boundary_points = create_boundary_points(domain_size, points_per_side)
    logger.info(f"Created {len(boundary_points)} boundary points")
    
    # Create interior points on a perfectly regular grid
    interior_points = create_uniform_grid(
        domain_size - interior_spacing/2,  # Shrink slightly to avoid boundary
        interior_spacing,
        jitter=0.01  # Very small jitter for numerical stability
    )
    logger.info(f"Created {len(interior_points)} interior points")
    
    # Create segments for boundary constraint
    segments = create_segments(boundary_points)
    
    # Combine all points (boundary must come first for segment indices)
    all_points = np.vstack((boundary_points, interior_points))
    
    # Calculate max area if not provided
    if max_area is None:
        # Triangle with side length = spacing
        # Area = (sqrt(3)/4) * spacing^2
        max_area = 0.433 * interior_spacing * interior_spacing
    
    # Prepare options string for Triangle
    options = f'pq{min_angle}a{max_area}'
    logger.info(f"Triangulating with options: {options}")
    
    # Create input for Triangle
    tri_input = {
        'vertices': all_points,
        'segments': segments
    }
    
    # Triangulate
    result = tr.triangulate(tri_input, options)
    logger.info(f"Created {len(result['triangles'])} triangles")
    
    # Plot the input points
    plt.figure(figsize=(12, 12))
    plt.scatter(interior_points[:, 0], interior_points[:, 1], c='b', s=10, label='Interior Points')
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='r', s=20, label='Boundary Points')
    
    # Show the boundary
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f"Uniform Grid Points ({len(interior_points)} interior + {len(boundary_points)} boundary)")
    plt.savefig(os.path.join(RESULTS_DIR, "uniform_triangulation_input.png"), dpi=300)
    plt.close()
    
    # Plot the triangulation
    plt.figure(figsize=(12, 12))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
               result['triangles'], 'b-', lw=0.5)
    
    # Show the boundary
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"Uniform Triangulation ({len(result['triangles'])} triangles)")
    plt.savefig(os.path.join(RESULTS_DIR, "uniform_triangulation.png"), dpi=300)
    plt.close()
    
    # Create a zoomed view
    plt.figure(figsize=(12, 12))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
               result['triangles'], 'b-', lw=0.5)
    
    # Show the boundary
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    # Zoom to corner
    plt.xlim(5, 10)
    plt.ylim(5, 10)
    
    plt.grid(True)
    plt.title("Zoomed View - Uniform Triangulation")
    plt.savefig(os.path.join(RESULTS_DIR, "uniform_triangulation_zoomed.png"), dpi=300)
    plt.close()
    
    return result

def create_regular_triangular_mesh(domain_size=10, spacing=1.0):
    """
    Create a highly regular triangular mesh using a triangular grid pattern.
    
    This creates a mesh with nearly equilateral triangles of the same size.
    
    Args:
        domain_size: Size of the square domain
        spacing: Spacing between points
        
    Returns:
        Triangulation result
    """
    # Number of points in each direction
    n = int(2 * domain_size / spacing) + 3  # Add margin points
    
    # Create rectangular grid
    x = np.linspace(-domain_size - spacing, domain_size + spacing, n)
    y = np.linspace(-domain_size - spacing, domain_size + spacing, n)
    
    # Convert to triangular grid by offsetting every other row
    points = []
    for i, yi in enumerate(y):
        offset = 0.5 * spacing if i % 2 else 0
        for xi in x:
            points.append([xi + offset, yi])
    
    points = np.array(points)
    
    # Filter to keep only points in or near the domain
    margin = spacing * 0.1
    mask = ((points[:, 0] >= -domain_size - margin) & 
            (points[:, 0] <= domain_size + margin) & 
            (points[:, 1] >= -domain_size - margin) & 
            (points[:, 1] <= domain_size + margin))
    
    filtered_points = points[mask]
    
    # Create boundary points - exactly on the boundary
    boundary_points_per_side = int(2 * domain_size / spacing) + 1
    boundary_points = create_boundary_points(domain_size, boundary_points_per_side)
    
    # Create segments for the boundary
    segments = create_segments(boundary_points)
    
    # Identify interior vs exterior points
    interior_mask = ((filtered_points[:, 0] > -domain_size) & 
                     (filtered_points[:, 0] < domain_size) & 
                     (filtered_points[:, 1] > -domain_size) & 
                     (filtered_points[:, 1] < domain_size))
    
    interior_points = filtered_points[interior_mask]
    
    # Combine points - boundary first, then interior
    all_points = np.vstack((boundary_points, interior_points))
    
    # Triangulate
    options = 'pq30'  # Constrained quality triangulation
    logger.info(f"Triangulating triangular grid with options: {options}")
    
    # Create input for Triangle
    tri_input = {
        'vertices': all_points,
        'segments': segments
    }
    
    # Triangulate
    result = tr.triangulate(tri_input, options)
    logger.info(f"Created {len(result['triangles'])} triangles")
    
    # Plot the input points
    plt.figure(figsize=(12, 12))
    plt.scatter(interior_points[:, 0], interior_points[:, 1], c='b', s=10, label='Interior Grid')
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='r', s=20, label='Boundary Points')
    
    # Show the boundary
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f"Triangular Grid Points ({len(interior_points)} interior + {len(boundary_points)} boundary)")
    plt.savefig(os.path.join(RESULTS_DIR, "triangular_grid_input.png"), dpi=300)
    plt.close()
    
    # Plot the triangulation
    plt.figure(figsize=(12, 12))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
               result['triangles'], 'b-', lw=0.5)
    
    # Show the boundary
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"Regular Triangular Mesh ({len(result['triangles'])} triangles)")
    plt.savefig(os.path.join(RESULTS_DIR, "regular_triangular_mesh.png"), dpi=300)
    plt.close()
    
    # Create a zoomed view
    plt.figure(figsize=(12, 12))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
               result['triangles'], 'b-', lw=0.5)
    
    # Show the boundary
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    # Zoom to corner
    plt.xlim(5, 10)
    plt.ylim(5, 10)
    
    plt.grid(True)
    plt.title("Zoomed View - Regular Triangular Mesh")
    plt.savefig(os.path.join(RESULTS_DIR, "regular_triangular_mesh_zoomed.png"), dpi=300)
    plt.close()
    
    return result

def run_all_triangulations():
    """Run all triangulation methods."""
    # 1. Standard uniform triangulation
    triangulate_uniform(domain_size=10, min_angle=30)
    
    # 2. Regular triangular mesh (more equilateral triangles)
    create_regular_triangular_mesh(domain_size=10, spacing=0.75)

if __name__ == "__main__":
    run_all_triangulations()
    plt.close('all')  # Close all figures 