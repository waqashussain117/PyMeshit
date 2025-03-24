"""
Better triangulation with proper triangle elements.

This script creates a proper triangulation with well-defined triangular elements,
avoiding the quadrilateral patterns seen in the previous attempt.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import os
import logging

# Set up logging and output directory
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BetterTriangulation")
RESULTS_DIR = "triangulation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_boundary_points(domain_size=10, points_per_side=15):
    """
    Create evenly spaced points along the boundary.
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

def create_interior_points(domain_size=10, density=20):
    """
    Create interior points with random distribution to avoid grid patterns.
    This ensures proper triangulation without artificial quadrilateral patterns.
    """
    # Total number of points to generate
    n_points = density * density
    
    # Create random points within domain
    rng = np.random.RandomState(42)  # Use fixed seed for reproducibility
    x = rng.uniform(-domain_size + 0.5, domain_size - 0.5, n_points)
    y = rng.uniform(-domain_size + 0.5, domain_size - 0.5, n_points)
    
    points = np.column_stack((x, y))
    
    return points

def create_segments(boundary_points):
    """
    Create segments connecting boundary points in sequence.
    """
    n_points = len(boundary_points)
    segments = []
    
    for i in range(n_points):
        segments.append([i, (i + 1) % n_points])
    
    return np.array(segments)

def triangulate_better(domain_size=10, min_angle=20, max_area=None):
    """
    Create a proper triangulation with real triangular elements.
    
    Args:
        domain_size: Size of the square domain
        min_angle: Minimum angle for quality constraint
        max_area: Maximum triangle area
    """
    # Create boundary points
    boundary_points = create_boundary_points(domain_size, points_per_side=20)
    logger.info(f"Created {len(boundary_points)} boundary points")
    
    # Create interior points with random distribution
    interior_points = create_interior_points(domain_size, density=20)
    logger.info(f"Created {len(interior_points)} interior points")
    
    # Create segments for boundary constraint
    segments = create_segments(boundary_points)
    
    # Combine all points (boundary must come first for segment indices)
    all_points = np.vstack((boundary_points, interior_points))
    
    # Calculate max area if not provided
    if max_area is None:
        max_area = (domain_size * domain_size) / 200  # Will create roughly 400 triangles
    
    # Prepare options for Triangle
    # 'p' - planar straight line graph
    # 'q' - quality mesh generation with minimum angle
    # 'a' - maximum area constraint
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
    plt.scatter(interior_points[:, 0], interior_points[:, 1], c='b', s=10, alpha=0.5, label='Interior Points')
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='r', s=20, label='Boundary Points')
    
    # Show the boundary
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f"Input Points ({len(interior_points)} interior + {len(boundary_points)} boundary)")
    plt.savefig(os.path.join(RESULTS_DIR, "better_triangulation_input.png"), dpi=300)
    plt.close()
    
    # Plot the triangulation
    plt.figure(figsize=(12, 12))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
               result['triangles'], 'b-', lw=0.5)
    
    # Show boundary
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"Better Triangulation ({len(result['triangles'])} triangles)")
    plt.savefig(os.path.join(RESULTS_DIR, "better_triangulation.png"), dpi=300)
    plt.close()
    
    # Create a zoomed view
    plt.figure(figsize=(12, 12))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
               result['triangles'], 'b-', lw=0.5)
    
    # Show boundary
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    # Zoom to corner
    plt.xlim(5, 10)
    plt.ylim(5, 10)
    
    plt.grid(True)
    plt.title("Zoomed View - Better Triangulation")
    plt.savefig(os.path.join(RESULTS_DIR, "better_triangulation_zoomed.png"), dpi=300)
    plt.close()
    
    return result

def triangulate_structured_mesh(domain_size=10, base_size=1.0, min_angle=20):
    """
    Create a structured triangular mesh by diagonally dividing a quadrilateral grid.
    This ensures a very clean triangulation with consistent triangle shapes.
    
    Args:
        domain_size: Size of the square domain
        base_size: Base size of elements
        min_angle: Minimum angle for quality
    """
    # Create a grid of points with consistent spacing
    n = int(2 * domain_size / base_size) + 1
    
    # Create grid points
    x = np.linspace(-domain_size, domain_size, n)
    y = np.linspace(-domain_size, domain_size, n)
    xx, yy = np.meshgrid(x, y)
    
    # Convert to point array
    interior_points = []
    
    # Keep only interior points
    for i in range(1, n-1):  # Skip boundary
        for j in range(1, n-1):  # Skip boundary
            interior_points.append([xx[i, j], yy[i, j]])
    
    interior_points = np.array(interior_points)
    
    # Create boundary with more points than grid corners
    boundary_points = create_boundary_points(domain_size, points_per_side=n)
    
    # Create segments for boundary constraint
    segments = create_segments(boundary_points)
    
    # Combine all points (boundary must come first for segment indices)
    all_points = np.vstack((boundary_points, interior_points))
    
    # Prepare options for Triangle
    # 'p' - planar straight line graph
    # 'q' - quality mesh generation with minimum angle
    # 'D' - Deterministic triangulation for better ordering
    options = f'pq{min_angle}D'
    logger.info(f"Triangulating structured mesh with options: {options}")
    
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
    plt.title(f"Structured Grid Points ({len(interior_points)} interior + {len(boundary_points)} boundary)")
    plt.savefig(os.path.join(RESULTS_DIR, "structured_mesh_input.png"), dpi=300)
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
    plt.title(f"Structured Triangular Mesh ({len(result['triangles'])} triangles)")
    plt.savefig(os.path.join(RESULTS_DIR, "structured_mesh.png"), dpi=300)
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
    plt.title("Zoomed View - Structured Triangular Mesh")
    plt.savefig(os.path.join(RESULTS_DIR, "structured_mesh_zoomed.png"), dpi=300)
    plt.close()
    
    return result

def run_all_triangulations():
    """Run all triangulation methods."""
    # 1. Better triangulation with randomized points
    triangulate_better(domain_size=10, min_angle=20, max_area=1.0)
    
    # 2. Structured triangular mesh with clean diagonal patterns
    triangulate_structured_mesh(domain_size=10, base_size=1.0, min_angle=20)

if __name__ == "__main__":
    run_all_triangulations()
    plt.close('all')  # Close all figures 