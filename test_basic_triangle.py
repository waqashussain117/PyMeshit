"""
Basic triangulation test using Triangle library directly.

This script demonstrates using the Triangle library directly for mesh generation,
similar to what MeshIt would do for the initial triangulation step.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import os
import logging

# Set up logging and output directory
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BasicTriangle")
RESULTS_DIR = "triangulation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_square_boundary(size=10, points_per_edge=20):
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

def create_interior_points(size=10, spacing=2.0):
    """Create a regular grid of interior points."""
    # Calculate number of points in each direction
    n_points = int(2 * size / spacing) + 1
    
    # Create interior grid of points
    x = np.linspace(-size + spacing, size - spacing, n_points - 2)
    y = np.linspace(-size + spacing, size - spacing, n_points - 2)
    xx, yy = np.meshgrid(x, y)
    
    # Flatten to get points array
    points = np.column_stack((xx.flatten(), yy.flatten()))
    
    return points

def compute_segments(boundary_points):
    """Create segments connecting boundary points in order."""
    n_points = len(boundary_points)
    segments = []
    
    for i in range(n_points):
        segments.append([i, (i + 1) % n_points])
    
    return np.array(segments)

def run_basic_triangulation():
    """Run basic triangulation using Triangle directly."""
    # Parameters
    domain_size = 10
    boundary_points_per_edge = 15
    interior_spacing = 2.0
    
    # Create boundary and interior points
    boundary_points = create_square_boundary(size=domain_size, points_per_edge=boundary_points_per_edge)
    interior_points = create_interior_points(size=domain_size, spacing=interior_spacing)
    
    # Combine points (boundary points first, then interior)
    all_points = np.vstack((boundary_points, interior_points))
    
    # Create segments for the boundary
    segments = compute_segments(boundary_points)
    
    # Set up Triangle input
    tri_input = {
        'vertices': all_points,
        'segments': segments
    }
    
    # Test with three different triangulation approaches
    
    # 1. Basic Delaunay triangulation - just p (PSLG)
    logger.info("Running basic Delaunay triangulation...")
    tri_result_basic = tr.triangulate(tri_input, 'p')
    
    # 2. Quality triangulation with minimum angle constraint
    logger.info("Running quality constrained triangulation (min angle)...")
    tri_result_quality = tr.triangulate(tri_input, 'pq25')  # Min angle 25 degrees
    
    # 3. Area constrained triangulation
    logger.info("Running area constrained triangulation...")
    # Calculate base size from domain
    base_size = interior_spacing * 1.5
    area_constraint = base_size * base_size * 0.5
    tri_result_area = tr.triangulate(tri_input, f'pa{area_constraint}')
    
    # 4. MeshIt-style triangulation (combined quality and area constraints)
    logger.info("Running MeshIt-style initial triangulation...")
    tri_result_meshit = tr.triangulate(tri_input, f'pq20a{area_constraint}')
    
    # Plot input points
    plt.figure(figsize=(12, 12))
    plt.scatter(all_points[:, 0], all_points[:, 1], c='r', s=5)
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    plt.axis('equal')
    plt.grid(True)
    plt.title('Input Points')
    plt.savefig(os.path.join(RESULTS_DIR, 'basic_triangle_input.png'))
    plt.close()
    
    # Plot all triangulation results
    triangulation_types = [
        ('basic', tri_result_basic, 'Basic Delaunay'),
        ('quality', tri_result_quality, 'Quality Constrained (min angle=25°)'),
        ('area', tri_result_area, f'Area Constrained (max area={area_constraint:.2f})'),
        ('meshit', tri_result_meshit, 'MeshIt-style (quality + area)')
    ]
    
    for label, result, title in triangulation_types:
        if 'triangles' in result:
            plt.figure(figsize=(12, 12))
            plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
                      result['triangles'], 'b-', lw=0.5)
            
            # Show boundary in green
            plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
                     np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
            
            # Also show input points
            plt.scatter(all_points[:, 0], all_points[:, 1], c='r', s=3, alpha=0.3)
            
            plt.axis('equal')
            plt.grid(True)
            plt.title(f'{title} ({len(result["triangles"])} triangles)')
            plt.savefig(os.path.join(RESULTS_DIR, f'basic_triangle_{label}.png'), dpi=300)
            plt.close()
            
            logger.info(f"{title}: Created {len(result['triangles'])} triangles")
    
    logger.info("Basic triangulation tests completed")

if __name__ == "__main__":
    run_basic_triangulation()
    plt.close('all')  # Close all figures 