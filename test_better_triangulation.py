"""
Better triangulation test with more evenly sized triangles.

This script creates a more robust triangulation with proper mesh quality by 
using refined interior point distribution and improved Triangle options.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import os
import logging
import math

# Set up logging and output directory
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BetterTriangulation")
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

def create_hexagonal_grid(size=10, spacing=2.0):
    """
    Create a hexagonal grid of interior points.
    Hexagonal grids tend to produce more equilateral triangles.
    """
    # Calculate x and y ranges
    x_range = np.arange(-size + spacing, size - spacing, spacing)
    y_range = np.arange(-size + spacing, size - spacing, spacing * math.sqrt(3)/2)
    
    points = []
    for i, y in enumerate(y_range):
        # Offset every other row
        offset = spacing/2 if i % 2 else 0
        for x in x_range:
            # Skip if this would be right at the edge
            if abs(x + offset) >= size - spacing/2 or abs(y) >= size - spacing/2:
                continue
                
            points.append([x + offset, y])
    
    return np.array(points)

def compute_segments(boundary_points):
    """Create segments connecting boundary points in order."""
    n_points = len(boundary_points)
    segments = []
    
    for i in range(n_points):
        segments.append([i, (i + 1) % n_points])
    
    return np.array(segments)

def run_better_triangulation():
    """Run improved triangulation for better mesh quality."""
    # Parameters
    domain_size = 10
    boundary_points_per_edge = 15  # More boundary points for better edge quality
    interior_spacing = 1.0  # Slightly smaller spacing for more triangles
    
    # Create boundary points
    boundary_points = create_square_boundary(size=domain_size, points_per_edge=boundary_points_per_edge)
    
    # Create hexagonal grid of interior points - gives better triangulation
    interior_points = create_hexagonal_grid(size=domain_size, spacing=interior_spacing)
    
    # Combine points (boundary points first, then interior)
    all_points = np.vstack((boundary_points, interior_points))
    
    # Create segments for the boundary
    segments = compute_segments(boundary_points)
    
    # Set up Triangle input
    tri_input = {
        'vertices': all_points,
        'segments': segments
    }
    
    # Run triangulation with carefully chosen options
    # p: Use PSLG (preserve boundary segments)
    # q: Quality mesh generation with minimum angle
    # a: Maximum triangle area constraint
    # D: Conforming Delaunay (can create better triangles)
    logger.info("Running triangulation with optimal settings...")
    
    # Calculate target triangle area based on spacing
    # Using equilateral triangle area formula
    target_area = (math.sqrt(3)/4) * (interior_spacing**2)
    
    # Run triangulation with different quality settings
    results = []
    
    # 1. Best quality (30° min angle)
    best_quality_result = tr.triangulate(tri_input, f'pqDa{target_area}')
    results.append(('best', best_quality_result, 'High Quality (q30)'))
    
    # 2. Medium quality triangulation
    medium_quality_result = tr.triangulate(tri_input, f'pq20Da{target_area}')
    results.append(('medium', medium_quality_result, 'Medium Quality (q20)'))
    
    # 3. With additional interior Steiner points (allows Triangle to add points)
    with_steiner_result = tr.triangulate(tri_input, f'pq30a{target_area}')
    results.append(('steiner', with_steiner_result, 'With Steiner Points'))
    
    # Plot input points
    plt.figure(figsize=(12, 12))
    plt.scatter(all_points[:, 0], all_points[:, 1], c='r', s=5)
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    plt.axis('equal')
    plt.grid(True)
    plt.title('Hexagonal Grid Input Points')
    plt.savefig(os.path.join(RESULTS_DIR, 'better_triangle_input.png'))
    plt.close()
    
    # Plot triangulation results
    for label, result, title in results:
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
            plt.savefig(os.path.join(RESULTS_DIR, f'better_triangle_{label}.png'), dpi=300)
            plt.close()
            
            logger.info(f"{title}: Created {len(result['triangles'])} triangles")
    
    logger.info("Better triangulation tests completed")
    
    # Calculate mesh quality statistics for the best result
    if 'triangles' in best_quality_result:
        triangles = best_quality_result['triangles']
        vertices = best_quality_result['vertices']
        
        # Calculate quality metrics
        qualities = []
        areas = []
        
        for t in triangles:
            # Get triangle vertices
            v1, v2, v3 = vertices[t[0]], vertices[t[1]], vertices[t[2]]
            
            # Calculate edge lengths
            e1 = np.linalg.norm(v2 - v1)
            e2 = np.linalg.norm(v3 - v2)
            e3 = np.linalg.norm(v1 - v3)
            
            # Calculate area using Heron's formula
            s = (e1 + e2 + e3) / 2
            area = np.sqrt(s * (s - e1) * (s - e2) * (s - e3))
            areas.append(area)
            
            # Calculate quality metric (ratio of inradius to circumradius)
            # For an equilateral triangle this is 0.5, lower is worse
            quality = (4 * area) / (e1 * e2 * e3)
            qualities.append(quality)
        
        # Log statistics
        logger.info(f"Mesh quality statistics:")
        logger.info(f"  - Number of triangles: {len(triangles)}")
        logger.info(f"  - Mean triangle quality: {np.mean(qualities):.4f} (higher is better)")
        logger.info(f"  - Min triangle quality: {np.min(qualities):.4f}")
        logger.info(f"  - Max triangle quality: {np.max(qualities):.4f}")
        logger.info(f"  - Mean triangle area: {np.mean(areas):.4f}")
        logger.info(f"  - Area standard deviation: {np.std(areas):.4f}")

if __name__ == "__main__":
    run_better_triangulation()
    plt.close('all')  # Close all figures 