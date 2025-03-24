"""
Delaunay triangulation with front-based refinement.

This script creates a triangulation using Delaunay with front-based refinement,
which gives a more uniform mesh with high-quality triangles.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
from scipy.spatial import Delaunay
import os
import logging
import math

# Set up logging and output directory
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DelaunayRefinement")
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

def create_interior_points(boundary_points, target_spacing=1.0, jitter=0.1):
    """
    Create interior points using front-based method with padding from the boundary.
    This produces a more uniform point distribution.
    """
    # Find bounding box
    min_x, min_y = np.min(boundary_points, axis=0)
    max_x, max_y = np.max(boundary_points, axis=0)
    
    # Add margin to stay away from boundary
    margin = target_spacing
    min_x += margin
    min_y += margin
    max_x -= margin
    max_y -= margin
    
    # Compute grid size with extra density
    nx = int((max_x - min_x) / target_spacing) + 1
    ny = int((max_y - min_y) / target_spacing) + 1
    
    # Create dense grid covering the interior
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    xx, yy = np.meshgrid(x, y)
    
    # Apply small jitter for better triangulation
    if jitter > 0:
        xx = xx + np.random.uniform(-jitter * target_spacing, jitter * target_spacing, xx.shape)
        yy = yy + np.random.uniform(-jitter * target_spacing, jitter * target_spacing, yy.shape)
    
    # Flatten to get points array
    points = np.column_stack((xx.flatten(), yy.flatten()))
    
    return points

def create_circular_point_distribution(radius=10.0, rings=12, points_per_ring=30):
    """
    Create a more circular point distribution which often gives better triangulation.
    Points are positioned in concentric rings with higher density near the center.
    """
    points = []
    
    # Center point
    points.append([0, 0])
    
    # Create rings of points
    for ring in range(1, rings + 1):
        # Radius for this ring - increase density toward center
        r = radius * (ring / rings) ** 1.5
        
        # Points on this ring
        n_points = points_per_ring * ring // 2
        
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y])
    
    return np.array(points)

def compute_segments(boundary_points):
    """Create segments connecting boundary points in order."""
    n_points = len(boundary_points)
    segments = []
    
    for i in range(n_points):
        segments.append([i, (i + 1) % n_points])
    
    return np.array(segments)

def run_delaunay_refinement():
    """Run Delaunay triangulation with refinement."""
    # Parameters
    domain_size = 10
    boundary_points_per_edge = 10  # Fewer boundary points for a cleaner look
    target_spacing = 1.5  # Target spacing between interior points
    
    # Create boundary points
    boundary_points = create_square_boundary(size=domain_size, points_per_edge=boundary_points_per_edge)
    
    # Test multiple point distribution methods
    # 1. Grid-based interior points
    interior_grid = create_interior_points(boundary_points, target_spacing=target_spacing, jitter=0.15)
    
    # 2. Circular distribution
    interior_circular = create_circular_point_distribution(radius=domain_size-1, rings=10, points_per_ring=20)
    
    # 3. Combined approach - circular plus grid
    interior_combined = np.vstack((
        interior_circular,
        create_interior_points(boundary_points, target_spacing=target_spacing*2, jitter=0.3)
    ))
    
    # Create complete point sets
    all_points_grid = np.vstack((boundary_points, interior_grid))
    all_points_circular = np.vstack((boundary_points, interior_circular))
    all_points_combined = np.vstack((boundary_points, interior_combined))
    
    # Create segments for the boundary
    segments = compute_segments(boundary_points)
    
    # Run triangulation with Triangle library (compare methods)
    logger.info("Running triangulation with different point distributions...")
    
    # Set up Triangle inputs
    tri_inputs = [
        ('grid', {'vertices': all_points_grid, 'segments': segments}),
        ('circular', {'vertices': all_points_circular, 'segments': segments}),
        ('combined', {'vertices': all_points_combined, 'segments': segments})
    ]
    
    # Run triangulations
    results = []
    
    for label, tri_input in tri_inputs:
        # Constrained Delaunay triangulation - guaranteed to preserve boundary
        cdt_result = tr.triangulate(tri_input, 'pz')
        results.append((f'{label}_cdt', cdt_result, f'Constrained Delaunay ({label})'))
        
        # Quality triangulation - better uniform triangle shapes
        quality_result = tr.triangulate(tri_input, 'pzq30')
        results.append((f'{label}_quality', quality_result, f'Quality Triangulation ({label})'))
    
    # Plot input point distributions
    point_sets = [
        ('grid', all_points_grid),
        ('circular', all_points_circular),
        ('combined', all_points_combined)
    ]
    
    for label, points in point_sets:
        plt.figure(figsize=(12, 12))
        plt.scatter(points[:, 0], points[:, 1], c='r', s=5)
        plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
                 np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
        plt.axis('equal')
        plt.grid(True)
        plt.title(f'{label.capitalize()} Point Distribution')
        plt.savefig(os.path.join(RESULTS_DIR, f'delaunay_{label}_input.png'))
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
            
            # Don't plot points to keep it cleaner
            # plt.scatter(result['vertices'][:, 0], result['vertices'][:, 1], c='r', s=3, alpha=0.3)
            
            plt.axis('equal')
            plt.grid(True)
            plt.title(f'{title} ({len(result["triangles"])} triangles)')
            plt.savefig(os.path.join(RESULTS_DIR, f'delaunay_{label}.png'), dpi=300)
            plt.close()
            
            logger.info(f"{title}: Created {len(result['triangles'])} triangles")
    
    logger.info("Delaunay triangulation tests completed")

if __name__ == "__main__":
    run_delaunay_refinement()
    plt.close('all')  # Close all figures 