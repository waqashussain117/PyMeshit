"""
Comprehensive test script for comparing different triangulation methods.

This script tests both uniform and gradient-based triangulations with
various parameter settings, saving all results to the triangulation_results folder.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from meshit.triangle_direct import DirectTriangleWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TriangulationTest")

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

def triangulate_domain(gradient=1.0, grid_spacing=2.0, points_per_edge=10, 
                      create_feature_points=True, create_transition=True,
                      custom_options=None, label="test"):
    """
    Triangulate a square domain with the specified parameters.
    
    Args:
        gradient: Gradient control parameter (default: 1.0)
        grid_spacing: Spacing between interior points (default: 2.0)
        points_per_edge: Number of points per boundary edge (default: 10)
        create_feature_points: Whether to create feature points (default: True)
        create_transition: Whether to create transition points (default: True)
        custom_options: Optional custom Triangle options
        label: Label to use in filenames
        
    Returns:
        Dictionary with triangulation results
    """
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
    
    # Create wrapper and set parameters
    base_size = grid_spacing * 1.5
    wrapper = DirectTriangleWrapper(gradient=gradient, base_size=base_size)
    
    if custom_options:
        wrapper.triangle_opts = custom_options
    
    # Log configuration
    features_status = "enabled" if create_feature_points else "disabled"
    transition_status = "enabled" if create_transition else "disabled"
    options_status = f"custom: {custom_options}" if custom_options else "default"
    
    logger.info(f"Triangulating {label} with: gradient={gradient}, "
                f"feature points {features_status}, transition points {transition_status}, "
                f"options {options_status}")
    
    # Plot input
    plt.figure(figsize=(10, 10))
    plt.scatter(all_points[:, 0], all_points[:, 1], c='b', s=5)
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'r-')
    plt.axis('equal')
    plt.grid(True)
    plt.title(f'Input Domain - {label} (g={gradient})')
    plt.savefig(os.path.join(RESULTS_DIR, f'input_{label}_g{gradient}.png'))
    plt.close()
    
    # Triangulate
    result = wrapper.triangulate(
        all_points, segments,
        create_feature_points=create_feature_points,
        create_transition=create_transition
    )
    
    if result is not None and 'triangles' in result:
        # Plot triangulation result
        plt.figure(figsize=(12, 12))
        
        # Plot triangles
        plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
                    result['triangles'], 'b-', lw=0.5)
        
        # Plot boundary
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'g-', lw=1.5)
        
        # Plot original points with low opacity
        plt.scatter(all_points[:, 0], all_points[:, 1], c='r', s=3, alpha=0.3)
        
        plt.axis('equal')
        plt.grid(True)
        plt.title(f'{label} Triangulation (g={gradient}, {len(result["triangles"])} triangles)')
        plt.savefig(os.path.join(RESULTS_DIR, f'tri_{label}_g{gradient}.png'), dpi=300)
        plt.close()
        
        logger.info(f"Created {len(result['triangles'])} triangles with {len(result['vertices'])} vertices")
    else:
        logger.error(f"Triangulation failed for {label}")
    
    return result

def run_comprehensive_tests():
    """Run a comprehensive set of triangulation tests with different parameters."""
    # Test uniform triangulation (gradient = 0)
    triangulate_domain(
        gradient=0.0, 
        create_feature_points=False, 
        create_transition=False,
        custom_options="pzYYq30", 
        label="uniform"
    )
    
    # Test mild gradient (gradient = 0.5)
    triangulate_domain(
        gradient=0.5, 
        create_feature_points=True, 
        create_transition=True,
        label="mild_gradient"
    )
    
    # Test standard gradient (gradient = 1.0)
    triangulate_domain(
        gradient=1.0, 
        create_feature_points=True, 
        create_transition=True,
        label="standard_gradient"
    )
    
    # Test strong gradient (gradient = 2.0)
    triangulate_domain(
        gradient=2.0, 
        create_feature_points=True, 
        create_transition=True,
        label="strong_gradient"
    )
    
    # Test with feature points but no transition layer
    triangulate_domain(
        gradient=1.0, 
        create_feature_points=True, 
        create_transition=False,
        label="no_transition"
    )
    
    # Test with denser boundary points
    triangulate_domain(
        gradient=1.0, 
        points_per_edge=20,  # More boundary points
        create_feature_points=True, 
        create_transition=True,
        label="dense_boundary"
    )
    
    # Test with denser interior points
    triangulate_domain(
        gradient=1.0, 
        grid_spacing=1.0,  # Smaller spacing = more points
        create_feature_points=True, 
        create_transition=True,
        label="dense_interior"
    )
    
    logger.info("All triangulation tests completed. Results saved to triangulation_results folder.")

if __name__ == "__main__":
    run_comprehensive_tests()
    plt.close('all')  # Close all figures 