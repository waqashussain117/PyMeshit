"""
Improved uniform triangulation test with randomized point distribution.

This script creates a better triangulation by using randomly perturbed points
instead of a perfect grid, which produces more natural triangulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from meshit.triangle_direct import DirectTriangleWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ImprovedTriangulation")

# Ensure results directory exists
RESULTS_DIR = "triangulation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_random_points(num_points=200, min_x=-10, max_x=10, min_y=-10, max_y=10, 
                        boundary_padding=0.5):
    """Create randomly distributed points within a rectangular domain."""
    # Generate random points with padding from the boundary
    x = np.random.uniform(min_x + boundary_padding, max_x - boundary_padding, num_points)
    y = np.random.uniform(min_y + boundary_padding, max_y - boundary_padding, num_points)
    
    return np.column_stack((x, y))

def create_perturbed_grid(size=10, spacing=1.0, noise_level=0.3):
    """Create a grid of points with random perturbation for more natural triangulation."""
    # Calculate number of points in each direction
    n_points = int(2 * size / spacing) + 1
    
    # Create regular grid
    x = np.linspace(-size, size, n_points)
    y = np.linspace(-size, size, n_points)
    xx, yy = np.meshgrid(x, y)
    
    # Add random perturbation
    xx = xx + np.random.uniform(-noise_level * spacing, noise_level * spacing, xx.shape)
    yy = yy + np.random.uniform(-noise_level * spacing, noise_level * spacing, yy.shape)
    
    # Flatten to get points array
    points = np.column_stack((xx.flatten(), yy.flatten()))
    
    return points

def create_square_boundary(size=10, points_per_edge=15):
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

def run_improved_triangulation():
    """Run improved triangulation with randomized points and better parameters."""
    # Parameters
    domain_size = 10
    points_per_edge = 20  # More boundary points
    interior_spacing = 2.0
    noise_level = 0.4  # Add significant noise for natural triangulation
    
    logger.info("Creating improved triangulation...")
    
    # Create boundary points
    boundary_points = create_square_boundary(size=domain_size, points_per_edge=points_per_edge)
    
    # Create two sets of interior points for comparison
    regular_grid = create_perturbed_grid(size=domain_size-1, spacing=interior_spacing, noise_level=0)
    perturbed_grid = create_perturbed_grid(size=domain_size-1, spacing=interior_spacing, noise_level=noise_level)
    
    # Also create completely random points
    random_points = create_random_points(num_points=300, 
                                        min_x=-domain_size, max_x=domain_size,
                                        min_y=-domain_size, max_y=domain_size)
    
    # Create segments for boundary
    segments = compute_square_segments(boundary_points)
    
    # Map segments to their position in the combined array
    boundary_indices = np.arange(len(boundary_points))
    segment_pairs = []
    for i in range(len(segments)):
        segment_pairs.append([boundary_indices[segments[i][0]], 
                            boundary_indices[segments[i][1]]])
    segments = np.array(segment_pairs)
    
    # 1. Test with regular grid
    logger.info("Testing with regular grid...")
    all_points_regular = np.vstack((boundary_points, regular_grid))
    
    # Create wrapper for uniform triangulation
    uniform_wrapper = DirectTriangleWrapper(gradient=0.0, base_size=interior_spacing)
    uniform_wrapper.triangle_opts = "pzq25"  # Use quality constraint without u option
    
    # Triangulate
    regular_result = uniform_wrapper.triangulate(
        all_points_regular, segments,
        create_feature_points=False,
        create_transition=False
    )
    
    # Plot regular grid result
    plt.figure(figsize=(12, 12))
    plt.scatter(all_points_regular[:, 0], all_points_regular[:, 1], c='r', s=5)
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'g-', lw=1.5)
    plt.axis('equal')
    plt.grid(True)
    plt.title('Regular Grid Input')
    plt.savefig(os.path.join(RESULTS_DIR, 'improved_regular_input.png'))
    plt.close()
    
    if regular_result is not None and 'triangles' in regular_result:
        plt.figure(figsize=(12, 12))
        plt.triplot(regular_result['vertices'][:, 0], regular_result['vertices'][:, 1],
                  regular_result['triangles'], 'b-', lw=0.5)
        plt.scatter(all_points_regular[:, 0], all_points_regular[:, 1], c='r', s=3, alpha=0.3)
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'g-', lw=1.5)
        plt.axis('equal')
        plt.grid(True)
        plt.title(f'Regular Grid Triangulation ({len(regular_result["triangles"])} triangles)')
        plt.savefig(os.path.join(RESULTS_DIR, 'improved_regular_triangulation.png'), dpi=300)
        plt.close()
        
        logger.info(f"Created {len(regular_result['triangles'])} triangles with regular grid")
    
    # 2. Test with perturbed grid
    logger.info("Testing with perturbed grid...")
    all_points_perturbed = np.vstack((boundary_points, perturbed_grid))
    
    # Triangulate
    perturbed_result = uniform_wrapper.triangulate(
        all_points_perturbed, segments,
        create_feature_points=False,
        create_transition=False
    )
    
    # Plot perturbed grid result
    plt.figure(figsize=(12, 12))
    plt.scatter(all_points_perturbed[:, 0], all_points_perturbed[:, 1], c='r', s=5)
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'g-', lw=1.5)
    plt.axis('equal')
    plt.grid(True)
    plt.title('Perturbed Grid Input')
    plt.savefig(os.path.join(RESULTS_DIR, 'improved_perturbed_input.png'))
    plt.close()
    
    if perturbed_result is not None and 'triangles' in perturbed_result:
        plt.figure(figsize=(12, 12))
        plt.triplot(perturbed_result['vertices'][:, 0], perturbed_result['vertices'][:, 1],
                  perturbed_result['triangles'], 'b-', lw=0.5)
        plt.scatter(all_points_perturbed[:, 0], all_points_perturbed[:, 1], c='r', s=3, alpha=0.3)
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'g-', lw=1.5)
        plt.axis('equal')
        plt.grid(True)
        plt.title(f'Perturbed Grid Triangulation ({len(perturbed_result["triangles"])} triangles)')
        plt.savefig(os.path.join(RESULTS_DIR, 'improved_perturbed_triangulation.png'), dpi=300)
        plt.close()
        
        logger.info(f"Created {len(perturbed_result['triangles'])} triangles with perturbed grid")
    
    # 3. Test with random points
    logger.info("Testing with random points...")
    all_points_random = np.vstack((boundary_points, random_points))
    
    # Triangulate
    random_result = uniform_wrapper.triangulate(
        all_points_random, segments,
        create_feature_points=False,
        create_transition=False
    )
    
    # Plot random points result
    plt.figure(figsize=(12, 12))
    plt.scatter(all_points_random[:, 0], all_points_random[:, 1], c='r', s=5)
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'g-', lw=1.5)
    plt.axis('equal')
    plt.grid(True)
    plt.title('Random Points Input')
    plt.savefig(os.path.join(RESULTS_DIR, 'improved_random_input.png'))
    plt.close()
    
    if random_result is not None and 'triangles' in random_result:
        plt.figure(figsize=(12, 12))
        plt.triplot(random_result['vertices'][:, 0], random_result['vertices'][:, 1],
                  random_result['triangles'], 'b-', lw=0.5)
        plt.scatter(all_points_random[:, 0], all_points_random[:, 1], c='r', s=3, alpha=0.3)
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'g-', lw=1.5)
        plt.axis('equal')
        plt.grid(True)
        plt.title(f'Random Points Triangulation ({len(random_result["triangles"])} triangles)')
        plt.savefig(os.path.join(RESULTS_DIR, 'improved_random_triangulation.png'), dpi=300)
        plt.close()
        
        logger.info(f"Created {len(random_result['triangles'])} triangles with random points")
    
    logger.info("Completed improved triangulation tests")

if __name__ == "__main__":
    run_improved_triangulation()
    plt.close('all')  # Close all figures 