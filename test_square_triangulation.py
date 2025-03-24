"""
Test script for triangulating a square domain with regularly spaced points.

This script generates a square grid of points and triangulates it using our
improved triangulation methods to test mesh uniformity and boundary transitions.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.spatial import ConvexHull
from meshit.triangle_direct import DirectTriangleWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SquareTriangulationTest")

def create_square_grid(size=10, spacing=1.0, add_noise=False, noise_level=0.1):
    """
    Create a regular grid of points within a square.
    
    Args:
        size: Half-width of the square (total width will be 2*size)
        spacing: Spacing between points in the grid
        add_noise: Whether to add random noise to the grid positions
        noise_level: Amount of noise to add (as a fraction of spacing)
        
    Returns:
        NumPy array of points
    """
    # Calculate number of points in each direction (reduce for faster testing)
    n_points = int(2 * size / spacing) + 1
    
    # Create regular grid
    x = np.linspace(-size, size, n_points)
    y = np.linspace(-size, size, n_points)
    xx, yy = np.meshgrid(x, y)
    
    # Flatten to get points array
    points = np.column_stack((xx.flatten(), yy.flatten()))
    
    # Add random noise if requested
    if add_noise:
        noise = (np.random.random(points.shape) - 0.5) * noise_level * spacing
        points += noise
    
    return points

def create_square_boundary(size=10, num_points=40):
    """
    Create points along the boundary of a square.
    
    Args:
        size: Half-width of the square
        num_points: Total number of points to create on the boundary
        
    Returns:
        NumPy array of points along the square boundary
    """
    # Calculate points per edge (approximately)
    points_per_edge = max(int(num_points / 4), 2)
    
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
    """
    Create segment pairs connecting boundary points in order.
    
    Args:
        boundary_points: NumPy array of points along the boundary
        
    Returns:
        NumPy array of segment indices
    """
    n_points = len(boundary_points)
    segments = []
    
    for i in range(n_points):
        segments.append([i, (i + 1) % n_points])
    
    return np.array(segments)

def triangulate_square(gradient=1.0, grid_spacing=2.0):
    """
    Generate and triangulate a square domain with regular grid.
    
    Args:
        gradient: Mesh size gradient control parameter
        grid_spacing: Spacing between grid points (higher = fewer points)
        
    Returns:
        Dictionary with triangulation results
    """
    logger.info(f"Testing square triangulation with gradient={gradient}, spacing={grid_spacing}")
    
    # Create boundary points (fewer points for faster testing)
    boundary_points = create_square_boundary(size=10, num_points=40)
    
    # Create interior grid points (with larger spacing for fewer points)
    interior_points = create_square_grid(size=9, spacing=grid_spacing)
    
    # Combine points
    all_points = np.vstack((boundary_points, interior_points))
    
    # Create boundary segments
    segments = compute_square_segments(boundary_points)
    
    # Reset boundary point indices to match their position in the combined array
    boundary_indices = np.arange(len(boundary_points))
    segment_pairs = []
    for i in range(len(segments)):
        segment_pairs.append([boundary_indices[segments[i][0]], 
                             boundary_indices[segments[i][1]]])
    
    segments = np.array(segment_pairs)
    
    # Calculate base mesh size from domain size
    base_size = grid_spacing * 2
    
    # Create DirectTriangleWrapper instance
    wrapper = DirectTriangleWrapper(gradient=gradient, base_size=base_size)
    
    # Generate feature points automatically from boundary
    logger.info(f"Triangulating square domain with {len(all_points)} points...")
    
    # Plot input points
    plt.figure(figsize=(10, 10))
    plt.scatter(all_points[:, 0], all_points[:, 1], c='b', s=5)
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'r-')
    plt.axis('equal')
    plt.grid(True)
    plt.title(f'Square Domain Input (g={gradient})')
    plt.savefig(f'square_input_g{gradient}.png')
    
    # Triangulate
    result = wrapper.triangulate(all_points, segments)
    
    if result is not None and 'triangles' in result:
        # Plot triangulation
        plt.figure(figsize=(12, 12))
        
        # Plot triangles
        plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
                    result['triangles'], 'b-', lw=0.5)
        
        # Plot original points
        plt.scatter(all_points[:, 0], all_points[:, 1], c='r', s=5, alpha=0.5)
        
        # Plot boundary
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'g-', lw=1.5)
        
        plt.axis('equal')
        plt.grid(True)
        plt.title(f'Square Domain Triangulation (g={gradient}, {len(result["triangles"])} triangles)')
        plt.savefig(f'square_triangulation_g{gradient}.png')
        
        logger.info(f"Created {len(result['triangles'])} triangles with {len(result['vertices'])} vertices")
    else:
        logger.error("Triangulation failed")
    
    return result

def test_with_various_gradients():
    """Test triangulation with various gradient values"""
    # Test with different gradient values and a larger spacing (fewer points)
    for gradient in [0.5, 1.0, 2.0]:
        triangulate_square(gradient=gradient, grid_spacing=2.0)

if __name__ == "__main__":
    test_with_various_gradients()
    plt.close('all')  # Close all figures to avoid showing them 