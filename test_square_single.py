"""
Simplified test script for triangulating a square domain.

This test focuses on a single gradient value to produce a quick result.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from meshit.triangle_direct import DirectTriangleWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SquareSingleTest")

def create_square_grid(size=10, spacing=2.0):
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

def main():
    """Main function to test square triangulation."""
    # Fixed parameters
    gradient = 1.5  # Middle gradient value
    grid_spacing = 2.5  # Larger spacing for fewer points
    points_per_edge = 10  # Fewer boundary points
    
    logger.info(f"Testing square triangulation with gradient={gradient}")
    
    # Create boundary points
    boundary_points = create_square_boundary(size=10, points_per_edge=points_per_edge)
    
    # Create interior grid points
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
    
    # Log some info
    logger.info(f"Triangulating square domain with {len(all_points)} points...")
    logger.info(f"Using base_size={base_size}, gradient={gradient}")
    
    # Plot input points
    plt.figure(figsize=(10, 10))
    plt.scatter(all_points[:, 0], all_points[:, 1], c='b', s=10)
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'r-')
    plt.axis('equal')
    plt.grid(True)
    plt.title(f'Square Domain Input (g={gradient})')
    plt.savefig('square_input.png')
    
    # Triangulate
    result = wrapper.triangulate(all_points, segments)
    
    if result is not None and 'triangles' in result:
        # Plot triangulation
        plt.figure(figsize=(12, 12))
        
        # Plot triangles
        plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
                    result['triangles'], 'b-', lw=0.5)
        
        # Plot original points
        plt.scatter(all_points[:, 0], all_points[:, 1], c='r', s=8, alpha=0.5)
        
        # Plot boundary
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'g-', lw=1.5)
        
        plt.axis('equal')
        plt.grid(True)
        plt.title(f'Square Domain Triangulation (g={gradient}, {len(result["triangles"])} triangles)')
        plt.savefig('square_triangulation.png')
        
        logger.info(f"Created {len(result['triangles'])} triangles with {len(result['vertices'])} vertices")
    else:
        logger.error("Triangulation failed")

if __name__ == "__main__":
    main() 