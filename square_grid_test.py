"""
Square Grid Test

This script creates a uniform grid triangulation on a square boundary.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Grid-Test")

try:
    from meshit.triangle_direct import DirectTriangleWrapper
    logger.info("Successfully imported DirectTriangleWrapper")
except ImportError as e:
    logger.error(f"Failed to import DirectTriangleWrapper: {e}")
    exit(1)

def create_square_boundary(size=100.0):
    """Create a square boundary with the given size"""
    # Create a square boundary
    points = np.array([
        [-size, -size],  # bottom-left
        [size, -size],   # bottom-right
        [size, size],    # top-right
        [-size, size]    # top-left
    ])
    
    # Create segments connecting adjacent points
    segments = np.array([
        [0, 1],  # bottom
        [1, 2],  # right
        [2, 3],  # top
        [3, 0]   # left
    ])
    
    return points, segments

def create_grid_points(size=100.0, spacing=20.0):
    """Create a grid of interior points"""
    # Determine the number of points in each direction
    n = int(2 * size / spacing)
    
    # Create a grid of points
    x = np.linspace(-size + spacing, size - spacing, n)
    y = np.linspace(-size + spacing, size - spacing, n)
    
    # Create a mesh grid
    X, Y = np.meshgrid(x, y)
    
    # Reshape to a list of points
    points = np.vstack([X.flatten(), Y.flatten()]).T
    
    return points

def run_triangulation(boundary_size=100.0, grid_spacing=20.0, min_angle=25.0, base_size_factor=5.0):
    """Run triangulation on a square with a grid of interior points"""
    # Create boundary
    boundary_points, segments = create_square_boundary(boundary_size)
    
    # Create grid points
    grid_points = create_grid_points(boundary_size, grid_spacing)
    
    # Combine boundary and grid points
    all_points = np.vstack([boundary_points, grid_points])
    
    # Calculate base size
    base_size = 2 * boundary_size / base_size_factor
    
    # Create a DirectTriangleWrapper
    triangulator = DirectTriangleWrapper(
        gradient=1.0,  # No gradient for uniform mesh
        min_angle=min_angle,
        base_size=base_size
    )
    
    print(f"Parameters: boundary_size={boundary_size}, base_size={base_size}, min_angle={min_angle}")
    
    # Run triangulation
    start_time = time.time()
    
    triangulation_result = triangulator.triangulate(
        points=all_points,
        segments=segments,
        uniform=True
    )
    
    elapsed_time = time.time() - start_time
    
    # Extract results
    vertices = triangulation_result['vertices']
    triangles = triangulation_result['triangles']
    
    print(f"Triangulation complete in {elapsed_time:.2f}s:")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Triangles: {len(triangles)}")
    
    # Calculate edge statistics
    edge_lengths = []
    for tri in triangles:
        v1, v2, v3 = tri
        p1, p2, p3 = vertices[v1], vertices[v2], vertices[v3]
        
        edge_lengths.extend([
            np.linalg.norm(p2 - p1),
            np.linalg.norm(p3 - p2),
            np.linalg.norm(p1 - p3)
        ])
    
    mean_edge = np.mean(edge_lengths)
    edge_std = np.std(edge_lengths)
    uniformity = edge_std / mean_edge
    
    print(f"  Mean edge length: {mean_edge:.4f}")
    print(f"  Edge std: {edge_std:.4f}")
    print(f"  Uniformity: {uniformity:.4f}")
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot triangles using triplot
    ax.triplot(vertices[:, 0], vertices[:, 1], triangles, 'b-', alpha=0.5, linewidth=0.5)
    
    # Plot boundary
    ax.plot(
        np.append(boundary_points[:, 0], boundary_points[0, 0]),
        np.append(boundary_points[:, 1], boundary_points[0, 1]),
        'r-', linewidth=2.5, label='Boundary'
    )
    
    # Plot grid points
    ax.scatter(grid_points[:, 0], grid_points[:, 1], c='green', alpha=0.3, s=10, label='Grid Points')
    
    # Set limits with some margin
    margin = 0.1 * boundary_size
    ax.set_xlim(-boundary_size - margin, boundary_size + margin)
    ax.set_ylim(-boundary_size - margin, boundary_size + margin)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Triangulation ({len(triangles)} triangles, {len(vertices)} vertices)')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')
    
    plt.savefig("square_grid_triangulation.png", dpi=150)
    plt.show()
    
    return triangulation_result, fig, ax

if __name__ == "__main__":
    # Run triangulation with default parameters
    result, fig, ax = run_triangulation(
        boundary_size=100.0,
        grid_spacing=20.0,
        min_angle=25.0,
        base_size_factor=5.0
    ) 