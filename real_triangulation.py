"""
Real triangulation with proper triangular elements.

This script creates a true triangulation (not a diagonal grid pattern)
by using proper Delaunay triangulation with quality constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import os
import logging
import math

# Set up logging and output directory
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealTriangulation")
RESULTS_DIR = "triangulation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_boundary_points(domain_size=10, num_points=60):
    """
    Create points along the boundary of a square.
    
    Args:
        domain_size: Size of the square domain
        num_points: Total number of points on the boundary
    
    Returns:
        Array of boundary point coordinates
    """
    # Calculate points per side
    points_per_side = num_points // 4
    
    # Create arrays for each side
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

def create_well_spaced_interior(domain_size=10, min_spacing=1.0, jitter=0.2):
    """
    Create well-spaced interior points using a Poisson-disk like approach.
    
    Args:
        domain_size: Size of the square domain
        min_spacing: Minimum spacing between points
        jitter: Amount of randomness to add
        
    Returns:
        Array of interior point coordinates
    """
    # Margin to keep away from the boundary
    margin = 0.5
    
    # Compute number of cells in the grid
    cell_size = min_spacing / math.sqrt(2)
    grid_size = int(2 * (domain_size - margin) / cell_size) + 1
    
    # Initialize grid to track occupied cells
    grid = {}
    
    # List to store generated points
    points = []
    
    # Number of attempts to place a point near an existing one
    k = 30
    
    # Create random initial point
    rng = np.random.RandomState(42)  # For reproducibility
    x = rng.uniform(-domain_size + margin, domain_size - margin)
    y = rng.uniform(-domain_size + margin, domain_size - margin)
    initial_point = np.array([x, y])
    
    points.append(initial_point)
    
    # Add to grid
    ix = int((x + domain_size - margin) / cell_size)
    iy = int((y + domain_size - margin) / cell_size)
    grid[(ix, iy)] = 0  # Store index in the points list
    
    # Active list of points (indices)
    active = [0]
    
    # While there are active points
    while active:
        # Choose a random active point
        idx = rng.choice(len(active))
        active_idx = active[idx]
        active_point = points[active_idx]
        
        # Try to generate new points around the active point
        success = False
        for _ in range(k):
            # Generate random point at distance between min_spacing and 2*min_spacing
            theta = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(min_spacing, 2 * min_spacing)
            
            new_point = active_point + radius * np.array([np.cos(theta), np.sin(theta)])
            
            # Add jitter
            if jitter > 0:
                new_point += rng.uniform(-jitter * min_spacing, jitter * min_spacing, 2)
            
            # Check if the point is within the domain (with margin)
            if (np.abs(new_point[0]) >= domain_size - margin or 
                np.abs(new_point[1]) >= domain_size - margin):
                continue
                
            # Determine grid cell
            ix = int((new_point[0] + domain_size - margin) / cell_size)
            iy = int((new_point[1] + domain_size - margin) / cell_size)
            
            # Check neighborhood (3x3 cells)
            valid = True
            for nx in range(max(0, ix-1), min(grid_size, ix+2)):
                for ny in range(max(0, iy-1), min(grid_size, iy+2)):
                    if (nx, ny) in grid:
                        # Check distance to the point in this cell
                        neighbor_idx = grid[(nx, ny)]
                        neighbor = points[neighbor_idx]
                        if np.linalg.norm(new_point - neighbor) < min_spacing:
                            valid = False
                            break
                if not valid:
                    break
                    
            if valid:
                # Add the new point
                points.append(new_point)
                grid[(ix, iy)] = len(points) - 1
                active.append(len(points) - 1)
                success = True
                break
        
        # If we couldn't add new points, this active point is done
        if not success:
            active.pop(idx)
    
    return np.array(points)

def create_segments(boundary_points):
    """
    Create segments connecting boundary points in sequence.
    
    Args:
        boundary_points: Array of boundary point coordinates
        
    Returns:
        Array of segment indices
    """
    n_points = len(boundary_points)
    segments = []
    
    for i in range(n_points):
        segments.append([i, (i + 1) % n_points])
    
    return np.array(segments)

def triangulate_real(min_angle=30, max_area=1.0):
    """
    Perform real triangulation with quality constraints.
    
    Args:
        min_angle: Minimum angle constraint (degrees)
        max_area: Maximum triangle area constraint
        
    Returns:
        Triangulation result
    """
    # Domain parameters
    domain_size = 10
    
    # Create boundary points
    boundary_points = create_boundary_points(domain_size, num_points=60)
    logger.info(f"Created {len(boundary_points)} boundary points")
    
    # Create interior points with good spacing
    interior_points = create_well_spaced_interior(domain_size, min_spacing=1.0, jitter=0.3)
    logger.info(f"Created {len(interior_points)} interior points")
    
    # Create segments for boundary
    segments = create_segments(boundary_points)
    
    # Combine boundary and interior points
    all_points = np.vstack((boundary_points, interior_points))
    
    # Prepare triangulation input
    tri_input = {
        'vertices': all_points,
        'segments': segments
    }
    
    # First pass - basic conforming Delaunay without quality constraints
    basic_options = 'p'  # Preserves the PSLG (segments)
    logger.info(f"Initial triangulation with options: {basic_options}")
    
    basic_result = tr.triangulate(tri_input, basic_options)
    logger.info(f"Initial triangulation: {len(basic_result['triangles'])} triangles")
    
    # Second pass - add quality constraints
    quality_options = f'pq{min_angle}a{max_area}'
    logger.info(f"Quality triangulation with options: {quality_options}")
    
    quality_result = tr.triangulate(basic_result, quality_options)
    logger.info(f"Final triangulation: {len(quality_result['triangles'])} triangles")
    
    # Plot input points
    plt.figure(figsize=(12, 12))
    plt.scatter(interior_points[:, 0], interior_points[:, 1], c='b', s=20, alpha=0.6, label='Interior Points')
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='r', s=30, label='Boundary Points')
    
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f"Input Points for Real Triangulation")
    plt.savefig(os.path.join(RESULTS_DIR, "real_triangulation_input.png"), dpi=300)
    plt.close()
    
    # Plot initial triangulation
    plt.figure(figsize=(12, 12))
    plt.triplot(basic_result['vertices'][:, 0], basic_result['vertices'][:, 1],
               basic_result['triangles'], 'b-', lw=0.5)
    
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"Initial Delaunay Triangulation ({len(basic_result['triangles'])} triangles)")
    plt.savefig(os.path.join(RESULTS_DIR, "real_triangulation_initial.png"), dpi=300)
    plt.close()
    
    # Plot final triangulation
    plt.figure(figsize=(12, 12))
    plt.triplot(quality_result['vertices'][:, 0], quality_result['vertices'][:, 1],
               quality_result['triangles'], 'b-', lw=0.5)
    
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"Real Triangulation (min angle={min_angle}°, {len(quality_result['triangles'])} triangles)")
    plt.savefig(os.path.join(RESULTS_DIR, "real_triangulation.png"), dpi=300)
    plt.close()
    
    # Create zoomed view
    plt.figure(figsize=(12, 12))
    plt.triplot(quality_result['vertices'][:, 0], quality_result['vertices'][:, 1],
               quality_result['triangles'], 'b-', lw=0.5)
    
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    # Zoom to corner
    plt.xlim(5, 10)
    plt.ylim(5, 10)
    
    plt.grid(True)
    plt.title(f"Zoomed View - Real Triangulation")
    plt.savefig(os.path.join(RESULTS_DIR, "real_triangulation_zoomed.png"), dpi=300)
    plt.close()
    
    return quality_result

def triangulate_with_density_control(min_angle=30, quality_level='high'):
    """
    Create a triangulation with density control around specific features.
    
    Args:
        min_angle: Minimum angle constraint (degrees)
        quality_level: 'high', 'medium', or 'low'
        
    Returns:
        Triangulation result
    """
    # Domain parameters
    domain_size = 10
    
    # Set quality parameters based on level
    if quality_level == 'high':
        min_angle = 30
        max_area_global = 0.5
        max_area_feature = 0.2
    elif quality_level == 'medium':
        min_angle = 25
        max_area_global = 1.0
        max_area_feature = 0.4
    else:  # 'low'
        min_angle = 20
        max_area_global = 2.0
        max_area_feature = 0.8
    
    # Create boundary points
    boundary_points = create_boundary_points(domain_size, num_points=80)
    logger.info(f"Created {len(boundary_points)} boundary points")
    
    # Create interior points with good spacing
    base_interior_points = create_well_spaced_interior(domain_size, min_spacing=1.2, jitter=0.2)
    logger.info(f"Created {len(base_interior_points)} base interior points")
    
    # Define feature points (regions of interest where we want higher density)
    features = [
        {'position': np.array([5, 5]), 'radius': 3.0},
        {'position': np.array([-5, -5]), 'radius': 3.0},
        {'position': np.array([-5, 5]), 'radius': 2.0},
        {'position': np.array([5, -5]), 'radius': 2.0},
    ]
    
    # Add denser points around features
    feature_points = []
    for feature in features:
        # Number of points based on feature radius
        num_points = int(30 * feature['radius'])
        
        # Create denser points around the feature
        for _ in range(num_points):
            # Random angle
            angle = np.random.uniform(0, 2 * np.pi)
            # Random radius (more points closer to the center)
            r = feature['radius'] * np.sqrt(np.random.uniform(0, 1))
            
            # Calculate position
            x = feature['position'][0] + r * np.cos(angle)
            y = feature['position'][1] + r * np.sin(angle)
            
            # Ensure inside domain
            if abs(x) < domain_size - 0.5 and abs(y) < domain_size - 0.5:
                feature_points.append([x, y])
    
    feature_points = np.array(feature_points)
    logger.info(f"Created {len(feature_points)} additional feature points")
    
    # Combine all interior points
    interior_points = np.vstack((base_interior_points, feature_points))
    
    # Create segments for boundary
    segments = create_segments(boundary_points)
    
    # Combine boundary and interior points
    all_points = np.vstack((boundary_points, interior_points))
    
    # Define a hole for the triangulation (optional)
    # For example, a small hole at the center
    #hole = np.array([[0, 0]])
    
    # Prepare triangulation input
    tri_input = {
        'vertices': all_points,
        'segments': segments,
        #'holes': hole
    }
    
    # Triangulate with quality constraints
    quality_options = f'pq{min_angle}a{max_area_global}'
    logger.info(f"Triangulating with options: {quality_options}")
    
    result = tr.triangulate(tri_input, quality_options)
    logger.info(f"Created {len(result['triangles'])} triangles")
    
    # Plot input points
    plt.figure(figsize=(12, 12))
    plt.scatter(base_interior_points[:, 0], base_interior_points[:, 1], c='b', s=10, alpha=0.5, label='Base Points')
    plt.scatter(feature_points[:, 0], feature_points[:, 1], c='r', s=10, alpha=0.5, label='Feature Points')
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='g', s=20, label='Boundary Points')
    
    # Draw features
    for feature in features:
        circle = plt.Circle(feature['position'], feature['radius'], fill=False, color='orange', alpha=0.7)
        plt.gca().add_patch(circle)
    
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f"Input Points with Feature Regions")
    plt.savefig(os.path.join(RESULTS_DIR, "density_control_input.png"), dpi=300)
    plt.close()
    
    # Plot final triangulation
    plt.figure(figsize=(12, 12))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
               result['triangles'], 'b-', lw=0.5)
    
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    # Draw features lightly
    for feature in features:
        circle = plt.Circle(feature['position'], feature['radius'], fill=False, color='orange', alpha=0.3)
        plt.gca().add_patch(circle)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"Triangulation with Density Control ({len(result['triangles'])} triangles)")
    plt.savefig(os.path.join(RESULTS_DIR, f"density_control_{quality_level}.png"), dpi=300)
    plt.close()
    
    return result

if __name__ == "__main__":
    # Run the real triangulation
    triangulate_real(min_angle=30, max_area=1.0)
    
    # Run the density-controlled triangulation with different quality levels
    triangulate_with_density_control(quality_level='high') 