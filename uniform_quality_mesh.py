"""
Uniform quality triangular mesh generator.

This script creates high-quality, uniform triangular meshes by using
Poisson-disk sampling for point distribution and strict quality
constraints in the Triangle library.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import os
import logging
import math
import random
from scipy.spatial import cKDTree

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UniformMesh")

# Make sure output directory exists
RESULTS_DIR = "triangulation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def poisson_disk_sampling(width=20.0, r=1.0, k=30, margin=0.0, seed=42):
    """
    Generate points using Poisson-disk sampling for highly uniform distribution.
    
    This is Bridson's algorithm for Poisson-disk sampling, which ensures
    no two points are closer than distance r, creating a very regular pattern.
    
    Args:
        width: Width of the domain (centered at origin)
        r: Minimum distance between points
        k: Number of attempts to place new points near existing ones
        margin: Margin to avoid boundary
        seed: Random seed for reproducibility
        
    Returns:
        Array of point coordinates
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Adjusted dimensions with margin
    domain_min = -width/2 + margin
    domain_max = width/2 - margin
    domain_size = domain_max - domain_min
    
    # Cell size for acceleration grid
    cell_size = r / np.sqrt(2)
    grid_size = int(domain_size / cell_size) + 1
    
    # Initialize grid (mapping grid coordinates to point indices)
    grid = {}
    
    # List of generated points
    points = []
    
    # Active list (points that still have space around them)
    active = []
    
    # Initial point
    first_point = np.array([
        random.uniform(domain_min, domain_max),
        random.uniform(domain_min, domain_max)
    ])
    
    points.append(first_point)
    active.append(0)
    
    # Add to grid
    cell_x = int((first_point[0] - domain_min) / cell_size)
    cell_y = int((first_point[1] - domain_min) / cell_size)
    grid[(cell_x, cell_y)] = 0
    
    # Main loop
    while active:
        # Pick a random active point
        active_idx = random.choice(active)
        active_point = points[active_idx]
        
        # Try to find a valid new point around this point
        found = False
        
        for _ in range(k):
            # Generate random point at distance between r and 2r
            theta = random.uniform(0, 2 * np.pi)
            radius = random.uniform(r, 2 * r)
            
            new_point = active_point + radius * np.array([np.cos(theta), np.sin(theta)])
            
            # Check if in domain
            if (new_point[0] < domain_min or new_point[0] > domain_max or
                new_point[1] < domain_min or new_point[1] > domain_max):
                continue
            
            # Get grid position
            cell_x = int((new_point[0] - domain_min) / cell_size)
            cell_y = int((new_point[1] - domain_min) / cell_size)
            
            # Check if point is valid (not too close to existing points)
            valid = True
            
            # Check nearby cells
            for nx in range(max(0, cell_x-2), min(grid_size, cell_x+3)):
                for ny in range(max(0, cell_y-2), min(grid_size, cell_y+3)):
                    if (nx, ny) in grid:
                        existing_point_idx = grid[(nx, ny)]
                        existing_point = points[existing_point_idx]
                        
                        # Check distance
                        if np.linalg.norm(new_point - existing_point) < r:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Add new point
                points.append(new_point)
                new_idx = len(points) - 1
                active.append(new_idx)
                grid[(cell_x, cell_y)] = new_idx
                found = True
                break
        
        if not found:
            # Remove point from active list
            active.remove(active_idx)
    
    return np.array(points)

def create_boundary_points(width=20.0, spacing=0.5):
    """
    Create evenly spaced points along a square boundary.
    
    Args:
        width: Width of the square
        spacing: Spacing between boundary points
        
    Returns:
        Array of boundary point coordinates
    """
    half_width = width / 2
    
    # Calculate number of points per side (minimum 4 corners)
    points_per_side = max(2, int(width / spacing))
    
    # Create points on each side
    t = np.linspace(-half_width, half_width, points_per_side)
    
    top = np.column_stack([t, np.ones_like(t) * half_width])
    right = np.column_stack([np.ones_like(t) * half_width, np.flip(t)])
    bottom = np.column_stack([np.flip(t), np.ones_like(t) * -half_width])
    left = np.column_stack([np.ones_like(t) * -half_width, t])
    
    # Remove duplicated corners
    boundary = np.vstack([top[:-1], right[:-1], bottom[:-1], left[:-1]])
    
    return boundary

def create_segments(num_points):
    """
    Create segments connecting consecutive boundary points.
    
    Args:
        num_points: Number of boundary points
        
    Returns:
        Array of segment indices
    """
    segments = np.column_stack([
        np.arange(num_points),
        np.roll(np.arange(num_points), -1)
    ])
    return segments

def optimal_triangulation(width=20.0, point_spacing=1.0, min_angle=33, max_area=None):
    """
    Generate a highly uniform triangular mesh with optimal quality.
    
    Args:
        width: Width of the domain
        point_spacing: Spacing between interior points
        min_angle: Minimum angle constraint (degrees)
        max_area: Maximum triangle area constraint
        
    Returns:
        Dict with triangulation results
    """
    # Create boundary points with smaller spacing for better boundary representation
    boundary_points = create_boundary_points(width, spacing=point_spacing*0.5)
    logger.info(f"Created {len(boundary_points)} boundary points")
    
    # Create interior points using Poisson-disk sampling
    # Use a margin to avoid placing points too close to boundary
    margin = point_spacing * 0.51
    interior_points = poisson_disk_sampling(width, r=point_spacing, 
                                          margin=margin, seed=42)
    logger.info(f"Created {len(interior_points)} interior points using Poisson-disk sampling")
    
    # Create boundary segments
    segments = create_segments(len(boundary_points))
    
    # Combine points
    all_points = np.vstack([boundary_points, interior_points])
    
    # Create Triangle options string
    triangle_opts = 'pzq' + str(min_angle)
    if max_area is not None:
        triangle_opts += 'a' + str(max_area)
    
    # Prepare input for Triangle
    triangle_data = {
        'vertices': all_points,
        'segments': segments
    }
    
    # Run triangulation
    logger.info(f"Triangulating with options: {triangle_opts}")
    result = tr.triangulate(triangle_data, triangle_opts)
    
    num_triangles = len(result['triangles'])
    num_vertices = len(result['vertices'])
    logger.info(f"Created {num_triangles} triangles and {num_vertices} vertices")
    
    # Create visualization
    plt.figure(figsize=(12, 12))
    
    # Plot triangulation
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
               result['triangles'], 'b-', lw=0.5)
    
    # Draw boundary
    plt.plot(np.append(boundary_points[:, 0], boundary_points[0, 0]),
            np.append(boundary_points[:, 1], boundary_points[0, 1]), 
            'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"Uniform Triangular Mesh ({num_triangles} triangles)")
    plt.savefig(os.path.join(RESULTS_DIR, "uniform_optimal_mesh.png"), dpi=300)
    plt.close()
    
    # Plot input points
    plt.figure(figsize=(12, 12))
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], 
               c='r', s=20, label='Boundary Points')
    plt.scatter(interior_points[:, 0], interior_points[:, 1], 
               c='b', s=10, alpha=0.7, label='Interior Points')
    
    # Draw boundary
    plt.plot(np.append(boundary_points[:, 0], boundary_points[0, 0]),
            np.append(boundary_points[:, 1], boundary_points[0, 1]), 
            'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f"Input Points for Triangulation")
    plt.savefig(os.path.join(RESULTS_DIR, "uniform_optimal_input.png"), dpi=300)
    plt.close()
    
    return result

def analyze_mesh_quality(result):
    """
    Analyze and visualize mesh quality metrics.
    
    Args:
        result: Triangle result dictionary with 'vertices' and 'triangles'
        
    Returns:
        Dict with quality metrics
    """
    vertices = result['vertices']
    triangles = result['triangles']
    num_triangles = len(triangles)
    
    # Calculate quality metrics
    min_angles = []
    max_angles = []
    aspect_ratios = []
    areas = []
    
    for tri in triangles:
        v1, v2, v3 = vertices[tri]
        
        # Calculate edge lengths
        e1 = np.linalg.norm(v2 - v3)
        e2 = np.linalg.norm(v1 - v3)
        e3 = np.linalg.norm(v1 - v2)
        
        # Calculate angles
        try:
            a1 = math.acos(min(1.0, max(-1.0, (e2**2 + e3**2 - e1**2) / (2 * e2 * e3))))
            a2 = math.acos(min(1.0, max(-1.0, (e1**2 + e3**2 - e2**2) / (2 * e1 * e3))))
            a3 = math.acos(min(1.0, max(-1.0, (e1**2 + e2**2 - e3**2) / (2 * e1 * e2))))
            
            angles = [a1, a2, a3]
            min_angles.append(min(angles))
            max_angles.append(max(angles))
        except:
            logger.warning(f"Error calculating angles for triangle {tri}")
            continue
            
        # Calculate aspect ratio (ratio of longest to shortest edge)
        if min(e1, e2, e3) > 0:
            aspect_ratios.append(max(e1, e2, e3) / min(e1, e2, e3))
        
        # Calculate area using cross product
        v1_v2 = v2 - v1
        v1_v3 = v3 - v1
        # Handles 2D vectors safely
        if len(v1_v2) == 2:
            area = 0.5 * abs(v1_v2[0]*v1_v3[1] - v1_v2[1]*v1_v3[0])
        else:
            area = 0.5 * abs(np.cross(v1_v2, v1_v3))
        areas.append(area)
    
    # Convert angles to degrees
    min_angles_deg = [a * 180 / math.pi for a in min_angles]
    max_angles_deg = [a * 180 / math.pi for a in max_angles]
    
    # Calculate statistics
    quality = {
        "num_triangles": num_triangles,
        "min_angle_min": min(min_angles_deg),
        "min_angle_mean": sum(min_angles_deg) / num_triangles,
        "max_angle_max": max(max_angles_deg),
        "max_angle_mean": sum(max_angles_deg) / num_triangles,
        "area_min": min(areas),
        "area_max": max(areas),
        "area_mean": sum(areas) / num_triangles,
        "area_std": np.std(areas),
        "aspect_ratio_min": min(aspect_ratios),
        "aspect_ratio_max": max(aspect_ratios),
        "aspect_ratio_mean": sum(aspect_ratios) / num_triangles
    }
    
    # Log statistics
    logger.info("Mesh quality statistics:")
    logger.info(f"  Number of triangles: {quality['num_triangles']}")
    logger.info(f"  Min angle (min/mean): {quality['min_angle_min']:.2f}° / {quality['min_angle_mean']:.2f}°")
    logger.info(f"  Max angle (max/mean): {quality['max_angle_max']:.2f}° / {quality['max_angle_mean']:.2f}°")
    logger.info(f"  Triangle area (min/max/mean): {quality['area_min']:.4f} / {quality['area_max']:.4f} / {quality['area_mean']:.4f}")
    logger.info(f"  Area std deviation: {quality['area_std']:.4f} (uniformity measure)")
    logger.info(f"  Aspect ratio (min/max/mean): {quality['aspect_ratio_min']:.2f} / {quality['aspect_ratio_max']:.2f} / {quality['aspect_ratio_mean']:.2f}")
    
    # Create histograms for visualization
    
    # Minimum angle histogram
    plt.figure(figsize=(10, 6))
    plt.hist(min_angles_deg, bins=30, alpha=0.7, color='blue')
    plt.axvline(x=quality['min_angle_mean'], color='r', linestyle='--', 
               label=f"Mean: {quality['min_angle_mean']:.2f}°")
    plt.xlabel("Minimum Angle (degrees)")
    plt.ylabel("Count")
    plt.title("Distribution of Minimum Triangle Angles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "uniform_mesh_min_angles.png"), dpi=300)
    plt.close()
    
    # Area histogram
    plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=30, alpha=0.7, color='green')
    plt.axvline(x=quality['area_mean'], color='r', linestyle='--',
               label=f"Mean: {quality['area_mean']:.4f}")
    plt.xlabel("Triangle Area")
    plt.ylabel("Count")
    plt.title("Distribution of Triangle Areas")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "uniform_mesh_areas.png"), dpi=300)
    plt.close()
    
    # Aspect ratio histogram
    plt.figure(figsize=(10, 6))
    plt.hist(aspect_ratios, bins=30, alpha=0.7, color='purple')
    plt.axvline(x=quality['aspect_ratio_mean'], color='r', linestyle='--',
               label=f"Mean: {quality['aspect_ratio_mean']:.2f}")
    plt.xlabel("Aspect Ratio (max edge / min edge)")
    plt.ylabel("Count")
    plt.title("Distribution of Triangle Aspect Ratios")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "uniform_mesh_aspect_ratios.png"), dpi=300)
    plt.close()
    
    # Create color-coded quality map
    plt.figure(figsize=(12, 12))
    
    # Create triangulation with color based on minimum angle
    triang = plt.tripcolor(vertices[:, 0], vertices[:, 1], triangles, 
                          min_angles_deg, cmap='viridis', vmin=20, vmax=60)
    plt.colorbar(triang, label="Minimum Angle (degrees)")
    
    # Draw boundary
    boundary_points = vertices[:len(create_segments(len(vertices)))]
    plt.plot(np.append(boundary_points[:, 0], boundary_points[0, 0]),
            np.append(boundary_points[:, 1], boundary_points[0, 1]), 
            'k-', lw=1.0)
    
    plt.axis('equal')
    plt.title("Mesh Quality Visualization (color = minimum angle)")
    plt.savefig(os.path.join(RESULTS_DIR, "uniform_mesh_quality_map.png"), dpi=300)
    plt.close()
    
    return quality

def create_honeycomb_points(width=20.0, spacing=1.0, jitter=0.0):
    """
    Create interior points in a honeycomb (hexagonal) pattern for optimal packing.
    
    Args:
        width: Width of the domain
        spacing: Base spacing between points
        jitter: Random perturbation factor (0.0 = perfect grid)
        
    Returns:
        Array of point coordinates
    """
    half_width = width / 2.0
    
    # Honeycomb pattern has two axes:
    # - x axis with spacing 'spacing'
    # - y axis with spacing 'spacing * sqrt(3)/2'
    x_spacing = spacing
    y_spacing = spacing * np.sqrt(3) / 2
    
    # Calculate number of points in each dimension
    nx = int(width / x_spacing) + 2
    ny = int(width / y_spacing) + 2
    
    points = []
    for i in range(-nx//2, nx//2 + 1):
        for j in range(-ny//2, ny//2 + 1):
            # Offset every other row
            if j % 2 == 0:
                x = i * x_spacing
            else:
                x = (i + 0.5) * x_spacing
                
            y = j * y_spacing
            
            # Apply jitter if requested
            if jitter > 0:
                x += np.random.uniform(-jitter * spacing, jitter * spacing)
                y += np.random.uniform(-jitter * spacing, jitter * spacing)
            
            # Add point if inside domain (with margin)
            if abs(x) < half_width - 0.01 and abs(y) < half_width - 0.01:
                points.append([x, y])
    
    return np.array(points)

def generate_uniform_mesh(pattern='poisson', width=20.0, spacing=1.0, min_angle=33):
    """
    Generate a uniform triangular mesh using different point patterns.
    
    Args:
        pattern: Point distribution pattern ('poisson', 'honeycomb', or 'grid')
        width: Width of the domain
        spacing: Spacing between points
        min_angle: Minimum angle constraint
        
    Returns:
        Dict with triangulation results
    """
    # Create boundary points
    boundary_points = create_boundary_points(width, spacing=spacing*0.5)
    logger.info(f"Created {len(boundary_points)} boundary points")
    
    # Create interior points based on selected pattern
    if pattern == 'poisson':
        logger.info("Using Poisson-disk sampling for point distribution")
        interior_points = poisson_disk_sampling(width, r=spacing, 
                                              margin=spacing*0.51, seed=42)
    elif pattern == 'honeycomb':
        logger.info("Using honeycomb pattern for point distribution")
        interior_points = create_honeycomb_points(width, spacing=spacing, jitter=0.05)
    else:  # grid
        logger.info("Using grid pattern for point distribution")
        x = np.arange(-width/2 + spacing, width/2 - spacing, spacing)
        y = np.arange(-width/2 + spacing, width/2 - spacing, spacing)
        xx, yy = np.meshgrid(x, y)
        interior_points = np.column_stack([xx.flatten(), yy.flatten()])
        
        # Add small jitter to avoid numerical issues
        interior_points += np.random.uniform(-0.05*spacing, 0.05*spacing, 
                                          size=interior_points.shape)
    
    logger.info(f"Created {len(interior_points)} interior points")
    
    # Create boundary segments
    segments = create_segments(len(boundary_points))
    
    # Combine points
    all_points = np.vstack([boundary_points, interior_points])
    
    # Create Triangle options
    triangle_opts = f'pzq{min_angle}'
    
    # Prepare triangle input data
    triangle_data = {
        'vertices': all_points,
        'segments': segments
    }
    
    # Run triangulation
    logger.info(f"Triangulating with options: {triangle_opts}")
    result = tr.triangulate(triangle_data, triangle_opts)
    
    # Log results
    num_triangles = len(result['triangles'])
    num_vertices = len(result['vertices'])
    logger.info(f"Created {num_triangles} triangles and {num_vertices} vertices")
    
    # Create visualization
    plt.figure(figsize=(12, 12))
    
    # Plot triangulation
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
               result['triangles'], 'b-', lw=0.5)
    
    # Draw boundary
    plt.plot(np.append(boundary_points[:, 0], boundary_points[0, 0]),
            np.append(boundary_points[:, 1], boundary_points[0, 1]), 
            'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"Uniform Triangular Mesh - {pattern.capitalize()} ({num_triangles} triangles)")
    plt.savefig(os.path.join(RESULTS_DIR, f"uniform_{pattern}_mesh.png"), dpi=300)
    plt.close()
    
    # Plot input points
    plt.figure(figsize=(12, 12))
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], 
               c='r', s=20, label='Boundary Points')
    plt.scatter(interior_points[:, 0], interior_points[:, 1], 
               c='b', s=10, alpha=0.7, label='Interior Points')
    
    # Draw boundary
    plt.plot(np.append(boundary_points[:, 0], boundary_points[0, 0]),
            np.append(boundary_points[:, 1], boundary_points[0, 1]), 
            'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f"Input Points - {pattern.capitalize()} Distribution")
    plt.savefig(os.path.join(RESULTS_DIR, f"uniform_{pattern}_input.png"), dpi=300)
    plt.close()
    
    return result

if __name__ == "__main__":
    logger.info("Starting uniform quality mesh generation")
    
    # Generate mesh with optimal quality
    result = optimal_triangulation(width=20.0, point_spacing=1.0, min_angle=33)
    
    # Analyze mesh quality
    analyze_mesh_quality(result)
    
    # Generate meshes with different patterns for comparison
    poisson_result = generate_uniform_mesh(pattern='poisson', width=20.0, spacing=1.0)
    honeycomb_result = generate_uniform_mesh(pattern='honeycomb', width=20.0, spacing=1.0)
    grid_result = generate_uniform_mesh(pattern='grid', width=20.0, spacing=1.0)
    
    logger.info("Mesh generation complete") 