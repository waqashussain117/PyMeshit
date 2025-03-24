"""
Real mesh triangulation using Triangle library directly.

This script demonstrates proper triangulation using the Triangle library
with the custom wrapper to control mesh quality and density.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import os
import logging
import math
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealTriangle")

# Make sure triangulation results directory exists
RESULTS_DIR = "triangulation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_square_boundary(size=10.0, num_points=60):
    """
    Create a square boundary with evenly distributed points.
    
    Args:
        size: Half-size of the square (total width = 2*size)
        num_points: Total number of points on the boundary
        
    Returns:
        Array of boundary points
    """
    # Calculate points per side (making sure it's evenly distributed)
    points_per_side = num_points // 4
    
    # Create points for each side
    t = np.linspace(0, 1, points_per_side, endpoint=False)
    
    # Top side (left to right)
    top_x = -size + 2 * size * t
    top_y = np.ones_like(top_x) * size
    
    # Right side (top to bottom)
    right_y = size - 2 * size * t
    right_x = np.ones_like(right_y) * size
    
    # Bottom side (right to left) 
    bottom_x = size - 2 * size * t
    bottom_y = np.ones_like(bottom_x) * (-size)
    
    # Left side (bottom to top)
    left_y = -size + 2 * size * t
    left_x = np.ones_like(left_y) * (-size)
    
    # Combine all sides
    x = np.concatenate([top_x, right_x, bottom_x, left_x])
    y = np.concatenate([top_y, right_y, bottom_y, left_y])
    
    # Stack into 2D array
    boundary_points = np.column_stack([x, y])
    
    return boundary_points

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

def create_uniform_interior_points(size=10.0, spacing=1.0, jitter=0.0):
    """
    Create interior points with uniform spacing.
    
    Args:
        size: Half-size of the square (total width = 2*size)
        spacing: Spacing between points
        jitter: Amount of random displacement to apply (0.0 = grid)
        
    Returns:
        Array of interior points
    """
    # Calculate the number of points in each direction
    n_points = int(2 * size / spacing)
    
    # Create a grid of points
    x = np.linspace(-size + spacing, size - spacing, n_points)
    y = np.linspace(-size + spacing, size - spacing, n_points)
    xx, yy = np.meshgrid(x, y)
    
    # Add random jitter if requested
    if jitter > 0:
        xx += np.random.uniform(-jitter, jitter, xx.shape)
        yy += np.random.uniform(-jitter, jitter, yy.shape)
    
    # Reshape to n x 2 array
    points = np.column_stack([xx.flatten(), yy.flatten()])
    
    return points

def create_variable_density_points(size=10.0, min_spacing=0.5, max_spacing=2.0, features=None):
    """
    Create interior points with variable density.
    
    Args:
        size: Half-size of the square domain
        min_spacing: Minimum spacing between points (near features)
        max_spacing: Maximum spacing between points (away from features)
        features: List of feature points and their influence radiuses
        
    Returns:
        Array of interior points
    """
    if features is None:
        features = [
            {"pos": np.array([5, 5]), "radius": 3.0},
            {"pos": np.array([-5, -5]), "radius": 3.0},
        ]
    
    # Start with a coarse grid
    coarse_points = create_uniform_interior_points(size, max_spacing, jitter=0.2)
    
    # Function to determine local spacing at a point
    def get_local_spacing(p):
        # Start with maximum spacing
        local_spacing = max_spacing
        
        # Check distance to each feature
        for feature in features:
            dist = np.linalg.norm(p - feature["pos"])
            if dist < feature["radius"]:
                # Linear transition from min to max spacing
                t = dist / feature["radius"]
                spacing = min_spacing + (max_spacing - min_spacing) * t
                local_spacing = min(local_spacing, spacing)
        
        return local_spacing
    
    # Add more points where needed
    all_points = list(coarse_points)
    
    # Add dense points near features
    for feature in features:
        # Number of points based on feature size
        n_points = int(30 * (feature["radius"] / min_spacing))
        
        for _ in range(n_points):
            # Random angle
            angle = np.random.uniform(0, 2 * np.pi)
            # Random radius (more points closer to center)
            r = feature["radius"] * np.sqrt(np.random.uniform(0, 1))
            
            x = feature["pos"][0] + r * np.cos(angle)
            y = feature["pos"][1] + r * np.sin(angle)
            
            # Make sure inside domain
            if abs(x) < size - 0.5 and abs(y) < size - 0.5:
                all_points.append([x, y])
    
    return np.array(all_points)

def triangle_quality_options(min_angle=30, max_area=None, use_quality=True):
    """
    Generate Triangle option string.
    
    Args:
        min_angle: Minimum angle constraint (degrees)
        max_area: Maximum triangle area
        use_quality: Whether to apply quality constraints
        
    Returns:
        Option string for the Triangle library
    """
    options = 'pz'  # Preserve input (p), zero-indexing (z)
    
    if use_quality:
        options += f'q{min_angle}'  # Minimum angle
        
        if max_area is not None:
            options += f'a{max_area}'  # Maximum area
    
    return options

def triangulate_with_triangle(points, segments, options, plot=True, prefix='triangle'):
    """
    Triangulate using Triangle library.
    
    Args:
        points: Vertex coordinates
        segments: Boundary segments
        options: Triangle option string
        plot: Whether to create visualization plots
        prefix: Prefix for saved files
        
    Returns:
        Dict with triangulation results
    """
    # Create triangle input dictionary
    triangle_data = {
        'vertices': points,
        'segments': segments
    }
    
    # Run triangulation
    logger.info(f"Triangulating with options: {options}")
    result = tr.triangulate(triangle_data, options)
    
    # Log results
    logger.info(f"Created {len(result['triangles'])} triangles from {len(result['vertices'])} vertices")
    
    # Create visualization plots if requested
    if plot:
        # Plot input
        plt.figure(figsize=(10, 10))
        plt.scatter(points[:len(segments), 0], points[:len(segments), 1], 
                   c='r', s=30, label='Boundary Points')
        if len(points) > len(segments):
            plt.scatter(points[len(segments):, 0], points[len(segments):, 1], 
                       c='b', s=10, label='Interior Points')
        
        # Draw boundary
        for s in segments:
            plt.plot([points[s[0]][0], points[s[1]][0]], 
                     [points[s[0]][1], points[s[1]][1]], 'g-', lw=1.5)
        
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title(f"Input for Triangle ({len(points)} points)")
        plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_input.png"), dpi=300)
        plt.close()
        
        # Plot triangulation
        plt.figure(figsize=(10, 10))
        plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
                   result['triangles'], 'b-', lw=0.5)
        
        # Draw boundary
        boundary_points = points[:len(segments)]
        plt.plot(np.append(boundary_points[:, 0], boundary_points[0, 0]),
                np.append(boundary_points[:, 1], boundary_points[0, 1]), 
                'g-', lw=1.5)
        
        plt.axis('equal')
        plt.grid(True)
        plt.title(f"Triangle Mesh ({len(result['triangles'])} triangles)")
        plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_triangulation.png"), dpi=300)
        plt.close()
    
    return result

def analyze_triangle_quality(triangles, vertices):
    """
    Analyze the quality of triangles in the mesh.
    
    Args:
        triangles: Triangle indices
        vertices: Vertex coordinates
        
    Returns:
        Dict with quality metrics
    """
    n_triangles = len(triangles)
    min_angles = []
    max_angles = []
    areas = []
    aspect_ratios = []
    
    for tri in triangles:
        v1, v2, v3 = vertices[tri]
        
        # Calculate edge lengths
        e1 = np.linalg.norm(v2 - v3)
        e2 = np.linalg.norm(v1 - v3)
        e3 = np.linalg.norm(v1 - v2)
        
        # Calculate angles
        a1 = math.acos((e2**2 + e3**2 - e1**2) / (2 * e2 * e3))
        a2 = math.acos((e1**2 + e3**2 - e2**2) / (2 * e1 * e3))
        a3 = math.acos((e1**2 + e2**2 - e3**2) / (2 * e1 * e2))
        
        angles = [a1, a2, a3]
        min_angles.append(min(angles))
        max_angles.append(max(angles))
        
        # Calculate aspect ratio (ratio of longest to shortest edge)
        aspect_ratios.append(max(e1, e2, e3) / min(e1, e2, e3))
        
        # Calculate area using cross product
        v1_v2 = v2 - v1
        v1_v3 = v3 - v1
        area = 0.5 * abs(np.cross(v1_v2, v1_v3))
        areas.append(area)
    
    # Convert to degrees
    min_angles_deg = [a * 180 / math.pi for a in min_angles]
    max_angles_deg = [a * 180 / math.pi for a in max_angles]
    
    quality = {
        "num_triangles": n_triangles,
        "min_angle_min": min(min_angles_deg),
        "min_angle_mean": sum(min_angles_deg) / n_triangles,
        "max_angle_max": max(max_angles_deg),
        "max_angle_mean": sum(max_angles_deg) / n_triangles,
        "area_min": min(areas),
        "area_max": max(areas),
        "area_mean": sum(areas) / n_triangles,
        "aspect_ratio_min": min(aspect_ratios),
        "aspect_ratio_max": max(aspect_ratios),
        "aspect_ratio_mean": sum(aspect_ratios) / n_triangles
    }
    
    return quality

def run_high_quality_triangulation():
    """Create a high-quality triangulation with Triangle."""
    # Parameters
    domain_size = 10.0
    num_boundary_points = 60
    
    # Create boundary points and segments
    boundary_points = create_square_boundary(domain_size, num_boundary_points)
    segments = create_segments(len(boundary_points))
    
    # Create interior points with some randomness
    interior_points = create_uniform_interior_points(domain_size, spacing=1.0, jitter=0.4)
    
    # Combine boundary and interior points
    all_points = np.vstack([boundary_points, interior_points])
    
    # First pass: basic triangulation
    basic_options = triangle_quality_options(use_quality=False)
    basic_result = triangulate_with_triangle(all_points, segments, basic_options, 
                                          plot=True, prefix="basic")
    
    # Second pass: quality triangulation
    quality_options = triangle_quality_options(min_angle=30, max_area=1.0, use_quality=True)
    quality_result = triangulate_with_triangle(all_points, segments, quality_options, 
                                            plot=True, prefix="quality")
    
    # Analyze triangle quality
    basic_quality = analyze_triangle_quality(basic_result['triangles'], basic_result['vertices'])
    logger.info(f"Basic triangulation quality stats:")
    logger.info(f"  Triangles: {basic_quality['num_triangles']}")
    logger.info(f"  Min angle (min/mean): {basic_quality['min_angle_min']:.2f}° / {basic_quality['min_angle_mean']:.2f}°")
    logger.info(f"  Max angle (max/mean): {basic_quality['max_angle_max']:.2f}° / {basic_quality['max_angle_mean']:.2f}°")
    logger.info(f"  Area (min/max/mean): {basic_quality['area_min']:.2f} / {basic_quality['area_max']:.2f} / {basic_quality['area_mean']:.2f}")
    
    quality_quality = analyze_triangle_quality(quality_result['triangles'], quality_result['vertices'])
    logger.info(f"Quality triangulation quality stats:")
    logger.info(f"  Triangles: {quality_quality['num_triangles']}")
    logger.info(f"  Min angle (min/mean): {quality_quality['min_angle_min']:.2f}° / {quality_quality['min_angle_mean']:.2f}°")
    logger.info(f"  Max angle (max/mean): {quality_quality['max_angle_max']:.2f}° / {quality_quality['max_angle_mean']:.2f}°")
    logger.info(f"  Area (min/max/mean): {quality_quality['area_min']:.2f} / {quality_quality['area_max']:.2f} / {quality_quality['area_mean']:.2f}")
    
    return quality_result

def run_density_controlled_triangulation():
    """Create a triangulation with varying density."""
    # Parameters
    domain_size = 10.0
    num_boundary_points = 80
    
    # Define features (areas of higher density)
    features = [
        {"pos": np.array([5, 5]), "radius": 3.0},
        {"pos": np.array([-5, -5]), "radius": 3.0},
        {"pos": np.array([-5, 5]), "radius": 2.0},
        {"pos": np.array([5, -5]), "radius": 2.0},
    ]
    
    # Create boundary points and segments
    boundary_points = create_square_boundary(domain_size, num_boundary_points)
    segments = create_segments(len(boundary_points))
    
    # Create interior points with variable density
    interior_points = create_variable_density_points(domain_size, 
                                                   min_spacing=0.3, 
                                                   max_spacing=1.5,
                                                   features=features)
    
    # Combine boundary and interior points
    all_points = np.vstack([boundary_points, interior_points])
    
    # Run triangulation
    options = triangle_quality_options(min_angle=25, max_area=None, use_quality=True)
    result = triangulate_with_triangle(all_points, segments, options, 
                                      plot=True, prefix="density")
    
    # Create visualization with features
    plt.figure(figsize=(10, 10))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
               result['triangles'], 'b-', lw=0.5)
    
    # Draw boundary
    plt.plot(np.append(boundary_points[:, 0], boundary_points[0, 0]),
            np.append(boundary_points[:, 1], boundary_points[0, 1]), 
            'g-', lw=1.5)
    
    # Draw features
    for feature in features:
        circle = plt.Circle(feature["pos"], feature["radius"], 
                           fill=False, color='r', alpha=0.5, linestyle='--')
        plt.gca().add_patch(circle)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"Density-Controlled Triangulation ({len(result['triangles'])} triangles)")
    plt.savefig(os.path.join(RESULTS_DIR, "density_features.png"), dpi=300)
    plt.close()
    
    # Create zoomed-in views of dense regions
    for i, feature in enumerate(features):
        plt.figure(figsize=(8, 8))
        plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
                   result['triangles'], 'b-', lw=0.5)
        
        # Draw boundary of original square
        plt.plot(np.append(boundary_points[:, 0], boundary_points[0, 0]),
                np.append(boundary_points[:, 1], boundary_points[0, 1]), 
                'g-', lw=1.5)
        
        # Set limits to feature
        center = feature["pos"]
        radius = feature["radius"]
        plt.xlim(center[0] - radius - 1, center[0] + radius + 1)
        plt.ylim(center[1] - radius - 1, center[1] + radius + 1)
        
        # Draw feature circle
        circle = plt.Circle(feature["pos"], feature["radius"], 
                           fill=False, color='r', alpha=0.5, linestyle='--')
        plt.gca().add_patch(circle)
        
        plt.grid(True)
        plt.title(f"Zoom: Feature {i+1} Region")
        plt.savefig(os.path.join(RESULTS_DIR, f"density_zoom_{i+1}.png"), dpi=300)
        plt.close()
    
    return result

def run_steiner_points_test():
    """Test the effect of Steiner points in Triangle."""
    # Parameters
    domain_size = 10.0
    num_boundary_points = 40
    
    # Create boundary points and segments
    boundary_points = create_square_boundary(domain_size, num_boundary_points)
    segments = create_segments(len(boundary_points))
    
    # Create very sparse interior points
    interior_points = create_uniform_interior_points(domain_size, spacing=3.0, jitter=0.3)
    
    # Combine boundary and interior points
    all_points = np.vstack([boundary_points, interior_points])
    
    # Without Steiner points
    no_steiner_options = 'pzq30Y'  # Y = no Steiner points on boundary
    no_steiner_result = triangulate_with_triangle(all_points, segments, no_steiner_options, 
                                                plot=True, prefix="no_steiner")
    
    # With Steiner points
    steiner_options = 'pzq30'  # Allowing Steiner points
    steiner_result = triangulate_with_triangle(all_points, segments, steiner_options, 
                                             plot=True, prefix="with_steiner")
    
    # Compare quality
    no_steiner_quality = analyze_triangle_quality(no_steiner_result['triangles'], 
                                                no_steiner_result['vertices'])
    steiner_quality = analyze_triangle_quality(steiner_result['triangles'], 
                                             steiner_result['vertices'])
    
    logger.info("Triangle quality comparison: No Steiner vs With Steiner")
    logger.info(f"  No Steiner: {no_steiner_quality['num_triangles']} triangles, " + 
               f"min angle {no_steiner_quality['min_angle_min']:.2f}°")
    logger.info(f"  With Steiner: {steiner_quality['num_triangles']} triangles, " + 
               f"min angle {steiner_quality['min_angle_min']:.2f}°")
    
    return no_steiner_result, steiner_result

if __name__ == "__main__":
    # Run high-quality triangulation
    logger.info("Running high-quality triangulation test...")
    run_high_quality_triangulation()
    
    # Run density-controlled triangulation
    logger.info("\nRunning density-controlled triangulation test...")
    run_density_controlled_triangulation()
    
    # Run Steiner points test
    logger.info("\nRunning Steiner points test...")
    run_steiner_points_test()
    
    logger.info("\nAll triangulation tests complete!") 