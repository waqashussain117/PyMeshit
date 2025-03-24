"""
Hybrid triangulation using direct Triangle C++ triunsuitable callback.

This script implements a more direct approach to MeshIt-style triangulation by
using the Triangle library with direct C++ callback integration.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import math
import os
import logging
import random
import time
import sys

# Try to import the triangle callback module
try:
    from meshit.triangle_callback import initialize_gradient_control
    HAVE_CALLBACK = True
except ImportError:
    HAVE_CALLBACK = False
    print("Warning: Triangle callback module not available. Using pure Python implementation.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HybridTriangulation")

# Make sure output directory exists
RESULTS_DIR = "hybrid_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_nonuniform_polygon_boundary(num_sides=5, radius=10.0, num_points_per_side=10):
    """
    Create a polygon boundary with the specified number of sides with non-uniform distribution.
    
    Args:
        num_sides: Number of sides for the polygon
        radius: Base radius of the circumscribed circle
        num_points_per_side: Number of points to distribute along each side
        
    Returns:
        Array of boundary point coordinates
    """
    # Create the corner vertices of the polygon with slight randomization
    corners = []
    for i in range(num_sides):
        # Add some randomness to the radius
        rand_radius = radius * (0.9 + 0.2 * random.random())
        angle = 2 * math.pi * i / num_sides + (math.pi/2)  # Start from top
        x = rand_radius * math.cos(angle)
        y = rand_radius * math.sin(angle)
        corners.append((x, y))
    
    # Add points along each side with non-uniform distribution
    boundary_points = []
    for i in range(num_sides):
        start = corners[i]
        end = corners[(i + 1) % num_sides]
        
        # Add corner point
        boundary_points.append(start)
        
        # Generate intermediate points along this side
        for j in range(1, num_points_per_side - 1):
            # Linear distribution for hybrid approach
            t = j / (num_points_per_side - 1)
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            
            # Add slight randomization
            perturb = 0.05 * radius
            x += random.uniform(-perturb, perturb)
            y += random.uniform(-perturb, perturb)
            
            boundary_points.append((x, y))
    
    return np.array(boundary_points)

def create_segments(boundary_points):
    """Create segments connecting consecutive boundary points."""
    num_points = len(boundary_points)
    return np.column_stack([
        np.arange(num_points),
        np.roll(np.arange(num_points), -1)
    ])

def compute_hulls(points):
    """
    Compute both the convex hull and a smoothed hull for better triangulation.
    
    Returns:
        Dictionary with hull information
    """
    # For a simple pentagon, we'll use the boundary points as the hull
    hull = points
    segments = create_segments(hull)
    
    return {
        'hull_points': hull,
        'segments': segments
    }

def create_feature_points(boundary, base_size, num_features=8):
    """
    Create feature points to guide mesh refinement.
    
    Args:
        boundary: Boundary points
        base_size: Base mesh size
        num_features: Number of feature points
        
    Returns:
        Tuple of (feature points, feature sizes)
    """
    # Calculate centroid
    centroid = np.mean(boundary, axis=0)
    
    # Find maximum distance from centroid to any boundary point
    max_dist = max([np.linalg.norm(p - centroid) for p in boundary])
    
    feature_points = []
    feature_sizes = []
    
    # Create feature points at strategic locations
    
    # Center feature
    feature_points.append(centroid)
    # Larger size at center
    feature_sizes.append(base_size * 0.8)
    
    # Boundary features
    num_boundary_features = num_features - 1
    indices = np.linspace(0, len(boundary) - 1, num_boundary_features, dtype=int)
    
    for idx in indices:
        point = boundary[idx]
        # Offset slightly inward
        vec = centroid - point
        vec_len = np.linalg.norm(vec)
        if vec_len > 1e-8:
            normalized = vec / vec_len
            # Move 10% inward
            inward_pt = point + normalized * (0.1 * vec_len)
            feature_points.append(inward_pt)
            # Smaller size near boundary
            feature_sizes.append(base_size * 0.3)
    
    return np.array(feature_points), np.array(feature_sizes)

def hybrid_triangulation(points, segments, gradient=2.0, feature_points=None, feature_sizes=None):
    """
    Hybrid triangulation approach combining standard Triangle with C++ callback.
    
    Args:
        points: Input points
        segments: Boundary segments
        gradient: Gradient parameter
        feature_points: Feature points (optional)
        feature_sizes: Feature sizes (optional)
        
    Returns:
        Triangulation results
    """
    # Calculate base size from bounding box diagonal
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    diagonal = np.sqrt(np.sum((max_coords - min_coords)**2))
    base_size = diagonal / 15.0  # MeshIt's standard scaling
    
    # Adjust quality parameter based on gradient as in MeshIt
    min_angle = 20.0
    if gradient > 1.0:
        min_angle = max(20.0 - (gradient - 1.0) * 5.0, 10.0)
    
    # Area constraint
    area_constraint = base_size * base_size * 0.5
    
    # Setup input for Triangle
    tri_input = {
        'vertices': points,
        'segments': segments
    }
    
    if HAVE_CALLBACK and feature_points is not None and feature_sizes is not None:
        # Use the C++ triunsuitable callback
        logger.info(f"Using C++ triunsuitable callback with gradient {gradient}")
        
        # Initialize the C++ GradientControl with our feature points
        initialize_gradient_control(
            gradient,
            base_size * base_size,  # Square of base size
            feature_points,
            feature_sizes
        )
        
        # Triangle options
        # p = PSLG (use segments)
        # q = quality constraint
        # a = area constraint
        # z = number vertices from zero
        # u = use custom triunsuitable function via callback
        options = f'pzq{min_angle}a{area_constraint}u'
        
        logger.info(f"Triangle options: {options}")
        
        try:
            # Run triangulation with C++ callback
            result = tr.triangulate(tri_input, options)
            logger.info(f"Triangulation completed with {len(result['triangles'])} triangles")
            return result
        except Exception as e:
            logger.error(f"Error using C++ callback: {e}")
            logger.info("Falling back to standard Triangle")
    
    # If C++ callback failed or not available, use standard Triangle
    logger.info(f"Using standard Triangle with options: pzq{min_angle}a{area_constraint}")
    result = tr.triangulate(tri_input, f'pzq{min_angle}a{area_constraint}')
    logger.info(f"Standard triangulation completed with {len(result['triangles'])} triangles")
    return result

def run_hybrid_triangulation():
    """Run hybrid triangulation tests with different gradient values."""
    # Generate polygon boundary
    boundary_points = create_nonuniform_polygon_boundary(
        num_sides=5,
        radius=10.0,
        num_points_per_side=12
    )
    logger.info(f"Created {len(boundary_points)} boundary points")
    
    # Compute hulls and segments
    hull_data = compute_hulls(boundary_points)
    hull_points = hull_data['hull_points']
    segments = hull_data['segments']
    
    # Calculate base size
    min_coords = np.min(hull_points, axis=0)
    max_coords = np.max(hull_points, axis=0)
    diagonal = np.sqrt(np.sum((max_coords - min_coords)**2))
    base_size = diagonal / 15.0
    
    # Create feature points
    feature_points, feature_sizes = create_feature_points(
        hull_points, 
        base_size,
        num_features=8
    )
    logger.info(f"Created {len(feature_points)} feature points")
    
    # Plot the input
    plt.figure(figsize=(12, 12))
    plt.scatter(hull_points[:, 0], hull_points[:, 1], c='red', s=30)
    plt.scatter(feature_points[:, 0], feature_points[:, 1], c='green', s=40, marker='*')
    
    # Draw segments
    for seg in segments:
        plt.plot(
            [hull_points[seg[0]][0], hull_points[seg[1]][0]],
            [hull_points[seg[0]][1], hull_points[seg[1]][1]],
            'r-', lw=1.0
        )
    
    plt.axis('equal')
    plt.title(f"Input for Hybrid Triangulation")
    plt.savefig(os.path.join(RESULTS_DIR, "hybrid_input.png"), dpi=300)
    plt.close()
    
    # Run triangulation with different gradient values
    gradient_values = [1.0, 2.0, 3.0]
    
    for gradient in gradient_values:
        # Run triangulation
        result = hybrid_triangulation(
            hull_points,
            segments,
            gradient=gradient,
            feature_points=feature_points,
            feature_sizes=feature_sizes
        )
        
        # Plot result
        plt.figure(figsize=(12, 12))
        
        # Plot triangles
        plt.triplot(
            result['vertices'][:, 0],
            result['vertices'][:, 1],
            result['triangles'],
            'b-', lw=0.5
        )
        
        # Mark boundary points
        plt.scatter(hull_points[:, 0], hull_points[:, 1], c='red', s=15)
        
        # Mark feature points
        plt.scatter(feature_points[:, 0], feature_points[:, 1], c='green', s=20, marker='*')
        
        plt.axis('equal')
        plt.title(f"Hybrid Triangle (g={gradient}, {len(result['triangles'])} triangles)")
        plt.savefig(os.path.join(RESULTS_DIR, f"hybrid_triangle_g{gradient}.png"), dpi=300)
        plt.close()
    
    return hull_points, feature_points, feature_sizes, result

def run_densified_hybrid_triangulation():
    """
    Run hybrid triangulation with densified boundaries and interior points.
    This helps to better match MeshIt's results by providing more
    points for the triangulation.
    """
    # Generate dense polygon boundary
    boundary_points = create_nonuniform_polygon_boundary(
        num_sides=5,
        radius=10.0,
        num_points_per_side=20  # Increased density
    )
    logger.info(f"Created {len(boundary_points)} dense boundary points")
    
    # Create segments
    segments = create_segments(boundary_points)
    
    # Add some interior points with random distribution
    # This helps the triangulation match MeshIt's results
    centroid = np.mean(boundary_points, axis=0)
    max_radius = max([np.linalg.norm(p - centroid) for p in boundary_points])
    
    # Create interior points with random distribution
    num_interior = 300
    interior_points = []
    for _ in range(num_interior):
        # Random angle and distance from center
        angle = 2 * math.pi * random.random()
        # Random distance (use power law distribution to get more points near center)
        distance = max_radius * random.random()**0.5 * 0.9
        
        x = centroid[0] + distance * math.cos(angle)
        y = centroid[1] + distance * math.sin(angle)
        
        # Add some jitter
        x += random.uniform(-0.1, 0.1) * max_radius
        y += random.uniform(-0.1, 0.1) * max_radius
        
        interior_points.append([x, y])
    
    # Combine boundary and interior points
    interior_points = np.array(interior_points)
    all_points = np.vstack([boundary_points, interior_points])
    logger.info(f"Total points: {len(all_points)}")
    
    # Calculate base size
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    diagonal = np.sqrt(np.sum((max_coords - min_coords)**2))
    base_size = diagonal / 15.0
    
    # Create feature points
    feature_points, feature_sizes = create_feature_points(
        boundary_points, 
        base_size,
        num_features=8
    )
    logger.info(f"Created {len(feature_points)} feature points")
    
    # Plot the input
    plt.figure(figsize=(12, 12))
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='red', s=15)
    plt.scatter(interior_points[:, 0], interior_points[:, 1], c='blue', s=5, alpha=0.5)
    plt.scatter(feature_points[:, 0], feature_points[:, 1], c='green', s=30, marker='*')
    
    # Draw segments
    for seg in segments:
        plt.plot(
            [boundary_points[seg[0]][0], boundary_points[seg[1]][0]],
            [boundary_points[seg[0]][1], boundary_points[seg[1]][1]],
            'r-', lw=0.8
        )
    
    plt.axis('equal')
    plt.title(f"Input for Dense Hybrid Triangulation")
    plt.savefig(os.path.join(RESULTS_DIR, "hybrid_dense_input.png"), dpi=300)
    plt.close()
    
    # Create new segments that only connect boundary points (not interior)
    segments_with_interior = np.column_stack([
        np.arange(len(boundary_points)),
        np.roll(np.arange(len(boundary_points)), -1)
    ])
    
    # Run triangulation with different gradient values
    gradient_values = [1.0, 2.0, 3.0]
    
    for gradient in gradient_values:
        # Run triangulation
        result = hybrid_triangulation(
            all_points,
            segments_with_interior,
            gradient=gradient,
            feature_points=feature_points,
            feature_sizes=feature_sizes
        )
        
        # Plot result
        plt.figure(figsize=(12, 12))
        
        # Plot triangles
        plt.triplot(
            result['vertices'][:, 0],
            result['vertices'][:, 1],
            result['triangles'],
            'b-', lw=0.5
        )
        
        # Mark boundary points
        plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='red', s=10)
        
        # Mark feature points
        plt.scatter(feature_points[:, 0], feature_points[:, 1], c='green', s=20, marker='*')
        
        plt.axis('equal')
        plt.title(f"Dense Hybrid Triangle (g={gradient}, {len(result['triangles'])} triangles)")
        plt.savefig(os.path.join(RESULTS_DIR, f"hybrid_dense_triangle_g{gradient}.png"), dpi=300)
        plt.close()
    
    return boundary_points, interior_points, feature_points, feature_sizes

def generate_meshit_style_points(num_sides=5, radius=10.0, boundary_density=20, interior_density=300):
    """
    Generate points in a style that closely matches MeshIt's output.
    
    This function generates both boundary and interior points with specific
    distributions to match MeshIt's triangulation patterns.
    """
    # Generate polygon vertices
    vertices = []
    for i in range(num_sides):
        angle = 2 * math.pi * i / num_sides + (math.pi/2)  # Start from top
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertices.append((x, y))
    
    # Generate boundary points
    boundary_points = []
    for i in range(num_sides):
        start = vertices[i]
        end = vertices[(i + 1) % num_sides]
        
        # Add vertex
        boundary_points.append(start)
        
        # Add intermediate points
        num_intermediate = boundary_density // num_sides
        for j in range(1, num_intermediate):
            t = j / num_intermediate
            # Non-linear distribution - more points near vertices
            t_adjusted = 0.5 * (1 - math.cos(math.pi * t))
            
            x = start[0] + t_adjusted * (end[0] - start[0])
            y = start[1] + t_adjusted * (end[1] - start[1])
            
            # Add slight random variation perpendicular to edge
            edge_vec = (end[0] - start[0], end[1] - start[1])
            edge_len = math.sqrt(edge_vec[0]**2 + edge_vec[1]**2)
            
            if edge_len > 1e-8:
                # Normalize edge vector
                edge_vec = (edge_vec[0]/edge_len, edge_vec[1]/edge_len)
                
                # Perpendicular vector
                perp_vec = (-edge_vec[1], edge_vec[0])
                
                # Small random offset
                offset = 0.05 * random.random() * radius
                
                x += perp_vec[0] * offset
                y += perp_vec[1] * offset
            
            boundary_points.append((x, y))
    
    boundary_points = np.array(boundary_points)
    
    # Generate interior points
    centroid = np.mean(boundary_points, axis=0)
    max_radius = 0.95 * np.max([np.linalg.norm(p - centroid) for p in boundary_points])
    
    # MeshIt uses a combination of:
    # 1. Random points with organic distribution
    # 2. Structured points with some randomization
    
    # Create circles of points with randomization
    interior_points = []
    
    # Multiple concentric rings of points
    num_rings = 6
    points_per_ring = interior_density // num_rings
    
    for ring in range(num_rings):
        ring_radius = max_radius * (ring + 1) / (num_rings + 1)
        
        for i in range(points_per_ring):
            angle = 2 * math.pi * i / points_per_ring
            
            # Add randomness to angle and radius
            angle_jitter = random.uniform(-0.2, 0.2)
            radius_jitter = random.uniform(-0.1, 0.1) * ring_radius
            
            actual_angle = angle + angle_jitter
            actual_radius = ring_radius + radius_jitter
            
            x = centroid[0] + actual_radius * math.cos(actual_angle)
            y = centroid[1] + actual_radius * math.sin(actual_angle)
            
            interior_points.append((x, y))
    
    # Add some completely random points
    num_random = interior_density // 3
    for _ in range(num_random):
        angle = 2 * math.pi * random.random()
        
        # Use sqrt for uniform distribution in the disk
        radius_factor = math.sqrt(random.random())
        actual_radius = max_radius * radius_factor
        
        x = centroid[0] + actual_radius * math.cos(angle)
        y = centroid[1] + actual_radius * math.sin(angle)
        
        interior_points.append((x, y))
    
    interior_points = np.array(interior_points)
    
    return boundary_points, interior_points

def run_meshit_exact_triangulation():
    """
    Run triangulation that exactly mimics MeshIt's approach.
    
    This uses a combination of:
    1. MeshIt-style point generation
    2. Direct C++ callback for gradient control
    3. Triangle's exact options as in MeshIt
    """
    # Generate MeshIt-style boundary and interior points
    boundary_points, interior_points = generate_meshit_style_points(
        num_sides=5,
        radius=10.0,
        boundary_density=50,
        interior_density=400
    )
    logger.info(f"Created {len(boundary_points)} boundary points and {len(interior_points)} interior points")
    
    # Create segments
    segments = create_segments(boundary_points)
    
    # Combine boundary and interior points
    all_points = np.vstack([boundary_points, interior_points])
    
    # Calculate base size
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    diagonal = np.sqrt(np.sum((max_coords - min_coords)**2))
    base_size = diagonal / 15.0
    
    # Create feature points
    feature_points, feature_sizes = create_feature_points(
        boundary_points, 
        base_size,
        num_features=10  # More feature points for better control
    )
    logger.info(f"Created {len(feature_points)} feature points")
    
    # Plot the input
    plt.figure(figsize=(12, 12))
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='red', s=15)
    plt.scatter(interior_points[:, 0], interior_points[:, 1], c='blue', s=5, alpha=0.3)
    plt.scatter(feature_points[:, 0], feature_points[:, 1], c='green', s=30, marker='*')
    
    # Draw segments
    for seg in segments:
        plt.plot(
            [boundary_points[seg[0]][0], boundary_points[seg[1]][0]],
            [boundary_points[seg[0]][1], boundary_points[seg[1]][1]],
            'r-', lw=0.8
        )
    
    plt.axis('equal')
    plt.title(f"Input for MeshIt-Exact Triangulation")
    plt.savefig(os.path.join(RESULTS_DIR, "meshit_exact_input.png"), dpi=300)
    plt.close()
    
    # Create new segments that only connect boundary points
    segments_with_interior = np.column_stack([
        np.arange(len(boundary_points)),
        np.roll(np.arange(len(boundary_points)), -1)
    ])
    
    # Run triangulation with different gradient values
    gradient_values = [1.0, 2.0, 3.0]
    
    for gradient in gradient_values:
        # Adjust minimum angle based on gradient exactly as in MeshIt
        min_angle = 20.0
        if gradient > 1.0:
            min_angle = max(20.0 - (gradient - 1.0) * 5.0, 10.0)
        
        # Area constraint
        area_constraint = base_size * base_size * 0.5
        
        # Triangulation input
        tri_input = {
            'vertices': all_points,
            'segments': segments_with_interior
        }
        
        # Initialize the C++ GradientControl
        if HAVE_CALLBACK:
            initialize_gradient_control(
                gradient,
                base_size * base_size,  # Squared mesh size
                feature_points,
                feature_sizes
            )
            
            # MeshIt exact options: pzqXXau
            options = f'pzq{min_angle}a{area_constraint}u'
            logger.info(f"Using MeshIt exact options: {options}")
            
            try:
                result = tr.triangulate(tri_input, options)
            except Exception as e:
                logger.error(f"Error with MeshIt exact options: {e}")
                # Fall back to standard options
                options = f'pzq{min_angle}a{area_constraint}'
                logger.info(f"Falling back to: {options}")
                result = tr.triangulate(tri_input, options)
        else:
            # If callback not available, use standard Triangle
            options = f'pzq{min_angle}a{area_constraint}'
            logger.info(f"Using standard Triangle options: {options}")
            result = tr.triangulate(tri_input, options)
        
        logger.info(f"Triangulation created {len(result['triangles'])} triangles")
        
        # Plot result
        plt.figure(figsize=(12, 12))
        
        # Plot triangles
        plt.triplot(
            result['vertices'][:, 0],
            result['vertices'][:, 1],
            result['triangles'],
            'b-', lw=0.5
        )
        
        # Mark boundary points
        plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='red', s=10)
        
        # Mark feature points
        plt.scatter(feature_points[:, 0], feature_points[:, 1], c='green', s=20, marker='*')
        
        plt.axis('equal')
        plt.title(f"MeshIt-Exact Triangle (g={gradient}, {len(result['triangles'])} triangles)")
        plt.savefig(os.path.join(RESULTS_DIR, f"meshit_exact_g{gradient}.png"), dpi=300)
        plt.close()
    
    return boundary_points, interior_points, feature_points, feature_sizes

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    logger.info("Running MeshIt-style triangulation tests")
    
    # Run basic hybrid triangulation
    run_hybrid_triangulation()
    
    # Run densified hybrid triangulation
    run_densified_hybrid_triangulation()
    
    # Run MeshIt exact triangulation
    run_meshit_exact_triangulation()
    
    logger.info("All triangulation tests completed") 