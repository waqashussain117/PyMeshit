"""
Direct standard triangulation mimicking MeshIt's core C++ implementation.

This script creates a natural triangulation similar to the direct_standard_tri_g2.0.png example,
using an organic point distribution, proper feature points, and gradient-based refinement.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import math
import os
import logging
import random
from scipy.spatial import Delaunay
from matplotlib.path import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DirectTriangulation")

# Make sure output directory exists
RESULTS_DIR = "triangulation_results"
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
        
        # Generate non-uniformly distributed points along this side
        for j in range(num_points_per_side):
            # Non-linear distribution - more points near corners
            t = j / num_points_per_side
            
            # Apply non-uniform distribution (cubed function favors edges)
            if random.random() > 0.5:
                t = t**1.5  # Cluster near start
            else:
                t = 1.0 - (1.0 - t)**1.5  # Cluster near end
                
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            
            # Add slight random variation in position (perpendicular to edge)
            edge_vec = (end[0] - start[0], end[1] - start[1])
            edge_len = math.sqrt(edge_vec[0]**2 + edge_vec[1]**2)
            
            if edge_len > 0:
                # Normalize edge vector
                edge_vec = (edge_vec[0]/edge_len, edge_vec[1]/edge_len)
                
                # Perpendicular vector
                perp_vec = (-edge_vec[1], edge_vec[0])
                
                # Random offset perpendicular to edge
                rand_offset = 0.2 * random.random() - 0.1  # Small random offset
                
                x += perp_vec[0] * rand_offset
                y += perp_vec[1] * rand_offset
            
            boundary_points.append((x, y))
    
    return np.array(boundary_points)

def get_polygon_path(boundary_points):
    """
    Create a proper polygon Path for point-in-polygon testing.
    
    Args:
        boundary_points: Points along the boundary
        
    Returns:
        Matplotlib Path object for point testing
    """
    # We need a convex hull or ordered points for the path
    # For our case, since points are already ordered, we can use them directly
    return Path(boundary_points)

def create_organic_interior_points(boundary_points, num_points=150, min_distance=0.5):
    """
    Create interior points with a natural, organic distribution similar to MeshIt.
    
    Args:
        boundary_points: Boundary points defining the domain
        num_points: Target number of interior points
        min_distance: Minimum distance between points
        
    Returns:
        Array of interior point coordinates
    """
    # Calculate bounding box and centroid
    min_x = np.min(boundary_points[:, 0])
    max_x = np.max(boundary_points[:, 0])
    min_y = np.min(boundary_points[:, 1])
    max_y = np.max(boundary_points[:, 1])
    centroid = np.mean(boundary_points, axis=0)
    
    # Create a proper polygon path for testing
    polygon = get_polygon_path(boundary_points)
    
    # For MeshIt-like results, we'll use a combination of:
    # 1. Randomly placed points with minimum distance constraints
    # 2. Radial distribution from centroid with randomization
    # 3. Some points placed along interior lines between key boundary points
    
    # 1. Generate randomly distributed points with constraints
    random_points = []
    attempts = 0
    max_attempts = num_points * 20
    
    while len(random_points) < num_points//3 and attempts < max_attempts:
        # Generate a random point within the bounding box
        x = min_x + (max_x - min_x) * random.random()
        y = min_y + (max_y - min_y) * random.random()
        point = np.array([x, y])
        
        # Check if the point is inside the polygon
        if not polygon.contains_point((x, y)):
            attempts += 1
            continue
            
        # Check if the point is far enough from existing points
        too_close = False
        for existing_point in random_points:
            distance = np.linalg.norm(point - existing_point)
            if distance < min_distance:
                too_close = True
                break
                
        # Also check distance to boundary points
        if not too_close:
            for boundary_point in boundary_points:
                distance = np.linalg.norm(point - boundary_point)
                if distance < min_distance:
                    too_close = True
                    break
                
        if not too_close:
            random_points.append(point)
            
        attempts += 1
    
    # 2. Generate radially distributed points from centroid
    radial_points = []
    max_radius = max([np.linalg.norm(bp - centroid) for bp in boundary_points]) * 0.8
    num_radial = num_points//3
    
    for i in range(num_radial):
        # Random angle and radius (use sqrt for uniform distribution in the area)
        angle = 2 * math.pi * random.random()
        # Use a distribution that favors mid-range distances
        r = max_radius * (0.2 + 0.6 * random.random())
        
        x = centroid[0] + r * math.cos(angle)
        y = centroid[1] + r * math.sin(angle)
        
        # Skip if outside the polygon
        if not polygon.contains_point((x, y)):
            continue
            
        radial_points.append(np.array([x, y]))
    
    # 3. Generate points along key interior lines
    # Select a subset of boundary points to connect
    num_boundary = len(boundary_points)
    step = max(1, num_boundary // 8)  # Connect roughly 8 boundary points
    
    interior_lines_points = []
    for i in range(0, num_boundary, step):
        start = boundary_points[i]
        # Connect to a point roughly opposite 
        end_idx = (i + num_boundary//2) % num_boundary
        end = boundary_points[end_idx]
        
        # Add points along this interior line
        num_line_points = 4
        for j in range(1, num_line_points):
            t = j / (num_line_points + 1)
            # Avoid points too close to boundary
            t = 0.3 + 0.4 * t  # Keep between 30% and 70% along line
            
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            
            # Skip if outside polygon
            if not polygon.contains_point((x, y)):
                continue
                
            # Add some randomness perpendicular to the line
            line_vec = (end[0] - start[0], end[1] - start[1])
            line_len = math.sqrt(line_vec[0]**2 + line_vec[1]**2)
            
            if line_len > 0:
                # Normalize
                line_vec = (line_vec[0]/line_len, line_vec[1]/line_len)
                
                # Perpendicular vector
                perp_vec = (-line_vec[1], line_vec[0])
                
                # Random offset perpendicular to line
                rand_offset = 0.5 * random.random()
                if random.random() > 0.5:
                    rand_offset = -rand_offset
                    
                x += perp_vec[0] * rand_offset
                y += perp_vec[1] * rand_offset
                
                interior_lines_points.append(np.array([x, y]))
    
    # Combine all point sets
    all_interior = []
    if random_points:
        all_interior.extend(random_points)
    if radial_points:
        all_interior.extend(radial_points)
    if interior_lines_points:
        all_interior.extend(interior_lines_points)
    
    return np.array(all_interior) if all_interior else np.empty((0, 2))

def create_segments(boundary_points):
    """Create segments connecting consecutive boundary points."""
    num_points = len(boundary_points)
    return np.column_stack([
        np.arange(num_points),
        np.roll(np.arange(num_points), -1)
    ])

def create_feature_points(boundary_points, interior_points, num_features=5):
    """
    Create feature points to control mesh density in MeshIt style.
    
    Args:
        boundary_points: Boundary points of the domain
        interior_points: Interior points
        num_features: Number of feature points to create
        
    Returns:
        Tuple of (feature points, feature sizes)
    """
    # Calculate domain statistics
    centroid = np.mean(boundary_points, axis=0)
    max_radius = max([np.linalg.norm(bp - centroid) for bp in boundary_points])
    bounding_box_diag = max_radius * 2
    base_size = bounding_box_diag / 15.0  # MeshIt's rule of thumb
    
    feature_points = []
    feature_sizes = []
    
    # Add feature points near boundary with smaller sizes
    num_boundary_features = num_features // 2
    boundary_indices = np.linspace(0, len(boundary_points) - 1, num_boundary_features, dtype=int)
    
    for idx in boundary_indices:
        point = boundary_points[idx]
        # Move slightly inward from boundary
        vec_to_center = centroid - point
        vec_len = np.linalg.norm(vec_to_center)
        if vec_len > 1e-8:
            normalized = vec_to_center / vec_len
            # Move 10% inward
            inward_point = point + normalized * (vec_len * 0.1)
            feature_points.append(inward_point)
            # Smaller size near boundary for refinement
            feature_sizes.append(base_size * 0.3)
    
    # Add interior feature points with variable sizes
    num_interior_features = num_features - num_boundary_features
    
    if len(interior_points) > 0 and num_interior_features > 0:
        # Use k-means like approach to distribute feature points 
        selected_indices = []
        
        # Start with point closest to center for one feature
        if num_interior_features == 1:
            distances = [np.linalg.norm(pt - centroid) for pt in interior_points]
            selected_indices.append(np.argmin(distances))
        else:
            # For multiple features, try to space them out
            # Start with a point near center
            distances = [np.linalg.norm(pt - centroid) for pt in interior_points]
            first_idx = np.argmin(distances)
            selected_indices.append(first_idx)
            
            # Then select points farthest from already selected points
            while len(selected_indices) < num_interior_features:
                max_min_dist = -1
                best_idx = -1
                
                for i in range(len(interior_points)):
                    if i in selected_indices:
                        continue
                        
                    # Find minimum distance to any selected point
                    min_dist = float('inf')
                    for sel_idx in selected_indices:
                        dist = np.linalg.norm(interior_points[i] - interior_points[sel_idx])
                        min_dist = min(min_dist, dist)
                    
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = i
                
                if best_idx >= 0:
                    selected_indices.append(best_idx)
        
        # Add selected interior points as features
        for idx in selected_indices:
            feature_points.append(interior_points[idx])
            
            # Size increases toward the center (based on distance from centroid)
            dist_ratio = np.linalg.norm(interior_points[idx] - centroid) / max_radius
            size_factor = 0.4 + 0.3 * (1 - dist_ratio)  # 0.4 to 0.7
            feature_sizes.append(base_size * size_factor)
    
    # Add centroid as a feature with the largest size
    feature_points.append(centroid)
    feature_sizes.append(base_size * 0.8)  # Larger size at center
    
    return np.array(feature_points), np.array(feature_sizes)

def meshit_style_triangulation(vertices, segments, feature_points, feature_sizes, gradient=2.0):
    """
    Perform MeshIt-style triangulation with gradient control.
    
    Args:
        vertices: Input vertices (boundary + interior)
        segments: Input segments connecting boundary vertices
        feature_points: Feature points controlling mesh density
        feature_sizes: Sizes associated with feature points
        gradient: Gradient parameter controlling transition speed
        
    Returns:
        Dict with triangulation results
    """
    # Calculate base mesh size from bounding box
    all_points = np.vstack([vertices, feature_points])
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    diagonal = np.sqrt(np.sum((max_coords - min_coords) ** 2))
    base_size = diagonal / 15.0  # MeshIt's scaling
    
    # Adjust quality parameter based on gradient as in MeshIt
    min_angle = 20.0
    if gradient > 1.0:
        min_angle = max(20.0 - (gradient - 1.0) * 5.0, 10.0)
    
    # Prepare input for Triangle
    triangle_data = {
        'vertices': vertices,
        'segments': segments
    }
    
    # Define Triangle options
    # p = PSLG (use segments)
    # q = quality mesh with given minimum angle
    # a = impose maximum triangle area constraint
    # z = number vertices from zero
    # For MeshIt-like results use a generous quality constraint
    area_constraint = base_size * base_size * 0.5
    options = f'pzq{min_angle:.1f}a{area_constraint:.8f}'
    
    logger.info(f"Performing standard triangulation with options: {options}")
    
    # Run triangulation
    result = tr.triangulate(triangle_data, options)
    
    num_triangles = len(result['triangles'])
    num_vertices = len(result['vertices'])
    logger.info(f"Created {num_triangles} triangles from {num_vertices} vertices")
    
    return result

def run_direct_standard_triangulation():
    """Run direct standard triangulation with organic point distribution."""
    # Generate polygon boundary with non-uniform distribution
    boundary_points = create_nonuniform_polygon_boundary(
        num_sides=5, 
        radius=10.0, 
        num_points_per_side=10
    )
    num_boundary_points = len(boundary_points)
    logger.info(f"Created {num_boundary_points} boundary points for polygon")
    
    # Create segments
    segments = create_segments(boundary_points)
    
    # Create organic interior points with MeshIt-like distribution
    interior_points = create_organic_interior_points(
        boundary_points, 
        num_points=200, 
        min_distance=0.8
    )
    logger.info(f"Generated {len(interior_points)} organic interior points")
    
    # Create feature points to control mesh density
    feature_points, feature_sizes = create_feature_points(
        boundary_points, 
        interior_points, 
        num_features=8
    )
    logger.info(f"Created {len(feature_points)} feature points for mesh control")
    
    # Combine boundary and interior points
    all_points = np.vstack([boundary_points, interior_points])
    
    # Plot input (points and features)
    plt.figure(figsize=(12, 12))
    
    # Plot boundary and interior points
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='red', s=20)
    plt.scatter(interior_points[:, 0], interior_points[:, 1], c='blue', s=10, alpha=0.7)
    
    # Plot feature points
    plt.scatter(feature_points[:, 0], feature_points[:, 1], c='green', s=30, marker='*')
    
    # Draw boundary
    for seg in segments:
        plt.plot([all_points[seg[0]][0], all_points[seg[1]][0]],
                [all_points[seg[0]][1], all_points[seg[1]][1]],
                'g-', lw=0.5)
    
    plt.axis('equal')
    plt.title(f"Input Points and Features for Triangulation")
    plt.savefig(os.path.join(RESULTS_DIR, f"meshit_points_and_features.png"), dpi=300)
    plt.close()
    
    # Run MeshIt-style triangulations with different gradients
    for gradient in [1.0, 2.0, 3.0]:
        result = meshit_style_triangulation(
            all_points, 
            segments, 
            feature_points, 
            feature_sizes, 
            gradient=gradient
        )
        
        # Plot triangulation
        plt.figure(figsize=(12, 12))
        
        # Plot triangles
        plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
                   result['triangles'], 'b-', lw=0.5)
        
        # Mark boundary points
        plt.scatter(boundary_points[:, 0], boundary_points[:, 1],
                   c='red', s=15)
        
        # Mark feature points
        plt.scatter(feature_points[:, 0], feature_points[:, 1],
                   c='green', s=20, marker='*')
        
        plt.axis('equal')
        plt.title(f"Standard Triangle (g={gradient}, {len(result['triangles'])} triangles)")
        plt.savefig(os.path.join(RESULTS_DIR, f"direct_standard_tri_g{gradient}.png"), dpi=300)
        plt.close()
    
    return boundary_points, interior_points, feature_points, feature_sizes

if __name__ == "__main__":
    # Set random seed for reproducibility 
    random.seed(42)
    np.random.seed(42)
    
    logger.info("Running direct standard triangulation")
    run_direct_standard_triangulation()
    logger.info("Triangulation complete") 