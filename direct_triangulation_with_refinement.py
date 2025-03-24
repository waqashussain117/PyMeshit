"""
Direct triangulation with gradient-based refinement.

This script creates a high-quality triangulation with uniform triangles
and proper transition between boundary and interior regions.
"""

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import os
import logging
import math

# Set up logging and output directory
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DirectTriangulation")
RESULTS_DIR = "triangulation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_interior_points(n=20, domain_size=10, jitter=0.0, boundary_points=None):
    """
    Generate interior points with more density near the boundary.
    This helps create a smoother transition in triangle sizes.
    """
    # Basic grid
    x = np.linspace(-domain_size + 1, domain_size - 1, n)
    y = np.linspace(-domain_size + 1, domain_size - 1, n)
    xx, yy = np.meshgrid(x, y)
    
    # Apply small jitter for better triangulation
    if jitter > 0:
        xx = xx + np.random.uniform(-jitter, jitter, xx.shape)
        yy = yy + np.random.uniform(-jitter, jitter, yy.shape)
    
    # Combine coordinates
    base_points = np.column_stack((xx.flatten(), yy.flatten()))
    
    # If boundary points are provided, add additional points near the boundary
    if boundary_points is not None:
        # Calculate centroid of domain
        centroid = np.mean(boundary_points, axis=0)
        
        # Add transition points progressively inward from the boundary
        transition_layers = 3
        transition_points = []
        
        for layer in range(1, transition_layers + 1):
            # Calculate inset factor based on layer (closer to boundary = smaller factor)
            inset_factor = layer / (transition_layers + 1)
            
            # Create points along the inset boundary
            for i, point in enumerate(boundary_points):
                # Vector from centroid to boundary point
                vector = point - centroid
                
                # Scale the vector to create inset point
                inset_point = centroid + vector * (1 - inset_factor * 0.2)
                
                # Add some perpendicular jitter
                if i % 2 == 0 and layer < transition_layers:
                    # Create perpendicular vector
                    perp_vector = np.array([-vector[1], vector[0]])
                    perp_vector = perp_vector / np.linalg.norm(perp_vector) * domain_size * 0.05
                    
                    # Add points slightly offset from the main inset point
                    transition_points.append(inset_point + perp_vector)
                    transition_points.append(inset_point - perp_vector)
                
                transition_points.append(inset_point)
        
        # Combine base grid with transition points
        all_interior = np.vstack([base_points, transition_points])
        
        # Remove points that are too close to each other
        cleaned_points = remove_close_points(all_interior, min_distance=domain_size/n*0.5)
        return cleaned_points
    
    return base_points

def remove_close_points(points, min_distance=0.5):
    """Remove points that are too close to each other."""
    if len(points) < 2:
        return points
    
    keep_indices = [0]  # Always keep the first point
    
    for i in range(1, len(points)):
        # Check if this point is too close to any point we're keeping
        too_close = False
        for j in keep_indices:
            dist = np.sqrt(np.sum((points[i] - points[j])**2))
            if dist < min_distance:
                too_close = True
                break
        
        if not too_close:
            keep_indices.append(i)
    
    return points[keep_indices]

def generate_boundary_points(n_points=60, domain_size=10):
    """Generate points along the boundary of a square with higher density at corners."""
    # Points per side, with more points near corners
    points_per_side = n_points // 4
    
    # Create more densely packed boundary points using a non-linear distribution
    # This concentrates points near corners for better quality
    t = np.linspace(0, 1, points_per_side)
    t = t**0.7  # Non-linear distribution - creates higher density at corners
    
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

def add_feature_points(boundary_points, domain_size=10, gradient=2.0):
    """
    Add feature points at strategic locations to control triangle sizes.
    The gradient parameter controls how quickly the triangles grow in size
    as we move away from the boundary.
    """
    feature_points = []
    feature_sizes = []
    
    # Calculate centroid of the domain
    centroid = np.mean(boundary_points, axis=0)
    
    # Base point size for the center of the domain
    base_size = domain_size / 8.0
    
    # Add feature points along the boundary with small size
    boundary_size = base_size * 0.5  # Make boundary triangles smaller
    for point in boundary_points:
        feature_points.append(point)
        feature_sizes.append(boundary_size)
    
    # Add the centroid with largest size
    feature_points.append(centroid)
    feature_sizes.append(base_size * 1.5)  # Allow larger triangles in the center
    
    # Add transition points between boundary and center
    num_rings = int(3 * gradient)  # More rings for higher gradients
    points_per_ring = 8  # Use 8 points per ring for a good distribution
    
    for ring in range(1, num_rings + 1):
        # Calculate the radius of this ring
        radius_factor = ring / num_rings
        ring_radius = domain_size * radius_factor * 0.7  # Scale to stay inside boundary
        
        # Calculate size for this ring's points, based on position between boundary and center
        # Use cubic easing function for a smooth transition
        t = radius_factor
        t = t * t * (3 - 2 * t)  # Cubic easing function
        ring_size = boundary_size + t * (base_size - boundary_size)
        
        # Create points around the ring
        for i in range(points_per_ring):
            angle = 2 * np.pi * i / points_per_ring
            x = centroid[0] + ring_radius * np.cos(angle)
            y = centroid[1] + ring_radius * np.sin(angle)
            
            # Skip if point is outside the domain
            if abs(x) > domain_size or abs(y) > domain_size:
                continue
                
            feature_points.append([x, y])
            feature_sizes.append(ring_size)
    
    return np.array(feature_points), np.array(feature_sizes)

def compute_segments(boundary_points):
    """Create segments connecting boundary points in order."""
    n_points = len(boundary_points)
    segments = []
    
    for i in range(n_points):
        segments.append([i, (i + 1) % n_points])
    
    return np.array(segments)

def triangulate_with_refinement(gradient=2.0, min_angle=30):
    """
    Perform direct triangulation with refinement based on feature points.
    
    Args:
        gradient: Controls how quickly triangle size increases from boundary to interior
        min_angle: Minimum angle constraint for quality
    """
    # Parameters
    domain_size = 10
    n_boundary = 60
    interior_density = 8  # Fewer base points since we'll add feature points
    
    # Generate boundary points
    boundary_points = generate_boundary_points(n_boundary, domain_size)
    logger.info(f"Generated {len(boundary_points)} boundary points")
    
    # Generate interior points
    interior_points = generate_interior_points(interior_density, domain_size, jitter=0.2, 
                                             boundary_points=boundary_points)
    logger.info(f"Generated {len(interior_points)} interior points")
    
    # Create segments for the boundary
    segments = compute_segments(boundary_points)
    
    # Create feature points and sizes for refinement
    feature_points, feature_sizes = add_feature_points(boundary_points, domain_size, gradient)
    logger.info(f"Generated {len(feature_points)} feature points")
    
    # Combine all points - boundary points must come first for segment indices to be correct
    # Interior points can follow in any order
    all_points = np.vstack((boundary_points, interior_points))
    
    # Prepare options string for Triangle
    options = 'pq'  # p for PSLG (constrained Delaunay triangulation)
    
    if min_angle is not None:
        options += f'{min_angle}'  # Minimum angle constraint
    
    # We'll use feature points for size control instead of a fixed max area
    # This is more similar to how MeshIt's C++ implementation works
    
    logger.info(f"Triangulating with gradient {gradient} and options: {options}")
    
    # Create input for Triangle
    tri_input = {
        'vertices': all_points,
        'segments': segments
    }
    
    # First pass - basic triangulation
    result = tr.triangulate(tri_input, options)
    logger.info(f"Initial triangulation: {len(result['triangles'])} triangles")
    
    # Second pass - add feature points to control triangle sizes
    # For each triangle, check if it's too large based on feature points
    refinement_points = []
    
    for tri in result['triangles']:
        # Get triangle vertices
        v1 = result['vertices'][tri[0]]
        v2 = result['vertices'][tri[1]]
        v3 = result['vertices'][tri[2]]
        
        # Calculate centroid
        centroid = (v1 + v2 + v3) / 3.0
        
        # Calculate edge lengths
        edge_lengths = [
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v3 - v2),
            np.linalg.norm(v1 - v3)
        ]
        max_edge = max(edge_lengths)
        
        # Determine target size at this location
        target_size = domain_size  # Start with a large value
        
        for i, feat_point in enumerate(feature_points):
            # Feature size
            feat_size = feature_sizes[i]
            
            # Distance to feature point
            dist = np.linalg.norm(centroid - feat_point)
            
            # Calculate target size based on distance and gradient
            # Similar to the MeshIt approach
            sq_dist = dist * dist
            sq_grad = gradient * gradient
            influence_radius = gradient * (domain_size / 6)
            
            if dist < influence_radius:
                # Scale from feature size outward
                target_size_here = feat_size + dist / gradient
                
                # Take minimum size
                target_size = min(target_size, target_size_here)
        
        # If triangle is too big, add a refinement point
        if max_edge > target_size * 1.2:  # Allow some tolerance
            # Add circumcenter as refinement point
            # This is standard practice for Delaunay refinement
            
            # Calculate circumcenter
            a = np.vstack((v1, v2, v3))
            
            # Calculate vectors AB and AC
            ab = a[1] - a[0]
            ac = a[2] - a[0]
            
            # Calculate determinant
            D = 2 * (ab[0] * ac[1] - ab[1] * ac[0])
            
            # Skip degenerate triangles
            if abs(D) < 1e-10:
                continue
                
            # Calculate squared norms
            abSq = np.dot(ab, ab)
            acSq = np.dot(ac, ac)
            
            # Calculate circumcenter
            ux = (ac[1] * abSq - ab[1] * acSq) / D
            uy = (ab[0] * acSq - ac[0] * abSq) / D
            
            # Calculate circumcenter coordinates
            circumcenter = a[0] + np.array([ux, uy])
            
            # Skip if circumcenter is outside the domain
            if abs(circumcenter[0]) > domain_size or abs(circumcenter[1]) > domain_size:
                # Use centroid instead
                refinement_points.append(centroid)
            else:
                refinement_points.append(circumcenter)
    
    if len(refinement_points) > 0:
        logger.info(f"Adding {len(refinement_points)} refinement points")
        
        # Add refinement points to vertices
        all_points_with_refinement = np.vstack((all_points, refinement_points))
        
        # Re-triangulate with all points
        tri_input_refined = {
            'vertices': all_points_with_refinement,
            'segments': segments
        }
        
        result = tr.triangulate(tri_input_refined, options)
        logger.info(f"Final triangulation: {len(result['triangles'])} triangles")
    
    # Save input points visualization
    plt.figure(figsize=(12, 12))
    plt.scatter(interior_points[:, 0], interior_points[:, 1], c='gray', s=5, alpha=0.5, label='Interior Points')
    plt.scatter(feature_points[:, 0], feature_points[:, 1], c='r', s=30, label='Feature Points')
    
    # Draw differently sized circles for feature points to show size influence
    for i, point in enumerate(feature_points):
        circle = plt.Circle((point[0], point[1]), feature_sizes[i], 
                           fill=False, color='orange', alpha=0.3)
        plt.gca().add_patch(circle)
    
    # Draw boundary
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5, label='Boundary')
    
    # Draw refinement points if any
    if len(refinement_points) > 0:
        refinement_array = np.array(refinement_points)
        plt.scatter(refinement_array[:, 0], refinement_array[:, 1], 
                   c='b', s=10, label='Refinement Points')
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f"Input Distribution (Gradient = {gradient})")
    plt.savefig(os.path.join(RESULTS_DIR, f"direct_refined_input_g{gradient}.png"), dpi=300)
    plt.close()
    
    # Plot triangulation
    plt.figure(figsize=(12, 12))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
              result['triangles'], 'b-', lw=0.5)
    
    # Show boundary
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title(f"Standard Triangle (g={gradient}, {len(result['triangles'])} triangles)")
    plt.savefig(os.path.join(RESULTS_DIR, f"direct_standard_tri_g{gradient}.png"), dpi=300)
    plt.close()
    
    # Create a closer view of the triangulation near the boundary
    plt.figure(figsize=(12, 12))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1],
              result['triangles'], 'b-', lw=0.5)
    
    # Show boundary
    plt.plot(np.vstack([boundary_points, boundary_points[0]]).T[0], 
             np.vstack([boundary_points, boundary_points[0]]).T[1], 'g-', lw=1.5)
    
    # Zoom to top-right corner
    plt.xlim(5, 10)
    plt.ylim(5, 10)
    
    plt.grid(True)
    plt.title(f"Zoomed View - Gradient={gradient}")
    plt.savefig(os.path.join(RESULTS_DIR, f"direct_standard_tri_g{gradient}_zoomed.png"), dpi=300)
    plt.close()
    
    return result

def run_comparison():
    """Run triangulation with different gradient values for comparison."""
    # Only run with gradient 2.0 as requested
    gradients = [2.0]
    
    for gradient in gradients:
        logger.info(f"Running triangulation with gradient {gradient}")
        triangulate_with_refinement(gradient=gradient)

if __name__ == "__main__":
    run_comparison()
    plt.close('all')  # Close all figures 