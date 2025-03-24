"""
Test script for MeshIt workflow using Python bindings.

This script tests a full MeshIt workflow:
1. Create a planar surface
2. Compute convex hull
3. Perform coarse segmentation
4. Run triangulation

This helps us test if the direct Triangle callback implementation works correctly
in a more realistic scenario.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import meshit
from meshit.extensions import (
    create_surface_from_points,
    calculate_planar_convex_hull,
    enhanced_calculate_convex_hull,
    align_intersections_to_convex_hull,
    calculate_constraints,
    triangulate_with_triangle,
    run_coarse_triangulation
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MeshItTest")

def create_planar_test_surface():
    """Create a test planar surface with scattered points"""
    # Generate a set of scattered points
    np.random.seed(42)  # For reproducibility
    
    # Generate boundary points (a flower-like shape)
    theta = np.linspace(0, 2*np.pi, 30)
    r = 10 + 2 * np.sin(5*theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)
    
    boundary_points = np.column_stack((x, y, z))
    
    # Generate random interior points
    num_interior = 50
    interior_points = []
    
    for _ in range(num_interior):
        # Random point within the approximate boundary
        angle = np.random.random() * 2 * np.pi
        radius = np.random.random() * 8  # Smaller than boundary radius
        interior_points.append([
            radius * np.cos(angle), 
            radius * np.sin(angle), 
            0
        ])
    
    interior_points = np.array(interior_points)
    
    # Combine boundary and interior points
    all_points = np.vstack((boundary_points, interior_points))
    
    # Create surface from points
    logger.info(f"Creating surface with {len(all_points)} points")
    surface = create_surface_from_points(all_points, "PlanarSurface", "Planar")
    
    # Plot the original points
    plt.figure(figsize=(10, 8))
    plt.scatter(all_points[:, 0], all_points[:, 1], c='b', marker='o')
    plt.title('Original Points')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('original_points.png')
    logger.info("Saved original points plot to original_points.png")
    
    return surface

def test_convex_hull(surface):
    """Test computing the convex hull of the surface"""
    logger.info("Computing convex hull...")
    
    # Calculate convex hull
    calculate_planar_convex_hull(surface)
    
    # Verify convex hull points
    hull_points = np.array([[p.x, p.y, p.z] for p in surface.convex_hull])
    logger.info(f"Convex hull has {len(hull_points)} points")
    
    # Plot convex hull
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original points
    original_points = np.array([[p.x, p.y, p.z] for p in surface.vertices])
    ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], 
               c='b', marker='o', label='Original Points')
    
    # Plot convex hull - as a loop
    hull_plus_first = np.vstack((hull_points, hull_points[0:1]))
    ax.plot(hull_plus_first[:, 0], hull_plus_first[:, 1], hull_plus_first[:, 2], 
            'r-', linewidth=2, label='Convex Hull')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Surface with Convex Hull')
    ax.legend()
    
    plt.savefig('convex_hull.png')
    logger.info("Saved convex hull plot to convex_hull.png")
    
    # Also plot in 2D for clarity
    plt.figure(figsize=(10, 8))
    plt.scatter(original_points[:, 0], original_points[:, 1], c='b', marker='o', label='Original Points')
    
    hull_2d = np.array([[p.x, p.y] for p in surface.convex_hull])
    hull_plus_first_2d = np.vstack((hull_2d, hull_2d[0:1]))
    plt.plot(hull_plus_first_2d[:, 0], hull_plus_first_2d[:, 1], 'r-', linewidth=2, label='Convex Hull')
    
    plt.axis('equal')
    plt.grid(True)
    plt.title('Surface with Convex Hull (2D)')
    plt.legend()
    plt.savefig('convex_hull_2d.png')
    logger.info("Saved 2D convex hull plot to convex_hull_2d.png")
    
    return hull_points

def test_coarse_segmentation(surface):
    """Test coarse segmentation"""
    logger.info("Testing coarse segmentation...")
    
    # Run coarse segmentation operations from MeshIt
    align_intersections_to_convex_hull(surface)
    calculate_constraints(surface)
    
    # Log constraints information
    if hasattr(surface, 'constraints'):
        logger.info(f"Surface has {len(surface.constraints)} constraints")
    else:
        logger.warning("No constraints found on surface after coarse segmentation")
    
    return surface

def test_triangulation(surface, gradient=2.0):
    """Test the triangulation with the given gradient"""
    logger.info(f"Testing triangulation with gradient={gradient}...")
    
    # Check if we have direct Triangle callback support
    has_direct = False
    try:
        from meshit.triangle_direct import DirectTriangleWrapper
        has_direct = True
        logger.info("DirectTriangleWrapper is available!")
    except ImportError:
        logger.warning("DirectTriangleWrapper is not available, using fallback")
    
    # Get surface points as NumPy array
    surface_points = np.array([[v.x, v.y, v.z] for v in surface.vertices])
    
    # For a more realistic test, extract the 2D projection (since it's planar)
    points_2d = surface_points[:, :2]
    
    # Get convex hull points
    hull_points = np.array([[p.x, p.y] for p in surface.convex_hull])
    
    # If we have constraints, use them
    constraints = []
    if hasattr(surface, 'constraints'):
        for con in surface.constraints:
            constraints.append([con.start, con.end])
    
    logger.info(f"Surface has {len(points_2d)} points and {len(constraints)} constraints")
    
    # Debug: Visualize input to triangulation
    plt.figure(figsize=(10, 8))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c='b', marker='o', label='Input Points')
    
    if len(hull_points) > 0:
        hull_plus_first = np.vstack((hull_points, hull_points[0:1]))
        plt.plot(hull_plus_first[:, 0], hull_plus_first[:, 1], 'r-', linewidth=2, label='Convex Hull')
    
    if len(constraints) > 0:
        for con in constraints:
            start_idx = con[0]
            end_idx = con[1]
            start_pt = points_2d[start_idx]
            end_pt = points_2d[end_idx]
            plt.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 'g-', linewidth=2, label='Constraint')
    
    plt.axis('equal')
    plt.grid(True)
    plt.title(f'Input to Triangulation (gradient={gradient})')
    plt.legend()
    plt.savefig(f'triangulation_input_g{gradient}.png')
    logger.info(f"Saved triangulation input plot to triangulation_input_g{gradient}.png")
    
    # Triangulate with debug info
    logger.info("Calling triangulate_with_triangle...")
    
    # Directly use the wrapper to get more debug info
    if has_direct:
        try:
            # IMPROVED DIRECT TRIANGULATION APPROACH
            import triangle as tr
            from scipy.spatial import ConvexHull
            
            # Get a real convex hull rather than a rectangle
            if len(points_2d) > 3:
                try:
                    # Calculate proper convex hull
                    hull = ConvexHull(points_2d)
                    hull_indices = hull.vertices
                    real_hull_points = points_2d[hull_indices]
                    logger.info(f"Using scipy ConvexHull with {len(real_hull_points)} points")
                    
                    # Create segments from hull points
                    segments = np.array([[i, (i+1) % len(hull_indices)] for i in range(len(hull_indices))])
                    
                    # Plot the actual convex hull we're using
                    plt.figure(figsize=(10, 8))
                    plt.scatter(points_2d[:, 0], points_2d[:, 1], c='b', marker='o', label='Input Points')
                    
                    for i, j in segments:
                        plt.plot([real_hull_points[i, 0], real_hull_points[j, 0]], 
                                 [real_hull_points[i, 1], real_hull_points[j, 1]], 'r-', lw=1)
                    
                    plt.axis('equal')
                    plt.title(f'Convex Hull Segments for Triangle (gradient={gradient})')
                    plt.savefig(f'hull_segments_g{gradient}.png')
                except Exception as e:
                    logger.error(f"Error computing scipy ConvexHull: {e}")
                    real_hull_points = hull_points
                    segments = np.array([[i, (i+1) % len(hull_points)] for i in range(len(hull_points))])
            else:
                real_hull_points = hull_points
                segments = np.array([[i, (i+1) % len(hull_points)] for i in range(len(hull_points))])
                
            # First try standard Triangle library for basic triangulation
            try:
                # Prepare vertices for triangulation - use all input points
                vertices_dict = {'vertices': points_2d}
                
                # Add segments from hull
                if len(segments) > 0:
                    # Map hull points back to indices in the original array
                    mapped_segments = []
                    for i, pt1 in enumerate(real_hull_points):
                        for j, pt2 in enumerate(points_2d):
                            if np.allclose(pt1, pt2, atol=1e-10):
                                mapped_segments.append(j)
                                break
                    
                    # Create segment pairs using mapped indices
                    seg_pairs = []
                    for i in range(len(mapped_segments)):
                        seg_pairs.append([mapped_segments[i], mapped_segments[(i+1) % len(mapped_segments)]])
                    
                    if seg_pairs:
                        vertices_dict['segments'] = np.array(seg_pairs)
                        logger.info(f"Using {len(seg_pairs)} segments as constraints")
                
                # Basic triangulation with quality constraints
                min_angle = 20.0
                if gradient > 1.0:
                    min_angle = max(20.0 - (gradient - 1.0) * 5.0, 10.0)
                    
                # Create strings for Triangle options
                tri_opts = f'pq{min_angle}a'
                
                # Run basic triangulation first
                basic_tri = tr.triangulate(vertices_dict, tri_opts)
                
                if 'triangles' in basic_tri:
                    logger.info(f"Basic Triangle triangulation created {len(basic_tri['triangles'])} triangles")
                    
                    # Plot basic triangulation
                    plt.figure(figsize=(12, 10))
                    plt.triplot(basic_tri['vertices'][:, 0], basic_tri['vertices'][:, 1], 
                                basic_tri['triangles'], 'b-', lw=0.5)
                    plt.scatter(points_2d[:, 0], points_2d[:, 1], c='r', s=10)
                    plt.axis('equal')
                    plt.title(f'Basic Triangle Library Triangulation (gradient={gradient})')
                    plt.savefig(f'basic_triangulation_g{gradient}.png')
                else:
                    logger.warning("Basic triangulation failed to produce triangles")
            except Exception as e:
                logger.error(f"Error in basic triangulation: {e}")
            
            # Initialize the direct wrapper
            wrapper = DirectTriangleWrapper(gradient=gradient)
            
            # Select feature points for refinement
            # For better results, use interior points + hull points
            feature_points = []
            feature_sizes = []
            
            # Add boundary points with small feature size
            boundary_size = 0.5
            for pt in real_hull_points:
                feature_points.append(pt)
                feature_sizes.append(boundary_size)
            
            # Add a few interior points with larger feature size
            centroid = np.mean(points_2d, axis=0)
            feature_points.append(centroid)
            feature_sizes.append(1.0)  # Larger size for interior
            
            # Convert to numpy arrays
            feature_points = np.array(feature_points)
            feature_sizes = np.array(feature_sizes)
            
            # Set feature points for refinement
            wrapper.set_feature_points(feature_points, feature_sizes)
            logger.info(f"Using {len(feature_points)} feature points for refinement")
            
            # Run the direct triangulation
            logger.info(f"Using direct triangulation with {len(points_2d)} points, {len(segments)} segments")
            
            # Make a copy of the hull points for triangle
            hull_vertices = np.array(real_hull_points)
            
            # Run triangulation with our direct wrapper
            tri = wrapper.triangulate(points_2d, segments)
            
            # Extract results
            vertices_out = tri['vertices']
            triangles = tri['triangles']
            
            logger.info(f"Direct triangulation complete: {len(triangles)} triangles, {len(vertices_out)} vertices")
            
            # Create a 3D version with z=0 for all points
            vertices = np.column_stack((vertices_out, np.zeros(len(vertices_out))))
            
        except Exception as e:
            logger.error(f"Error in direct triangulation: {e}")
            # Fall back to standard triangulation
            vertices, triangles = triangulate_with_triangle(surface, gradient=gradient)
    else:
        # Use the standard triangulation
        vertices, triangles = triangulate_with_triangle(surface, gradient=gradient)
    
    logger.info(f"Generated {len(triangles)} triangles")
    logger.info(f"Mesh has {len(vertices)} vertices")
    
    # Plot triangulation result
    plt.figure(figsize=(12, 10))
    plt.triplot(vertices[:, 0], vertices[:, 1], triangles, 'b-', lw=1.0, alpha=0.7)
    
    # Plot original points
    plt.plot(points_2d[:, 0], points_2d[:, 1], 'ro', ms=4)
    
    # Plot convex hull
    if len(hull_points) > 0:
        hull_plus_first = np.vstack((hull_points, hull_points[0:1]))
        plt.plot(hull_plus_first[:, 0], hull_plus_first[:, 1], 'g-', lw=2)
    
    # Plot constraints
    if len(constraints) > 0:
        for con in constraints:
            start_idx = con[0]
            end_idx = con[1]
            start_pt = points_2d[start_idx]
            end_pt = points_2d[end_idx]
            plt.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 'y-', lw=2)
    
    plt.axis('equal')
    plt.title(f'Triangulation with gradient={gradient}')
    plt.savefig(f'triangulation_g{gradient}.png')
    logger.info(f"Saved triangulation plot to triangulation_g{gradient}.png")
    
    return vertices, triangles

def run_model_workflow():
    """Run a complete MeshIt workflow to test Python bindings"""
    logger.info("Starting MeshIt workflow test...")
    
    # 1. Create a planar surface
    surface = create_planar_test_surface()
    
    # 2. Compute convex hull
    hull = test_convex_hull(surface)
    
    # 3. Perform coarse segmentation
    surface = test_coarse_segmentation(surface)
    
    # 4. Test triangulation with different gradients
    gradient_values = [0.5, 1.0, 2.0]
    for gradient in gradient_values:
        vertices, triangles = test_triangulation(surface, gradient=gradient)
    
    # 5. Try using run_coarse_triangulation from extensions
    logger.info("\nTesting run_coarse_triangulation...")
    
    # Create a minimal model for testing run_coarse_triangulation
    model = meshit.MeshItModel()
    meshit.add_surface_to_model(model, surface)
    
    # Run coarse triangulation
    def progress_callback(message):
        logger.info(f"Progress: {message}")
    
    run_coarse_triangulation(model, progress_callback, gradient=2.0)
    
    # Check if triangulation was performed
    if hasattr(surface, 'triangles') and len(surface.triangles) > 0:
        logger.info(f"Coarse triangulation generated {len(surface.triangles)} triangles")
    else:
        logger.warning("No triangles found after run_coarse_triangulation")
    
    plt.show()

if __name__ == "__main__":
    # Configure logging for the triangulation module as well
    logging.getLogger('meshit.triangle_direct').setLevel(logging.DEBUG)
    logging.getLogger('DirectTriangleWrapper').setLevel(logging.DEBUG)
    
    run_model_workflow() 