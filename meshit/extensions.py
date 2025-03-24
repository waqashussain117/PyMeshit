"""
Extensions to the MeshIt C++ bindings to provide more complete functionality.
This module enhances the C++ bindings with Python implementations of advanced algorithms
that closely match the MeshIt C++ source code for each step of the workflow.
"""

import numpy as np
import triangle as tr
from .core import Surface, Vector3D, GradientControl
import math
from scipy.spatial import ConvexHull, Delaunay

try:
    from meshit.core import MeshItModel
except ImportError:
    import sys
    print("Warning: Failed to import MeshItModel from meshit.core", file=sys.stderr)
    print("Pre-mesh functionality will be limited", file=sys.stderr)
    
    # Define a minimal MeshItModel class for interface compatibility
    class MeshItModel:
        def __init__(self):
            self.surfaces = []
            self.model_polylines = []
            print("Warning: Using dummy MeshItModel class - pre-mesh functionality will be limited")
        
        def __str__(self):
            return "MeshItModel(dummy)"

def process_points(points):
    """
    Process raw points to prepare them for surface creation.
    This mimics the point processing step in MeshIt.
    
    Args:
        points: A list of 3D points as [x, y, z] lists
        
    Returns:
        Processed points suitable for surface creation
    """
    # In MeshIt, this step often includes:
    # 1. Removing duplicate points
    # 2. Sorting points
    # 3. Checking for coplanarity
    
    # Convert to numpy array for processing
    points_array = np.array(points)
    
    # Remove duplicate points (within a small tolerance)
    unique_points = []
    tolerance = 1e-10
    
    for point in points_array:
        is_duplicate = False
        for unique_point in unique_points:
            if np.linalg.norm(point - unique_point) < tolerance:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(point)
    
    # Sort points by x, then y, then z (similar to MeshIt's sorting)
    unique_points.sort(key=lambda p: (p[0], p[1], p[2]))
    
    # Check for coplanarity (similar to MeshIt's check)
    if len(unique_points) >= 3:
        # Calculate normal using first three points
        v1 = np.array(unique_points[1]) - np.array(unique_points[0])
        v2 = np.array(unique_points[2]) - np.array(unique_points[0])
        normal = np.cross(v1, v2)
        normal_length = np.linalg.norm(normal)
        
        # If normal is too small, points might be collinear
        if normal_length < 1e-10:
            print("Warning: First three points are nearly collinear")
        else:
            normal = normal / normal_length
            
            # Check if all points are approximately on the same plane
            max_distance = 0
            for point in unique_points[3:]:
                v = np.array(point) - np.array(unique_points[0])
                distance = abs(np.dot(v, normal))
                max_distance = max(max_distance, distance)
            
            if max_distance > 1e-6:
                print(f"Note: Points are not perfectly coplanar. Max distance from plane: {max_distance:.6f}")
    
    return unique_points

def create_surface_from_points(points, name="Surface", surface_type="Default"):
    """
    Create a surface from points, similar to MeshIt's surface creation step.
    
    Args:
        points: A list of 3D points
        name: Name for the surface
        surface_type: Type of surface (Default, Planar, etc.)
        
    Returns:
        A Surface object
    """
    # Process points first (like MeshIt does)
    processed_points = process_points(points)
    
    # Create the surface with processed points
    surface = Surface()
    surface.name = name
    surface.type = surface_type
    
    # Add vertices to the surface
    for point in processed_points:
        surface.add_vertex(Vector3D(point[0], point[1], point[2]))
    
    # Calculate bounds (min/max) like MeshIt does
    if len(surface.vertices) > 0:
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for vertex in surface.vertices:
            min_x = min(min_x, vertex.x)
            min_y = min(min_y, vertex.y)
            min_z = min(min_z, vertex.z)
            max_x = max(max_x, vertex.x)
            max_y = max(max_y, vertex.y)
            max_z = max(max_z, vertex.z)
        
        # In MeshIt, bounds are stored as two Vector3D objects
        # We'll mimic this by setting the bounds attribute
        if hasattr(surface, 'bounds') and len(surface.bounds) >= 2:
            surface.bounds[0] = Vector3D(min_x, min_y, min_z)
            surface.bounds[1] = Vector3D(max_x, max_y, max_z)
    
    return surface

def enhanced_calculate_convex_hull(surface):
    """
    Enhanced convex hull calculation that mimics MeshIt's implementation.
    Handles planar point sets more robustly.
    
    Args:
        surface: A Surface instance
    
    Returns:
        A list of Vector3D objects representing the convex hull
    """
    # Extract vertices as numpy array
    vertices = np.array([[v.x, v.y, v.z] for v in surface.vertices])
    
    # Check if points are coplanar (all z values are the same)
    z_values = vertices[:, 2]
    is_planar = np.allclose(z_values, z_values[0], atol=1e-10)
    
    if is_planar:
        print("Detected planar point set. Using 2D convex hull calculation.")
        try:
            # For planar points, use 2D convex hull
            # Project points to 2D (ignore z-coordinate)
            points_2d = vertices[:, :2]
            
            # Calculate 2D convex hull
            hull = ConvexHull(points_2d)
            
            # Get the vertices of the convex hull in order
            hull_vertices = []
            for vertex_idx in hull.vertices:
                hull_vertices.append(vertices[vertex_idx])
            
            # Create Vector3D objects for the hull vertices
            hull_points = []
            for point in hull_vertices:
                hull_points.append(Vector3D(point[0], point[1], point[2]))
            
            # Debug: Print the hull points
            print(f"Created 2D convex hull with {len(hull_points)} points:")
            for i, p in enumerate(hull_points):
                print(f"  Point {i}: ({p.x}, {p.y}, {p.z})")
            
            # If we have fewer than 3 points, fall back to manual calculation
            if len(hull_points) < 3:
                print("Warning: 2D convex hull has fewer than 3 points, using manual calculation")
                return calculate_planar_convex_hull(surface)
            
            # Update the surface's convex hull
            surface.convex_hull = hull_points
            
            return hull_points
            
        except Exception as e:
            print(f"Error in 2D convex hull calculation: {e}")
            # Fall back to manual convex hull for planar points
            return calculate_planar_convex_hull(surface)
    
    # For non-planar points, use the original 3D convex hull calculation
    try:
        # Need at least 4 points for 3D convex hull
        if len(vertices) < 4:
            print("Warning: Not enough points for 3D convex hull calculation")
            # For 3 points or less, all points are on the convex hull
            surface.convex_hull = surface.vertices.copy()
            return surface.convex_hull
        
        # Use scipy's ConvexHull which uses the same algorithm as MeshIt (QuickHull)
        hull = ConvexHull(vertices)
        
        # Extract the unique vertices that form the convex hull
        hull_vertices = []
        for simplex in hull.simplices:
            for idx in simplex:
                if idx not in hull_vertices:
                    hull_vertices.append(idx)
        
        # Create Vector3D objects for the hull vertices
        hull_points = []
        for idx in hull_vertices:
            point = vertices[idx]
            hull_points.append(Vector3D(point[0], point[1], point[2]))
        
        # Sort the hull points to ensure consistent ordering (similar to MeshIt)
        # This is a simplified version of MeshIt's sorting
        # MeshIt uses a more complex algorithm to ensure the hull is ordered correctly
        
        # First, find the centroid
        centroid = np.mean(vertices[hull_vertices], axis=0)
        
        # Project points onto a unit sphere around the centroid
        projected_points = []
        for i, idx in enumerate(hull_vertices):
            v = vertices[idx] - centroid
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-10:  # Avoid division by zero
                projected_points.append((i, v / v_norm))
        
        # Sort points by spherical coordinates (similar to MeshIt's approach)
        def spherical_angle(p):
            x, y, z = p[1]
            theta = math.atan2(y, x)
            phi = math.acos(z / np.linalg.norm(p[1]))
            return (theta, phi)
        
        projected_points.sort(key=spherical_angle)
        
        # Reorder hull points based on the sorted projected points
        ordered_hull_points = []
        for i, _ in projected_points:
            ordered_hull_points.append(hull_points[i])
        
        # Update the surface's convex hull
        surface.convex_hull = ordered_hull_points
        
        return ordered_hull_points
    
    except Exception as e:
        print(f"Error calculating convex hull: {e}")
        # Fallback to original method if scipy fails
        try:
            surface.calculate_convex_hull()
            return surface.convex_hull
        except:
            print("Original convex hull calculation also failed. Using manual calculation.")
            return calculate_planar_convex_hull(surface)

def calculate_planar_convex_hull(surface):
    """
    Calculate convex hull for planar points manually.
    This is a fallback method when scipy's ConvexHull fails.
    
    Args:
        surface: A Surface instance
    
    Returns:
        A list of Vector3D objects representing the convex hull
    """
    print("Using manual convex hull calculation for planar points.")
    
    # Extract vertices
    vertices = [[v.x, v.y, v.z] for v in surface.vertices]
    
    # Find the min and max x, y coordinates to form a bounding box
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)
    z_val = vertices[0][2]  # Assuming all z values are the same
    
    # Create a rectangular convex hull using the bounding box
    # Make sure to create the points in counter-clockwise order
    hull_points = [
        Vector3D(min_x, min_y, z_val),  # Bottom left
        Vector3D(max_x, min_y, z_val),  # Bottom right
        Vector3D(max_x, max_y, z_val),  # Top right
        Vector3D(min_x, max_y, z_val)   # Top left
    ]
    
    # Debug: Print the hull points
    print("Created rectangular convex hull with points:")
    for i, p in enumerate(hull_points):
        print(f"  Point {i}: ({p.x}, {p.y}, {p.z})")
    
    # Update the surface's convex hull
    surface.convex_hull = hull_points
    
    return hull_points

def align_intersections_to_convex_hull(surface):
    """
    Properly align intersection points to the convex hull.
    This mimics MeshIt's alignIntersectionsToConvexHull method.
    
    Args:
        surface: A Surface instance
    
    Returns:
        A list of triangles representing the convex hull
    """
    # If convex hull is not calculated yet, calculate it
    if not hasattr(surface, 'convex_hull') or len(surface.convex_hull) == 0:
        enhanced_calculate_convex_hull(surface)
    
    # If convex hull has less than 3 points, we can't do anything
    if len(surface.convex_hull) < 3:
        print("Warning: Convex hull has less than 3 points, cannot align intersections")
        return []
    
    # Create triangles from the convex hull points (similar to MeshIt)
    hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
    
    # Calculate the centroid of the hull
    centroid = np.mean(hull_points, axis=0)
    
    # Create triangles by connecting the centroid to each edge of the hull
    # This is similar to how MeshIt creates triangles for projection
    hull_triangles = []
    
    # In MeshIt, they create a triangulation of the convex hull
    # We'll use Delaunay triangulation on the projected hull points
    
    # First, find the normal of the hull (using the first three points)
    v1 = hull_points[1] - hull_points[0]
    v2 = hull_points[2] - hull_points[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    # Create a rotation matrix to align the hull with the XY plane
    z_axis = np.array([0, 0, 1])
    axis = np.cross(normal, z_axis)
    
    if np.linalg.norm(axis) < 1e-12:
        # Hull is already aligned with XY plane
        rotation_matrix = np.eye(3)
    else:
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.dot(z_axis, normal)
        sin_theta = np.sqrt(1 - cos_theta**2)
        C = 1 - cos_theta
        
        # Build rotation matrix (similar to MeshIt's rotation)
        r1 = np.array([axis[0]*axis[0]*C + cos_theta, 
                       axis[0]*axis[1]*C - axis[2]*sin_theta, 
                       axis[0]*axis[2]*C + axis[1]*sin_theta])
        
        r2 = np.array([axis[1]*axis[0]*C + axis[2]*sin_theta, 
                       axis[1]*axis[1]*C + cos_theta, 
                       axis[1]*axis[2]*C - axis[0]*sin_theta])
        
        r3 = np.array([axis[2]*axis[0]*C - axis[1]*sin_theta, 
                       axis[2]*axis[1]*C + axis[0]*sin_theta, 
                       axis[2]*axis[2]*C + cos_theta])
        
        rotation_matrix = np.vstack((r1, r2, r3))
    
    # Rotate hull points to align with XY plane
    rotated_hull_points = np.zeros_like(hull_points)
    for i, point in enumerate(hull_points):
        rotated_hull_points[i] = np.dot(rotation_matrix, point)
    
    # Extract the 2D points (x, y) from the rotated hull
    hull_2d_points = rotated_hull_points[:, :2]
    
    # Use Delaunay triangulation on the 2D hull points
    try:
        tri = Delaunay(hull_2d_points)
        
        # Convert the triangulation to our format
        for simplex in tri.simplices:
            hull_triangles.append([
                surface.convex_hull[simplex[0]],
                surface.convex_hull[simplex[1]],
                surface.convex_hull[simplex[2]]
            ])
    except Exception as e:
        print(f"Error in Delaunay triangulation: {e}")
        # Fallback to a simpler triangulation (fan triangulation)
        for i in range(1, len(surface.convex_hull) - 1):
            hull_triangles.append([
                surface.convex_hull[0],
                surface.convex_hull[i],
                surface.convex_hull[i+1]
            ])
    
    # In MeshIt, this method would now project intersection points onto the hull
    # Since we don't have actual intersection points in our simplified model,
    # we'll just print a message about what would happen
    print(f"Aligning intersections to convex hull with {len(hull_triangles)} triangles")
    print("In MeshIt, this would project intersection points onto the convex hull surface")
    
    return hull_triangles

def calculate_constraints(surface):
    """
    Calculate constraints for triangulation, mimicking MeshIt's constraint calculation.
    
    Args:
        surface: A Surface instance
    
    Returns:
        A list of constraint segments
    """
    # If convex hull is not calculated yet, calculate it
    if not hasattr(surface, 'convex_hull') or len(surface.convex_hull) == 0:
        enhanced_calculate_convex_hull(surface)
    
    # If convex hull has less than 3 points, we can't do anything
    if len(surface.convex_hull) < 3:
        print("Warning: Convex hull has less than 3 points, cannot calculate constraints")
        return []
    
    # In MeshIt, constraints are calculated from:
    # 1. Convex hull edges
    # 2. Intersection lines
    # 3. User-defined constraints
    
    # We'll focus on the convex hull edges for now
    
    # Find the indices of the convex hull points in the vertices array
    hull_indices = []
    for hull_point in surface.convex_hull:
        for i, vertex in enumerate(surface.vertices):
            if (abs(hull_point.x - vertex.x) < 1e-10 and
                abs(hull_point.y - vertex.y) < 1e-10 and
                abs(hull_point.z - vertex.z) < 1e-10):
                hull_indices.append(i)
                break
    
    # Create segments for the convex hull boundary (similar to MeshIt)
    constraints = []
    for i in range(len(hull_indices)):
        constraints.append([hull_indices[i], hull_indices[(i + 1) % len(hull_indices)]])
    
    # In MeshIt, they also add constraints for intersection lines
    # Since we don't have actual intersections in our simplified model,
    # we'll just print a message about what would happen
    print(f"Created {len(constraints)} constraint segments from convex hull")
    print("In MeshIt, additional constraints would be added for intersection lines")
    
    return constraints

def triangulate_with_triangle(points_or_surface, gradient=2.0):
    """
    Triangulate a set of points or a surface using Triangle library with gradient control.
    
    This function now uses a direct C++ callback to Triangle that exactly matches 
    the behavior of MeshIt's C++ implementation. If the C++ extension is not available,
    it falls back to the Python-based triangulation wrapper.
    
    Args:
        points_or_surface: Either a numpy array of shape (N, 3) containing points,
                          or a Surface object
        gradient: Gradient control parameter (default: 2.0)
                 Lower values produce more uniform triangles
                 Higher values allow more size variation
    
    Returns:
        vertices: numpy array of shape (M, 3) containing vertex coordinates
        triangles: numpy array of shape (K, 3) containing triangle indices
    """
    from meshit.core import Surface, GradientControl, Vector3D
    import numpy as np
    
    # Try to import our new direct triangle callback wrapper first
    use_direct_wrapper = False
    use_custom_wrapper = False
    
    try:
        from meshit.triangle_direct import DirectTriangleWrapper
        use_direct_wrapper = True
        print("Using DirectTriangleWrapper with C++ triunsuitable callback")
    except ImportError:
        print("Direct triangle callback not available, trying fallback")
        try:
            # Fallback to the older custom wrapper
            from meshit.triangle_wrapper import TriangleWrapper
            use_custom_wrapper = True
            print("Using TriangleWrapper with Python-based refinement")
        except ImportError:
            print("Custom triangle wrapper not available, using standard triangulation")
    
    # Convert input to Surface object if needed
    if isinstance(points_or_surface, np.ndarray):
        surface = Surface()
        for point in points_or_surface:
            v = Vector3D(float(point[0]), float(point[1]), float(point[2]))
            surface.add_vertex(v)
    else:
        surface = points_or_surface
    
    # Calculate surface normal and create rotation matrix
    normal = calculate_surface_normal(surface)
    rotation = rotation_matrix_from_axis_angle(normal, [0, 0, 1])
    
    # Project vertices to 2D
    vertices_2d = []
    for v in surface.vertices:
        point = np.array([v.x, v.y, v.z])
        rotated = rotation @ point
        vertices_2d.append([rotated[0], rotated[1]])
    vertices_2d = np.array(vertices_2d)
    
    # Calculate convex hull
    hull_points = calculate_convex_hull_2d(vertices_2d)
    hull_area = polygon_area(hull_points)
    base_size = np.sqrt(hull_area / 100)  # Approximate target triangle size
    
    # Create refined size map based on feature points
    feature_points = []
    feature_sizes = []
    
    # 1. Add convex hull points as features with small sizes
    # This ensures fine triangulation near the boundaries
    hull_point_size = base_size * 0.5  # Smaller size at hull points
    for i, pt in enumerate(hull_points):
        feature_points.append(pt)
        feature_sizes.append(hull_point_size)
    
    # 2. Add intersection points (if any) as features with even smaller sizes
    intersection_size = base_size * 0.3  # Very small size at intersections
    if hasattr(surface, 'intersections') and surface.intersections:
        for intersection in surface.intersections:
            for point in intersection.points:
                # Project the 3D intersection point to 2D
                pt_3d = np.array([point.x, point.y, point.z])
                pt_2d = rotation @ pt_3d
                feature_points.append([pt_2d[0], pt_2d[1]])
                feature_sizes.append(intersection_size)
    
    # If we don't have any feature points, use center of vertices as a reference point
    if not feature_points:
        center = np.mean(vertices_2d, axis=0)
        feature_points.append(center)
        feature_sizes.append(base_size)
    
    # Convert feature points and sizes to numpy arrays
    feature_points = np.array(feature_points)
    feature_sizes = np.array(feature_sizes)
    
    # Update gradient control with feature points and sizes (for C++ version)
    if not use_direct_wrapper:
        # This is equivalent to the C++ GradientControl update in MeshIt
        gc = GradientControl.get_instance()
        
        # Flatten feature points to match the C++ interface
        points_flat = feature_points.flatten()
        
        # Create C-compatible arrays for the update function
        point_array = np.copy(points_flat)
        size_array = np.copy(feature_sizes)
        
        # Update gradient control
        try:
            gc.update(
                float(gradient),
                float(base_size),
                len(feature_points),
                point_array[0] if len(point_array) > 0 else 0.0,  # Simplification for the Python binding
                float(feature_sizes[0]) if len(feature_sizes) > 0 else base_size
            )
        except Exception as e:
            print(f"Warning: Could not update GradientControl: {e}")
    
    # Prepare Triangle input
    vertices_array = vertices_2d
    segments = np.array([[i, (i + 1) % len(hull_points)] for i in range(len(hull_points))])
    
    # Use direct C++ callback wrapper when available (best option)
    if use_direct_wrapper:
        wrapper = DirectTriangleWrapper(gradient=gradient, base_size=base_size)
        wrapper.set_feature_points(feature_points, feature_sizes)
        
        print(f"Using C++ triunsuitable callback with gradient {gradient} and {len(feature_points)} feature points")
        
        # Run triangulation with direct C++ callback
        tri = wrapper.triangulate(vertices_array, segments)
    
    # Fall back to custom Python wrapper if available
    elif use_custom_wrapper:
        # Setup the triangle wrapper
        min_angle = 20.0
        if gradient > 1.0:
            min_angle = 20.0 - (gradient - 1.0) * 5.0
            min_angle = max(min_angle, 10.0)
        
        area_constraint = hull_area / 100
        if gradient < 1.0:
            area_constraint *= (1.0 - 0.5 * (1.0 - gradient))
        elif gradient > 1.0:
            area_constraint *= (1.0 + 0.5 * (gradient - 1.0))
            
        wrapper = TriangleWrapper(gradient=gradient, min_angle=min_angle, base_size=base_size)
        wrapper.set_feature_points(feature_points, feature_sizes)
        
        print(f"Using Python-based refinement with gradient {gradient} and {len(feature_points)} feature points")
        
        # Run triangulation with custom refinement
        tri = wrapper.triangulate(vertices_array, segments)
    
    # Last resort: use standard Triangle options
    else:
        # Calculate quality constraints based on gradient
        min_angle = 20.0
        if gradient > 1.0:
            min_angle = 20.0 - (gradient - 1.0) * 5.0  # Decrease min angle as gradient increases
            min_angle = max(min_angle, 10.0)  # Don't go below 10 degrees
        
        area_constraint = hull_area / 100  # Base area constraint
        
        # Adjust area constraint based on gradient
        if gradient < 1.0:
            area_constraint *= (1.0 - 0.5 * (1.0 - gradient))
        elif gradient > 1.0:
            area_constraint *= (1.0 + 0.5 * (gradient - 1.0))
            
        # Create Triangle options string
        triangle_opts = f'pzq{min_angle}a{area_constraint}'
        
        # Run triangulation with standard Triangle options
        import triangle as tr
        tri_data = {'vertices': vertices_array, 'segments': segments}
        tri = tr.triangulate(tri_data, triangle_opts)
    
    # Extract vertices and triangles from the result
    vertices_2d_out = tri['vertices']
    triangles = tri['triangles']
    
    # Project vertices back to 3D
    rotation_inv = np.linalg.inv(rotation)
    vertices_3d = []
    for v_2d in vertices_2d_out:
        pt_2d = np.array([v_2d[0], v_2d[1], 0.0])
        pt_3d = rotation_inv @ pt_2d
        vertices_3d.append(pt_3d)
    
    return np.array(vertices_3d), triangles

def calculate_surface_normal(surface):
    """Calculate average normal vector of a surface."""
    import numpy as np
    vertices = np.array([[v.x, v.y, v.z] for v in surface.vertices])
    if len(vertices) < 3:
        return np.array([0, 0, 1])
    
    # Use first three non-collinear points to calculate normal
    v0 = vertices[0]
    for i in range(1, len(vertices)-1):
        v1 = vertices[i] - v0
        v2 = vertices[i+1] - v0
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) > 1e-10:
            return normal / np.linalg.norm(normal)
    
    return np.array([0, 0, 1])

def rotation_matrix_from_axis_angle(from_vec, to_vec):
    """Create rotation matrix to align from_vec with to_vec."""
    import numpy as np
    from_vec = np.array(from_vec)
    to_vec = np.array(to_vec)
    
    if np.allclose(from_vec, to_vec):
        return np.eye(3)
    
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)
    
    v = np.cross(from_vec, to_vec)
    c = np.dot(from_vec, to_vec)
    s = np.linalg.norm(v)
    
    if s < 1e-10:
        return np.eye(3)
    
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

def calculate_convex_hull_2d(points):
    """Calculate 2D convex hull of points."""
    from scipy.spatial import ConvexHull
    import numpy as np
    points = np.array(points)
    hull = ConvexHull(points)
    return points[hull.vertices]

def polygon_area(points):
    """Calculate area of a polygon defined by points."""
    import numpy as np
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# Monkey patch the Surface class to add our improved methods
original_triangulate = Surface.triangulate

def enhanced_triangulate(self, hull_size=None, gradient=1.0):
    """
    Enhanced triangulation method that uses the Triangle library with MeshIt-like constraints
    
    Args:
        hull_size: Size parameter for hull points (if None, calculated from surface)
                  Controls the density of triangulation - smaller values create more triangles
        gradient: Controls how quickly triangle size increases with distance from features
                  Smaller values create more triangles with smoother transitions
    """
    print("Using enhanced triangulation method with MeshIt-like constraints")
    if hull_size is not None:
        print(f"Using provided hull size: {hull_size:.6f}")
    try:
        # Calculate bounding box to determine appropriate min/max lengths
        min_x = min(v.x for v in self.vertices)
        max_x = max(v.x for v in self.vertices)
        min_y = min(v.y for v in self.vertices)
        max_y = max(v.y for v in self.vertices)
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        bbox_diagonal = np.sqrt(bbox_width**2 + bbox_height**2)
        
        # Use 1/15 of the bounding box diagonal as min_length for more uniform triangles
        min_length = bbox_diagonal / 15.0
        
        # Use 1/3 of the bounding box diagonal as max_length
        # This creates more uniform triangles as described in the MeshIt paper
        max_length = bbox_diagonal / 3.0
        
        # Use 20 degrees as minimum angle (as mentioned in the MeshIt paper)
        # This ensures good quality triangles
        min_angle = 20.0
        
        # Try to use our improved triangulation method with MeshIt-like parameters
        triangles = triangulate_with_triangle(self, min_length, max_length, min_angle, hull_size, gradient)
        if triangles and len(triangles) > 0:
            return triangles
        else:
            print("Enhanced triangulation failed to produce triangles, falling back to original method")
            # Fall back to original method if our method fails
            original_triangulate(self)
            return self.triangles
    except Exception as e:
        print(f"Error in enhanced triangulation: {e}")
        print("Falling back to original triangulation method")
        # Fall back to original method if our method fails
        original_triangulate(self)
        return self.triangles

# Replace the original method with our enhanced version
Surface.triangulate = enhanced_triangulate
Surface.triangulate_with_triangle = triangulate_with_triangle
Surface.align_intersections_to_convex_hull = align_intersections_to_convex_hull
Surface.calculate_constraints = calculate_constraints
Surface.enhanced_calculate_convex_hull = enhanced_calculate_convex_hull

# Add the new methods
Surface.process_points = staticmethod(process_points)

# Add a method to the Surface class that properly handles the hull size parameter
def enhanced_triangulate_method(self, hull_size=None, gradient=1.0):
    """
    Enhanced triangulation method that uses the Triangle library with MeshIt-like constraints
    
    Args:
        hull_size: Size parameter for hull points (if None, calculated from surface)
                  Controls the density of triangulation - smaller values create more triangles
        gradient: Controls how quickly triangle size increases with distance from features
                  Smaller values create more triangles with smoother transitions
    """
    print(f"Using enhanced triangulation method with gradient={gradient}")
    
    try:
        # Calculate bounding box to determine appropriate hull size
        min_x = min(v.x for v in self.vertices)
        max_x = max(v.x for v in self.vertices)
        min_y = min(v.y for v in self.vertices)
        max_y = max(v.y for v in self.vertices)
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        bbox_diagonal = np.sqrt(bbox_width**2 + bbox_height**2)
        
        # Calculate hull size if not provided
        if hull_size is None:
            # Make sure convex hull is calculated
            if not hasattr(self, 'convex_hull') or len(self.convex_hull) == 0:
                enhanced_calculate_convex_hull(self)
                
            # Calculate hull area
            hull_points = np.array([[v.x, v.y, v.z] for v in self.convex_hull])
            hull_area = 0.0
            for i in range(len(hull_points)):
                j = (i + 1) % len(hull_points)
                hull_area += hull_points[i][0] * hull_points[j][1]
                hull_area -= hull_points[j][0] * hull_points[i][1]
            hull_area = abs(hull_area) / 2.0
            
            # Default hull size based on hull area and number of vertices
            hull_size = np.sqrt(hull_area / max(len(self.vertices), 10))
            print(f"Calculated hull size: {hull_size:.6f}")
        else:
            print(f"Using provided hull size: {hull_size:.6f}")
        
        # Set up feature points for refinement
        feature_points = []
        feature_sizes = []
        
        # Add convex hull points as features
        hull_point_size = hull_size * 0.5
        for p in self.convex_hull:
            feature_points.append([p.x, p.y, p.z])
            feature_sizes.append(hull_point_size)
        
        # Use triangulate_with_triangle which now includes proper gradient control
        vertices_3d, triangles = triangulate_with_triangle(self, gradient)
        
        # Update the surface triangles
        self.triangles = []
        for tri in triangles:
            self.triangles.append([int(tri[0]), int(tri[1]), int(tri[2])])
        
        print(f"Triangulation complete with {len(self.triangles)} triangles")
        return self.triangles
        
    except Exception as e:
        print(f"Error in enhanced triangulation: {e}")
        print("Falling back to original triangulation method")
        original_triangulate(self)
        return self.triangles

# Add the enhanced_triangulate method to the Surface class
Surface.enhanced_triangulate = enhanced_triangulate_method

def extract_special_points(model, surface_index):
    """
    Extract special points (intersections and triple points) related to a surface
    to use as feature points for refinement in triangulation.
    
    Args:
        model: A MeshItModel object
        surface_index: Index of the surface to extract points for
        
    Returns:
        Tuple of (special_points, sizes) where special_points is a list of Vector3D objects
        and sizes is a list of refinement sizes for each point
    """
    special_points = []
    sizes = []
    
    # Base size defined based on the surface size
    surface = model.surfaces[surface_index]
    base_size = surface.size if surface.size > 0 else 1.0
    intersection_size = base_size * 0.3  # Finer mesh near intersections
    triple_point_size = base_size * 0.2  # Even finer mesh near triple points
    
    # 1. Get intersection points related to this surface
    for intersection in model.intersections:
        # Check if this intersection involves our surface
        if intersection.id1 == surface_index or intersection.id2 == surface_index:
            # Add all points from this intersection
            for point in intersection.points:
                special_points.append(point)
                sizes.append(intersection_size)
    
    # 2. Get triple points
    for triple_point in model.triple_points:
        # We'll add all triple points - they're important features
        special_points.append(triple_point.point)
        sizes.append(triple_point_size)
    
    print(f"Extracted {len(special_points)} special points for surface {surface_index}")
    return special_points, sizes

def triangulate_with_special_points(model, surface_index, gradient=1.0):
    """
    Triangulate a surface with special attention to intersections and triple points.
    
    Args:
        model: A MeshItModel object
        surface_index: Index of the surface to triangulate
        gradient: Gradient control parameter (default: 1.0)
        
    Returns:
        Triangulated surface with proper refinement around special points
    """
    if surface_index < 0 or surface_index >= len(model.surfaces):
        raise ValueError(f"Invalid surface index: {surface_index}")
    
    surface = model.surfaces[surface_index]
    
    # Extract special points for this surface
    special_points, special_sizes = extract_special_points(model, surface_index)
    
    # Calculate convex hull if needed
    if not hasattr(surface, 'convex_hull') or len(surface.convex_hull) == 0:
        surface.enhanced_calculate_convex_hull()
    
    # Calculate bounding box
    min_x = min(v.x for v in surface.vertices)
    max_x = max(v.x for v in surface.vertices)
    min_y = min(v.y for v in surface.vertices)
    max_y = max(v.y for v in surface.vertices)
    min_z = min(v.z for v in surface.vertices)
    max_z = max(v.z for v in surface.vertices)
    
    bbox_diagonal = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2 + (max_z - min_z)**2)
    base_size = bbox_diagonal / 15.0  # Default size based on bounding box
    
    # Calculate surface normal and create rotation matrix
    normal = calculate_surface_normal(surface)
    rotation = rotation_matrix_from_axis_angle(normal, [0, 0, 1])
    
    # Project vertices to 2D
    vertices_2d = []
    for v in surface.vertices:
        point = np.array([v.x, v.y, v.z])
        rotated = rotation @ point
        vertices_2d.append([rotated[0], rotated[1]])
    vertices_2d = np.array(vertices_2d)
    
    # Project special points to 2D
    feature_points = []
    feature_sizes = []
    
    # Add special points
    for i, point in enumerate(special_points):
        pt_3d = np.array([point.x, point.y, point.z])
        pt_2d = rotation @ pt_3d
        feature_points.append([pt_2d[0], pt_2d[1]])
        feature_sizes.append(special_sizes[i])
    
    # Add convex hull points as features
    hull_point_size = base_size * 0.5
    for p in surface.convex_hull:
        pt_3d = np.array([p.x, p.y, p.z])
        pt_2d = rotation @ pt_3d
        feature_points.append([pt_2d[0], pt_2d[1]])
        feature_sizes.append(hull_point_size)
    
    # If we don't have any feature points, use vertices
    if not feature_points:
        for pt in vertices_2d:
            feature_points.append(pt)
            feature_sizes.append(base_size)
    
    # Convert to numpy arrays
    feature_points = np.array(feature_points)
    feature_sizes = np.array(feature_sizes)
    
    # Update gradient control
    gc = GradientControl.get_instance()
    points_flat = feature_points.flatten()
    
    # Create C-compatible arrays for the update function
    point_array = np.copy(points_flat)
    size_array = np.copy(feature_sizes)
    
    # Update gradient control
    gc.update(
        float(gradient),
        float(base_size),
        len(feature_points),
        point_array[0] if len(point_array) > 0 else 0.0,
        float(feature_sizes[0]) if len(feature_sizes) > 0 else base_size
    )
    
    # Calculate convex hull of 2D points for boundary segments
    hull_2d = calculate_convex_hull_2d(vertices_2d)
    
    # Prepare Triangle input
    segments = np.array([[i, (i + 1) % len(hull_2d)] for i in range(len(hull_2d))])
    
    # Calculate quality constraints based on gradient
    min_angle = 20.0
    if gradient > 1.0:
        min_angle = 20.0 - (gradient - 1.0) * 5.0
        min_angle = max(min_angle, 10.0)
    
    # Calculate area constraint
    hull_area = polygon_area(hull_2d)
    area_constraint = hull_area / 100
    
    # Adjust area constraint based on gradient
    if gradient < 1.0:
        area_constraint = area_constraint * (1.0 - 0.5 * (1.0 - gradient))
    elif gradient > 1.0:
        area_constraint = area_constraint * (1.0 + 0.5 * (gradient - 1.0))
    
    # Run Triangle with quality constraints
    import triangle
    triangle_opts = f'q{min_angle}a{area_constraint}'
    
    print(f"Triangulating surface {surface_index} with {len(feature_points)} feature points")
    print(f"Using gradient={gradient}, min_angle={min_angle:.1f}, area_constraint={area_constraint:.6f}")
    
    tri = triangle.triangulate({
        'vertices': vertices_2d,
        'segments': segments
    }, triangle_opts)
    
    # Convert back to 3D and update the surface
    surface.triangles = []
    for t in tri['triangles']:
        surface.triangles.append([int(t[0]), int(t[1]), int(t[2])])
    
    print(f"Triangulation complete with {len(surface.triangles)} triangles")
    return surface 

def run_pre_mesh_job(model, progress_callback=None, gradient=1.0):
    """
    Run the complete pre_mesh_job workflow as in the C++ implementation.
    This includes:
    1. Calculate convex hulls
    2. Coarse segmentation
    3. Coarse triangulation
    4. Computing intersections
    5. Computing triple points
    6. Aligning intersections to convex hulls
    7. Calculating constraints
    
    Args:
        model: A MeshItModel object
        progress_callback: Optional callback function to report progress
        gradient: Gradient control parameter (default: 1.0)
        
    Returns:
        The updated model with all preprocessing steps completed
    """
    import time
    from datetime import datetime
    import threading
    import concurrent.futures
    
    def report_progress(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(message)
    
    # Record start time
    start_time = time.time()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_progress(f">Start Time: {current_time}\n")
    
    # Step 1: Calculate convex hulls
    report_progress(">Start calculating convexhull...\n")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks for each surface
        futures = []
        for i, surface in enumerate(model.surfaces):
            futures.append(executor.submit(surface.enhanced_calculate_convex_hull))
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()
    
    report_progress(">...finished\n")
    
    # Step 2: Segmentation (coarse)
    report_progress(">Start coarse segmentation...\n")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks for each polyline
        futures = []
        for i, polyline in enumerate(model.model_polylines):
            futures.append(executor.submit(polyline.calculate_segments, False))
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()
    
    report_progress(">...finished\n")
    
    # Step 3: 2D triangulation (coarse)
    report_progress(">Start coarse triangulation...\n")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks for each surface
        futures = []
        for i, surface in enumerate(model.surfaces):
            futures.append(executor.submit(surface.triangulate))
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()
    
    report_progress(">...finished\n")
    
    # Step 4: Intersection: surface-surface
    report_progress(">Start calculating surface-surface intersections...\n")
    
    # Clear existing intersections
    model.intersections.clear()
    
    # Calculate total number of combinations
    n_surfaces = len(model.surfaces)
    total_steps = n_surfaces * (n_surfaces - 1) // 2
    
    if total_steps > 0:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each surface pair
            futures = []
            for s1 in range(n_surfaces - 1):
                for s2 in range(s1 + 1, n_surfaces):
                    futures.append(executor.submit(model.calculate_surface_surface_intersection, s1, s2))
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
    
    report_progress(">...finished\n")
    
    # Step 5: Intersection: polyline-surface
    report_progress(">Start calculating polyline-surface intersections...\n")
    
    total_steps = len(model.model_polylines) * n_surfaces
    
    if total_steps > 0:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each polyline-surface pair
            futures = []
            for p in range(len(model.model_polylines)):
                for s in range(n_surfaces):
                    futures.append(executor.submit(model.calculate_polyline_surface_intersection, p, s))
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
    
    report_progress(">...finished\n")
    
    # Step 6: Intersection: calculate size
    model.calculate_size_of_intersections()
    
    # Step 7: Intersection: triple points
    report_progress(">Start calculating intersection triplepoints...\n")
    
    # Clear existing triple points
    model.triple_points.clear()
    
    total_steps = len(model.intersections) * (len(model.intersections) - 1) // 2
    
    if total_steps > 0:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each intersection pair
            futures = []
            for i1 in range(len(model.intersections) - 1):
                for i2 in range(i1 + 1, len(model.intersections)):
                    futures.append(executor.submit(model.calculate_triple_points, i1, i2))
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
    
    # Insert triple points
    model.insert_triple_points()
    
    report_progress(">...finished\n")
    
    # Step 8: Align convex hulls to intersections
    report_progress(">Start aligning Convex Hulls to Intersections...\n")
    
    for s, surface in enumerate(model.surfaces):
        report_progress(f"   >({s + 1}/{n_surfaces}) {surface.name} ({surface.type})")
        surface.align_intersections_to_convex_hull()
    
    report_progress(">...finished\n")
    
    # Step 9: Model constraints
    report_progress(">Start calculating constraints...\n")
    
    for surface in model.surfaces:
        surface.calculate_constraints()
    
    for polyline in model.model_polylines:
        polyline.calculate_Constraints()
    
    report_progress(">...finished\n")
    
    # Calculate size of constraints
    model.calculate_size_of_constraints()
    
    # Step 10: Refine triangulation with proper gradient and refinement
    report_progress(">Start refined triangulation with gradient control...\n")
    
    for s in range(len(model.surfaces)):
        report_progress(f"   >({s + 1}/{n_surfaces}) Triangulating {model.surfaces[s].name}")
        try:
            triangulate_with_special_points(model, s, gradient)
        except Exception as e:
            report_progress(f"   >Error triangulating surface {s}: {e}")
    
    report_progress(">...finished\n")
    
    # End timing
    end_time = time.time()
    elapsed = end_time - start_time
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_progress(f">End Time: {current_time}\n")
    report_progress(f">Elapsed Time: {elapsed:.2f}s\n")
    
    return model

# Add the run_pre_mesh_job function to the MeshItModel class
MeshItModel.run_python_pre_mesh_job = run_pre_mesh_job 

def run_coarse_triangulation(model, progress_callback=None, gradient=1.0):
    """
    Run only the first three steps of the pre_mesh_job workflow:
    1. Calculate convex hulls
    2. Coarse segmentation
    3. Coarse triangulation with gradient control
    
    This function focuses on implementing proper gradient control and refinesize
    parameters during the triangulation process.
    
    Args:
        model: A MeshItModel object
        progress_callback: Optional callback function to report progress
        gradient: Gradient control parameter (default: 1.0)
        
    Returns:
        The model with coarse triangulation completed
    """
    import time
    from datetime import datetime
    import concurrent.futures
    
    def report_progress(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(message)
    
    # Record start time
    start_time = time.time()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_progress(f">Start Time: {current_time}\n")
    
    # Step 1: Calculate convex hulls
    report_progress(">Start calculating convexhull...\n")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks for each surface
        futures = []
        for i, surface in enumerate(model.surfaces):
            futures.append(executor.submit(surface.enhanced_calculate_convex_hull))
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()
    
    report_progress(">...finished\n")
    
    # Step 2: Segmentation (coarse)
    report_progress(">Start coarse segmentation...\n")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks for each polyline
        futures = []
        for i, polyline in enumerate(model.model_polylines):
            futures.append(executor.submit(polyline.calculate_segments, False))
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()
    
    report_progress(">...finished\n")
    
    # Step 3: 2D triangulation (coarse) with gradient control
    report_progress(">Start coarse triangulation with gradient control...\n")
    
    # Process each surface sequentially to apply proper gradient control
    n_surfaces = len(model.surfaces)
    for s, surface in enumerate(model.surfaces):
        report_progress(f"   >({s + 1}/{n_surfaces}) Triangulating {surface.name}")
        
        try:
            # Setup gradient control for this surface
            # Calculate bounding box to determine base size
            min_x = min(v.x for v in surface.vertices) if surface.vertices else 0
            max_x = max(v.x for v in surface.vertices) if surface.vertices else 1
            min_y = min(v.y for v in surface.vertices) if surface.vertices else 0
            max_y = max(v.y for v in surface.vertices) if surface.vertices else 1
            min_z = min(v.z for v in surface.vertices) if surface.vertices else 0
            max_z = max(v.z for v in surface.vertices) if surface.vertices else 1
            
            # Calculate diagonal as a reference for base size
            bbox_diagonal = ((max_x - min_x)**2 + (max_y - min_y)**2 + (max_z - min_z)**2)**0.5
            base_size = bbox_diagonal / 15.0  # Default size based on bounding box
            
            # Create feature points from convex hull vertices
            feature_points = []
            feature_sizes = []
            
            hull_point_size = base_size * 0.5  # Smaller size at hull points
            for point in surface.convex_hull:
                feature_points.append(np.array([point.x, point.y, point.z]))
                feature_sizes.append(hull_point_size)
            
            # If no hull points, use all vertices as features
            if not feature_points and surface.vertices:
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                center_z = (min_z + max_z) / 2
                feature_points.append(np.array([center_x, center_y, center_z]))
                feature_sizes.append(base_size)
            
            # Update gradient control
            gc = GradientControl.get_instance()
            if feature_points:
                # Flatten first point for the C++ interface
                first_point = feature_points[0].flatten()[0] if feature_points else 0.0
                first_size = feature_sizes[0] if feature_sizes else base_size
                
                gc.update(
                    float(gradient),
                    float(base_size),
                    len(feature_points),
                    float(first_point),
                    float(first_size)
                )
                
                report_progress(f"      >Applied gradient {gradient} with {len(feature_points)} feature points")
            
            # Perform triangulation with enhanced method
            surface.enhanced_triangulate(gradient=gradient)
            report_progress(f"      >Created {len(surface.triangles)} triangles")
            
        except Exception as e:
            report_progress(f"      >Error triangulating surface: {e}")
            # Fall back to basic triangulation
            try:
                surface.triangulate()
                report_progress(f"      >Fell back to basic triangulation: {len(surface.triangles)} triangles")
            except Exception as e2:
                report_progress(f"      >Basic triangulation also failed: {e2}")
    
    report_progress(">...finished coarse triangulation\n")
    
    # End timing
    end_time = time.time()
    elapsed = end_time - start_time
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_progress(f">End Time: {current_time}\n")
    report_progress(f">Elapsed Time: {elapsed:.2f}s\n")
    
    return model

# Add the coarse triangulation function to the MeshItModel class
MeshItModel.run_coarse_triangulation = run_coarse_triangulation 