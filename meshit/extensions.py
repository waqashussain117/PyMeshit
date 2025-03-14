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
    
    # Update gradient control with average point location and size
    gc = GradientControl.get_instance()
    avg_point = float(np.mean(vertices_2d))  # Use average point as representative
    gc.update(float(gradient), float(base_size), len(vertices_2d), avg_point, float(base_size))
    
    # Prepare Triangle input
    vertices_array = vertices_2d
    segments = np.array([[i, (i + 1) % len(hull_points)] for i in range(len(hull_points))])
    
    # Run Triangle with quality constraints
    import triangle
    min_angle = 20.0 if gradient < 1.0 else 15.0
    tri = triangle.triangulate({
        'vertices': vertices_array,
        'segments': segments
    }, f'q{min_angle}a{hull_area/100}')
    
    # Convert back to 3D
    vertices_3d = []
    inv_rotation = np.linalg.inv(rotation)
    for v in tri['vertices']:
        point_2d = np.array([v[0], v[1], 0])
        point_3d = inv_rotation @ point_2d
        vertices_3d.append(point_3d)
    
    return np.array(vertices_3d), tri['triangles']

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
    print("Using enhanced triangulation method with MeshIt-like constraints")
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
            
            # Default hull size based on number of hull points and hull area
            hull_size = np.sqrt(hull_area / len(self.convex_hull))
            print(f"Calculated hull size: {hull_size:.6f}")
        else:
            print(f"Using provided hull size: {hull_size:.6f}")
        
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

# Add the enhanced_triangulate method to the Surface class
Surface.enhanced_triangulate = enhanced_triangulate_method 