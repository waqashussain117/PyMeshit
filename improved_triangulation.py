import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
from scipy.spatial import ConvexHull
import meshit
from meshit.core._meshit import Vector3D

def generate_grid_points(n=5):
    """Generate a grid of points in the XY plane."""
    points = []
    for x in np.linspace(0, 10, n):
        for y in np.linspace(0, 10, n):
            points.append([x, y, 0])
    return points

def improved_triangulate_with_triangle(surface, hull_size=2.0):
    """
    Improved triangulation that properly handles new vertices created by Triangle.
    This matches MeshIt's approach of creating new vertices during triangulation.
    
    Args:
        surface: The surface to triangulate
        hull_size: Controls the density of triangles. Smaller values create more triangles.
    
    Returns:
        The updated surface with triangles
    """
    print(f"=== Triangulating with hull_size = {hull_size} ===")
    
    # Make sure we have a convex hull
    if not surface.convex_hull:
        meshit.extensions.enhanced_calculate_convex_hull(surface)
    
    # Get the convex hull points and vertices
    hull_points = surface.convex_hull
    vertices = surface.vertices
    
    # Calculate the normal of the surface
    if len(hull_points) >= 3:
        normal = Vector3D.normal(hull_points[0], hull_points[1], hull_points[2])
    else:
        # Default to Z-axis if we don't have enough points
        normal = Vector3D(0, 0, 1)
    
    # Create a rotation matrix to align the surface with the XY plane
    z_axis = Vector3D(0, 0, 1)
    rotation_axis = Vector3D.cross(normal, z_axis)
    
    # If the normal is already aligned with Z, no rotation needed
    if rotation_axis.length() < 1e-10:
        rotation_matrix = np.eye(3)
    else:
        # Normalize the rotation axis
        rotation_axis = rotation_axis.normalized()
        
        # Calculate the rotation angle
        cos_angle = Vector3D.dot(normal, z_axis)
        sin_angle = rotation_axis.length()
        
        # Create the rotation matrix (Rodrigues' rotation formula)
        K = np.array([
            [0, -rotation_axis.z, rotation_axis.y],
            [rotation_axis.z, 0, -rotation_axis.x],
            [-rotation_axis.y, rotation_axis.x, 0]
        ])
        rotation_matrix = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
    
    # Project the vertices to 2D
    vertices_2d = []
    for vertex in vertices:
        # Apply rotation to align with XY plane
        rotated = np.dot(rotation_matrix, np.array([vertex.x, vertex.y, vertex.z]))
        vertices_2d.append([rotated[0], rotated[1]])
    
    # Project the hull points to 2D
    hull_2d = []
    for point in hull_points:
        rotated = np.dot(rotation_matrix, np.array([point.x, point.y, point.z]))
        hull_2d.append([rotated[0], rotated[1]])
    
    # Create segments for the convex hull boundary
    segments = []
    for i in range(len(hull_points)):
        # Find the indices of the hull points in the vertices list
        idx1 = -1
        idx2 = -1
        for j, vertex in enumerate(vertices):
            if abs(vertex.x - hull_points[i].x) < 1e-10 and \
               abs(vertex.y - hull_points[i].y) < 1e-10 and \
               abs(vertex.z - hull_points[i].z) < 1e-10:
                idx1 = j
                break
        
        next_i = (i + 1) % len(hull_points)
        for j, vertex in enumerate(vertices):
            if abs(vertex.x - hull_points[next_i].x) < 1e-10 and \
               abs(vertex.y - hull_points[next_i].y) < 1e-10 and \
               abs(vertex.z - hull_points[next_i].z) < 1e-10:
                idx2 = j
                break
        
        if idx1 >= 0 and idx2 >= 0:
            segments.append([idx1, idx2])
    
    # Prepare data for Triangle library
    vertices_dict = dict(vertices=np.array(vertices_2d))
    vertices_dict['segments'] = np.array(segments)
    
    # Calculate the maximum area based on hull_size
    # Calculate the area of the convex hull
    hull_area = 0
    if len(hull_2d) >= 3:
        # Use the shoelace formula to calculate the area
        x = [p[0] for p in hull_2d]
        y = [p[1] for p in hull_2d]
        hull_area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(len(hull_2d) - 1)) + 
                             x[-1] * y[0] - x[0] * y[-1])
    
    # IMPORTANT: In MeshIt, smaller hull_size means smaller triangles
    # The max_area is inversely proportional to hull_size^2
    max_area = hull_area / (hull_size * hull_size)
    
    # For very small hull_size values, we need to limit the max_area
    # to avoid creating too many triangles
    if max_area > hull_area:
        max_area = hull_area / 4.0  # Reasonable default
    
    print(f"Hull area: {hull_area:.2f}, Max triangle area: {max_area:.2f}")
    
    # Triangulate with quality constraints
    options = f'pq30a{max_area}'
    print(f"Triangle options: {options}")
    triangulation = tr.triangulate(vertices_dict, options)
    
    # Extract triangulation results
    tri_vertices = triangulation['vertices']
    tri_triangles = triangulation['triangles']
    
    print(f"Triangulation created {len(tri_vertices)} vertices and {len(tri_triangles)} triangles")
    
    # IMPORTANT: In MeshIt, new vertices created by Triangle are added to the surface
    # This is the key difference from our previous implementation
    
    # Create a mapping from Triangle vertices to surface vertices
    vertex_map = {}
    
    # First, map the original vertices
    for i, vertex_2d in enumerate(vertices_2d):
        for j, tri_vertex in enumerate(tri_vertices):
            if np.allclose(vertex_2d, tri_vertex, atol=1e-10):
                vertex_map[j] = i
                break
    
    # Then, add new vertices created by Triangle to the surface
    # and update the mapping
    original_vertex_count = len(vertices)
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    
    for i, tri_vertex in enumerate(tri_vertices):
        if i not in vertex_map:
            # This is a new vertex created by Triangle
            # Project it back to 3D using the inverse rotation
            tri_vertex_3d = np.dot(inverse_rotation_matrix, np.array([tri_vertex[0], tri_vertex[1], 0]))
            new_vertex = Vector3D(tri_vertex_3d[0], tri_vertex_3d[1], tri_vertex_3d[2])
            
            # Add the new vertex to the surface
            surface.add_vertex(new_vertex)
            
            # Update the mapping
            vertex_map[i] = len(surface.vertices) - 1
    
    # Create triangles using the updated vertex indices
    triangles = []
    for triangle in tri_triangles:
        v1 = vertex_map.get(triangle[0], -1)
        v2 = vertex_map.get(triangle[1], -1)
        v3 = vertex_map.get(triangle[2], -1)
        
        # Only add valid triangles
        if v1 >= 0 and v2 >= 0 and v3 >= 0:
            triangles.append([v1, v2, v3])
    
    print(f"Created {len(triangles)} triangles after mapping")
    
    # Update the surface triangles
    surface.triangles = triangles
    
    return triangles

def visualize_improved_triangulation(surface, hull_size):
    """Visualize the improved triangulation."""
    # Extract vertices and triangles
    vertices = np.array([[v.x, v.y, v.z] for v in surface.vertices])
    triangles = surface.triangles
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the triangulation
    for triangle in triangles:
        v1 = vertices[triangle[0], :2]
        v2 = vertices[triangle[1], :2]
        v3 = vertices[triangle[2], :2]
        triangle_vertices = np.array([v1, v2, v3])
        ax.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], 
                color='tan', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Plot the original points (first 25 vertices)
    original_count = 25  # Assuming 5x5 grid
    ax.scatter(vertices[:original_count, 0], vertices[:original_count, 1], 
              color='blue', s=50, zorder=3, label='Original Points')
    
    # Plot the new points created by Triangle
    if len(vertices) > original_count:
        ax.scatter(vertices[original_count:, 0], vertices[original_count:, 1], 
                  color='green', s=30, zorder=3, label='New Points')
    
    # Plot the convex hull
    hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
    hull_points_closed = np.vstack([hull_points, hull_points[0]])  # Close the loop
    ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2, label='Convex Hull')
    
    # Set title and labels
    ax.set_title(f"Improved Triangulation (hull_size={hull_size}, {len(triangles)} triangles, {len(vertices)} vertices)", fontsize=16)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(f"improved_triangulation_{hull_size}.png")
    plt.show()

def compare_hull_sizes():
    """Compare triangulations with different hull sizes."""
    hull_sizes = [0.5, 1.0, 2.0, 4.0]
    
    # Create a multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, hull_size in enumerate(hull_sizes):
        ax = axes[i]
        
        # Generate points
        points = generate_grid_points()
        
        # Create surface
        surface = meshit.extensions.create_surface_from_points(points)
        
        # Calculate convex hull
        meshit.extensions.enhanced_calculate_convex_hull(surface)
        
        # Triangulate with improved method
        improved_triangulate_with_triangle(surface, hull_size)
        
        # Extract vertices and triangles
        vertices = np.array([[v.x, v.y, v.z] for v in surface.vertices])
        triangles = surface.triangles
        
        # Plot the triangulation
        for triangle in triangles:
            v1 = vertices[triangle[0], :2]
            v2 = vertices[triangle[1], :2]
            v3 = vertices[triangle[2], :2]
            triangle_vertices = np.array([v1, v2, v3])
            ax.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], 
                    color='tan', alpha=0.7, edgecolor='black', linewidth=1)
        
        # Plot the original points (first 25 vertices)
        original_count = 25  # Assuming 5x5 grid
        ax.scatter(vertices[:original_count, 0], vertices[:original_count, 1], 
                  color='blue', s=30, zorder=3)
        
        # Plot the new points created by Triangle
        if len(vertices) > original_count:
            ax.scatter(vertices[original_count:, 0], vertices[original_count:, 1], 
                      color='green', s=15, zorder=3)
        
        # Plot the convex hull
        hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
        hull_points_closed = np.vstack([hull_points, hull_points[0]])  # Close the loop
        ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2)
        
        # Set title
        ax.set_title(f"hull_size = {hull_size} ({len(triangles)} triangles, {len(vertices)} vertices)", fontsize=12)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add overall title
    fig.suptitle("Comparison of Triangulations with Different Hull Sizes", fontsize=16)
    
    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig("hull_size_comparison.png")
    plt.show()

if __name__ == "__main__":
    # Generate points
    points = generate_grid_points()
    
    # Create surface
    surface = meshit.extensions.create_surface_from_points(points)
    
    # Calculate convex hull
    meshit.extensions.enhanced_calculate_convex_hull(surface)
    
    # Triangulate with improved method
    hull_size = 0.5  # Using a smaller hull_size to create more triangles
    improved_triangulate_with_triangle(surface, hull_size)
    
    # Visualize the triangulation
    visualize_improved_triangulation(surface, hull_size)
    
    # Compare different hull sizes
    compare_hull_sizes() 