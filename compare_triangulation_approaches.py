import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
from scipy.spatial import ConvexHull
import meshit

def generate_grid_points(n=5):
    """Generate a grid of points in the XY plane."""
    points = []
    for x in np.linspace(0, 10, n):
        for y in np.linspace(0, 10, n):
            points.append([x, y, 0])
    return points

def direct_triangulation(points_2d, hull_size=2.0):
    """
    Triangulate points directly using the Triangle library.
    This is the approach used in direct_triangulation.py.
    """
    print(f"\n=== Direct Triangulation with hull_size = {hull_size} ===")
    
    # Compute the convex hull
    hull = ConvexHull(points_2d)
    
    # Create segments for the convex hull boundary
    segments = []
    for i in range(len(hull.vertices)):
        segments.append([hull.vertices[i], hull.vertices[(i + 1) % len(hull.vertices)]])
    
    # Prepare data for Triangle library
    vertices = dict(vertices=points_2d)
    vertices['segments'] = np.array(segments)
    
    # Calculate the hull area
    hull_area = hull.volume  # In 2D, volume is area
    
    # Calculate the maximum triangle area
    max_area = hull_area / (hull_size * hull_size)
    print(f"Hull area: {hull_area:.2f}, Max triangle area: {max_area:.2f}")
    
    # Triangulate with quality constraints
    options = f'pq30a{max_area}'
    print(f"Triangle options: {options}")
    triangulation = tr.triangulate(vertices, options)
    
    # Extract triangulation results
    tri_vertices = triangulation['vertices']
    tri_triangles = triangulation['triangles']
    
    print(f"Triangulation created {len(tri_vertices)} vertices and {len(tri_triangles)} triangles")
    
    return {
        'vertices': tri_vertices,
        'triangles': tri_triangles,
        'hull': hull,
        'max_area': max_area
    }

def meshit_triangulation(points_3d, hull_size=2.0):
    """
    Triangulate points using the MeshIt approach.
    This is the approach used in meshit/extensions.py.
    """
    print(f"\n=== MeshIt Triangulation with hull_size = {hull_size} ===")
    
    # Create a surface from the points
    surface = meshit.extensions.create_surface_from_points(points_3d)
    
    # Calculate the convex hull
    meshit.extensions.enhanced_calculate_convex_hull(surface)
    
    # Triangulate the surface
    triangles = meshit.extensions.triangulate_with_triangle(surface, hull_size=hull_size)
    
    # Extract vertices and hull points
    vertices = np.array([[v.x, v.y, v.z] for v in surface.vertices])
    hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
    
    # Create a 2D convex hull for visualization
    points_2d = vertices[:, :2]
    hull = ConvexHull(points_2d)
    
    print(f"Created {len(triangles)} triangles")
    
    return {
        'vertices': vertices,
        'triangles': triangles,
        'hull': hull,
        'surface': surface
    }

def visualize_comparison(direct_result, meshit_result, hull_size):
    """Visualize the comparison between direct and MeshIt triangulation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Direct Triangulation
    ax1.set_title(f"Direct Triangulation (hull_size={hull_size}, {len(direct_result['triangles'])} triangles)")
    
    # Plot the triangulation
    tri_vertices = direct_result['vertices']
    tri_triangles = direct_result['triangles']
    
    for triangle in tri_triangles:
        v1 = tri_vertices[triangle[0]]
        v2 = tri_vertices[triangle[1]]
        v3 = tri_vertices[triangle[2]]
        triangle_vertices = np.array([v1, v2, v3])
        ax1.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], 
                color='tan', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Plot the original points
    original_points = np.array(generate_grid_points())[:, :2]
    ax1.scatter(original_points[:, 0], original_points[:, 1], color='blue', s=50, zorder=3)
    
    # Plot the convex hull
    hull = direct_result['hull']
    hull_points = original_points[hull.vertices]
    hull_points_closed = np.vstack([hull_points, hull_points[0]])  # Close the loop
    ax1.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2)
    
    ax1.set_aspect('equal')
    
    # MeshIt Triangulation
    ax2.set_title(f"MeshIt Triangulation (hull_size={hull_size}, {len(meshit_result['triangles'])} triangles)")
    
    # Plot the triangulation
    vertices = meshit_result['vertices']
    triangles = meshit_result['triangles']
    
    for triangle in triangles:
        v1 = vertices[triangle[0], :2]
        v2 = vertices[triangle[1], :2]
        v3 = vertices[triangle[2], :2]
        triangle_vertices = np.array([v1, v2, v3])
        ax2.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], 
                color='tan', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Plot the original points
    ax2.scatter(vertices[:, 0], vertices[:, 1], color='blue', s=50, zorder=3)
    
    # Plot the convex hull
    hull = meshit_result['hull']
    hull_points = vertices[:, :2][hull.vertices]
    hull_points_closed = np.vstack([hull_points, hull_points[0]])  # Close the loop
    ax2.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2)
    
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f"triangulation_comparison_{hull_size}.png")
    plt.show()

def explain_differences():
    """Explain the key differences between the two triangulation approaches."""
    print("\n=== Key Differences Between Triangulation Approaches ===")
    print("1. Vertex Mapping:")
    print("   - Direct approach: Uses all vertices created by Triangle, including new ones")
    print("   - MeshIt approach: Only uses original vertices, discarding triangles with new vertices")
    print("\n2. Triangle Filtering:")
    print("   - Direct approach: Keeps all triangles created by Triangle")
    print("   - MeshIt approach: Filters out triangles that use vertices not in the original set")
    print("\n3. Area Calculation:")
    print("   - Direct approach: Uses hull_area / (hull_size * hull_size) for max_area")
    print("   - MeshIt approach: Uses the same formula but may have different hull area calculation")
    print("\n4. Vertex Projection:")
    print("   - Direct approach: Works directly with 2D points")
    print("   - MeshIt approach: Projects 3D points to 2D, then maps back to 3D")
    print("\nConclusion:")
    print("The MeshIt approach produces fewer triangles because it only uses the original vertices,")
    print("while the direct approach allows Triangle to create new vertices for better triangulation.")
    print("This is why direct triangulation produces more triangles with better quality.")

if __name__ == "__main__":
    # Set the hull size
    hull_size = 2.0
    
    # Generate points
    points_3d = generate_grid_points()
    points_2d = np.array(points_3d)[:, :2]
    
    # Perform direct triangulation
    direct_result = direct_triangulation(points_2d, hull_size)
    
    # Perform MeshIt triangulation
    meshit_result = meshit_triangulation(points_3d, hull_size)
    
    # Visualize the comparison
    visualize_comparison(direct_result, meshit_result, hull_size)
    
    # Explain the differences
    explain_differences() 