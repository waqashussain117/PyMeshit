import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

def generate_grid_points(n=5):
    """Generate a grid of points in the XY plane."""
    points = []
    for x in np.linspace(0, 10, n):
        for y in np.linspace(0, 10, n):
            points.append([x, y, 0])  # Adding Z=0 for 3D
    return np.array(points)

def is_planar(points, tol=1e-8):
    """Check if points are approximately coplanar."""
    if len(points) < 4:
        return True
    
    # Take the first three non-collinear points to define a plane
    p0 = points[0]
    
    # Find a point that's not collinear with p0
    for i in range(1, len(points)):
        p1 = points[i]
        v1 = p1 - p0
        if np.linalg.norm(v1) > tol:
            break
    else:
        return True  # All points are at the same location
    
    # Find a point that's not collinear with p0 and p1
    for i in range(1, len(points)):
        p2 = points[i]
        v2 = p2 - p0
        if np.linalg.norm(np.cross(v1, v2)) > tol:
            break
    else:
        return True  # All points are collinear
    
    # Calculate the normal of the plane
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    # Check if all points are on this plane
    for p in points:
        dist = np.abs(np.dot(p - p0, normal))
        if dist > tol:
            return False
    
    return True

def triangulate_with_triangle(points_3d, hull_size=2.0):
    """
    Triangulate a set of 3D points using the Triangle library.
    This function mimics what should happen in MeshIt's triangulation.
    """
    print(f"=== Triangulating with hull_size = {hull_size} ===")
    print(f"Input: {len(points_3d)} 3D points")
    
    # Check if points are planar
    planar = is_planar(points_3d)
    if planar:
        print("Detected a planar point set")
    
    # Step 1: Calculate the convex hull
    if planar:
        # For planar points, use a 2D convex hull
        points_2d = points_3d[:, :2]  # Project to XY plane
        hull_2d = ConvexHull(points_2d)
        hull_vertices = hull_2d.vertices
        print(f"Created a 2D convex hull with {len(hull_vertices)} points")
    else:
        # For non-planar points, use a 3D convex hull
        hull_3d = ConvexHull(points_3d)
        hull_vertices = hull_3d.vertices
        print(f"Created a 3D convex hull with {len(hull_vertices)} points")
    
    # Step 2: Project points to 2D for triangulation
    # For this example, we know it's the XY plane, but in general:
    if planar:
        # For planar points, we can directly use the XY coordinates
        points_2d = points_3d[:, :2]
    else:
        # For non-planar points, we need to project to the best-fitting plane
        # This would involve calculating the normal and creating a rotation matrix
        # For simplicity, we'll just use the XY coordinates in this example
        points_2d = points_3d[:, :2]
    
    # Step 3: Prepare data for Triangle library
    vertices = dict(vertices=points_2d)
    
    # Add segments for the convex hull boundary
    segments = []
    for i in range(len(hull_vertices)):
        segments.append([hull_vertices[i], hull_vertices[(i + 1) % len(hull_vertices)]])
    vertices['segments'] = np.array(segments)
    
    # Step 4: Triangulate with Triangle library
    # Calculate the maximum area based on hull_size
    hull_area = ConvexHull(points_2d).volume  # In 2D, volume is area
    max_area = hull_area / (hull_size * hull_size)
    print(f"Hull area: {hull_area:.2f}, Max triangle area: {max_area:.2f}")
    
    # Triangulate with quality constraints
    options = f'pq30a{max_area}'
    print(f"Triangle options: {options}")
    triangulation = tr.triangulate(vertices, options)
    
    # Step 5: Extract triangulation results
    tri_vertices_2d = triangulation['vertices']
    tri_triangles = triangulation['triangles']
    
    print(f"Triangulation created {len(tri_vertices_2d)} vertices and {len(tri_triangles)} triangles")
    
    # Step 6: Map 2D vertices back to 3D
    # For new vertices created by Triangle, we need to set Z=0
    tri_vertices_3d = np.zeros((len(tri_vertices_2d), 3))
    tri_vertices_3d[:, :2] = tri_vertices_2d
    
    # For original vertices, we should use the original Z values
    # But since our example is flat (Z=0), this doesn't matter
    
    # Step 7: Create a mapping from original points to triangulation vertices
    # This is crucial for MeshIt to correctly reference the vertices
    vertex_map = {}
    for i, point in enumerate(points_3d):
        # Find the corresponding vertex in the triangulation
        # In a real implementation, we'd need to handle floating point precision
        for j, tri_vertex in enumerate(tri_vertices_3d):
            if np.allclose(point[:2], tri_vertex[:2], atol=1e-10):
                vertex_map[i] = j
                break
    
    print(f"Mapped {len(vertex_map)} original vertices to triangulation vertices")
    
    return {
        'original_points_3d': points_3d,
        'hull_vertices': hull_vertices,
        'tri_vertices_2d': tri_vertices_2d,
        'tri_vertices_3d': tri_vertices_3d,
        'tri_triangles': tri_triangles,
        'vertex_map': vertex_map,
        'hull_size': hull_size,
        'max_area': max_area,
        'planar': planar
    }

def visualize_triangulation(result):
    """Visualize the triangulation in both 2D and 3D."""
    # Extract data from result
    points_3d = result['original_points_3d']
    hull_vertices = result['hull_vertices']
    tri_vertices_2d = result['tri_vertices_2d']
    tri_vertices_3d = result['tri_vertices_3d']
    tri_triangles = result['tri_triangles']
    hull_size = result['hull_size']
    
    # Create a figure with 2 subplots (2D and 3D views)
    fig = plt.figure(figsize=(15, 7))
    
    # 2D View
    ax1 = fig.add_subplot(121)
    ax1.set_title(f"2D Triangulation (hull_size={hull_size}, {len(tri_triangles)} triangles)")
    
    # Plot the triangulation
    for triangle in tri_triangles:
        v1 = tri_vertices_2d[triangle[0]]
        v2 = tri_vertices_2d[triangle[1]]
        v3 = tri_vertices_2d[triangle[2]]
        triangle_vertices = np.array([v1, v2, v3])
        ax1.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], 
                color='tan', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Plot the original points
    points_2d = points_3d[:, :2]
    ax1.scatter(points_2d[:, 0], points_2d[:, 1], color='blue', s=50, zorder=3, label='Original Points')
    
    # Plot the additional points created during triangulation
    if len(tri_vertices_2d) > len(points_2d):
        additional_points = tri_vertices_2d[len(points_2d):]
        ax1.scatter(additional_points[:, 0], additional_points[:, 1], 
                  color='green', s=30, zorder=3, label='Added Points')
    
    # Plot the convex hull
    hull_points = points_2d[hull_vertices]
    hull_points_closed = np.vstack([hull_points, hull_points[0]])  # Close the loop
    ax1.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2, label='Convex Hull')
    
    ax1.set_aspect('equal')
    ax1.legend()
    
    # 3D View
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"3D Triangulation (hull_size={hull_size}, {len(tri_triangles)} triangles)")
    
    # Plot the triangulation
    for triangle in tri_triangles:
        v1 = tri_vertices_3d[triangle[0]]
        v2 = tri_vertices_3d[triangle[1]]
        v3 = tri_vertices_3d[triangle[2]]
        triangle_vertices = np.array([v1, v2, v3])
        ax2.plot_trisurf(triangle_vertices[:, 0], triangle_vertices[:, 1], triangle_vertices[:, 2], 
                        color='tan', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Plot the original points
    ax2.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
               color='blue', s=50, label='Original Points')
    
    # Set equal aspect ratio
    ax2.set_box_aspect([1, 1, 0.1])  # Slightly compressed in Z
    
    # Set labels
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

def compare_triangulations(hull_sizes):
    """Compare triangulations with different hull sizes."""
    # Generate points
    points = generate_grid_points()
    
    # Create a multi-panel figure
    n_sizes = len(hull_sizes)
    n_cols = 2
    n_rows = (n_sizes + 1) // 2  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten() if n_sizes > 1 else [axes]  # Handle single subplot case
    
    # Process each hull size
    for i, hull_size in enumerate(hull_sizes):
        ax = axes[i]
        
        # Triangulate
        result = triangulate_with_triangle(points, hull_size)
        
        # Extract data
        tri_vertices_2d = result['tri_vertices_2d']
        tri_triangles = result['tri_triangles']
        
        # Plot the triangulation
        for triangle in tri_triangles:
            v1 = tri_vertices_2d[triangle[0]]
            v2 = tri_vertices_2d[triangle[1]]
            v3 = tri_vertices_2d[triangle[2]]
            triangle_vertices = np.array([v1, v2, v3])
            ax.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], 
                    color='tan', alpha=0.7, edgecolor='black', linewidth=1)
        
        # Plot the original points
        points_2d = points[:, :2]
        ax.scatter(points_2d[:, 0], points_2d[:, 1], color='blue', s=30, zorder=3)
        
        # Plot the convex hull
        hull_vertices = result['hull_vertices']
        hull_points = points_2d[hull_vertices]
        hull_points_closed = np.vstack([hull_points, hull_points[0]])  # Close the loop
        ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2)
        
        # Set title
        ax.set_title(f"hull_size = {hull_size} ({len(tri_triangles)} triangles)", fontsize=12)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Add overall title
    fig.suptitle("Comparison of Triangulations with Different Hull Sizes", fontsize=16)
    
    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.show()

def explain_meshit_implementation():
    """Print an explanation of how to implement this in MeshIt."""
    print("\n=== How to Implement in MeshIt ===")
    print("Based on the above demonstration, here's how to implement triangulation in MeshIt:")
    print("\n1. Check if the points are coplanar")
    print("2. Calculate the convex hull (2D for planar points, 3D for non-planar)")
    print("3. Project the points to 2D using the normal of the best-fitting plane")
    print("4. Calculate the maximum triangle area based on the hull size parameter")
    print("5. Use the Triangle library to triangulate the 2D points")
    print("6. Map the triangulation vertices back to the original 3D points")
    print("7. Create triangles using the original point indices")
    print("\nKey Issues to Fix in MeshIt:")
    print("- Handle planar point sets properly")
    print("- Ensure the hull_size parameter correctly controls triangle density")
    print("- Handle the mapping between Triangle's vertices and MeshIt's vertices")
    print("- Filter out invalid triangles (those with indices outside the valid range)")
    print("- Ensure the triangulation respects the convex hull boundary")

if __name__ == "__main__":
    # Ask user which visualization to show
    print("1. Single triangulation")
    print("2. Compare multiple triangulations")
    print("3. Just explain the MeshIt implementation")
    choice = input("Enter your choice (1, 2, or 3): ")
    
    if choice == "1":
        # Ask user for hull size
        try:
            hull_size = float(input("Enter hull size (e.g., 0.5, 1.0, 2.0, 4.0): "))
        except ValueError:
            hull_size = 2.0
            print(f"Invalid input. Using default hull size: {hull_size}")
        
        # Generate points
        points = generate_grid_points()
        
        # Triangulate
        result = triangulate_with_triangle(points, hull_size)
        
        # Visualize
        visualize_triangulation(result)
        
        # Explain implementation
        explain_meshit_implementation()
        
    elif choice == "2":
        # Define hull sizes to compare
        hull_sizes = [0.5, 1.0, 2.0, 4.0]
        
        # Compare triangulations
        compare_triangulations(hull_sizes)
        
        # Explain implementation
        explain_meshit_implementation()
        
    else:
        # Just explain the implementation
        explain_meshit_implementation() 