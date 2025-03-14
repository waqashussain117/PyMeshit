import numpy as np
import matplotlib.pyplot as plt
import triangle as tr

def generate_grid_points(n=5):
    """Generate a grid of points in the XY plane."""
    points = []
    for x in np.linspace(0, 10, n):
        for y in np.linspace(0, 10, n):
            points.append([x, y])
    return np.array(points)

def triangulate_points(points, max_area):
    """Triangulate points using the Triangle library."""
    # Create a dictionary with the vertices
    vertices = dict(vertices=points)
    
    # Compute the convex hull
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    
    # Create segments for the convex hull boundary
    segments = []
    for i in range(len(hull.vertices)):
        segments.append([hull.vertices[i], hull.vertices[(i + 1) % len(hull.vertices)]])
    
    # Add segments to the dictionary
    vertices['segments'] = np.array(segments)
    
    # Triangulate with a maximum area constraint
    options = f'pq30a{max_area}'
    print(f"Triangulating with options: {options}")
    triangulation = tr.triangulate(vertices, options)
    
    return triangulation, hull

def visualize_triangulation(points, triangulation, hull, max_area):
    """Visualize the triangulation."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get the triangulation vertices and triangles
    tri_vertices = triangulation['vertices']
    tri_triangles = triangulation['triangles']
    
    print(f"Original points: {len(points)}")
    print(f"Triangulation vertices: {len(tri_vertices)}")
    print(f"Number of triangles: {len(tri_triangles)}")
    
    # Plot the triangulation directly
    for triangle in tri_triangles:
        v1 = tri_vertices[triangle[0]]
        v2 = tri_vertices[triangle[1]]
        v3 = tri_vertices[triangle[2]]
        triangle_vertices = np.array([v1, v2, v3])
        ax.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], 
                color='tan', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Plot the original points
    ax.scatter(points[:, 0], points[:, 1], color='blue', s=50, zorder=3, label='Original Points')
    
    # Plot the additional points created during triangulation
    if len(tri_vertices) > len(points):
        additional_points = tri_vertices[len(points):]
        ax.scatter(additional_points[:, 0], additional_points[:, 1], 
                  color='green', s=30, zorder=3, label='Added Points')
    
    # Plot the convex hull
    hull_points = points[hull.vertices]
    hull_points_closed = np.vstack([hull_points, hull_points[0]])  # Close the loop
    ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2, label='Convex Hull')
    
    # Set title and labels
    ax.set_title(f"Triangulation with max_area = {max_area} ({len(tri_triangles)} triangles)", fontsize=16)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def compare_triangulations(max_areas):
    """Compare triangulations with different maximum area constraints."""
    # Generate points
    points = generate_grid_points()
    
    # Create a multi-panel figure
    n_areas = len(max_areas)
    n_cols = 2
    n_rows = (n_areas + 1) // 2  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten() if n_areas > 1 else [axes]  # Handle single subplot case
    
    # Process each maximum area
    for i, max_area in enumerate(max_areas):
        ax = axes[i]
        
        # Triangulate
        triangulation, hull = triangulate_points(points, max_area)
        
        # Get the triangulation vertices and triangles
        tri_vertices = triangulation['vertices']
        tri_triangles = triangulation['triangles']
        
        # Plot the triangulation directly
        for triangle in tri_triangles:
            v1 = tri_vertices[triangle[0]]
            v2 = tri_vertices[triangle[1]]
            v3 = tri_vertices[triangle[2]]
            triangle_vertices = np.array([v1, v2, v3])
            ax.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], 
                    color='tan', alpha=0.7, edgecolor='black', linewidth=1)
        
        # Plot the original points
        ax.scatter(points[:, 0], points[:, 1], color='blue', s=30, zorder=3)
        
        # Plot the convex hull
        hull_points = points[hull.vertices]
        hull_points_closed = np.vstack([hull_points, hull_points[0]])  # Close the loop
        ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2)
        
        # Set title
        ax.set_title(f"max_area = {max_area} ({len(tri_triangles)} triangles)", fontsize=12)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Add overall title
    fig.suptitle("Comparison of Triangulations with Different Maximum Area Constraints", fontsize=16)
    
    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.show()

if __name__ == "__main__":
    # Ask user which visualization to show
    print("1. Single triangulation")
    print("2. Compare multiple triangulations")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # Ask user for maximum area
        try:
            max_area = float(input("Enter maximum triangle area (e.g., 0.5, 1.0, 2.0, 4.0): "))
        except ValueError:
            max_area = 2.0
            print(f"Invalid input. Using default maximum area: {max_area}")
        
        # Generate points
        points = generate_grid_points()
        
        # Triangulate
        triangulation, hull = triangulate_points(points, max_area)
        
        # Visualize
        visualize_triangulation(points, triangulation, hull, max_area)
    else:
        # Define maximum areas to compare
        max_areas = [0.5, 1.0, 2.0, 4.0]
        
        # Compare triangulations
        compare_triangulations(max_areas) 