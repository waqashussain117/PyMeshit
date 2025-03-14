import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
from scipy.spatial import ConvexHull

def generate_grid_points(n=5):
    """Generate a grid of points in the XY plane."""
    points = []
    for x in np.linspace(0, 10, n):
        for y in np.linspace(0, 10, n):
            points.append([x, y])
    return np.array(points)

def triangulate_with_area_constraint(points, max_area=1.0):
    """
    Triangulate points with a maximum area constraint.
    Smaller max_area values create more triangles.
    """
    print(f"Triangulating with max_area = {max_area}")
    
    # Compute the convex hull
    hull = ConvexHull(points)
    
    # Create segments for the convex hull boundary
    segments = []
    for i in range(len(hull.vertices)):
        segments.append([hull.vertices[i], hull.vertices[(i + 1) % len(hull.vertices)]])
    
    # Prepare data for Triangle library
    vertices = dict(vertices=points)
    vertices['segments'] = np.array(segments)
    
    # Triangulate with quality constraints
    # p: Preserve the boundary segments
    # q: Quality mesh generation with minimum angle of 30 degrees
    # a: Maximum triangle area constraint
    options = f'pq30a{max_area}'
    print(f"Triangle options: {options}")
    triangulation = tr.triangulate(vertices, options)
    
    # Extract triangulation results
    tri_vertices = triangulation['vertices']
    tri_triangles = triangulation['triangles']
    
    print(f"Triangulation created {len(tri_vertices)} vertices and {len(tri_triangles)} triangles")
    print(f"Original points: {len(points)}")
    print(f"New points added: {len(tri_vertices) - len(points)}")
    
    return triangulation, hull

def visualize_triangulation(triangulation, hull, max_area):
    """Visualize the triangulation."""
    # Extract data
    tri_vertices = triangulation['vertices']
    tri_triangles = triangulation['triangles']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the triangulation
    for triangle in tri_triangles:
        v1 = tri_vertices[triangle[0]]
        v2 = tri_vertices[triangle[1]]
        v3 = tri_vertices[triangle[2]]
        triangle_vertices = np.array([v1, v2, v3])
        ax.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], 
                color='tan', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Plot the original points
    original_count = 25  # Assuming 5x5 grid
    ax.scatter(tri_vertices[:original_count, 0], tri_vertices[:original_count, 1], 
              color='blue', s=50, zorder=3, label='Original Points')
    
    # Plot the new points created by Triangle
    if len(tri_vertices) > original_count:
        ax.scatter(tri_vertices[original_count:, 0], tri_vertices[original_count:, 1], 
                  color='green', s=30, zorder=3, label='New Points')
    
    # Plot the convex hull
    hull_points = tri_vertices[hull.vertices]
    hull_points_closed = np.vstack([hull_points, hull_points[0]])  # Close the loop
    ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2, label='Convex Hull')
    
    # Set title and labels
    ax.set_title(f"Triangulation with max_area = {max_area}\n({len(tri_triangles)} triangles, {len(tri_vertices)} vertices)", fontsize=16)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(f"direct_triangulation_{max_area}.png")
    plt.show()

def compare_area_constraints():
    """Compare triangulations with different maximum area constraints."""
    max_areas = [10.0, 5.0, 1.0, 0.5]
    
    # Create a multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Generate points
    points = generate_grid_points()
    
    for i, max_area in enumerate(max_areas):
        ax = axes[i]
        
        # Triangulate
        triangulation, hull = triangulate_with_area_constraint(points, max_area)
        
        # Extract data
        tri_vertices = triangulation['vertices']
        tri_triangles = triangulation['triangles']
        
        # Plot the triangulation
        for triangle in tri_triangles:
            v1 = tri_vertices[triangle[0]]
            v2 = tri_vertices[triangle[1]]
            v3 = tri_vertices[triangle[2]]
            triangle_vertices = np.array([v1, v2, v3])
            ax.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], 
                    color='tan', alpha=0.7, edgecolor='black', linewidth=1)
        
        # Plot the original points
        original_count = len(points)
        ax.scatter(tri_vertices[:original_count, 0], tri_vertices[:original_count, 1], 
                  color='blue', s=30, zorder=3)
        
        # Plot the new points created by Triangle
        if len(tri_vertices) > original_count:
            ax.scatter(tri_vertices[original_count:, 0], tri_vertices[original_count:, 1], 
                      color='green', s=15, zorder=3)
        
        # Plot the convex hull
        hull_points = points[hull.vertices]
        hull_points_closed = np.vstack([hull_points, hull_points[0]])  # Close the loop
        ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2)
        
        # Set title
        ax.set_title(f"max_area = {max_area}\n({len(tri_triangles)} triangles, {len(tri_vertices)} vertices)", fontsize=12)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add overall title
    fig.suptitle("Comparison of Triangulations with Different Maximum Area Constraints", fontsize=16)
    
    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig("area_constraint_comparison.png")
    plt.show()

if __name__ == "__main__":
    # Generate points
    points = generate_grid_points()
    
    # Triangulate with a very small area constraint to create many triangles
    max_area = 0.5  # This should create many triangles
    triangulation, hull = triangulate_with_area_constraint(points, max_area)
    
    # Visualize the triangulation
    visualize_triangulation(triangulation, hull, max_area)
    
    # Compare different area constraints
 