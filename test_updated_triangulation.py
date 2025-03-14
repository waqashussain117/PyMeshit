import numpy as np
import matplotlib.pyplot as plt
import meshit
from meshit.core._meshit import Vector3D

def generate_grid_points(n=5):
    """Generate a grid of points in the XY plane."""
    points = []
    for x in np.linspace(0, 10, n):
        for y in np.linspace(0, 10, n):
            points.append([x, y, 0])
    return points

def test_triangulation_with_hull_size(hull_size):
    """Test triangulation with a specific hull size."""
    print(f"\n=== Testing triangulation with hull_size = {hull_size} ===")
    
    # Step 1: Create points
    print("Step 1: Creating points")
    raw_points = generate_grid_points()
    print(f"Created {len(raw_points)} points")
    
    # Step 2: Create surface
    print("Step 2: Creating surface")
    surface = meshit.extensions.create_surface_from_points(raw_points)
    
    # Step 3: Calculate convex hull
    print("Step 3: Calculating convex hull")
    meshit.extensions.enhanced_calculate_convex_hull(surface)
    
    # Step 4: Triangulate
    print("Step 4: Triangulating")
    triangles = meshit.extensions.triangulate_with_triangle(surface, hull_size=hull_size)
    
    # Ensure we have the complete output for all hull sizes
    original_vertex_count = 25  # 5x5 grid
    new_vertex_count = len(surface.vertices) - original_vertex_count
    print(f"Created {len(triangles)} valid triangles after mapping")
    print(f"Surface now has {len(surface.vertices)} vertices ({new_vertex_count} new vertices added)")
    
    # Step 5: Visualize
    print("Step 5: Visualizing")
    visualize_triangulation(surface, hull_size)
    
    return surface

def visualize_triangulation(surface, hull_size):
    """Visualize the triangulation."""
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
    ax.set_title(f"Triangulation with hull_size = {hull_size}\n({len(triangles)} triangles, {len(vertices)} vertices)", fontsize=16)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(f"updated_triangulation_{hull_size}.png")
    plt.show()

def compare_hull_sizes():
    """Compare triangulations with different hull sizes."""
    hull_sizes = [0.5, 1.0, 2.0, 4.0]
    
    # Create a multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, hull_size in enumerate(hull_sizes):
        ax = axes[i]
        
        # Create points
        raw_points = generate_grid_points()
        
        # Create surface
        surface = meshit.extensions.create_surface_from_points(raw_points)
        
        # Calculate convex hull
        meshit.extensions.enhanced_calculate_convex_hull(surface)
        
        # Triangulate
        meshit.extensions.triangulate_with_triangle(surface, hull_size=hull_size)
        
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
        ax.set_title(f"hull_size = {hull_size}\n({len(triangles)} triangles, {len(vertices)} vertices)", fontsize=12)
        
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
    # Test with different hull sizes
    hull_sizes = [0.5, 1.0, 2.0, 4.0]
    
    for hull_size in hull_sizes:
        test_triangulation_with_hull_size(hull_size)
    
    # Compare different hull sizes
    compare_hull_sizes() 