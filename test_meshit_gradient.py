import numpy as np
import matplotlib.pyplot as plt
from meshit.extensions import create_surface_from_points, triangulate_with_triangle

def generate_input_points():
    """Generate a simple set of 5 points on a flat surface."""
    points = np.array([
        [-1, -1, 0],  # Bottom left
        [1, -1, 0],   # Bottom right
        [1, 1, 0],    # Top right
        [-1, 1, 0],   # Top left
        [0, 0, 0]     # Center
    ])
    return points

def visualize_step(points, edges=None, triangles=None, title=""):
    """Visualize points, edges, and triangles in 2D."""
    plt.figure(figsize=(8, 8))
    plt.grid(True)
    
    # Plot points
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=50, label='Points')
    
    # Plot edges if provided
    if edges is not None:
        for edge in edges:
            p1, p2 = points[edge]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=1, label='_')
    
    # Plot triangles if provided
    if triangles is not None:
        for triangle in triangles:
            for i in range(3):
                p1 = points[triangle[i]]
                p2 = points[triangle[(i + 1) % 3]]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=0.5, label='_')
    
    plt.title(title)
    plt.axis('equal')
    plt.legend()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

def main():
    # Step 1: Generate input points
    points = generate_input_points()
    print("Step 1: Generated input points")
    visualize_step(points, title="Input Points")
    
    # Step 2: Create surface
    surface = create_surface_from_points(points)
    print("Step 2: Created surface")
    
    # Step 3: Calculate convex hull
    surface.calculate_convex_hull()
    print("Step 3: Calculated convex hull")
    
    # Create edges for convex hull visualization
    hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
    hull_edges = [[i, (i + 1) % len(hull_points)] for i in range(len(hull_points))]
    visualize_step(points, edges=hull_edges, title="Surface with Convex Hull")
    
    # Step 4: Triangulate with different gradients
    for gradient in [0.5, 2.0]:
        print(f"\nTriangulating with gradient = {gradient}")
        vertices, triangles = triangulate_with_triangle(surface, gradient=gradient)
        
        # Convert vertices to numpy array for visualization
        vertices_array = np.array([[v[0], v[1], v[2]] for v in vertices])
        
        # Calculate edge statistics
        edge_lengths = []
        for triangle in triangles:
            for i in range(3):
                p1 = vertices[triangle[i]]
                p2 = vertices[triangle[(i + 1) % 3]]
                edge_length = np.sqrt(np.sum((p1 - p2) ** 2))
                edge_lengths.append(edge_length)
        
        print(f"Number of triangles: {len(triangles)}")
        print(f"Edge lengths: Average = {np.mean(edge_lengths):.3f}, "
              f"Min = {np.min(edge_lengths):.3f}, Max = {np.max(edge_lengths):.3f}")
        
        visualize_step(vertices_array, triangles=triangles, 
                      title=f"Triangulation (gradient={gradient})")

if __name__ == "__main__":
    main() 