import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import meshit

def generate_grid_points(n=5):
    """Generate a grid of points in the XY plane."""
    points = []
    for x in np.linspace(0, 10, n):
        for y in np.linspace(0, 10, n):
            points.append([x, y, 0])
    return points

def visualize_triangulation(hull_size):
    """Visualize triangulation with a specific hull size using Matplotlib."""
    print(f"\n=== Visualizing triangulation with hull_size = {hull_size} ===")
    
    # Step 1: Create points
    raw_points = generate_grid_points()
    
    # Step 2: Create surface
    surface = meshit.extensions.create_surface_from_points(raw_points)
    
    # Step 3: Calculate convex hull
    meshit.extensions.enhanced_calculate_convex_hull(surface)
    
    # Step 4: Triangulate
    triangles = meshit.extensions.triangulate_with_triangle(surface, hull_size=hull_size)
    print(f"Created {len(triangles)} triangles with hull_size = {hull_size}")
    
    # Extract points and hull points
    points = np.array([[v.x, v.y, v.z] for v in surface.vertices])
    hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Filter out triangles with invalid indices
    valid_triangles = []
    num_vertices = len(points)
    for triangle in triangles:
        if all(idx < num_vertices for idx in triangle):
            valid_triangles.append(triangle)
        else:
            print(f"Skipping triangle with invalid indices: {triangle}")
    
    print(f"Using {len(valid_triangles)} valid triangles out of {len(triangles)} total")
    
    # Plot the triangulation
    for triangle in valid_triangles:
        # Get the vertices of the triangle
        v1 = points[triangle[0], :2]  # Only use x, y coordinates
        v2 = points[triangle[1], :2]
        v3 = points[triangle[2], :2]
        
        # Create a polygon for the triangle
        triangle_vertices = np.array([v1, v2, v3])
        
        # Plot the triangle
        ax.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], 
                color='tan', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Plot the original points
    ax.scatter(points[:, 0], points[:, 1], color='blue', s=50, zorder=3)
    
    # Plot the convex hull
    hull_points_closed = np.vstack([hull_points, hull_points[0]])  # Close the loop
    ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2)
    
    # Set title and labels
    ax.set_title(f"Triangulation with hull_size = {hull_size} ({len(valid_triangles)} triangles)", fontsize=16)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    return surface, valid_triangles

def compare_triangulations(hull_sizes):
    """Compare triangulations with different hull sizes using Matplotlib."""
    print(f"\n=== Comparing triangulations with different hull sizes ===")
    
    # Create a multi-panel figure
    n_sizes = len(hull_sizes)
    n_cols = 2
    n_rows = (n_sizes + 1) // 2  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten() if n_sizes > 1 else [axes]  # Handle single subplot case
    
    # Process each hull size
    for i, hull_size in enumerate(hull_sizes):
        ax = axes[i]
        
        # Create triangulation
        print(f"Creating triangulation with hull_size = {hull_size}")
        
        # Step 1: Create points
        raw_points = generate_grid_points()
        
        # Step 2: Create surface
        surface = meshit.extensions.create_surface_from_points(raw_points)
        
        # Step 3: Calculate convex hull
        meshit.extensions.enhanced_calculate_convex_hull(surface)
        
        # Step 4: Triangulate
        triangles = meshit.extensions.triangulate_with_triangle(surface, hull_size=hull_size)
        print(f"Created {len(triangles)} triangles with hull_size = {hull_size}")
        
        # Extract points and hull points
        points = np.array([[v.x, v.y, v.z] for v in surface.vertices])
        hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
        
        # Filter out triangles with invalid indices
        valid_triangles = []
        num_vertices = len(points)
        for triangle in triangles:
            if all(idx < num_vertices for idx in triangle):
                valid_triangles.append(triangle)
        
        print(f"Using {len(valid_triangles)} valid triangles out of {len(triangles)} total")
        
        # Plot the triangulation
        for triangle in valid_triangles:
            # Get the vertices of the triangle
            v1 = points[triangle[0], :2]  # Only use x, y coordinates
            v2 = points[triangle[1], :2]
            v3 = points[triangle[2], :2]
            
            # Create a polygon for the triangle
            triangle_vertices = np.array([v1, v2, v3])
            
            # Plot the triangle
            ax.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], 
                    color='tan', alpha=0.7, edgecolor='black', linewidth=1)
        
        # Plot the original points
        ax.scatter(points[:, 0], points[:, 1], color='blue', s=30, zorder=3)
        
        # Plot the convex hull
        hull_points_closed = np.vstack([hull_points, hull_points[0]])  # Close the loop
        ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2)
        
        # Set title
        ax.set_title(f"hull_size = {hull_size} ({len(valid_triangles)} triangles)", fontsize=12)
        
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

if __name__ == "__main__":
    # Ask user which visualization to show
    print("1. Single triangulation")
    print("2. Compare multiple triangulations")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # Ask user for hull size
        try:
            hull_size = float(input("Enter hull size (e.g., 0.5, 1.0, 2.0, 4.0): "))
        except ValueError:
            hull_size = 2.0
            print(f"Invalid input. Using default hull size: {hull_size}")
        
        # Visualize triangulation
        visualize_triangulation(hull_size)
    else:
        # Define hull sizes to compare
        hull_sizes = [0.5, 1.0, 2.0, 4.0]
        
        # Compare triangulations
        compare_triangulations(hull_sizes) 