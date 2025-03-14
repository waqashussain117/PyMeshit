import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import meshit
from meshit.core._meshit import Vector3D, Surface
import math
import os

def generate_grid_points(n=5, z_variation=False):
    """
    Generate a grid of points in the XY plane.
    
    Args:
        n: Number of points in each dimension
        z_variation: If True, add some variation to z coordinates
    
    Returns:
        List of [x, y, z] points
    """
    points = []
    for x in np.linspace(0, 10, n):
        for y in np.linspace(0, 10, n):
            if z_variation:
                # Add some variation to z to create a non-planar surface
                z = 0.2 * math.sin(x) * math.cos(y)
            else:
                z = 0
            points.append([x, y, z])
    return points

def create_output_directory():
    """Create output directory for visualization files."""
    if not os.path.exists("workflow_output"):
        os.makedirs("workflow_output")
    return "workflow_output"

def visualize_points(points, title="Raw Points", filename=None):
    """
    Visualize points in 3D.
    
    Args:
        points: List of [x, y, z] points
        title: Title for the plot
        filename: If provided, save the plot to this file
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    
    # Plot points
    ax.scatter(x, y, z, c='blue', marker='o', s=50)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 0.2])
    
    # Save if filename is provided
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def visualize_surface(surface, title="Surface", show_vertices=True, 
                     show_convex_hull=True, show_triangles=True, filename=None):
    """
    Visualize a surface with its vertices, convex hull, and triangulation.
    
    Args:
        surface: Surface object
        title: Title for the plot
        show_vertices: Whether to show vertices
        show_convex_hull: Whether to show convex hull
        show_triangles: Whether to show triangulation
        filename: If provided, save the plot to this file
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract vertices
    vertices = np.array([[v.x, v.y, v.z] for v in surface.vertices])
    
    # Plot vertices
    if show_vertices and len(vertices) > 0:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  c='blue', marker='o', s=50, label='Vertices')
    
    # Plot convex hull
    if show_convex_hull and hasattr(surface, 'convex_hull') and len(surface.convex_hull) > 0:
        hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
        # Connect hull points to form a polygon
        for i in range(len(hull_points)):
            j = (i + 1) % len(hull_points)
            ax.plot([hull_points[i, 0], hull_points[j, 0]],
                   [hull_points[i, 1], hull_points[j, 1]],
                   [hull_points[i, 2], hull_points[j, 2]],
                   'r-', linewidth=2)
        ax.scatter(hull_points[:, 0], hull_points[:, 1], hull_points[:, 2], 
                  c='red', marker='^', s=100, label='Convex Hull')
    
    # Plot triangulation
    if show_triangles and hasattr(surface, 'triangles') and len(surface.triangles) > 0:
        for triangle in surface.triangles:
            v1 = vertices[triangle[0]]
            v2 = vertices[triangle[1]]
            v3 = vertices[triangle[2]]
            
            # Create a polygon for the triangle
            verts = [v1, v2, v3]
            ax.plot_trisurf([v[0] for v in verts], [v[1] for v in verts], [v[2] for v in verts],
                           color='tan', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add legend
    if (show_vertices and len(vertices) > 0) or (show_convex_hull and len(surface.convex_hull) > 0):
        ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 0.2])
    
    # Save if filename is provided
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def visualize_triangulation_2d(surface, hull_size, filename=None):
    """
    Visualize triangulation in 2D with quality metrics.
    
    Args:
        surface: Surface object with triangulation
        hull_size: Hull size used for triangulation
        filename: If provided, save the plot to this file
    """
    # Extract vertices and triangles
    vertices = np.array([[v.x, v.y, v.z] for v in surface.vertices])
    triangles = surface.triangles
    
    # Calculate quality metrics
    min_angles = []
    aspect_ratios = []
    
    for triangle in triangles:
        v1 = surface.vertices[triangle[0]]
        v2 = surface.vertices[triangle[1]]
        v3 = surface.vertices[triangle[2]]
        
        # Calculate edge lengths
        e1 = math.sqrt((v2.x - v1.x)**2 + (v2.y - v1.y)**2 + (v2.z - v1.z)**2)
        e2 = math.sqrt((v3.x - v2.x)**2 + (v3.y - v2.y)**2 + (v3.z - v2.z)**2)
        e3 = math.sqrt((v1.x - v3.x)**2 + (v1.y - v3.y)**2 + (v1.z - v3.z)**2)
        
        # Calculate angles using law of cosines
        angle1 = math.degrees(math.acos(min(1.0, max(-1.0, (e1**2 + e3**2 - e2**2) / (2 * e1 * e3)))))
        angle2 = math.degrees(math.acos(min(1.0, max(-1.0, (e1**2 + e2**2 - e3**2) / (2 * e1 * e2)))))
        angle3 = math.degrees(math.acos(min(1.0, max(-1.0, (e2**2 + e3**2 - e1**2) / (2 * e2 * e3)))))
        
        min_angles.append(min(angle1, angle2, angle3))
        
        # Calculate aspect ratio
        edges = sorted([e1, e2, e3])
        aspect_ratio = edges[2] / edges[0] if edges[0] > 0 else float('inf')
        aspect_ratios.append(aspect_ratio)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot triangulation with color based on minimum angle
    for i, triangle in enumerate(triangles):
        v1 = vertices[triangle[0], :2]
        v2 = vertices[triangle[1], :2]
        v3 = vertices[triangle[2], :2]
        triangle_vertices = np.array([v1, v2, v3])
        
        # Color based on minimum angle
        min_angle = min_angles[i]
        if min_angle < 20:
            color = 'red'
            alpha = 0.7
        elif min_angle < 30:
            color = 'orange'
            alpha = 0.7
        else:
            color = 'green'
            alpha = 0.7
            
        ax1.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], 
                color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
    
    # Plot the convex hull
    hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
    hull_points_closed = np.vstack([hull_points, hull_points[0]])  # Close the loop
    ax1.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2)
    
    # Plot original points and new points
    original_count = 25  # Assuming 5x5 grid
    ax1.scatter(vertices[:original_count, 0], vertices[:original_count, 1], 
              color='blue', s=30, zorder=3, label='Original Points')
    
    if len(vertices) > original_count:
        ax1.scatter(vertices[original_count:, 0], vertices[original_count:, 1], 
                  color='green', s=15, zorder=3, label='New Points')
    
    # Set title and labels
    ax1.set_title(f"Triangle Quality (hull_size={hull_size})\nRed: <20°, Orange: 20-30°, Green: >30°", fontsize=12)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect('equal')
    ax1.legend()
    
    # Plot histogram of minimum angles
    ax2.hist(min_angles, bins=18, range=(0, 90), alpha=0.7, color='blue')
    ax2.axvline(x=20, color='r', linestyle='--', label='MeshIt Min (20°)')
    ax2.axvline(x=30, color='g', linestyle='--', label='Good Quality (30°)')
    ax2.set_title(f"Minimum Angle Distribution (hull_size={hull_size})", fontsize=12)
    ax2.set_xlabel("Minimum Angle (degrees)")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    
    # Add text with quality statistics
    stats_text = (
        f"Triangle Count: {len(triangles)}\n"
        f"Vertex Count: {len(surface.vertices)}\n"
        f"Min Angle (min/mean/max): {min(min_angles):.1f}°/{sum(min_angles)/len(min_angles):.1f}°/{max(min_angles):.1f}°\n"
        f"Poor Angle (<20°): {sum(1 for a in min_angles if a < 20)} triangles ({sum(1 for a in min_angles if a < 20)/len(min_angles)*100:.1f}%)"
    )
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save if filename is provided
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def compare_hull_sizes_visualization(results, output_dir):
    """
    Create comparison visualization for different hull sizes.
    
    Args:
        results: List of (hull_size, surface, stats) tuples
        output_dir: Directory to save output files
    """
    # Create a multi-panel figure for 2D comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (hull_size, surface, stats) in enumerate(results):
        ax = axes[i]
        
        # Extract vertices and triangles
        vertices = np.array([[v.x, v.y, v.z] for v in surface.vertices])
        triangles = surface.triangles
        
        # Plot triangulation
        for triangle in triangles:
            v1 = vertices[triangle[0], :2]
            v2 = vertices[triangle[1], :2]
            v3 = vertices[triangle[2], :2]
            triangle_vertices = np.array([v1, v2, v3])
            ax.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], 
                    color='tan', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Plot the convex hull
        hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
        hull_points_closed = np.vstack([hull_points, hull_points[0]])  # Close the loop
        ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-', linewidth=2)
        
        # Plot original points and new points
        original_count = 25  # Assuming 5x5 grid
        ax.scatter(vertices[:original_count, 0], vertices[:original_count, 1], 
                  color='blue', s=30, zorder=3)
        
        if len(vertices) > original_count:
            ax.scatter(vertices[original_count:, 0], vertices[original_count:, 1], 
                      color='green', s=15, zorder=3)
        
        # Set title
        ax.set_title(f"hull_size = {hull_size}\n{len(triangles)} triangles, {len(vertices)} vertices\nMin Angle: {stats['min_angle_mean']:.1f}°", fontsize=12)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add overall title
    fig.suptitle("Comparison of Triangulations with Different Hull Sizes", fontsize=16)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig(f"{output_dir}/hull_size_comparison.png")
    plt.close()
    
    # Create bar charts for comparison metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    hull_sizes = [r[0] for r in results]
    triangle_counts = [len(r[1].triangles) for r in results]
    vertex_counts = [len(r[1].vertices) for r in results]
    min_angle_means = [r[2]['min_angle_mean'] for r in results]
    aspect_ratio_means = [r[2]['aspect_ratio_mean'] for r in results]
    
    # Plot triangle count
    ax1.bar(hull_sizes, triangle_counts, color='blue', alpha=0.7)
    ax1.set_title("Triangle Count")
    ax1.set_xlabel("Hull Size")
    ax1.set_ylabel("Count")
    
    # Plot vertex count
    ax2.bar(hull_sizes, vertex_counts, color='green', alpha=0.7)
    ax2.set_title("Vertex Count")
    ax2.set_xlabel("Hull Size")
    ax2.set_ylabel("Count")
    
    # Plot mean minimum angle
    ax3.bar(hull_sizes, min_angle_means, color='red', alpha=0.7)
    ax3.set_title("Mean Minimum Angle")
    ax3.set_xlabel("Hull Size")
    ax3.set_ylabel("Degrees")
    ax3.axhline(y=20, color='r', linestyle='--', label='MeshIt Min (20°)')
    ax3.legend()
    
    # Plot mean aspect ratio
    ax4.bar(hull_sizes, aspect_ratio_means, color='purple', alpha=0.7)
    ax4.set_title("Mean Aspect Ratio")
    ax4.set_xlabel("Hull Size")
    ax4.set_ylabel("Ratio")
    
    # Add overall title
    fig.suptitle("Triangulation Quality Metrics Comparison", fontsize=16)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig(f"{output_dir}/quality_metrics_comparison.png")
    plt.close()

def run_complete_workflow(hull_sizes=[0.5, 1.0, 2.0, 4.0], output_dir="workflow_output"):
    """
    Run the complete MeshIt workflow from point generation to triangulation.
    
    Args:
        hull_sizes: List of hull sizes to test
        output_dir: Directory to save output files
    """
    print("=== Starting Complete MeshIt Workflow Test ===")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: Generate points
    print("\nStep 1: Generating points")
    raw_points = generate_grid_points(n=5, z_variation=False)
    print(f"Generated {len(raw_points)} points")
    
    # Visualize raw points
    visualize_points(raw_points, title="Raw Points", 
                    filename=f"{output_dir}/step1_raw_points.png")
    
    # Results to store for comparison
    results = []
    
    # Test each hull size
    for hull_size in hull_sizes:
        print(f"\n=== Testing hull_size = {hull_size} ===")
        
        # Step 2: Create surface from points
        print("\nStep 2: Creating surface from points")
        surface = meshit.extensions.create_surface_from_points(raw_points)
        print(f"Created surface with {len(surface.vertices)} vertices")
        
        # Visualize surface points
        visualize_surface(surface, title=f"Surface Points", 
                         show_convex_hull=False, show_triangles=False,
                         filename=f"{output_dir}/step2_surface_points_{hull_size}.png")
        
        # Step 3: Calculate convex hull
        print("\nStep 3: Calculating convex hull")
        meshit.extensions.enhanced_calculate_convex_hull(surface)
        print(f"Calculated convex hull with {len(surface.convex_hull)} points")
        
        # Visualize surface with convex hull
        visualize_surface(surface, title=f"Surface with Convex Hull (hull_size={hull_size})", 
                         show_triangles=False,
                         filename=f"{output_dir}/step3_surface_with_hull_{hull_size}.png")
        
        # Step 4: Triangulate
        print("\nStep 4: Triangulating")
        triangles = meshit.extensions.triangulate_with_triangle(surface, hull_size=hull_size)
        
        # Calculate quality metrics
        min_angles = []
        aspect_ratios = []
        
        for triangle in triangles:
            v1 = surface.vertices[triangle[0]]
            v2 = surface.vertices[triangle[1]]
            v3 = surface.vertices[triangle[2]]
            
            # Calculate edge lengths
            e1 = math.sqrt((v2.x - v1.x)**2 + (v2.y - v1.y)**2 + (v2.z - v1.z)**2)
            e2 = math.sqrt((v3.x - v2.x)**2 + (v3.y - v2.y)**2 + (v3.z - v2.z)**2)
            e3 = math.sqrt((v1.x - v3.x)**2 + (v1.y - v3.y)**2 + (v1.z - v3.z)**2)
            
            # Calculate angles using law of cosines
            angle1 = math.degrees(math.acos(min(1.0, max(-1.0, (e1**2 + e3**2 - e2**2) / (2 * e1 * e3)))))
            angle2 = math.degrees(math.acos(min(1.0, max(-1.0, (e1**2 + e2**2 - e3**2) / (2 * e1 * e2)))))
            angle3 = math.degrees(math.acos(min(1.0, max(-1.0, (e2**2 + e3**2 - e1**2) / (2 * e2 * e3)))))
            
            min_angles.append(min(angle1, angle2, angle3))
            
            # Calculate aspect ratio
            edges = sorted([e1, e2, e3])
            aspect_ratio = edges[2] / edges[0] if edges[0] > 0 else float('inf')
            aspect_ratios.append(aspect_ratio)
        
        # Calculate statistics
        stats = {
            'min_angle_min': min(min_angles) if min_angles else 0,
            'min_angle_mean': sum(min_angles) / len(min_angles) if min_angles else 0,
            'min_angle_max': max(min_angles) if min_angles else 0,
            'aspect_ratio_min': min(aspect_ratios) if aspect_ratios else 0,
            'aspect_ratio_mean': sum(aspect_ratios) / len(aspect_ratios) if aspect_ratios else 0,
            'aspect_ratio_max': max(aspect_ratios) if aspect_ratios else 0,
            'poor_angle_percentage': (sum(1 for a in min_angles if a < 20) / len(min_angles) * 100) if min_angles else 0
        }
        
        print(f"Created {len(triangles)} triangles")
        print(f"Surface now has {len(surface.vertices)} vertices")
        print(f"Minimum angle (min/mean/max): {stats['min_angle_min']:.2f}°/{stats['min_angle_mean']:.2f}°/{stats['min_angle_max']:.2f}°")
        print(f"Aspect ratio (min/mean/max): {stats['aspect_ratio_min']:.2f}/{stats['aspect_ratio_mean']:.2f}/{stats['aspect_ratio_max']:.2f}")
        print(f"Poor angle percentage (<20°): {stats['poor_angle_percentage']:.2f}%")
        
        # Visualize triangulated surface in 3D
        visualize_surface(surface, title=f"Triangulated Surface (hull_size={hull_size})", 
                         filename=f"{output_dir}/step4_triangulated_surface_3d_{hull_size}.png")
        
        # Visualize triangulation quality in 2D
        visualize_triangulation_2d(surface, hull_size, 
                                  filename=f"{output_dir}/step4_triangulation_quality_{hull_size}.png")
        
        # Store results for comparison
        results.append((hull_size, surface, stats))
    
    # Step 5: Compare results
    print("\nStep 5: Comparing results")
    compare_hull_sizes_visualization(results, output_dir)
    
    # Print summary table
    print("\n=== Quality Comparison Summary ===")
    print(f"{'Hull Size':<10} {'Triangles':<10} {'Vertices':<10} {'Min Angle':<10} {'Poor Angle %':<12} {'Aspect Ratio':<12}")
    print("-" * 70)
    
    for hull_size, surface, stats in results:
        print(f"{hull_size:<10.1f} {len(surface.triangles):<10d} {len(surface.vertices):<10d} "
              f"{stats['min_angle_mean']:<10.2f} {stats['poor_angle_percentage']:<12.2f} "
              f"{stats['aspect_ratio_mean']:<12.2f}")
    
    print("\n=== Complete MeshIt Workflow Test Completed ===")
    print(f"Output files saved to {output_dir}/")
    
    return results

if __name__ == "__main__":
    # Run the complete workflow
    run_complete_workflow() 