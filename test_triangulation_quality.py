import numpy as np
import matplotlib.pyplot as plt
import meshit
from meshit.core._meshit import Vector3D
import math

def generate_grid_points(n=5):
    """Generate a grid of points in the XY plane."""
    points = []
    for x in np.linspace(0, 10, n):
        for y in np.linspace(0, 10, n):
            points.append([x, y, 0])
    return points

def calculate_triangle_quality(vertices, triangle):
    """
    Calculate quality metrics for a triangle.
    
    Args:
        vertices: List of Vector3D objects
        triangle: List of 3 vertex indices
        
    Returns:
        Dictionary containing quality metrics:
        - min_angle: Minimum angle in degrees
        - max_angle: Maximum angle in degrees
        - aspect_ratio: Ratio of longest to shortest edge
        - area: Area of the triangle
    """
    # Extract triangle vertices
    v1 = vertices[triangle[0]]
    v2 = vertices[triangle[1]]
    v3 = vertices[triangle[2]]
    
    # Calculate edge lengths
    e1 = math.sqrt((v2.x - v1.x)**2 + (v2.y - v1.y)**2 + (v2.z - v1.z)**2)
    e2 = math.sqrt((v3.x - v2.x)**2 + (v3.y - v2.y)**2 + (v3.z - v2.z)**2)
    e3 = math.sqrt((v1.x - v3.x)**2 + (v1.y - v3.y)**2 + (v1.z - v3.z)**2)
    
    # Sort edges
    edges = sorted([e1, e2, e3])
    
    # Calculate angles using law of cosines
    angle1 = math.degrees(math.acos(min(1.0, max(-1.0, (e1**2 + e3**2 - e2**2) / (2 * e1 * e3)))))
    angle2 = math.degrees(math.acos(min(1.0, max(-1.0, (e1**2 + e2**2 - e3**2) / (2 * e1 * e2)))))
    angle3 = math.degrees(math.acos(min(1.0, max(-1.0, (e2**2 + e3**2 - e1**2) / (2 * e2 * e3)))))
    
    angles = [angle1, angle2, angle3]
    
    # Calculate aspect ratio (longest edge / shortest edge)
    aspect_ratio = edges[2] / edges[0] if edges[0] > 0 else float('inf')
    
    # Calculate area using Heron's formula
    s = (e1 + e2 + e3) / 2
    area = math.sqrt(max(0, s * (s - e1) * (s - e2) * (s - e3)))
    
    return {
        'min_angle': min(angles),
        'max_angle': max(angles),
        'aspect_ratio': aspect_ratio,
        'area': area
    }

def analyze_triangulation_quality(surface):
    """
    Analyze the quality of triangulation for a surface.
    
    Args:
        surface: Surface object with triangulation
        
    Returns:
        Dictionary containing quality statistics
    """
    if not surface.triangles:
        return None
    
    quality_metrics = []
    
    for triangle in surface.triangles:
        metrics = calculate_triangle_quality(surface.vertices, triangle)
        quality_metrics.append(metrics)
    
    # Calculate statistics
    min_angles = [m['min_angle'] for m in quality_metrics]
    max_angles = [m['max_angle'] for m in quality_metrics]
    aspect_ratios = [m['aspect_ratio'] for m in quality_metrics]
    areas = [m['area'] for m in quality_metrics]
    
    stats = {
        'min_angle': {
            'min': min(min_angles),
            'max': max(min_angles),
            'mean': sum(min_angles) / len(min_angles),
            'median': sorted(min_angles)[len(min_angles) // 2]
        },
        'max_angle': {
            'min': min(max_angles),
            'max': max(max_angles),
            'mean': sum(max_angles) / len(max_angles),
            'median': sorted(max_angles)[len(max_angles) // 2]
        },
        'aspect_ratio': {
            'min': min(aspect_ratios),
            'max': max(aspect_ratios),
            'mean': sum(aspect_ratios) / len(aspect_ratios),
            'median': sorted(aspect_ratios)[len(aspect_ratios) // 2]
        },
        'area': {
            'min': min(areas),
            'max': max(areas),
            'mean': sum(areas) / len(areas),
            'median': sorted(areas)[len(areas) // 2],
            'total': sum(areas)
        },
        'triangle_count': len(surface.triangles),
        'vertex_count': len(surface.vertices)
    }
    
    # Count triangles with poor quality (based on MeshIt standards)
    # MeshIt typically requires minimum angles > 20° for quality triangulation
    poor_angle_count = sum(1 for angle in min_angles if angle < 20)
    stats['poor_angle_percentage'] = (poor_angle_count / len(min_angles)) * 100
    
    # High aspect ratio indicates poor quality
    high_aspect_ratio_count = sum(1 for ratio in aspect_ratios if ratio > 10)
    stats['high_aspect_ratio_percentage'] = (high_aspect_ratio_count / len(aspect_ratios)) * 100
    
    return stats, quality_metrics

def visualize_triangulation_quality(surface, quality_metrics, hull_size):
    """
    Visualize the triangulation quality.
    
    Args:
        surface: Surface object with triangulation
        quality_metrics: List of quality metrics for each triangle
        hull_size: Hull size used for triangulation
    """
    # Extract vertices
    vertices = np.array([[v.x, v.y, v.z] for v in surface.vertices])
    triangles = surface.triangles
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot triangulation colored by minimum angle
    min_angles = [m['min_angle'] for m in quality_metrics]
    
    # Plot triangulation with color based on minimum angle
    for i, triangle in enumerate(triangles):
        v1 = vertices[triangle[0], :2]
        v2 = vertices[triangle[1], :2]
        v3 = vertices[triangle[2], :2]
        triangle_vertices = np.array([v1, v2, v3])
        
        # Color based on minimum angle (red for poor quality, blue for good quality)
        # MeshIt standard: minimum angle should be > 20°
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
    
    # Set title and labels
    ax1.set_title(f"Triangle Quality (hull_size={hull_size})\nRed: <20°, Orange: 20-30°, Green: >30°", fontsize=12)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect('equal')
    
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
    plt.savefig(f"triangulation_quality_{hull_size}.png")
    plt.close()

def test_triangulation_with_hull_size(hull_size):
    """Test triangulation with a specific hull size and analyze quality."""
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
    
    # Step 5: Analyze quality
    print("Step 5: Analyzing quality")
    stats, quality_metrics = analyze_triangulation_quality(surface)
    
    # Print quality statistics
    print(f"Triangle count: {stats['triangle_count']}")
    print(f"Vertex count: {stats['vertex_count']}")
    print(f"Minimum angle (min/mean/max): {stats['min_angle']['min']:.2f}°/{stats['min_angle']['mean']:.2f}°/{stats['min_angle']['max']:.2f}°")
    print(f"Maximum angle (min/mean/max): {stats['max_angle']['min']:.2f}°/{stats['max_angle']['mean']:.2f}°/{stats['max_angle']['max']:.2f}°")
    print(f"Aspect ratio (min/mean/max): {stats['aspect_ratio']['min']:.2f}/{stats['aspect_ratio']['mean']:.2f}/{stats['aspect_ratio']['max']:.2f}")
    print(f"Triangle area (min/mean/max): {stats['area']['min']:.2f}/{stats['area']['mean']:.2f}/{stats['area']['max']:.2f}")
    print(f"Poor angle percentage (<20°): {stats['poor_angle_percentage']:.2f}%")
    print(f"High aspect ratio percentage (>10): {stats['high_aspect_ratio_percentage']:.2f}%")
    
    # Step 6: Visualize quality
    print("Step 6: Visualizing quality")
    visualize_triangulation_quality(surface, quality_metrics, hull_size)
    
    return surface, stats, quality_metrics

def compare_hull_sizes_quality():
    """Compare triangulation quality with different hull sizes."""
    hull_sizes = [0.5, 1.0, 2.0, 4.0]
    results = []
    
    for hull_size in hull_sizes:
        surface, stats, _ = test_triangulation_with_hull_size(hull_size)
        results.append((hull_size, stats))
    
    # Create comparison figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    axes = [ax1, ax2, ax3, ax4]
    
    # Plot comparison metrics
    hull_size_values = [r[0] for r in results]
    min_angle_means = [r[1]['min_angle']['mean'] for r in results]
    triangle_counts = [r[1]['triangle_count'] for r in results]
    poor_angle_percentages = [r[1]['poor_angle_percentage'] for r in results]
    aspect_ratio_means = [r[1]['aspect_ratio']['mean'] for r in results]
    
    ax1.bar(hull_size_values, min_angle_means, color='blue', alpha=0.7)
    ax1.set_title("Mean Minimum Angle")
    ax1.set_xlabel("Hull Size")
    ax1.set_ylabel("Degrees")
    ax1.axhline(y=20, color='r', linestyle='--', label='MeshIt Min (20°)')
    ax1.legend()
    
    ax2.bar(hull_size_values, triangle_counts, color='green', alpha=0.7)
    ax2.set_title("Triangle Count")
    ax2.set_xlabel("Hull Size")
    ax2.set_ylabel("Count")
    
    ax3.bar(hull_size_values, poor_angle_percentages, color='red', alpha=0.7)
    ax3.set_title("Poor Angle Percentage (<20°)")
    ax3.set_xlabel("Hull Size")
    ax3.set_ylabel("Percentage")
    
    ax4.bar(hull_size_values, aspect_ratio_means, color='purple', alpha=0.7)
    ax4.set_title("Mean Aspect Ratio")
    ax4.set_xlabel("Hull Size")
    ax4.set_ylabel("Ratio")
    ax4.axhline(y=10, color='r', linestyle='--', label='Poor Quality (>10)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig("hull_size_quality_comparison.png")
    plt.close()
    
    # Create a summary table
    print("\n=== Quality Comparison Summary ===")
    print(f"{'Hull Size':<10} {'Triangles':<10} {'Vertices':<10} {'Min Angle':<10} {'Poor Angle %':<12} {'Aspect Ratio':<12}")
    print("-" * 70)
    
    for hull_size, stats in results:
        print(f"{hull_size:<10.1f} {stats['triangle_count']:<10d} {stats['vertex_count']:<10d} "
              f"{stats['min_angle']['mean']:<10.2f} {stats['poor_angle_percentage']:<12.2f} "
              f"{stats['aspect_ratio']['mean']:<12.2f}")

if __name__ == "__main__":
    # Test with different hull sizes and analyze quality
    compare_hull_sizes_quality() 