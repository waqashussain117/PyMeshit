import numpy as np
import matplotlib.pyplot as plt
from meshit import Surface, Vector3D, GradientControl
from meshit.extensions import triangulate_with_triangle

def generate_test_points(n=20):
    """Generate a grid of points in the XY plane with random perturbation."""
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    points = []
    
    # Add grid points with random perturbation
    for i in x:
        for j in y:
            # Add random perturbation up to 0.2 units
            dx = np.random.uniform(-0.2, 0.2)
            dy = np.random.uniform(-0.2, 0.2)
            points.append(Vector3D(i + dx, j + dy, 0))
    
    # Add some random points for more variation
    num_random = n * 2
    for _ in range(num_random):
        x_rand = np.random.uniform(0, 10)
        y_rand = np.random.uniform(0, 10)
        points.append(Vector3D(x_rand, y_rand, 0))
    
    return points

def test_triangulation(hull_size, gradient):
    """Test triangulation with specific hull_size and gradient values."""
    # Create surface with test points
    surface = Surface()
    points = generate_test_points()
    for point in points:
        surface.add_vertex(point)
    
    # Perform triangulation
    triangles = triangulate_with_triangle(surface, hull_size, gradient)
    
    return surface, triangles

def plot_triangulation(surface, triangles, title, subplot_pos):
    """Plot triangulation results with triangles colored by area and quality metrics."""
    plt.subplot(2, 2, subplot_pos)
    plt.title(title)
    
    # Plot vertices
    x = [v.x for v in surface.vertices]
    y = [v.y for v in surface.vertices]
    plt.plot(x, y, 'k.', markersize=2, label='Vertices')
    
    # Calculate triangle metrics
    areas = []
    angles = []
    aspect_ratios = []
    
    for triangle in triangles:
        v1, v2, v3 = [surface.vertices[i] for i in triangle]
        
        # Calculate triangle area
        area = abs((v2.x - v1.x) * (v3.y - v1.y) - (v3.x - v1.x) * (v2.y - v1.y)) / 2
        areas.append(area)
        
        # Calculate angles
        def get_angle(p1, p2, p3):
            v1 = np.array([p2.x - p1.x, p2.y - p1.y])
            v2 = np.array([p3.x - p1.x, p3.y - p1.y])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
        
        angles.extend([
            get_angle(v1, v2, v3),
            get_angle(v2, v3, v1),
            get_angle(v3, v1, v2)
        ])
        
        # Calculate aspect ratio (longest edge / shortest height)
        edges = [
            np.linalg.norm([v2.x - v1.x, v2.y - v1.y]),
            np.linalg.norm([v3.x - v2.x, v3.y - v2.y]),
            np.linalg.norm([v1.x - v3.x, v1.y - v3.y])
        ]
        max_edge = max(edges)
        min_height = 2 * area / max_edge
        aspect_ratio = max_edge / min_height
        aspect_ratios.append(aspect_ratio)
    
    # Plot triangles with color based on minimum angle
    min_angles = np.array([min(angles[i:i+3]) for i in range(0, len(angles), 3)])
    collection = plt.tripcolor([v.x for v in surface.vertices], 
                             [v.y for v in surface.vertices],
                             triangles,
                             min_angles,
                             cmap='viridis',
                             vmin=0,
                             vmax=60)
    
    if subplot_pos == 2:  # Add colorbar to the second subplot
        plt.colorbar(collection, label='Minimum Angle (degrees)')
    
    plt.axis('equal')
    plt.grid(True)
    if subplot_pos == 1:
        plt.legend()
    
    # Add quality metrics as text
    stats_text = f"Quality Metrics:\n"
    stats_text += f"Min angle: {min(angles):.1f}°\n"
    stats_text += f"Max angle: {max(angles):.1f}°\n"
    stats_text += f"Mean aspect ratio: {np.mean(aspect_ratios):.2f}\n"
    stats_text += f"Mean area: {np.mean(areas):.2f}"
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def compare_triangulations():
    """Compare triangulation results with different parameters."""
    plt.figure(figsize=(15, 12))
    
    # Test cases
    test_cases = [
        (2.0, 0.5, "Hull Size: 2.0, Gradient: 0.5"),
        (2.0, 2.0, "Hull Size: 2.0, Gradient: 2.0"),
        (1.0, 0.5, "Hull Size: 1.0, Gradient: 0.5"),
        (1.0, 2.0, "Hull Size: 1.0, Gradient: 2.0")
    ]
    
    # Run tests and plot results
    for i, (hull_size, gradient, title) in enumerate(test_cases, 1):
        surface, triangles = test_triangulation(hull_size, gradient)
        plot_triangulation(surface, triangles, title, i)
        print(f"\n{title}")
        print(f"Number of vertices: {len(surface.vertices)}")
        print(f"Number of triangles: {len(triangles)}")
    
    plt.suptitle("Triangulation Quality Comparison\nColors indicate minimum angle (degrees)", fontsize=14)
    plt.tight_layout()
    plt.savefig('triangulation_comparison.png')
    plt.show()  # Show the plot interactively

if __name__ == "__main__":
    print("Testing triangulation with GradientControl...")
    compare_triangulations() 