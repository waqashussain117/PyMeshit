import numpy as np
import pyvista as pv
from meshit import Vector3D, Surface
import meshit

# Configure PyVista for interactive display
pv.set_plot_theme("document")

def create_points_for_surface():
    """Create points for a simple surface"""
    # Create a grid of points in the XY plane
    points = []
    for x in np.linspace(0, 10, 5):
        for y in np.linspace(0, 10, 5):
            points.append([x, y, 0])
    
    return points

def visualize_convex_hull(surface, title="Convex Hull Visualization"):
    """Visualize a surface with its convex hull"""
    # Get the vertices
    points = np.array([[v.x, v.y, v.z] for v in surface.vertices])
    
    # Create a plotter
    plotter = pv.Plotter()
    plotter.set_background([0.318, 0.341, 0.431])  # MeshIt background color
    
    # Add a title
    plotter.add_text(title, position='upper_edge', font_size=16, color='white')
    
    # Show the points
    point_cloud = pv.PolyData(points)
    plotter.add_mesh(point_cloud, color='blue', point_size=8, render_points_as_spheres=True)
    
    # If convex hull exists, show it
    if hasattr(surface, 'convex_hull') and len(surface.convex_hull) > 0:
        hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
        
        # Print hull points for debugging
        print(f"Visualizing convex hull with {len(hull_points)} points:")
        for i, p in enumerate(hull_points):
            print(f"  Point {i}: ({p[0]}, {p[1]}, {p[2]})")
        
        # Create lines connecting the convex hull points
        lines = []
        n_points = len(hull_points)
        
        # Ensure we're creating a complete polygon by connecting all points
        for i in range(n_points):
            next_i = (i + 1) % n_points
            lines.append([2, i, next_i])
        
        # Create a polydata for the convex hull
        hull_polydata = pv.PolyData(hull_points)
        
        # Set the lines properly to ensure all points are connected
        if lines:
            hull_polydata.lines = np.hstack(lines)
        
        # Add the convex hull lines
        plotter.add_mesh(hull_polydata, color='red', line_width=3, render_lines_as_tubes=True)
        
        # Add the convex hull points
        plotter.add_mesh(pv.PolyData(hull_points), color='red', point_size=10, render_points_as_spheres=True)
    
    # Add axes with MeshIt-like styling
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z', line_width=2)
    
    # Show the plot
    plotter.show(title=title)

def fix_convex_hull(surface):
    """Fix the convex hull if it has issues"""
    if len(surface.convex_hull) == 0:
        print("No convex hull to fix")
        return
    
    # Check for duplicate points
    hull_points = [[p.x, p.y, p.z] for p in surface.convex_hull]
    unique_points = []
    for p in hull_points:
        if p not in unique_points:
            unique_points.append(p)
    
    if len(unique_points) < len(hull_points):
        print(f"Warning: Convex hull has {len(unique_points)} unique points instead of {len(hull_points)}")
        print("Fixing convex hull to create a proper rectangle")
        
        # Find the min and max x, y coordinates to form a bounding box
        min_x = min(v.x for v in surface.vertices)
        max_x = max(v.x for v in surface.vertices)
        min_y = min(v.y for v in surface.vertices)
        max_y = max(v.y for v in surface.vertices)
        z_val = surface.vertices[0].z  # Assuming all z values are the same
        
        # Create a rectangular convex hull using the bounding box
        surface.convex_hull = [
            Vector3D(min_x, min_y, z_val),  # Bottom left
            Vector3D(max_x, min_y, z_val),  # Bottom right
            Vector3D(max_x, max_y, z_val),  # Top right
            Vector3D(min_x, max_y, z_val)   # Top left
        ]
        
        print("Fixed convex hull with points:")
        for i, p in enumerate(surface.convex_hull):
            print(f"  Point {i}: ({p.x}, {p.y}, {p.z})")

def test_convex_hull():
    """Test the convex hull calculation and visualization"""
    print("\n=== Testing Convex Hull ===")
    
    # Create points for a surface
    points = create_points_for_surface()
    print(f"Created {len(points)} points")
    
    # Create a surface
    surface = meshit.create_surface(points, [], "TestSurface", "Planar")
    print(f"Created surface with {len(surface.vertices)} vertices")
    
    # Calculate convex hull
    print("\nCalculating convex hull...")
    surface.calculate_convex_hull()
    print(f"Calculated convex hull with {len(surface.convex_hull)} points")
    
    # Print the convex hull points
    print("Convex hull points:")
    for i, p in enumerate(surface.convex_hull):
        print(f"  Point {i}: ({p.x}, {p.y}, {p.z})")
    
    # Visualize the surface with convex hull
    print("\nVisualizing surface with original convex hull...")
    visualize_convex_hull(surface, "Original Convex Hull")
    
    # Fix the convex hull if needed
    print("\nFixing convex hull if needed...")
    fix_convex_hull(surface)
    
    # Visualize the surface with fixed convex hull
    print("\nVisualizing surface with fixed convex hull...")
    visualize_convex_hull(surface, "Fixed Convex Hull")
    
    print("\nConvex hull test completed successfully!")

if __name__ == "__main__":
    test_convex_hull() 