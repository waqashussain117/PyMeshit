import numpy as np
import pyvista as pv
import meshit

# Configure PyVista for interactive display
pv.set_plot_theme("document")
pv.global_theme.window_size = [1200, 800]
pv.global_theme.anti_aliasing = "fxaa"
pv.global_theme.show_scalar_bar = False
pv.global_theme.background = 'white'

def generate_grid_points(n=5):
    """Generate a grid of points in the XY plane."""
    points = []
    for x in np.linspace(0, 10, n):
        for y in np.linspace(0, 10, n):
            points.append([x, y, 0])
    return points

def visualize_triangulation(hull_size):
    """Visualize triangulation with a specific hull size."""
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
    
    # Create visualization data
    points = [[v.x, v.y, v.z] for v in surface.vertices]
    hull_points = [[v.x, v.y, v.z] for v in surface.convex_hull]
    
    # Create triangulated surface polydata
    faces = []
    for triangle in triangles:
        faces.append(3)  # Number of vertices in the face
        faces.extend(triangle)
    
    # Create the mesh
    mesh = pv.PolyData(points, faces=faces)
    
    # Create a plotter
    plotter = pv.Plotter(window_size=[1200, 800])
    
    # Add the triangulated mesh with edges
    plotter.add_mesh(mesh, color='tan', show_edges=True, edge_color='black', line_width=2)
    
    # Add the original points
    plotter.add_points(pv.PolyData(points), color='blue', point_size=10, render_points_as_spheres=True)
    
    # Add the convex hull
    hull_lines = []
    for i in range(len(hull_points)):
        hull_lines.append([2, i, (i + 1) % len(hull_points)])
    hull_lines = np.array(hull_lines).flatten()
    hull_pd = pv.PolyData(hull_points, lines=hull_lines)
    plotter.add_mesh(hull_pd, color='red', line_width=3)
    
    # Set the title
    plotter.add_title(f"Triangulation with hull_size = {hull_size} ({len(triangles)} triangles)", font_size=24)
    
    # Set the camera position for a good view
    plotter.view_xy()
    
    # Show the visualization (interactive)
    plotter.show()
    
    return surface, triangles

if __name__ == "__main__":
    # Ask user for hull size
    try:
        hull_size = float(input("Enter hull size (e.g., 0.5, 1.0, 2.0, 4.0): "))
    except ValueError:
        hull_size = 2.0
        print(f"Invalid input. Using default hull size: {hull_size}")
    
    # Visualize triangulation
    surface, triangles = visualize_triangulation(hull_size) 