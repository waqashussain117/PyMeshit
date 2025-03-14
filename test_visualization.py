import numpy as np
import pyvista as pv
import meshit
import os

# Configure PyVista for better visualization
pv.set_plot_theme("document")
pv.global_theme.window_size = [1200, 800]
pv.global_theme.anti_aliasing = "fxaa"  # Use fxaa anti-aliasing
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
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 800])
    
    # Add the triangulated mesh with edges
    plotter.add_mesh(mesh, color='tan', show_edges=True, edge_color='black', line_width=2)
    
    # Add the original points
    plotter.add_points(pv.PolyData(points), color='blue', point_size=10, render_points_as_spheres=True)
    
    # Set the title
    plotter.add_title(f"Triangulation with hull_size = {hull_size} ({len(triangles)} triangles)", font_size=24)
    
    # Set the camera position for a good view
    plotter.view_xy()
    
    # Save the visualization
    output_file = f"triangulation_hull_size_{hull_size}.png"
    plotter.screenshot(output_file)
    print(f"Saved visualization to {output_file}")
    plotter.close()
    
    return surface, triangles

if __name__ == "__main__":
    # Test with different hull sizes
    hull_sizes = [0.5, 1.0, 2.0, 4.0]
    
    for hull_size in hull_sizes:
        surface, triangles = visualize_triangulation(hull_size)
        print(f"Completed visualization with hull_size = {hull_size}, created {len(triangles)} triangles\n")
        print("=" * 50) 