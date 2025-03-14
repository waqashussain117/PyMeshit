import numpy as np
import meshit
import pyvista as pv

def generate_grid_points(n=5):
    """Generate a grid of points in the XY plane."""
    points = []
    for x in np.linspace(0, 10, n):
        for y in np.linspace(0, 10, n):
            points.append([x, y, 0])
    return points

def test_triangulation(hull_size=2.0):
    """Test triangulation with the original MeshIt approach."""
    print(f"\n=== Testing triangulation with hull_size = {hull_size} ===")
    
    # Step 1: Create points
    print("\nStep 1: Creating points")
    raw_points = generate_grid_points()
    print(f"Created {len(raw_points)} points")
    
    # Step 2: Create surface
    print("\nStep 2: Creating surface")
    surface = meshit.extensions.create_surface_from_points(raw_points)
    print(f"Created surface with {len(surface.vertices)} vertices")
    
    # Step 3: Calculate convex hull
    print("\nStep 3: Calculating convex hull")
    meshit.extensions.enhanced_calculate_convex_hull(surface)
    print(f"Calculated convex hull with {len(surface.convex_hull)} points")
    
    # Step 4: Triangulate
    print("\nStep 4: Triangulating")
    triangles = meshit.extensions.triangulate_with_triangle(surface, hull_size=hull_size)
    print(f"Triangulation completed with {len(triangles)} triangles")
    
    # Visualize the triangulation
    print("\nStep 5: Visualizing")
    
    # Create PyVista data structures
    points = [[v.x, v.y, v.z] for v in surface.vertices]
    hull_points = [[v.x, v.y, v.z] for v in surface.convex_hull]
    
    # Create points polydata
    points_pd = pv.PolyData(points)
    
    # Create convex hull polydata
    hull_lines = []
    for i in range(len(hull_points)):
        hull_lines.append([2, i, (i + 1) % len(hull_points)])
    hull_lines = np.array(hull_lines).flatten()
    hull_pd = pv.PolyData(hull_points, lines=hull_lines)
    
    # Create triangulated surface polydata
    faces = []
    for triangle in triangles:
        faces.append(3)  # Number of vertices in the face
        faces.extend(triangle)
    mesh = pv.PolyData(points, faces=faces)
    
    # Save the mesh to a VTK file
    mesh.save(f"triangulated_surface_{hull_size}.vtk")
    print(f"Saved triangulated surface to triangulated_surface_{hull_size}.vtk")
    
    # Create a multi-panel plotter
    plotter = pv.Plotter(shape=(2, 2), off_screen=True)
    
    # Panel 1: Original points
    plotter.subplot(0, 0)
    plotter.add_points(points_pd, color='blue', point_size=10, render_points_as_spheres=True)
    plotter.add_title("Original Points")
    
    # Panel 2: Points with convex hull
    plotter.subplot(0, 1)
    plotter.add_points(points_pd, color='blue', point_size=10, render_points_as_spheres=True)
    plotter.add_mesh(hull_pd, color='red', line_width=3)
    plotter.add_title("Points with Convex Hull")
    
    # Panel 3: Wireframe triangulation
    plotter.subplot(1, 0)
    plotter.add_mesh(mesh, color='tan', style='wireframe', line_width=2)
    plotter.add_points(points_pd, color='blue', point_size=10, render_points_as_spheres=True)
    plotter.add_title(f"Wireframe (hull_size={hull_size})")
    
    # Panel 4: Surface triangulation
    plotter.subplot(1, 1)
    plotter.add_mesh(mesh, color='tan', show_edges=True)
    plotter.add_title(f"Surface (hull_size={hull_size})")
    
    # Save the visualization
    plotter.screenshot(f"triangulation_visualization_{hull_size}.png")
    print(f"Saved visualization to triangulation_visualization_{hull_size}.png")
    plotter.close()
    
    # Create a 3D visualization
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color='tan', show_edges=True)
    plotter.add_title(f"3D View (hull_size={hull_size})")
    plotter.view_isometric()
    plotter.screenshot(f"triangulation_3d_{hull_size}.png")
    print(f"Saved 3D visualization to triangulation_3d_{hull_size}.png")
    plotter.close()
    
    return surface, triangles

if __name__ == "__main__":
    # Test with different hull sizes
    hull_sizes = [1.0, 2.0, 4.0]
    
    for hull_size in hull_sizes:
        surface, triangles = test_triangulation(hull_size)
        print(f"Completed test with hull_size = {hull_size}, created {len(triangles)} triangles\n")
        print("=" * 50) 