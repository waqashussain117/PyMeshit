import numpy as np
import pyvista as pv
import triangle as tr
from meshit import MeshItModel, Vector3D, Surface, Polyline
import meshit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time

# Configure PyVista for interactive display
pv.set_plot_theme("document")
try:
    # Try to use the most compatible rendering backend
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.anti_aliasing = True
    pv.global_theme.show_scalar_bar = False
except:
    print("Warning: Could not configure PyVista backend. Visualization may not work properly.")

# Define colors to match MeshIt C++ visualization
COLORS = {
    'background': [0.318, 0.341, 0.431],  # MeshIt background color
    'white': [1.0, 1.0, 1.0],
    'grey': [0.7, 0.7, 0.7],
    'yellow': [1.0, 1.0, 0.0],
    'red': [1.0, 0.0, 0.0],
    'green': [0.0, 1.0, 0.0],
    'blue': [0.0, 0.0, 1.0],
    'blue_trans': [0.0, 0.0, 1.0, 0.5],
    'green_trans': [0.0, 1.0, 0.0, 0.5],
    'red_trans': [1.0, 0.0, 0.0, 0.5],
    'yellow_trans': [1.0, 1.0, 0.0, 0.5],
    'surface': [0.8, 0.8, 0.6],  # Tan color for surfaces
    'convex_hull': [1.0, 0.0, 0.0],  # Red for convex hull
    'constraints': [0.0, 1.0, 0.0],  # Green for constraints
    'vertices': [0.0, 0.0, 1.0],  # Blue for vertices
    'edges': [0.0, 0.0, 0.0]  # Black for edges
}

def generate_points(num_points=25):
    """Generate random points in 3D space."""
    np.random.seed(42)  # For reproducibility
    points = []
    for _ in range(num_points):
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 10)
        z = 0  # Keep points in XY plane for simplicity
        points.append([x, y, z])
    return points

def visualize_points(points, title="Points Visualization", save_image=True):
    """Visualize points in 3D in a style similar to MeshIt"""
    # Convert points to numpy array
    points_array = np.array(points)
    
    # Create a plotter
    plotter = pv.Plotter(off_screen=save_image)
    
    # Add points to the plotter
    point_cloud = pv.PolyData(points_array)
    plotter.add_mesh(point_cloud, color='red', point_size=10, render_points_as_spheres=True)
    
    # Add coordinate axes
    plotter.add_axes()
    
    # Set camera position
    plotter.camera_position = [(20, 20, 20), (5, 5, 0), (0, 0, 1)]
    
    # Set background color
    plotter.background_color = 'white'
    
    # Add title
    plotter.add_text(title, position='upper_edge', font_size=12, color='black')
    
    # Save or show the visualization
    if save_image:
        filename = title.replace(" ", "_").replace(":", "") + ".png"
        plotter.screenshot(filename)
        print(f"Points visualization saved to {filename}")
    else:
        plotter.show(title=title)

def save_points_to_vtk(points, filename):
    """Save points to a VTK file."""
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Points data\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        # Add vertices (each point is a vertex)
        f.write(f"VERTICES {len(points)} {len(points) * 2}\n")
        for i in range(len(points)):
            f.write(f"1 {i}\n")
    
    print(f"Points saved to VTK file: {filename}")

def save_polyline_segments_to_vtk(polyline, filename):
    """Save polyline segments to a VTK file."""
    # Extract vertices and segments
    vertices = [[v.x, v.y, v.z] for v in polyline.vertices]
    segments = polyline.segments if hasattr(polyline, 'segments') else []
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Polyline segments\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(vertices)} float\n")
        
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        # Write segments as lines
        total_segments = len(segments)
        if total_segments > 0:
            # Each segment entry has format: 2 (for 2 points), point1_idx, point2_idx
            total_entries = total_segments * 3
            f.write(f"LINES {total_segments} {total_entries}\n")
            
            for segment in segments:
                if len(segment) >= 2:
                    f.write(f"2 {segment[0]} {segment[1]}\n")
        else:
            # If no segments, just connect consecutive points
            if len(vertices) > 1:
                total_lines = len(vertices) - 1
                total_entries = total_lines * 3
                f.write(f"LINES {total_lines} {total_entries}\n")
                
                for i in range(len(vertices) - 1):
                    f.write(f"2 {i} {i+1}\n")
    
    print(f"Polyline segments saved to VTK file: {filename}")

def print_polyline_details(polyline, index):
    """Print detailed information about a polyline."""
    print(f"\nPolyline {index} details:")
    
    # Print basic attributes
    print(f"  Number of vertices: {len(polyline.vertices)}")
    
    # Print first few vertices
    print("  Vertices (first 3):")
    for i, vertex in enumerate(polyline.vertices[:3]):
        print(f"    Vertex {i}: ({vertex.x}, {vertex.y}, {vertex.z})")
    
    # Print segments if available
    if hasattr(polyline, 'segments'):
        print(f"  Number of segments: {len(polyline.segments)}")
        print("  Segments (first 3):")
        for i, segment in enumerate(polyline.segments[:3]):
            print(f"    Segment {i}: {segment}")
    else:
        print("  No segments attribute found")
    
    # Print other attributes
    print("  Other attributes:")
    for attr in dir(polyline):
        if not attr.startswith('__') and not callable(getattr(polyline, attr)):
            try:
                value = getattr(polyline, attr)
                if attr not in ['vertices', 'segments']:
                    print(f"    {attr}: {value}")
            except:
                print(f"    {attr}: <error accessing attribute>")

def create_points_for_surface():
    """Create a set of points for testing surface creation."""
    # Create a 5x5 grid of points in the XY plane
    points = []
    for x in np.linspace(0, 10, 5):
        for y in np.linspace(0, 10, 5):
            points.append([x, y, 0])
    return points

def visualize_surface(surface, title="Surface Visualization", show_constraints=False, save_image=True):
    """Visualize a surface with its triangulation in a style similar to MeshIt"""
    # Get the vertices
    points = np.array([[v.x, v.y, v.z] for v in surface.vertices])
    
    # Create a plotter
    plotter = pv.Plotter(off_screen=save_image)
    plotter.set_background([0.318, 0.341, 0.431])  # MeshIt background color
    
    # Add a title
    plotter.add_text(title, position='upper_edge', font_size=16, color='white')
    
    # If triangulated, show the mesh
    if len(surface.triangles) > 0:
        faces = []
        for triangle in surface.triangles:
            faces.extend([3, triangle[0], triangle[1], triangle[2]])
        mesh = pv.PolyData(points, faces=faces)
        
        # Add the mesh with MeshIt-like styling
        plotter.add_mesh(mesh, color=[0.8, 0.8, 0.6], opacity=0.7, show_edges=True, 
                         edge_color='black', line_width=1)
    else:
        # Otherwise, just show the points
        point_cloud = pv.PolyData(points)
        plotter.add_mesh(point_cloud, color='blue', point_size=8, 
                         render_points_as_spheres=True)
    
    # If convex hull exists, show it
    if hasattr(surface, 'convex_hull') and len(surface.convex_hull) > 0:
        hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
        
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
        
        # Debug: Print hull points to verify
        print(f"Visualizing convex hull with {n_points} points:")
        for i, p in enumerate(hull_points):
            print(f"  Point {i}: ({p[0]}, {p[1]}, {p[2]})")
    
    # If constraints should be shown and they exist
    if show_constraints and hasattr(surface, 'calculate_constraints'):
        try:
            constraints = surface.calculate_constraints()
            print(f"Created {len(constraints)} constraint segments from convex hull")
            print("In MeshIt, additional constraints would be added for intersection lines")
            
            if constraints:
                # For visualization purposes, we'll create lines for the constraints
                hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
                
                # Create lines for the constraints
                constraint_lines = []
                for constraint in constraints:
                    if len(constraint) >= 2:
                        constraint_lines.append([2, constraint[0], constraint[1]])
                
                # Create a polydata for the constraints
                if constraint_lines:
                    constraint_polydata = pv.PolyData(points)  # Use all points, not just hull points
                    constraint_polydata.lines = np.hstack(constraint_lines)
                    
                    # Add the constraints
                    plotter.add_mesh(constraint_polydata, color='green', 
                                    line_width=2, render_lines_as_tubes=True)
        except Exception as e:
            print(f"Error visualizing constraints: {e}")
            print("Continuing without constraint visualization...")
    
    # Add axes with MeshIt-like styling
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z', line_width=2)
    
    if save_image:
        # Save the visualization as an image instead of showing it interactively
        image_filename = f"{title.replace(' ', '_').replace(':', '_')}.png"
        plotter.screenshot(image_filename, transparent_background=False)
        print(f"Visualization saved to {image_filename}")
    else:
        # Show the plot interactively
        plotter.show(title=title)

def save_surface_to_vtk(surface, filename):
    """Save a surface to a VTK file"""
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Surface data\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        
        # Write vertices
        f.write(f"POINTS {len(surface.vertices)} double\n")
        for v in surface.vertices:
            f.write(f"{v.x} {v.y} {v.z}\n")
        
        # Write triangles
        if hasattr(surface, 'triangles') and surface.triangles:
            num_triangles = len(surface.triangles)
            # Each triangle has 3 points plus 1 for the count
            size = num_triangles * 4
            f.write(f"POLYGONS {num_triangles} {size}\n")
            for t in surface.triangles:
                f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
        else:
            # If no triangles, just write vertices as points
            f.write(f"VERTICES {len(surface.vertices)} {len(surface.vertices) + 1}\n")
            f.write(f"{len(surface.vertices)}")
            for i in range(len(surface.vertices)):
                f.write(f" {i}")
            f.write("\n")
    
    print(f"Saved surface to {filename}")

def save_convex_hull_to_vtk(surface, filename):
    """Save a surface's convex hull to a VTK file"""
    if hasattr(surface, 'convex_hull') and len(surface.convex_hull) > 0:
        hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
        
        # Create a polydata for the convex hull
        hull_polydata = pv.PolyData(hull_points)
        
        # Create lines connecting the convex hull points
        lines = []
        for i in range(len(hull_points)):
            next_i = (i + 1) % len(hull_points)
            lines.append([2, i, next_i])
        
        # Add the lines to the polydata
        if lines:
            hull_polydata.lines = np.hstack(lines)
        
        # Save the polydata to a VTK file
        hull_polydata.save(filename, binary=False)
        print(f"Saved convex hull to {filename}")
    else:
        print(f"No convex hull available to save to {filename}")

def visualize_meshit_workflow(surface, interactive=True, save_image=True):
    """
    Visualize the MeshIt workflow with a style similar to the C++ MeshIt application.
    
    Args:
        surface: The Surface object to visualize
        interactive: Whether to show the visualization interactively
        save_image: Whether to save the visualization as an image
    """
    if not interactive and not save_image:
        return
    
    # Create a multi-window plotter to show all steps at once
    plotter = pv.Plotter(shape=(2, 2), off_screen=save_image)
    
    # Step 1: Raw points
    plotter.subplot(0, 0)
    points = np.array([[v.x, v.y, v.z] for v in surface.vertices])
    point_cloud = pv.PolyData(points)
    plotter.add_mesh(point_cloud, color='blue', point_size=8, render_points_as_spheres=True)
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z', line_width=2)
    plotter.add_text("Step 1: Raw Points", position='upper_edge', font_size=12, color='white')
    plotter.set_background([0.318, 0.341, 0.431])
    
    # Step 2: Surface from points
    plotter.subplot(0, 1)
    plotter.add_mesh(point_cloud, color='blue', point_size=8, render_points_as_spheres=True)
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z', line_width=2)
    plotter.add_text("Step 2: Surface from Points", position='upper_edge', font_size=12, color='white')
    plotter.set_background([0.318, 0.341, 0.431])
    
    # Step 3: Surface with convex hull
    plotter.subplot(1, 0)
    plotter.add_mesh(point_cloud, color='blue', point_size=8, render_points_as_spheres=True)
    
    # Add convex hull
    if hasattr(surface, 'convex_hull') and len(surface.convex_hull) > 0:
        hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
        
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
    
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z', line_width=2)
    plotter.add_text("Step 3: Surface with Convex Hull", position='upper_edge', font_size=12, color='white')
    plotter.set_background([0.318, 0.341, 0.431])
    
    # Step 4: Surface with triangulation
    plotter.subplot(1, 1)
    
    # If triangulated, show the mesh
    if len(surface.triangles) > 0:
        faces = []
        for triangle in surface.triangles:
            faces.extend([3, triangle[0], triangle[1], triangle[2]])
        mesh = pv.PolyData(points, faces=faces)
        
        # Add the mesh with MeshIt-like styling
        plotter.add_mesh(mesh, color=[0.8, 0.8, 0.6], opacity=0.7, show_edges=True, 
                         edge_color='black', line_width=1)
    
    # Add convex hull
    if hasattr(surface, 'convex_hull') and len(surface.convex_hull) > 0:
        hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
        
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
    
    # Add constraints
    if hasattr(surface, 'calculate_constraints'):
        constraints = surface.calculate_constraints()
        if constraints:
            # For visualization purposes, we'll create lines for the constraints
            hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
            
            # Create lines for the constraints
            constraint_lines = []
            for constraint in constraints:
                if len(constraint) >= 2:
                    constraint_lines.append([2, constraint[0], constraint[1]])
            
            # Create a polydata for the constraints
            if constraint_lines:
                constraint_polydata = pv.PolyData(hull_points)
                constraint_polydata.lines = np.hstack(constraint_lines)
                
                # Add the constraints
                plotter.add_mesh(constraint_polydata, color='green', line_width=2, render_lines_as_tubes=True)
    
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z', line_width=2)
    plotter.add_text("Step 4: Triangulated Surface", position='upper_edge', font_size=12, color='white')
    plotter.set_background([0.318, 0.341, 0.431])
    
    # Link all cameras for synchronized rotation/zoom
    plotter.link_views()
    
    if save_image:
        # Save the visualization as an image instead of showing it interactively
        image_filename = "MeshIt_Workflow_Visualization.png"
        plotter.screenshot(image_filename, transparent_background=False)
        print(f"Workflow visualization saved to {image_filename}")
    else:
        # Show the plot interactively
        plotter.show(title="MeshIt Workflow Visualization")

def visualize_meshit_3d(surface, title="MeshIt 3D Visualization", save_image=True):
    """
    Create a 3D visualization of the surface with proper lighting and materials.
    
    Args:
        surface: A Surface instance
        title: Title for the visualization
        save_image: Whether to save the visualization as an image
    """
    # Create a plotter
    plotter = pv.Plotter(off_screen=save_image)
    
    # Set background color
    plotter.background_color = 'white'
    
    # Add coordinate axes
    plotter.add_axes()
    
    # If the surface has triangles, visualize them
    if hasattr(surface, 'triangles') and len(surface.triangles) > 0:
        # Extract vertices and triangles
        vertices = np.array([[v.x, v.y, v.z] for v in surface.vertices])
        triangles = np.array(surface.triangles)
        
        # Create faces array for PyVista
        faces = []
        for triangle in triangles:
            faces.append(3)  # Number of vertices in the face
            faces.extend(triangle)
        
        # Create a PyVista PolyData object
        mesh = pv.PolyData(vertices, np.array(faces))
        
        # Add the mesh to the plotter with proper lighting
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, 
                         edge_color='black', specular=0.5, 
                         smooth_shading=True)
    
    # Add the convex hull if it exists
    if hasattr(surface, 'convex_hull') and len(surface.convex_hull) > 0:
        # Extract hull points
        hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
        
        # Create a polydata for the hull points
        hull_polydata = pv.PolyData(hull_points)
        
        # Add the hull points to the plotter
        plotter.add_mesh(hull_polydata, color='red', point_size=10, 
                         render_points_as_spheres=True)
        
        # Create convex hull edges
        hull_edges = []
        for i in range(len(hull_points)):
            # PyVista expects lines in the format [number_of_points, point1, point2, ...]
            hull_edges.append([2, i, (i + 1) % len(hull_points)])
        
        # Convert to numpy array and flatten for PyVista
        hull_edges = np.array(hull_edges).flatten()
        hull_vtk = pv.PolyData(hull_points, lines=hull_edges)
        
        # Add the hull lines to the plotter
        plotter.add_mesh(hull_vtk, color='red', line_width=3, 
                         render_lines_as_tubes=True)
    
    # Add constraints if they exist
    if hasattr(surface, 'calculate_constraints'):
        constraints = surface.calculate_constraints()
        if constraints:
            # For visualization purposes, we'll create lines for the constraints
            points = np.array([[v.x, v.y, v.z] for v in surface.vertices])
            
            # Create lines for the constraints
            constraint_lines = []
            for constraint in constraints:
                if len(constraint) >= 2:
                    constraint_lines.append([2, constraint[0], constraint[1]])
            
            # Create a polydata for the constraints
            if constraint_lines:
                constraint_polydata = pv.PolyData(points)
                constraint_polydata.lines = np.hstack(constraint_lines)
                
                # Add the constraints to the plotter
                plotter.add_mesh(constraint_polydata, color='green', 
                                 line_width=3, render_lines_as_tubes=True)
    
    # Set camera position for a good view
    plotter.camera_position = [(20, 20, 20), (5, 5, 0), (0, 0, 1)]
    
    # Add title
    plotter.add_text(title, position='upper_edge', font_size=12, color='black')
    
    # Save or show the visualization
    if save_image:
        filename = title.replace(" ", "_").replace(":", "").replace("=", "_") + ".png"
        plotter.screenshot(filename)
        print(f"3D visualization saved to {filename}")
    else:
        plotter.show(title=title)
    
    return plotter

def save_vtk(obj, filename):
    """Save object to VTK file."""
    obj.save(filename)
    print(f"Saved {filename}")

def save_visualization(plotter, filename):
    """Save visualization to image file."""
    plotter.screenshot(filename)
    print(f"Saved visualization to {filename}")

def test_hull_size(hull_size):
    """Test triangulation with a specific hull size."""
    print(f"\n=== Testing with hull_size = {hull_size:.2f} ===")
    
    # Step 1: Create raw points
    print("Step 1: Creating raw points")
    raw_points = generate_points()
    print(f"Created {len(raw_points)} raw points")
    
    # Save raw points to VTK
    raw_points_vtk = pv.PolyData(raw_points)
    save_vtk(raw_points_vtk, "step1_raw_points.vtk")
    
    # Visualize raw points
    plotter = pv.Plotter(off_screen=True)
    plotter.add_points(raw_points_vtk, color='blue', point_size=10, render_points_as_spheres=True)
    plotter.add_title("Raw Points")
    save_visualization(plotter, f"Step_1__Raw_Points.png")
    plotter.close()
    
    # Step 2: Process points and create surface
    print("\nStep 2: Processing points and creating surface")
    surface = meshit.extensions.create_surface_from_points(raw_points)
    print(f"Created surface with {len(surface.vertices)} vertices")
    
    # Save surface points to VTK
    surface_points = [[v.x, v.y, v.z] for v in surface.vertices]
    surface_points_vtk = pv.PolyData(surface_points)
    save_vtk(surface_points_vtk, "step2_surface_points.vtk")
    
    # Save surface to VTK
    surface_vtk = pv.PolyData(surface_points)
    save_vtk(surface_vtk, "step2_surface.vtk")
    
    # Visualize surface points
    plotter = pv.Plotter(off_screen=True)
    plotter.add_points(surface_points_vtk, color='red', point_size=10, render_points_as_spheres=True)
    plotter.add_title("Surface Points")
    save_visualization(plotter, f"Step_2__Surface_Points.png")
    plotter.close()
    
    # Step 3: Calculate convex hull
    print("\nStep 3: Calculating convex hull")
    meshit.extensions.enhanced_calculate_convex_hull(surface)
    print(f"Calculated convex hull with {len(surface.convex_hull)} points")
    
    # Save convex hull points to VTK
    hull_points = [[v.x, v.y, v.z] for v in surface.convex_hull]
    hull_points_vtk = pv.PolyData(hull_points)
    save_vtk(hull_points_vtk, "step3_convex_hull_points.vtk")
    
    # Create convex hull edges
    hull_edges = []
    for i in range(len(hull_points)):
        # PyVista expects lines in the format [number_of_points, point1, point2, ...]
        hull_edges.append([2, i, (i + 1) % len(hull_points)])
    
    # Convert to numpy array and flatten for PyVista
    hull_edges = np.array(hull_edges).flatten()
    hull_vtk = pv.PolyData(hull_points, lines=hull_edges)
    save_vtk(hull_vtk, "step3_convex_hull.vtk")
    
    # Visualize surface with convex hull
    plotter = pv.Plotter(off_screen=True)
    plotter.add_points(surface_points_vtk, color='red', point_size=10, render_points_as_spheres=True)
    plotter.add_mesh(hull_vtk, color='green', line_width=3)
    plotter.add_title("Surface with Convex Hull")
    save_visualization(plotter, f"Step_3__Surface_with_Convex_Hull.png")
    plotter.close()
    
    # Step 3.5: Align intersections to convex hull
    print("\nStep 3.5: Aligning intersections to convex hull")
    # This is a placeholder for the actual alignment step
    aligned_hull_vtk = hull_vtk.copy()
    save_vtk(aligned_hull_vtk, "step3_5_aligned_hull.vtk")
    
    # Step 3.6: Calculate constraints
    print("\nStep 3.6: Calculating constraints using extension method")
    # Create constraint segments from convex hull
    constraint_segments = []
    for i in range(len(hull_points)):
        constraint_segments.append([hull_points[i], hull_points[(i + 1) % len(hull_points)]])
    print(f"Created {len(constraint_segments)} constraint segments from convex hull")
    
    # Visualize surface with constraints
    plotter = pv.Plotter(off_screen=True)
    plotter.add_points(surface_points_vtk, color='red', point_size=10, render_points_as_spheres=True)
    
    # Add constraint lines
    for segment in constraint_segments:
        line = pv.Line(segment[0], segment[1])
        plotter.add_mesh(line, color='blue', line_width=3)
    
    plotter.add_title("Surface with Constraints")
    save_visualization(plotter, f"Step_3.6__Surface_with_Constraints.png")
    plotter.close()
    
    # Step 4: Triangulate the surface
    print("\nStep 4: Triangulating the surface")
    print(f"Using hull size: {hull_size:.6f}")
    
    # Use our improved triangulation function
    triangles = meshit.extensions.triangulate_with_triangle(surface, hull_size=hull_size)
    print(f"Created {len(triangles)} triangles")
    
    # Create triangulated surface
    faces = []
    for triangle in triangles:
        # Check if all indices are valid
        if max(triangle) >= len(surface.vertices):
            continue
        faces.append(3)  # Number of vertices in the face
        faces.extend(triangle)
    
    # Only create the triangulated surface if we have valid faces
    if faces:
        triangulated_surface_vtk = pv.PolyData(surface_points, faces=faces)
        save_vtk(triangulated_surface_vtk, "step4_triangulated_surface.vtk")
        
        # Create a full triangulated surface with all data
        triangulated_surface_full_vtk = triangulated_surface_vtk.copy()
        save_vtk(triangulated_surface_full_vtk, "step4_triangulated_surface_full.vtk")
        
        # Visualize triangulated surface
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(triangulated_surface_vtk, color='tan', show_edges=True)
        plotter.add_title(f"Triangulated Surface (hull_size={hull_size:.2f})")
        save_visualization(plotter, f"Step_4__Triangulated_Surface_(hull_size={hull_size:.2f}).png")
        plotter.close()
        
        # Create a 3D visualization
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(triangulated_surface_vtk, color='tan', show_edges=True)
        plotter.add_title(f"MeshIt 3D Visualization (hull_size {hull_size:.2f})")
        save_visualization(plotter, f"MeshIt_3D_Visualization_(hull_size_{hull_size:.2f}).png")
        plotter.close()
        
        # Create a workflow visualization
        plotter = pv.Plotter(shape=(2, 2), off_screen=True)
        
        plotter.subplot(0, 0)
        plotter.add_points(raw_points_vtk, color='blue', point_size=10, render_points_as_spheres=True)
        plotter.add_title("Step 1: Raw Points")
        
        plotter.subplot(0, 1)
        plotter.add_points(surface_points_vtk, color='red', point_size=10, render_points_as_spheres=True)
        plotter.add_title("Step 2: Surface Points")
        
        plotter.subplot(1, 0)
        plotter.add_points(surface_points_vtk, color='red', point_size=10, render_points_as_spheres=True)
        plotter.add_mesh(hull_vtk, color='green', line_width=3)
        plotter.add_title("Step 3: Convex Hull")
        
        plotter.subplot(1, 1)
        plotter.add_mesh(triangulated_surface_vtk, color='tan', show_edges=True)
        plotter.add_title(f"Step 4: Triangulation (hull_size={hull_size:.2f})")
        
        save_visualization(plotter, f"MeshIt_Workflow_Visualization_{hull_size:.2f}.png")
        plotter.close()
    else:
        print("Warning: No valid triangles found for visualization")
    
    # Calculate and print triangulation statistics
    print("\nTriangulation Statistics:")
    print(f"Total triangles: {len(triangles)}")
    
    # Calculate triangle areas
    areas = []
    for triangle in triangles:
        # Check if all indices are valid
        if max(triangle) >= len(surface.vertices):
            print(f"Warning: Triangle {triangle} has invalid indices (max index: {len(surface.vertices)-1})")
            continue
            
        v0 = np.array([surface.vertices[triangle[0]].x, surface.vertices[triangle[0]].y, surface.vertices[triangle[0]].z])
        v1 = np.array([surface.vertices[triangle[1]].x, surface.vertices[triangle[1]].y, surface.vertices[triangle[1]].z])
        v2 = np.array([surface.vertices[triangle[2]].x, surface.vertices[triangle[2]].y, surface.vertices[triangle[2]].z])
        
        # Calculate area using cross product
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        areas.append(area)
    
    if areas:
        print(f"Triangle areas: min={min(areas):.6f}, max={max(areas):.6f}, avg={sum(areas)/len(areas):.6f}")
        print(f"Total surface area: {sum(areas):.6f}")
    
    # Calculate edge lengths
    edge_lengths = []
    for triangle in triangles:
        # Check if all indices are valid
        if max(triangle) >= len(surface.vertices):
            continue
            
        v0 = np.array([surface.vertices[triangle[0]].x, surface.vertices[triangle[0]].y, surface.vertices[triangle[0]].z])
        v1 = np.array([surface.vertices[triangle[1]].x, surface.vertices[triangle[1]].y, surface.vertices[triangle[1]].z])
        v2 = np.array([surface.vertices[triangle[2]].x, surface.vertices[triangle[2]].y, surface.vertices[triangle[2]].z])
        
        edge_lengths.append(np.linalg.norm(v1 - v0))
        edge_lengths.append(np.linalg.norm(v2 - v1))
        edge_lengths.append(np.linalg.norm(v0 - v2))
    
    if edge_lengths:
        print(f"Edge lengths: min={min(edge_lengths):.6f}, max={max(edge_lengths):.6f}, avg={sum(edge_lengths)/len(edge_lengths):.6f}")
    
    print(f"\nCompleted testing with hull_size = {hull_size:.2f}")
    print("All VTK files and visualization images have been saved.")

def main():
    # Test with just one hull size for now
    hull_size = 2.0
    test_hull_size(hull_size)
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()