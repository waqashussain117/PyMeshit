import meshit
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
import os
import time

def generate_points():
    """Generate points for a surface and a polyline."""
    # Generate points for a surface (a plane with some noise)
    surface_points = []
    for x in np.linspace(-5, 5, 10):
        for y in np.linspace(-5, 5, 10):
            z = 0.2 * np.sin(x) * np.cos(y)  # Add some noise
            surface_points.append([x, y, z])
    
    # Generate points for a polyline (a spiral)
    polyline_points = []
    for t in np.linspace(0, 4*np.pi, 20):
        x = 3 * np.cos(t)
        y = 3 * np.sin(t)
        z = t / 2
        polyline_points.append([x, y, z])
    
    return surface_points, polyline_points

def visualize_points(points_list, labels, title="Point Visualization"):
    """Visualize points in 3D."""
    plotter = pv.Plotter()
    plotter.set_background('white')
    plotter.add_title(title, font_size=16)
    
    # Add each set of points with a different color
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, points in enumerate(points_list):
        if len(points) > 0:
            points_array = np.array(points)
            point_cloud = pv.PolyData(points_array)
            plotter.add_mesh(point_cloud, color=colors[i % len(colors)], 
                            point_size=10, render_points_as_spheres=True,
                            label=labels[i])
    
    # Add axes and legend
    plotter.add_axes()
    plotter.add_legend()
    
    # Show the visualization
    plotter.show()

def visualize_polyline_segments(polyline, title="Polyline Segments Visualization"):
    """Visualize a polyline with its segments."""
    plotter = pv.Plotter()
    plotter.set_background('white')
    plotter.add_title(title, font_size=16)
    
    # Convert vertices to numpy array
    vertices = np.array([[v.x, v.y, v.z] for v in polyline.vertices])
    point_cloud = pv.PolyData(vertices)
    
    # Add points
    plotter.add_mesh(point_cloud, color='blue', 
                    point_size=10, render_points_as_spheres=True)
    
    # Add segments as lines
    if polyline.segments:
        cells = []
        for segment in polyline.segments:
            cells.extend([2, segment[0], segment[1]])
        
        lines = pv.PolyData(vertices, np.array(cells))
        plotter.add_mesh(lines, color='red', line_width=2)
    
    # Add axes
    plotter.add_axes()
    
    # Show the visualization
    plotter.show()

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

def save_segments_to_vtk(polyline, filename):
    """Save polyline segments to a VTK file."""
    # Convert vertices to list of points
    points = [[v.x, v.y, v.z] for v in polyline.vertices]
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Segments data\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        # Add segments as lines
        if polyline.segments:
            f.write(f"LINES {len(polyline.segments)} {len(polyline.segments) * 3}\n")
            for segment in polyline.segments:
                f.write(f"2 {segment[0]} {segment[1]}\n")
    
    print(f"Segments saved to VTK file: {filename}")

def test_workflow(interactive=True):
    """Test the complete workflow: points → convex hull → coarse segmentation."""
    print("=" * 80)
    print("STEP 1: Generate Points")
    print("=" * 80)
    
    # Generate points
    surface_points, polyline_points = generate_points()
    
    print(f"Generated {len(surface_points)} points for the surface")
    print(f"Generated {len(polyline_points)} points for the polyline")
    
    # Visualize the points
    if interactive:
        print("\nVisualizing points...")
        visualize_points(
            [surface_points, polyline_points],
            ["Surface Points", "Polyline Points"],
            "Step 1: Raw Points"
        )
    
    # Save the points to VTK files
    save_points_to_vtk(surface_points, "step1_surface_points.vtk")
    save_points_to_vtk(polyline_points, "step1_polyline_points.vtk")
    
    print("\n" + "=" * 80)
    print("STEP 2: Compute Convex Hull")
    print("=" * 80)
    
    # Compute convex hull for the surface points
    print("Computing convex hull for surface points...")
    start_time = time.time()
    surface_hull = meshit.compute_convex_hull(surface_points)
    elapsed = time.time() - start_time
    print(f"Convex hull computation completed in {elapsed:.4f} seconds")
    print(f"Convex hull consists of {len(surface_hull)} points")
    
    # Visualize the points with their convex hull
    if interactive:
        print("\nVisualizing points with convex hull...")
        visualize_points(
            [surface_points, surface_hull, polyline_points],
            ["Surface Points", "Surface Hull", "Polyline Points"],
            "Step 2: Points with Convex Hull"
        )
    
    # Save the convex hull to a VTK file
    save_points_to_vtk(surface_hull, "step2_surface_hull.vtk")
    
    print("\n" + "=" * 80)
    print("STEP 3: Create Polyline")
    print("=" * 80)
    
    # Create a polyline using the create_polyline function
    polyline = meshit.create_polyline(polyline_points, "TestPolyline")
    polyline.size = 0.5  # Segment size for refinement
    
    print(f"Created polyline with {len(polyline.vertices)} vertices")
    
    # Print the first few vertices to verify
    if polyline.vertices:
        print("\nFirst 5 vertices:")
        for i, vertex in enumerate(polyline.vertices[:5]):
            print(f"  Vertex {i}: ({vertex.x}, {vertex.y}, {vertex.z})")
    
    # Print bounds if available
    if hasattr(polyline, 'bounds'):
        print(f"\nPolyline bounds: Min({polyline.bounds[0].x}, {polyline.bounds[0].y}, {polyline.bounds[0].z}), "
              f"Max({polyline.bounds[1].x}, {polyline.bounds[1].y}, {polyline.bounds[1].z})")
    
    print("\n" + "=" * 80)
    print("STEP 4: Perform Coarse Segmentation")
    print("=" * 80)
    
    # Check if the polyline has segments before segmentation
    print(f"Before segmentation: {len(polyline.segments)} segments")
    
    # Perform coarse segmentation
    print("\nPerforming coarse segmentation...")
    start_time = time.time()
    polyline.calculate_segments(False)  # False for coarse segmentation
    elapsed = time.time() - start_time
    
    # Check if the polyline has segments after segmentation
    print(f"After segmentation: {len(polyline.segments)} segments")
    print(f"Coarse segmentation completed in {elapsed:.4f} seconds")
    
    # Print the first few segments
    if polyline.segments:
        print("\nFirst 5 segments:")
        for i, segment in enumerate(polyline.segments[:5]):
            print(f"  Segment {i}: {segment}")
    
    # Visualize the polyline with segments
    if interactive and polyline.segments:
        print("\nVisualizing polyline with segments...")
        visualize_polyline_segments(polyline, "Step 4: Polyline with Segments")
    
    # Save the segments to a VTK file
    if polyline.segments:
        save_segments_to_vtk(polyline, "step4_polyline_segments.vtk")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return surface_points, surface_hull, polyline

if __name__ == "__main__":
    # Run the workflow test
    surface_points, surface_hull, polyline = test_workflow(interactive=True) 