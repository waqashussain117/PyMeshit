import meshit
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
import os
import time

def generate_polyline_points():
    """Generate points for a polyline that will be used for coarse segmentation."""
    # Create a spiral polyline
    polyline_points = []
    for t in np.linspace(0, 4*np.pi, 20):
        x = 5 * np.cos(t)
        y = 5 * np.sin(t)
        z = t
        polyline_points.append([x, y, z])
    
    return polyline_points

def visualize_points(points, title="Point Visualization"):
    """Visualize points in 3D."""
    plotter = pv.Plotter()
    plotter.set_background('white')
    plotter.add_title(title, font_size=16)
    
    # Convert points to numpy array
    points_array = np.array(points)
    point_cloud = pv.PolyData(points_array)
    
    # Add points
    plotter.add_mesh(point_cloud, color='blue', 
                    point_size=10, render_points_as_spheres=True)
    
    # Add lines connecting the points
    if len(points) > 1:
        cells = []
        for i in range(len(points) - 1):
            cells.extend([2, i, i+1])
        
        lines = pv.PolyData(points_array, np.array(cells))
        plotter.add_mesh(lines, color='red', line_width=2)
    
    # Add axes
    plotter.add_axes()
    
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
        
        # Add lines connecting the points
        if len(points) > 1:
            f.write(f"LINES {len(points) - 1} {(len(points) - 1) * 3}\n")
            for i in range(len(points) - 1):
                f.write(f"2 {i} {i+1}\n")
    
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

def test_coarse_segmentation(interactive=True):
    """Test the coarse segmentation step."""
    print("=" * 80)
    print("STEP 1: Generate Polyline Points")
    print("=" * 80)
    
    # Generate polyline points
    polyline_points = generate_polyline_points()
    
    print(f"Generated {len(polyline_points)} points for the polyline")
    
    # Visualize the points
    if interactive:
        print("\nVisualizing polyline points...")
        visualize_points(polyline_points, "Step 1: Polyline Points")
    
    # Save the points to a VTK file
    save_points_to_vtk(polyline_points, "step1_polyline_points.vtk")
    
    print("\n" + "=" * 80)
    print("STEP 2: Create Polyline Object")
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
    print("STEP 3: Perform Coarse Segmentation")
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
        visualize_polyline_segments(polyline, "Step 3: Polyline with Segments")
    
    # Save the segments to a VTK file
    if polyline.segments:
        save_segments_to_vtk(polyline, "step3_polyline_segments.vtk")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return polyline

if __name__ == "__main__":
    # Run the coarse segmentation test
    polyline = test_coarse_segmentation(interactive=True) 