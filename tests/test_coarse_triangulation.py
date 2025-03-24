#!/usr/bin/env python
"""
Test script for visualizing and testing coarse triangulation with gradient control.

This script demonstrates:
1. Loading or generating input points
2. Creating a surface
3. Computing convex hull
4. Coarse segmentation (for polylines)
5. Coarse triangulation with gradient control
6. Visualizing each step of the process

You can modify gradient and refinesize parameters to observe their effect on triangulation.
"""

import os
import sys

# Add the parent directory to the Python path so we can import modules more easily
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import numpy as np
import time
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import pyvista as pv
    HAS_PYVISTA = True
    # Configure PyVista for interactive display
    pv.set_plot_theme("document")
    try:
        pv.global_theme.jupyter_backend = "panel"
        pv.global_theme.smooth_shading = True
        pv.global_theme.anti_aliasing = True
        pv.global_theme.show_scalar_bar = False
    except:
        print("Warning: Could not configure PyVista backend. Visualization may not work properly.")
except ImportError:
    HAS_PYVISTA = False
    print("PyVista not found. Some visualizations will be limited.")

# Import MeshIt modules
try:
    from meshit.core import MeshItModel, Vector3D, Surface, Polyline, GradientControl
    from meshit import extensions
    print("Successfully imported MeshIt modules.")
except ImportError as e:
    print(f"Error importing MeshIt modules. Make sure MeshIt is installed: {e}")
    # Try a direct import approach as fallback
    try:
        import sys
        print(f"Python search paths: {sys.path}")
        print("Attempting alternative import method...")
        
        # Try an alternative import method
        import importlib.util
        spec = importlib.util.find_spec("meshit")
        if spec is not None:
            print(f"Found meshit at {spec.origin}")
            import meshit
            from meshit.core import MeshItModel, Vector3D, Surface, Polyline, GradientControl
            print("Alternative import successful.")
        else:
            print("Could not find meshit module.")
            sys.exit(1)
    except Exception as e2:
        print(f"Alternative import also failed: {e2}")
        sys.exit(1)

# Define colors for visualization
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

def generate_points(num_points=25, point_type='grid', noise=0.0):
    """
    Generate points for testing.
    
    Args:
        num_points: Approximate number of points to generate
        point_type: 'grid', 'random', or 'circle'
        noise: Amount of random noise to add (0.0-1.0)
        
    Returns:
        List of [x, y, z] points
    """
    points = []
    
    if point_type == 'grid':
        # Calculate grid size to get approximately num_points
        grid_size = int(np.sqrt(num_points))
        for x in np.linspace(0, 10, grid_size):
            for y in np.linspace(0, 10, grid_size):
                # Add some noise if requested
                if noise > 0:
                    x += np.random.uniform(-noise, noise)
                    y += np.random.uniform(-noise, noise)
                points.append([x, y, 0])
    
    elif point_type == 'random':
        np.random.seed(42)  # For reproducibility
        for _ in range(num_points):
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 10)
            z = 0  # Keep points in XY plane for simplicity
            points.append([x, y, z])
    
    elif point_type == 'circle':
        # Generate points in a circle
        radius = 5.0
        center = [5.0, 5.0, 0.0]
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            
            # Add some noise if requested
            if noise > 0:
                x += np.random.uniform(-noise, noise)
                y += np.random.uniform(-noise, noise)
            
            points.append([x, y, 0.0])
    
    return points

def load_points_from_file(filename):
    """
    Load points from a text file.
    Expected format: x y z on each line.
    
    Args:
        filename: Path to the file
        
    Returns:
        List of [x, y, z] points
    """
    points = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Skip empty lines and comments
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse coordinates
                coords = line.split()
                if len(coords) >= 3:
                    x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                    points.append([x, y, z])
                elif len(coords) == 2:
                    x, y = float(coords[0]), float(coords[1])
                    points.append([x, y, 0.0])  # Assume z=0 if not provided
        
        print(f"Loaded {len(points)} points from {filename}")
    except Exception as e:
        print(f"Error loading points from {filename}: {e}")
    
    return points

def save_points_to_file(points, filename):
    """
    Save points to a text file.
    
    Args:
        points: List of [x, y, z] points
        filename: Path to save the file
    """
    try:
        with open(filename, 'w') as f:
            f.write("# MeshIt test points - x y z\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        print(f"Saved {len(points)} points to {filename}")
    except Exception as e:
        print(f"Error saving points to {filename}: {e}")

def visualize_points(points, title="Points Visualization", save_image=False, output_dir="output"):
    """
    Visualize points in 3D.
    
    Args:
        points: List of [x, y, z] points
        title: Title for the visualization
        save_image: Whether to save the image to file
        output_dir: Directory to save images
    """
    if not HAS_PYVISTA:
        # Fall back to matplotlib
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        
        # Plot points
        ax.scatter(xs, ys, zs, color='blue', s=50)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Set view angle
        ax.view_init(30, 45)
        
        # Save or show
        if save_image:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}.png"))
            print(f"Saved visualization to {output_dir}/{title.replace(' ', '_')}.png")
        else:
            plt.show()
        
        plt.close()
        return
    
    # Use PyVista for better visualization
    points_array = np.array(points)
    
    # Create a plotter
    plotter = pv.Plotter(off_screen=save_image)
    
    # Add points to the plotter
    point_cloud = pv.PolyData(points_array)
    plotter.add_mesh(point_cloud, color='blue', point_size=10, render_points_as_spheres=True)
    
    # Add coordinate axes
    plotter.add_axes()
    
    # Set camera position
    plotter.camera_position = [(20, 20, 20), (5, 5, 0), (0, 0, 1)]
    
    # Set background color
    plotter.background_color = COLORS['background']
    
    # Add title
    plotter.add_text(title, position='upper_edge', font_size=16, color='white')
    
    # Save or show the visualization
    if save_image:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
        plotter.screenshot(filename)
        print(f"Saved visualization to {filename}")
    else:
        plotter.show(title=title)

def visualize_surface(surface, title="Surface Visualization", show_convex_hull=True, 
                      show_triangulation=True, save_image=False, output_dir="output"):
    """
    Visualize a surface with its triangulation and convex hull.
    
    Args:
        surface: A Surface object
        title: Title for the visualization
        show_convex_hull: Whether to show the convex hull
        show_triangulation: Whether to show the triangulation
        save_image: Whether to save the image to file
        output_dir: Directory to save images
    """
    if not HAS_PYVISTA:
        # Fall back to matplotlib for basic visualization
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract vertices
        vertices = [[v.x, v.y, v.z] for v in surface.vertices]
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        
        # Plot vertices
        ax.scatter(xs, ys, zs, color='blue', s=50, label='Vertices')
        
        # Plot convex hull if available and requested
        if show_convex_hull and hasattr(surface, 'convex_hull') and surface.convex_hull:
            hull_vertices = [[v.x, v.y, v.z] for v in surface.convex_hull]
            hull_xs = [v[0] for v in hull_vertices]
            hull_ys = [v[1] for v in hull_vertices]
            hull_zs = [v[2] for v in hull_vertices]
            
            # Connect the hull points with lines
            n_hull = len(hull_vertices)
            for i in range(n_hull):
                next_i = (i + 1) % n_hull
                ax.plot([hull_xs[i], hull_xs[next_i]], 
                        [hull_ys[i], hull_ys[next_i]], 
                        [hull_zs[i], hull_zs[next_i]], 
                        color='red', linestyle='-', linewidth=2)
            
            ax.scatter(hull_xs, hull_ys, hull_zs, color='red', s=80, label='Convex Hull')
        
        # Plot triangulation if available and requested
        if show_triangulation and surface.triangles:
            for tri in surface.triangles:
                tri_xs = [vertices[tri[0]][0], vertices[tri[1]][0], vertices[tri[2]][0], vertices[tri[0]][0]]
                tri_ys = [vertices[tri[0]][1], vertices[tri[1]][1], vertices[tri[2]][1], vertices[tri[0]][1]]
                tri_zs = [vertices[tri[0]][2], vertices[tri[1]][2], vertices[tri[2]][2], vertices[tri[0]][2]]
                ax.plot(tri_xs, tri_ys, tri_zs, color='black', linestyle='-', linewidth=1)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Add legend
        plt.legend()
        
        # Set view angle
        ax.view_init(30, 45)
        
        # Save or show
        if save_image:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}.png"))
            print(f"Saved visualization to {output_dir}/{title.replace(' ', '_')}.png")
        else:
            plt.show()
        
        plt.close()
        return
    
    # Use PyVista for better visualization
    # Get the vertices
    points = np.array([[v.x, v.y, v.z] for v in surface.vertices])
    
    # Create a plotter
    plotter = pv.Plotter(off_screen=save_image)
    plotter.set_background(COLORS['background'])
    
    # Add a title
    plotter.add_text(title, position='upper_edge', font_size=16, color='white')
    
    # If triangulated and requested, show the mesh
    if show_triangulation and surface.triangles:
        faces = []
        for triangle in surface.triangles:
            faces.extend([3, triangle[0], triangle[1], triangle[2]])
        
        mesh = pv.PolyData(points, faces=faces)
        
        # Add the mesh with styling
        plotter.add_mesh(mesh, color=COLORS['surface'], opacity=0.7, show_edges=True, 
                         edge_color='black', line_width=1)
    else:
        # Otherwise, just show the points
        point_cloud = pv.PolyData(points)
        plotter.add_mesh(point_cloud, color=COLORS['vertices'], point_size=8, 
                         render_points_as_spheres=True)
    
    # If convex hull exists and requested, show it
    if show_convex_hull and hasattr(surface, 'convex_hull') and surface.convex_hull:
        hull_points = np.array([[v.x, v.y, v.z] for v in surface.convex_hull])
        
        # Create lines connecting the convex hull points
        lines = []
        n_points = len(hull_points)
        
        for i in range(n_points):
            next_i = (i + 1) % n_points
            lines.append([2, i, next_i])
        
        # Create a polydata for the convex hull
        hull_polydata = pv.PolyData(hull_points)
        
        # Set the lines properly
        if lines:
            hull_polydata.lines = np.hstack(lines)
        
        # Add the convex hull lines
        plotter.add_mesh(hull_polydata, color=COLORS['convex_hull'], line_width=3, render_lines_as_tubes=True)
        
        # Add the convex hull points
        plotter.add_mesh(pv.PolyData(hull_points), color=COLORS['convex_hull'], point_size=10, 
                         render_points_as_spheres=True)
    
    # Add axes
    plotter.add_axes()
    
    # Set camera position
    plotter.camera_position = [(20, 20, 20), (5, 5, 0), (0, 0, 1)]
    
    # Save or show the visualization
    if save_image:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
        plotter.screenshot(filename)
        print(f"Saved visualization to {filename}")
    else:
        plotter.show(title=title)

def test_gradient_control(points, gradient_values=None, save_all=False, output_dir="output"):
    """
    Test triangulation with different gradient control values.
    
    Args:
        points: List of [x, y, z] points
        gradient_values: List of gradient values to test
        save_all: Whether to save all visualizations
        output_dir: Directory to save output files
    """
    if gradient_values is None:
        gradient_values = [0.5, 1.0, 2.0, 3.0]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a model
    model = MeshItModel()
    
    # Create a surface from points
    surface = Surface()
    surface.name = "TestSurface"
    surface.type = "Default"
    
    # Add vertices to the surface
    for point in points:
        surface.add_vertex(Vector3D(point[0], point[1], point[2]))
    
    # Add the surface to the model
    model.surfaces.append(surface)
    
    # Save the input points
    print(f"\nTesting with {len(points)} input points")
    visualize_points(points, title="Input Points", save_image=save_all, output_dir=output_dir)
    
    # Step 1: Calculate convex hull
    print("\nStep 1: Calculating convex hull...")
    start_time = time.time()
    surface.enhanced_calculate_convex_hull()
    elapsed = time.time() - start_time
    print(f"Convex hull calculated with {len(surface.convex_hull)} vertices in {elapsed:.3f} seconds")
    
    # Visualize the surface with convex hull
    visualize_surface(surface, title="Surface with Convex Hull", 
                     show_triangulation=False, save_image=save_all, output_dir=output_dir)
    
    # Test different gradient values
    print("\nStep 2: Testing triangulation with different gradient values...")
    
    # Create a dictionary to store results
    results = {}
    
    for gradient in gradient_values:
        print(f"\nTesting gradient = {gradient}")
        
        # Create a copy of the surface to test with this gradient
        test_surface = Surface()
        test_surface.name = f"Gradient_{gradient}"
        test_surface.type = "Default"
        
        # Copy vertices
        for vertex in surface.vertices:
            test_surface.add_vertex(Vector3D(vertex.x, vertex.y, vertex.z))
        
        # Copy convex hull
        test_surface.convex_hull = []
        for hull_point in surface.convex_hull:
            test_surface.convex_hull.append(Vector3D(hull_point.x, hull_point.y, hull_point.z))
        
        # Calculate bounding box for base size
        min_x = min(v.x for v in test_surface.vertices)
        max_x = max(v.x for v in test_surface.vertices)
        min_y = min(v.y for v in test_surface.vertices)
        max_y = max(v.y for v in test_surface.vertices)
        min_z = min(v.z for v in test_surface.vertices)
        max_z = max(v.z for v in test_surface.vertices)
        
        bbox_diagonal = ((max_x - min_x)**2 + (max_y - min_y)**2 + (max_z - min_z)**2)**0.5
        base_size = bbox_diagonal / 15.0
        
        # Prepare feature points for gradient control
        feature_points = []
        feature_sizes = []
        
        # Add convex hull points as features
        hull_point_size = base_size * 0.5
        for point in test_surface.convex_hull:
            feature_points.append([point.x, point.y, point.z])
            feature_sizes.append(hull_point_size)
        
        # Update gradient control
        gc = GradientControl.get_instance()
        if feature_points:
            # Flatten first point for the interface
            first_point = feature_points[0][0] if feature_points else 0.0
            first_size = feature_sizes[0] if feature_sizes else base_size
            
            gc.update(
                float(gradient),
                float(base_size),
                len(feature_points),
                float(first_point),
                float(first_size)
            )
            
            print(f"Applied gradient {gradient} with {len(feature_points)} feature points")
            print(f"Base size: {base_size:.4f}, hull point size: {hull_point_size:.4f}")
        
        # Perform triangulation with enhanced method
        start_time = time.time()
        test_surface.enhanced_triangulate(gradient=gradient)
        elapsed = time.time() - start_time
        
        # Store results
        triangle_count = len(test_surface.triangles)
        results[gradient] = {
            'triangle_count': triangle_count,
            'elapsed_time': elapsed
        }
        
        print(f"Created {triangle_count} triangles in {elapsed:.3f} seconds")
        
        # Visualize the triangulated surface
        visualize_surface(test_surface, 
                         title=f"Triangulation with Gradient {gradient}",
                         save_image=save_all, output_dir=output_dir)
    
    # Print comparison table
    print("\nResults comparison:")
    print("-" * 60)
    print(f"{'Gradient':<10} | {'Triangles':<10} | {'Time (s)':<10} | {'% Change':<10}")
    print("-" * 60)
    
    base_count = results[1.0]['triangle_count'] if 1.0 in results else None
    
    for gradient in sorted(results.keys()):
        result = results[gradient]
        triangle_count = result['triangle_count']
        elapsed = result['elapsed_time']
        
        if base_count:
            percent_change = (triangle_count - base_count) / base_count * 100
            print(f"{gradient:<10.1f} | {triangle_count:<10} | {elapsed:<10.3f} | {percent_change:<+10.2f}%")
        else:
            print(f"{gradient:<10.1f} | {triangle_count:<10} | {elapsed:<10.3f} | {'N/A':<10}")
    
    print("-" * 60)
    
    # Save the comparison results
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write("Gradient Control Testing Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input Points: {len(points)}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write(f"{'Gradient':<10} | {'Triangles':<10} | {'Time (s)':<10} | {'% Change':<10}\n")
        f.write("-" * 60 + "\n")
        
        for gradient in sorted(results.keys()):
            result = results[gradient]
            triangle_count = result['triangle_count']
            elapsed = result['elapsed_time']
            
            if base_count:
                percent_change = (triangle_count - base_count) / base_count * 100
                f.write(f"{gradient:<10.1f} | {triangle_count:<10} | {elapsed:<10.3f} | {percent_change:<+10.2f}%\n")
            else:
                f.write(f"{gradient:<10.1f} | {triangle_count:<10} | {elapsed:<10.3f} | {'N/A':<10}\n")
        
        f.write("-" * 60 + "\n")
    
    print(f"\nResults saved to {output_dir}/results.txt")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test coarse triangulation with gradient control")
    
    # Point generation options
    point_group = parser.add_argument_group("Point options")
    point_group.add_argument("--input-file", "-i", help="Input file with points (x y z per line)")
    point_group.add_argument("--num-points", "-n", type=int, default=25, 
                           help="Number of points to generate (default: 25)")
    point_group.add_argument("--point-type", "-p", choices=["grid", "random", "circle"],
                           default="grid", help="Type of points to generate (default: grid)")
    point_group.add_argument("--noise", type=float, default=0.0,
                           help="Amount of noise to add to generated points (0.0-1.0)")
    point_group.add_argument("--save-points", "-s", help="Save generated points to file")
    
    # Gradient testing options
    gradient_group = parser.add_argument_group("Gradient options")
    gradient_group.add_argument("--gradients", "-g", type=float, nargs="+",
                              default=[0.5, 1.0, 2.0, 3.0],
                              help="Gradient values to test (default: 0.5 1.0 2.0 3.0)")
    
    # Visualization options
    viz_group = parser.add_argument_group("Visualization options")
    viz_group.add_argument("--save-all", action="store_true",
                         help="Save all visualizations to files")
    viz_group.add_argument("--output-dir", "-o", default="output",
                         help="Output directory for saved files (default: output)")
    
    args = parser.parse_args()
    
    # Load or generate points
    if args.input_file:
        points = load_points_from_file(args.input_file)
    else:
        points = generate_points(args.num_points, args.point_type, args.noise)
    
    # Save points if requested
    if args.save_points:
        save_points_to_file(points, args.save_points)
    
    # Run gradient control testing
    test_gradient_control(points, args.gradients, args.save_all, args.output_dir)

if __name__ == "__main__":
    main() 