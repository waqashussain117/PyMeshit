#!/usr/bin/env python
"""
Simplified test script for visualizing coarse triangulation with gradient control.

This script demonstrates:
1. Generating test input points
2. Creating a simple surface representation
3. Visualizing with different gradient parameters
4. Creating comparison metrics

This script doesn't require the full MeshIt implementation.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from pathlib import Path
import csv

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

# Define a simple gradient control class for demonstration
class GradientControl:
    def __init__(self, gradient=1.0):
        self.gradient = gradient
    
    def __str__(self):
        return f"GradientControl(gradient={self.gradient})"

# Generate test points
def generate_points(num_points=25, point_type='grid', noise=0.0):
    """Generate test points in 3D space."""
    if point_type == 'grid':
        # Calculate a square grid dimension based on num_points
        grid_dim = int(np.ceil(np.sqrt(num_points)))
        x = np.linspace(-1, 1, grid_dim)
        y = np.linspace(-1, 1, grid_dim)
        xx, yy = np.meshgrid(x, y)
        xx = xx.flatten()[:num_points]
        yy = yy.flatten()[:num_points]
        zz = np.zeros_like(xx)
        
        # Add noise if requested
        if noise > 0:
            xx += np.random.normal(0, noise, xx.shape)
            yy += np.random.normal(0, noise, yy.shape)
            zz += np.random.normal(0, noise/5, zz.shape)  # Less noise in z
    
    elif point_type == 'random':
        xx = np.random.uniform(-1, 1, num_points)
        yy = np.random.uniform(-1, 1, num_points)
        zz = np.random.uniform(-0.1, 0.1, num_points)
    
    elif point_type == 'circle':
        theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        xx = np.cos(theta)
        yy = np.sin(theta)
        zz = np.zeros_like(xx)
        
        # Add noise if requested
        if noise > 0:
            xx += np.random.normal(0, noise/3, xx.shape)
            yy += np.random.normal(0, noise/3, yy.shape)
            zz += np.random.normal(0, noise/5, zz.shape)
    
    else:
        raise ValueError(f"Unknown point type: {point_type}")
    
    return np.column_stack((xx, yy, zz))

# Load points from file
def load_points(filename):
    """Load 3D points from a text file."""
    points = np.loadtxt(filename)
    if points.shape[1] < 3:
        # If only 2D points, add z=0
        zeros = np.zeros((points.shape[0], 1))
        points = np.hstack((points, zeros))
    return points

# Save points to file
def save_points(points, filename):
    """Save 3D points to a text file."""
    np.savetxt(filename, points, fmt='%.6f')
    print(f"Saved {len(points)} points to {filename}")

# Get triangulation for visualization based on gradient
def get_triangulation(points, gradient=1.0):
    """
    Generate a triangulation based on points and gradient.
    For this simplified version, we're using Delaunay triangulation
    but modifying the density based on gradient value.
    """
    from scipy.spatial import Delaunay
    
    # The gradient parameter affects how uniform the triangulation is
    # Lower gradient = more uniform triangles
    # Higher gradient = triangle size varies more
    
    # Add extra points based on gradient to simulate density control
    # For low gradient, add more points uniformly
    # For high gradient, add more points near the edges
    num_extra_points = int(50 / gradient)  # More points for lower gradient
    
    if gradient <= 1.0:
        # For low gradients, add more uniform points
        extra_points = np.random.uniform(-1, 1, (num_extra_points, 2))
        extra_points = np.column_stack((extra_points, np.zeros(num_extra_points)))
        all_points = np.vstack((points, extra_points))
    else:
        # For high gradients, no extra points needed
        all_points = points
    
    # Create triangulation
    if len(all_points) >= 3:
        # Project to 2D for triangulation if points are 3D
        if all_points.shape[1] > 2:
            tri = Delaunay(all_points[:, :2])
        else:
            tri = Delaunay(all_points)
        
        triangles = tri.simplices
        
        # For high gradients, randomly remove some triangles near the center
        if gradient > 1.0:
            # Calculate centroids
            centers = np.mean(all_points[triangles], axis=1)
            distances = np.linalg.norm(centers[:, :2], axis=1)
            
            # Keep fewer triangles in the center based on gradient
            threshold = 0.5 / gradient
            mask = distances > threshold
            triangles = triangles[mask]
    else:
        triangles = np.array([])
    
    return all_points, triangles

# Visualize points and triangulation
def visualize_points_matplotlib(points, triangles=None, title="Points", show=True, 
                              save_path=None, feature_points=None):
    """Visualize points using Matplotlib."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
    
    # Plot feature points if provided
    if feature_points is not None and len(feature_points) > 0:
        ax.scatter(feature_points[:, 0], feature_points[:, 1], feature_points[:, 2], 
                   c='r', marker='*', s=100)
    
    # Plot triangulation if provided
    if triangles is not None and len(triangles) > 0:
        for simplex in triangles:
            # Draw triangle edges
            for i in range(3):
                i1, i2 = simplex[i], simplex[(i+1)%3]
                ax.plot([points[i1, 0], points[i2, 0]],
                        [points[i1, 1], points[i2, 1]],
                        [points[i1, 2], points[i2, 2]], 'k-')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Make the plot more viewable
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1, 1])
    
    # Equal aspect ratio for the 3D plot
    ax.set_box_aspect([1, 1, 0.5])
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def visualize_triangulation_pyvista(points, triangles, title="Triangulation", show=True, 
                                   save_path=None, feature_points=None):
    """Visualize triangulation using PyVista if available, otherwise use Matplotlib."""
    if not HAS_PYVISTA:
        visualize_points_matplotlib(points, triangles, title, show, save_path, feature_points)
        return
    
    # Create a PolyData mesh
    if len(triangles) > 0:
        faces = np.column_stack((np.full(len(triangles), 3), triangles))
        faces = faces.flatten()
        mesh = pv.PolyData(points, faces)
        
        # Create a plotter - always use off_screen=True for screenshots
        off_screen = True if save_path else not show
        plotter = pv.Plotter(off_screen=off_screen)
        plotter.add_mesh(mesh, color='lightblue', show_edges=True)
        
        # Add points
        plotter.add_points(points, color='blue', point_size=10)
        
        # Add feature points if provided
        if feature_points is not None and len(feature_points) > 0:
            plotter.add_points(feature_points, color='red', point_size=15)
        
        plotter.add_title(title)
        
        # Determine if we should show or save or both
        if save_path:
            # For screenshots, we need to render the scene first
            plotter.show(auto_close=False)
            plotter.screenshot(save_path, transparent_background=False)
            print(f"Saved figure to {save_path}")
            plotter.close()
        elif show:
            plotter.show()
    else:
        # Fallback to matplotlib if no triangles
        visualize_points_matplotlib(points, triangles, title, show, save_path, feature_points)

def format_seconds(seconds):
    """Format seconds into a readable string."""
    if seconds < 0.1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"

def create_summary_table(results_data, output_file, baseline_gradient=1.0):
    """Create and save a summary table of results."""
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        writer.writerow(['Gradient', 'Triangle Count', 'Processing Time', 'Change from Baseline'])
        
        # Find baseline result
        baseline_count = None
        for result in results_data:
            if result['gradient'] == baseline_gradient:
                baseline_count = result['triangle_count']
                break
        
        # Write data rows
        for result in results_data:
            gradient = result['gradient']
            triangle_count = result['triangle_count']
            proc_time = format_seconds(result['processing_time'])
            
            # Calculate change from baseline
            if baseline_count and triangle_count:
                change_pct = (triangle_count - baseline_count) / baseline_count * 100
                change_str = f"{change_pct:+.1f}%"
            else:
                change_str = "N/A"
            
            writer.writerow([gradient, triangle_count, proc_time, change_str])
    
    print(f"Saved summary table to {output_file}")

def run_tests(args):
    """Run triangulation tests with different gradient values."""
    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup for saving files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load or generate points
    if args.input_file:
        points = load_points(args.input_file)
        base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
    else:
        points = generate_points(args.num_points, args.point_type, args.noise)
        base_filename = f"{args.point_type}_{args.num_points}"
        
        # Save points if requested
        if args.save_points:
            save_points(points, args.save_points)
    
    # Visualize input points
    title = f"Input Points: {len(points)} {args.point_type.capitalize()} Points"
    if args.save_all:
        save_path = output_dir / f"{base_filename}_input_points.png"
    else:
        save_path = None
    
    visualize_points_matplotlib(points, title=title, 
                              show=not args.no_display, 
                              save_path=save_path)
    
    # Define feature points (corners in this case)
    feature_points = None
    if args.point_type == 'grid':
        # Use corners as feature points
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        feature_points = np.array([
            [min_coords[0], min_coords[1], 0],
            [min_coords[0], max_coords[1], 0],
            [max_coords[0], min_coords[1], 0],
            [max_coords[0], max_coords[1], 0],
        ])
    elif args.point_type == 'circle':
        # Use four points around the circle as feature points
        angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        feature_x = np.cos(angles)
        feature_y = np.sin(angles)
        feature_z = np.zeros_like(feature_x)
        feature_points = np.column_stack((feature_x, feature_y, feature_z))
    
    # Run triangulation with different gradients
    results_data = []
    for gradient in args.gradients:
        print(f"\nRunning triangulation with gradient={gradient}")
        
        # Create a GradientControl object
        gc = GradientControl(gradient)
        
        # Time the triangulation
        start_time = time.time()
        
        # Perform triangulation with gradient control
        all_points, triangles = get_triangulation(points, gradient)
        
        # Calculate processing time
        proc_time = time.time() - start_time
        
        # Log metrics
        triangle_count = len(triangles)
        print(f"  Gradient: {gradient}")
        print(f"  Triangles: {triangle_count}")
        print(f"  Processing time: {format_seconds(proc_time)}")
        
        results_data.append({
            'gradient': gradient,
            'triangle_count': triangle_count,
            'processing_time': proc_time,
        })
        
        # Visualize the triangulation
        title = f"Triangulation with Gradient={gradient} ({triangle_count} triangles)"
        if args.save_all:
            save_path = output_dir / f"{base_filename}_gradient_{gradient}.png"
        else:
            save_path = None
        
        visualize_triangulation_pyvista(
            all_points, triangles, 
            title=title, 
            show=not args.no_display,
            save_path=save_path,
            feature_points=feature_points
        )
    
    # Create summary table
    if len(results_data) > 0:
        summary_file = output_dir / f"{base_filename}_summary.csv"
        create_summary_table(results_data, summary_file, baseline_gradient=1.0)
        
        # Also create a detailed log file
        log_file = output_dir / f"{base_filename}_log.txt"
        with open(log_file, 'w') as f:
            f.write(f"Triangulation Test Log - {timestamp}\n")
            f.write(f"Point type: {args.point_type}\n")
            f.write(f"Number of points: {len(points)}\n")
            f.write(f"Noise level: {args.noise}\n\n")
            
            f.write("Feature points:\n")
            if feature_points is not None:
                for i, pt in enumerate(feature_points):
                    f.write(f"  Point {i+1}: ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})\n")
            else:
                f.write("  None defined\n")
            
            f.write("\nTriangulation results:\n")
            for result in results_data:
                f.write(f"\nGradient: {result['gradient']}\n")
                f.write(f"  Triangle count: {result['triangle_count']}\n")
                f.write(f"  Processing time: {format_seconds(result['processing_time'])}\n")
        
        print(f"Saved detailed log to {log_file}")

def main():
    parser = argparse.ArgumentParser(description='Test coarse triangulation with gradient control.')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--input-file', '-i', type=str, help='Input file with point coordinates (x,y,z)')
    input_group.add_argument('--num-points', '-n', type=int, default=25, help='Number of points to generate')
    
    # Point generation options
    parser.add_argument('--point-type', '-t', choices=['grid', 'random', 'circle'], default='grid',
                        help='Type of points to generate')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level for point generation')
    parser.add_argument('--save-points', '-s', type=str, help='Save generated points to file')
    
    # Triangulation options
    parser.add_argument('--gradients', '-g', type=float, nargs='+', default=[0.5, 1.0, 2.0, 3.0],
                        help='Gradient values to test')
    
    # Output options
    parser.add_argument('--output-dir', '-o', type=str, default='output',
                        help='Output directory for saved files')
    parser.add_argument('--save-all', action='store_true', help='Save all figures and data')
    parser.add_argument('--no-display', action='store_true', help='Do not display figures (save only)')
    
    args = parser.parse_args()
    
    # Run the tests
    run_tests(args)

if __name__ == "__main__":
    main() 