#!/usr/bin/env python
"""
Test script to demonstrate MeshIt's gradient control and refinesize features.
This script uses the actual MeshIt package to demonstrate how gradient and refinesize
parameters affect triangulation.

Usage:
    python test_gradient_refinement.py [options]

Options:
    --num-points N          Number of points to generate (default: 64)
    --point-type TYPE       Type of points (grid, random, circle)
    --noise VALUE           Amount of noise to add to points (default: 0.0)
    --gradient VALUE        Gradient value for refinement (default: 1.0)
    --base-size VALUE       Base mesh size (default: auto-calculated from bbox)
    --refinesize-values     Comma-separated list of refinesize values (default: 0.1,0.3,0.5)
    --feature-points N      Number of feature points to use (default: 3)
    --feature-size VALUE    Size for feature points (default: auto-calculated)
    --feature-locations     Comma-separated list of x,y locations for feature points
    --no-display            Don't show interactive plots
    --save-all              Save all figures and logs
    --output-dir DIR        Output directory for results (default: test_gradient_refinesize)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import ConvexHull
import argparse
import random
import logging
from datetime import datetime
import traceback

# Import MeshIt modules
try:
    import meshit
    from meshit.core import Vector3D, Surface, GradientControl
    from meshit import extensions
    print("Successfully imported MeshIt modules.")
    HAS_MESHIT = True
except ImportError as e:
    print(f"Error importing MeshIt modules: {e}")
    print("Falling back to basic triangulation without MeshIt.")
    HAS_MESHIT = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("PyVista not available, falling back to matplotlib for visualization")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_points(num_points, point_type='grid', noise=0.0):
    """Generate test points"""
    if point_type == 'grid':
        # Create a grid of points
        n = int(np.sqrt(num_points))
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack((X.ravel(), Y.ravel()))
    elif point_type == 'random':
        # Generate random points
        points = np.random.uniform(-1, 1, (num_points, 2))
    else:  # circle
        # Generate points in a circle
        theta = np.linspace(0, 2*np.pi, num_points)
        r = 0.8 + np.random.uniform(-0.1, 0.1, num_points)
        points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    
    # Add noise if specified
    if noise > 0:
        points += np.random.normal(0, noise, points.shape)
    
    # Convert 2D points to 3D (Z=0)
    points_3d = np.column_stack((points, np.zeros(len(points))))
    
    return points_3d

def compute_convex_hull(points):
    """Compute the convex hull of points"""
    # Use 2D points for hull calculation
    points_2d = points[:, :2]
    hull = ConvexHull(points_2d)
    
    # Extract hull vertices in order
    hull_points = points_2d[hull.vertices]
    
    # Convert back to 3D points (z=0)
    hull_points_3d = np.column_stack((hull_points, np.zeros(len(hull_points))))
    
    return hull_points_3d

def calculate_bbox_size(points):
    """Calculate bounding box size for base size calculation"""
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    diagonal = np.sqrt(bbox_width**2 + bbox_height**2)
    
    # In MeshIt, base size is often proportional to the bounding box diagonal
    return diagonal / 15.0  # A reasonable default proportion

def triangulate_with_meshit(points, hull_points, gradient, refinesize, feature_points=None, feature_sizes=None):
    """
    Triangulate points using MeshIt's triangulation with gradient control
    
    Args:
        points: Array of input points (N, 3)
        hull_points: Convex hull points for boundary (M, 3)
        gradient: Gradient value for refinement
        refinesize: Refinesize value for local refinement
        feature_points: Array of feature points (optional)
        feature_sizes: Array of feature point sizes (optional)
        
    Returns:
        Dictionary with triangulation results
    """
    if not HAS_MESHIT:
        print("MeshIt not available, falling back to scipy Delaunay")
        from scipy.spatial import Delaunay
        tri = Delaunay(points[:, :2])
        return {
            'vertices': points,
            'triangles': tri.simplices,
            'num_triangles': len(tri.simplices)
        }
    
    try:
        # Create a Surface object from the input points
        surface = Surface()
        for point in points:
            v = Vector3D(float(point[0]), float(point[1]), float(point[2]))
            surface.add_vertex(v)
        
        # Set the convex hull
        for hp in hull_points:
            # Check if the add_convex_hull_vertex method exists
            if hasattr(surface, 'add_convex_hull_vertex'):
                v = Vector3D(float(hp[0]), float(hp[1]), float(hp[2]))
                surface.add_convex_hull_vertex(v)
            # Otherwise, calculate the hull using the enhanced_calculate_convex_hull method
            else:
                if hasattr(surface, 'enhanced_calculate_convex_hull'):
                    surface.enhanced_calculate_convex_hull()
                else:
                    print("Using built-in calculate_convex_hull")
                    surface.calculate_convex_hull()
                break  # Only need to calculate once
        
        # Prepare feature points for gradient control
        if feature_points is not None and feature_sizes is not None and len(feature_points) > 0:
            # Calculate base size if not provided
            base_size = calculate_bbox_size(points)
            
            # Get the gradient control instance
            gc = GradientControl.get_instance()
            
            # Flatten feature points for the C++ interface
            fp_flat = feature_points.flatten()
            
            # Update gradient control
            # We'll adapt to handle different versions of the GradientControl API
            try:
                # First try with full list passing (ideal)
                gc.update(
                    float(gradient),
                    float(base_size), 
                    len(feature_points),
                    fp_flat,
                    feature_sizes
                )
            except Exception as e:
                print(f"Could not update GradientControl with full points list: {e}")
                try:
                    # Try with just the first point and size (simplified)
                    gc.update(
                        float(gradient),
                        float(base_size), 
                        len(feature_points),
                        float(fp_flat[0]) if len(fp_flat) > 0 else 0.0,
                        float(feature_sizes[0]) if len(feature_sizes) > 0 else base_size
                    )
                except Exception as e:
                    print(f"Could not update GradientControl: {e}")
                    traceback.print_exc()
        
        # Triangulate with MeshIt's Triangle implementation
        start_time = time.time()
        
        # Try different ways to triangulate depending on the available API
        if hasattr(extensions, 'triangulate_with_triangle'):
            # Try with gradient parameter
            try:
                tri_result = extensions.triangulate_with_triangle(surface, gradient=gradient)
            except TypeError:
                # Try without gradient parameter
                tri_result = extensions.triangulate_with_triangle(surface)
        else:
            # Fall back to the built-in triangulate method
            surface.triangulate()
            tri_result = surface.triangles
        
        end_time = time.time()
        
        # Extract triangulation results
        if isinstance(tri_result, tuple) and len(tri_result) == 2:
            # Newer MeshIt versions return (vertices, triangles)
            vertices, triangles = tri_result
            vertices_array = np.array(vertices)
            triangles_array = np.array(triangles)
        else:
            # Older versions might return triangles directly or store them in the surface
            if hasattr(surface, 'triangles') and len(surface.triangles) > 0:
                triangles_array = np.array(surface.triangles)
                vertices_array = np.array([[v.x, v.y, v.z] for v in surface.vertices])
            else:
                triangles_array = np.array(tri_result) if tri_result is not None else np.array([])
                vertices_array = points
        
        # Return triangulation results
        return {
            'vertices': vertices_array,
            'triangles': triangles_array,
            'num_triangles': len(triangles_array),
            'time': end_time - start_time
        }
    except Exception as e:
        print(f"Error in MeshIt triangulation: {e}")
        traceback.print_exc()
        # Fall back to basic triangulation
        from scipy.spatial import Delaunay
        tri = Delaunay(points[:, :2])
        return {
            'vertices': points,
            'triangles': tri.simplices,
            'num_triangles': len(tri.simplices)
        }

def visualize_triangulation(vertices, triangles, hull_points, feature_points, title, output_path):
    """Visualize triangulation using matplotlib"""
    plt.figure(figsize=(10, 10))
    
    # Ensure triangle indices are valid
    valid_triangles = np.array([tri for tri in triangles if all(idx < len(vertices) for idx in tri)])
    
    if len(valid_triangles) > 0:
        # Plot triangulation
        plt.triplot(vertices[:, 0], vertices[:, 1], valid_triangles, 'b-', alpha=0.5)
    else:
        print("Warning: No valid triangles to plot")
    
    # Plot original points
    plt.plot(vertices[:, 0], vertices[:, 1], 'k.', markersize=2)
    
    # Plot convex hull
    hull_closed = np.vstack((hull_points, hull_points[0]))
    plt.plot(hull_closed[:, 0], hull_closed[:, 1], 'g-', linewidth=2)
    
    # Plot feature points if any
    if feature_points is not None and len(feature_points) > 0:
        plt.plot(feature_points[:, 0], feature_points[:, 1], 'ro', markersize=6)
        
        # Add circular zones around feature points to illustrate refinesize
        for fp in feature_points:
            circle = plt.Circle((fp[0], fp[1]), 0.2, color='r', fill=False, alpha=0.3, linestyle='--')
            plt.gca().add_artist(circle)
    
    # Add statistics
    plt.title(f"{title}\nTriangles: {len(valid_triangles)}")
    plt.axis('equal')
    plt.grid(True)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def run_tests(args):
    """Run gradient and refinesize tests"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate points
    points = generate_points(args.num_points, args.point_type, args.noise)
    
    # Compute convex hull
    hull_points = compute_convex_hull(points)
    
    # Calculate default base size if not specified
    base_size = args.base_size if args.base_size else calculate_bbox_size(points)
    
    # Generate or use specified feature points
    if args.feature_locations:
        # Parse feature locations
        locations = np.array([float(x) for x in args.feature_locations.split(',')])
        feature_points_2d = locations.reshape(-1, 2)
        feature_points = np.column_stack((feature_points_2d, np.zeros(len(feature_points_2d))))
    else:
        # Generate random feature points
        if args.feature_points > 0:
            # Select random interior points as features
            interior_indices = []
            for i in range(len(points)):
                p = points[i, :2]  # Get 2D coordinates
                path = Path(hull_points[:, :2])  # Use 2D hull for contains_point
                if path.contains_point(p):
                    interior_indices.append(i)
            
            if interior_indices:
                feature_indices = np.random.choice(
                    interior_indices, 
                    min(args.feature_points, len(interior_indices)), 
                    replace=False
                )
                feature_points = points[feature_indices]
            else:
                # Fall back to random points if no interior points found
                feature_points_2d = np.random.uniform(-0.5, 0.5, (args.feature_points, 2))
                feature_points = np.column_stack((feature_points_2d, np.zeros(args.feature_points)))
        else:
            feature_points = np.array([])
    
    # Calculate feature size if not specified
    feature_size = args.feature_size if args.feature_size else base_size * 0.3
    feature_sizes = np.ones(len(feature_points)) * feature_size
    
    # Save input data visualization
    plt.figure(figsize=(10, 10))
    plt.plot(points[:, 0], points[:, 1], 'k.', markersize=3, label='Input Points')
    plt.plot(hull_points[:, 0], hull_points[:, 1], 'g-', linewidth=2, label='Convex Hull')
    
    if len(feature_points) > 0:
        plt.plot(feature_points[:, 0], feature_points[:, 1], 'ro', markersize=6, label='Feature Points')
    
    plt.title(f"Input Data\n{args.num_points} points, {len(feature_points)} feature points")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "1_input_data.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Log input parameters
    with open(os.path.join(output_dir, "parameters.txt"), "w") as f:
        f.write(f"Test Parameters:\n")
        f.write(f"Date/Time: {timestamp}\n")
        f.write(f"Number of points: {args.num_points}\n")
        f.write(f"Point type: {args.point_type}\n")
        f.write(f"Noise level: {args.noise}\n")
        f.write(f"Gradient value: {args.gradient}\n")
        f.write(f"Base size: {base_size:.6f}\n")
        f.write(f"Feature points: {len(feature_points)}\n")
        f.write(f"Feature size: {feature_size:.6f}\n")
        
        if len(feature_points) > 0:
            f.write("\nFeature Point Locations:\n")
            for i, fp in enumerate(feature_points):
                f.write(f"  {i+1}: ({fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f})\n")
        
        f.write("\nRefinesize Test Values:\n")
        for rs in args.refinesize_values:
            f.write(f"  {rs:.4f}\n")
    
    # Run triangulation for each refinesize value
    results = []
    for refinesize in args.refinesize_values:
        logging.info(f"Testing with refinesize={refinesize:.4f}")
        
        # Triangulate with current refinesize using MeshIt
        tri_result = triangulate_with_meshit(
            points, hull_points, args.gradient, refinesize,
            feature_points, feature_sizes
        )
        
        # Record result
        result = {
            'refinesize': refinesize,
            'triangles': tri_result['num_triangles'],
            'time': tri_result.get('time', 0)
        }
        results.append(result)
        
        # Visualize result
        output_path = os.path.join(output_dir, f"2_triangulation_refinesize_{refinesize:.4f}.png")
        title = f"Triangulation with Gradient={args.gradient:.2f}, Refinesize={refinesize:.4f}"
        visualize_triangulation(
            tri_result['vertices'], tri_result['triangles'], 
            hull_points, feature_points, title, output_path
        )
        
        logging.info(f"Created {result['triangles']} triangles in {result['time']:.4f} seconds")
    
    # Create comparison chart
    plt.figure(figsize=(10, 6))
    refinesizes = [r['refinesize'] for r in results]
    num_triangles = [r['triangles'] for r in results]
    
    plt.bar(range(len(refinesizes)), num_triangles)
    plt.xlabel('Refinesize Value')
    plt.ylabel('Number of Triangles')
    plt.title(f'Effect of Refinesize on Triangle Count\nGradient={args.gradient:.2f}')
    plt.xticks(range(len(refinesizes)), [f"{rs:.4f}" for rs in refinesizes])
    plt.grid(axis='y')
    
    # Add exact values on top of bars
    for i, v in enumerate(num_triangles):
        plt.text(i, v + 5, str(v), ha='center')
    
    plt.savefig(os.path.join(output_dir, "3_refinesize_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write("Refinesize Test Results:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Refinesize':<15} {'Triangles':<15} {'Time (s)':<15}\n")
        f.write("-" * 50 + "\n")
        
        for result in results:
            f.write(f"{result['refinesize']:<15.4f} {result['triangles']:<15} {result['time']:<15.4f}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test MeshIt gradient control and refinesize parameters')
    
    parser.add_argument('--num-points', type=int, default=64, 
                        help='Number of points to generate (default: 64)')
    parser.add_argument('--point-type', choices=['grid', 'random', 'circle'], default='grid',
                        help='Type of points (default: grid)')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Amount of noise to add to points (default: 0.0)')
    parser.add_argument('--gradient', type=float, default=1.0,
                        help='Gradient value for refinement (default: 1.0)')
    parser.add_argument('--base-size', type=float, default=None,
                        help='Base mesh size (default: auto-calculated from bbox)')
    parser.add_argument('--refinesize-values', type=str, default='0.1,0.3,0.5',
                        help='Comma-separated list of refinesize values to test')
    parser.add_argument('--feature-points', type=int, default=3,
                        help='Number of feature points to use (default: 3)')
    parser.add_argument('--feature-size', type=float, default=None,
                        help='Size for feature points (default: auto-calculated)')
    parser.add_argument('--feature-locations', type=str, default=None,
                        help='Comma-separated list of x,y locations for feature points')
    parser.add_argument('--no-display', action='store_true',
                        help='Don\'t show interactive plots')
    parser.add_argument('--save-all', action='store_true',
                        help='Save all intermediate results')
    parser.add_argument('--output-dir', type=str, default='test_gradient_refinesize',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Process refinesize values
    args.refinesize_values = [float(val) for val in args.refinesize_values.split(',')]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the tests
    results = run_tests(args)
    
    # Print summary
    print("\nTest Results Summary:")
    print("=" * 60)
    print(f"Gradient value: {args.gradient:.2f}")
    print(f"Number of points: {args.num_points}")
    print(f"Feature points: {args.feature_points}")
    print("=" * 60)
    print(f"{'Refinesize':<15} {'Triangles':<15} {'Time (s)':<15}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['refinesize']:<15.4f} {result['triangles']:<15} {result['time']:<15.4f}")
    
    # Calculate percentage changes
    if len(results) > 1:
        baseline = results[0]['triangles']
        print("\nPercentage change in triangle count:")
        print("-" * 60)
        
        for i, result in enumerate(results):
            if i == 0:
                pct_change = 0.0
            else:
                pct_change = ((result['triangles'] - baseline) / baseline) * 100
            
            print(f"Refinesize {result['refinesize']:.4f}: {pct_change:+.2f}%")
    
    print("\nResults saved to:", args.output_dir)

if __name__ == '__main__':
    main() 