#!/usr/bin/env python
"""
Test script to demonstrate MeshIt's custom triangle refinement implementation.
This script shows the difference between using the standard Triangle library
and our custom refinement that mimics MeshIt's C++ triunsuitable function.

Usage:
    python test_custom_refinement.py
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Try to import MeshIt modules
try:
    import meshit
    from meshit.core import Surface, Vector3D, GradientControl
    from meshit import extensions
    try:
        from meshit.triangle_wrapper import TriangleWrapper
        HAS_CUSTOM_WRAPPER = True
    except ImportError:
        HAS_CUSTOM_WRAPPER = False
        print("Custom triangle wrapper not available - run will show standard approach")
        
    print("Successfully imported MeshIt modules.")
    HAS_MESHIT = True
except ImportError as e:
    print(f"Error importing MeshIt modules: {e}")
    print("Falling back to basic triangulation without MeshIt.")
    HAS_MESHIT = False
    HAS_CUSTOM_WRAPPER = False

def generate_points(num_points=100, point_type='grid', feature_points=3):
    """Generate test points with feature points"""
    # Create a grid of points
    if point_type == 'grid':
        n = int(np.sqrt(num_points))
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack((X.ravel(), Y.ravel()))
    elif point_type == 'random':
        # Random points in a square
        points = np.random.uniform(-1, 1, (num_points, 2))
    else:  # 'circle'
        # Points in a circle
        theta = np.linspace(0, 2*np.pi, num_points)
        r = np.random.uniform(0.7, 1.0, num_points)
        points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    
    # Generate feature points (smaller randomly placed points)
    feature_pos = np.random.uniform(-0.6, 0.6, (feature_points, 2))
    feature_sizes = np.random.uniform(0.05, 0.2, feature_points)
    
    return points, feature_pos, feature_sizes

def compute_convex_hull(points):
    """Compute convex hull of points"""
    hull = ConvexHull(points)
    return points[hull.vertices]

def triangulate_standard(points, hull_points, gradient, feature_points, feature_sizes):
    """Triangulate using the standard Triangle library"""
    import triangle
    
    # Calculate parameters based on bounding box and gradient
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0]) 
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    diag = np.sqrt(bbox_width**2 + bbox_height**2)
    base_size = diag / 15.0
    
    # Adjust min angle based on gradient
    min_angle = 20.0
    if gradient > 1.0:
        min_angle = 20.0 - (gradient - 1.0) * 5.0
        min_angle = max(min_angle, 10.0)
    
    # Adjust area constraint based on gradient
    hull_area = polygon_area(hull_points)
    area_constraint = hull_area / 100
    if gradient < 1.0:
        area_constraint *= (1.0 - 0.5 * (1.0 - gradient))
    elif gradient > 1.0:
        area_constraint *= (1.0 + 0.5 * (gradient - 1.0))
    
    # Set up triangle options
    triangle_opts = f'q{min_angle}a{area_constraint}'
    
    # Create segments from hull
    segments = np.array([[i, (i + 1) % len(hull_points)] for i in range(len(hull_points))])
    
    # Run triangulation
    start_time = time.time()
    result = triangle.triangulate({
        'vertices': points,
        'segments': segments
    }, triangle_opts)
    end_time = time.time()
    
    return result, end_time - start_time

def triangulate_custom(points, hull_points, gradient, feature_points, feature_sizes):
    """Triangulate using our custom triangle wrapper with refinement"""
    if not HAS_CUSTOM_WRAPPER:
        print("Custom wrapper not available, using standard triangulation")
        return triangulate_standard(points, hull_points, gradient, feature_points, feature_sizes)
    
    # Calculate parameters based on bounding box and gradient
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0]) 
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    diag = np.sqrt(bbox_width**2 + bbox_height**2)
    base_size = diag / 15.0
    
    # Adjust min angle based on gradient
    min_angle = 20.0
    if gradient > 1.0:
        min_angle = 20.0 - (gradient - 1.0) * 5.0
        min_angle = max(min_angle, 10.0)
    
    # Adjust area constraint based on gradient
    hull_area = polygon_area(hull_points)
    area_constraint = hull_area / 100
    if gradient < 1.0:
        area_constraint *= (1.0 - 0.5 * (1.0 - gradient))
    elif gradient > 1.0:
        area_constraint *= (1.0 + 0.5 * (gradient - 1.0))
    
    # Create segments from hull
    segments = np.array([[i, (i + 1) % len(hull_points)] for i in range(len(hull_points))])
    
    # Create and configure the triangle wrapper
    wrapper = TriangleWrapper(gradient=gradient, min_angle=min_angle, max_area=area_constraint)
    wrapper.base_size = base_size
    wrapper.set_feature_points(feature_points, feature_sizes)
    
    # Run triangulation with custom refinement
    start_time = time.time()
    result = wrapper.triangulate(points, segments)
    end_time = time.time()
    
    return result, end_time - start_time

def triangulate_meshit(points, hull_points, gradient, feature_points, feature_sizes):
    """Triangulate using MeshIt's extensions.triangulate_with_triangle"""
    if not HAS_MESHIT:
        print("MeshIt not available")
        return None, 0
    
    # Convert to 3D points
    points_3d = np.column_stack((points, np.zeros(len(points))))
    feature_points_3d = np.column_stack((feature_points, np.zeros(len(feature_points))))
    
    # Create Surface object
    surface = Surface()
    for point in points_3d:
        v = Vector3D(float(point[0]), float(point[1]), float(point[2]))
        surface.add_vertex(v)
    
    # Run triangulation with MeshIt
    start_time = time.time()
    try:
        result = extensions.triangulate_with_triangle(surface, gradient=gradient)
        vertices, triangles = result
        
        # Convert result to the same format as other methods
        tri_result = {
            'vertices': vertices[:, :2],
            'triangles': triangles
        }
        success = True
    except Exception as e:
        print(f"Error in MeshIt triangulation: {e}")
        tri_result = None
        success = False
    
    end_time = time.time()
    
    return tri_result, end_time - start_time

def polygon_area(points):
    """Calculate the area of a polygon"""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def plot_triangulation(points, hull_points, feature_points, feature_sizes, result, title, filename, show_centroid_circles=False):
    """Plot triangulation results"""
    plt.figure(figsize=(10, 10))
    
    # Plot original points
    plt.plot(points[:, 0], points[:, 1], 'k.', markersize=2, alpha=0.3, label='Points')
    
    # Plot convex hull
    hull_closed = np.vstack((hull_points, hull_points[0]))
    plt.plot(hull_closed[:, 0], hull_closed[:, 1], 'g-', linewidth=2, label='Convex Hull')
    
    # Plot feature points
    plt.plot(feature_points[:, 0], feature_points[:, 1], 'ro', markersize=6, label='Feature Points')
    
    # Add circles around feature points to illustrate size
    for i, fp in enumerate(feature_points):
        size = feature_sizes[i]
        circle = plt.Circle((fp[0], fp[1]), size, color='r', fill=False, alpha=0.3, linestyle='--')
        plt.gca().add_artist(circle)
        
        if show_centroid_circles:
            # Also add a larger circle showing gradient effect
            grad_circle = plt.Circle((fp[0], fp[1]), size * 3, color='r', fill=False, alpha=0.1, linestyle=':')
            plt.gca().add_artist(grad_circle)
    
    # Plot triangulation
    if result is not None:
        plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], result['triangles'], 'b-', alpha=0.5)
        plt.title(f"{title}\n{len(result['triangles'])} triangles")
    else:
        plt.title(f"{title}\nTriangulation failed")
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    # Save figure
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def run_comparison_test(gradient_values=[0.5, 1.0, 2.0], point_type='grid', output_dir='triangulation_comparison'):
    """Run comparison tests between standard, custom, and MeshIt triangulation"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate points
    num_points = 100
    num_features = 3
    points, feature_points, feature_sizes = generate_points(num_points, point_type, num_features)
    
    # Compute convex hull
    hull_points = compute_convex_hull(points)
    
    # Run comparison tests for each gradient value
    for gradient in gradient_values:
        print(f"\nTesting with gradient {gradient}")
        
        # Standard triangulation
        print("Running standard triangulation...")
        std_result, std_time = triangulate_standard(points, hull_points, gradient, feature_points, feature_sizes)
        print(f"Standard triangulation: {len(std_result['triangles'])} triangles in {std_time:.4f} seconds")
        
        # Custom triangulation
        print("Running custom triangulation...")
        custom_result, custom_time = triangulate_custom(points, hull_points, gradient, feature_points, feature_sizes)
        print(f"Custom triangulation: {len(custom_result['triangles'])} triangles in {custom_time:.4f} seconds")
        
        # Compare by number of triangles, not by array equality
        std_tri_count = len(std_result['triangles'])
        custom_tri_count = len(custom_result['triangles'])
        if custom_tri_count != std_tri_count:
            # Calculate difference percentage
            diff_pct = ((custom_tri_count - std_tri_count) / std_tri_count) * 100
            print(f"  Difference: {diff_pct:+.2f}% triangles")
        else:
            print("Custom triangulation produced same number of triangles as standard")
        
        # MeshIt triangulation
        print("Running MeshIt triangulation...")
        meshit_result, meshit_time = triangulate_meshit(points, hull_points, gradient, feature_points, feature_sizes)
        if meshit_result is not None:
            print(f"MeshIt triangulation: {len(meshit_result['triangles'])} triangles in {meshit_time:.4f} seconds")
        else:
            print("MeshIt triangulation failed or not available")
        
        # Plot results
        # Standard
        plot_triangulation(
            points, hull_points, feature_points, feature_sizes, 
            std_result, 
            f"Standard Triangulation (Gradient {gradient})",
            os.path.join(output_dir, f"standard_gradient_{gradient}.png")
        )
        
        # Custom
        if len(custom_result['triangles']) != len(std_result['triangles']):
            plot_triangulation(
                points, hull_points, feature_points, feature_sizes, 
                custom_result, 
                f"Custom Refinement (Gradient {gradient})",
                os.path.join(output_dir, f"custom_gradient_{gradient}.png"),
                show_centroid_circles=True
            )
        
        # MeshIt
        if meshit_result is not None:
            plot_triangulation(
                points, hull_points, feature_points, feature_sizes, 
                meshit_result, 
                f"MeshIt Triangulation (Gradient {gradient})",
                os.path.join(output_dir, f"meshit_gradient_{gradient}.png")
            )
    
    # Write comparison summary
    with open(os.path.join(output_dir, "comparison_summary.txt"), "w") as f:
        f.write("Triangulation Comparison Summary\n")
        f.write("===============================\n\n")
        f.write(f"Number of points: {num_points}\n")
        f.write(f"Number of feature points: {num_features}\n")
        f.write(f"Point type: {point_type}\n\n")
        
        f.write("Results by gradient value:\n")
        for gradient in gradient_values:
            f.write(f"\nGradient = {gradient}\n")
            f.write("-" * 50 + "\n")
            
            # Standard
            std_result, std_time = triangulate_standard(points, hull_points, gradient, feature_points, feature_sizes)
            f.write(f"Standard: {len(std_result['triangles'])} triangles in {std_time:.4f}s\n")
            
            # Custom
            custom_result, custom_time = triangulate_custom(points, hull_points, gradient, feature_points, feature_sizes)
            f.write(f"Custom:   {len(custom_result['triangles'])} triangles in {custom_time:.4f}s\n")
            
            std_tri_count = len(std_result['triangles'])
            custom_tri_count = len(custom_result['triangles'])
            if custom_tri_count != std_tri_count:
                diff_pct = ((custom_tri_count - std_tri_count) / std_tri_count) * 100
                f.write(f"          Difference: {diff_pct:+.2f}% triangles\n")
            else:
                f.write("          (Same number of triangles as standard)\n")
            
            # MeshIt
            meshit_result, meshit_time = triangulate_meshit(points, hull_points, gradient, feature_points, feature_sizes)
            if meshit_result is not None:
                f.write(f"MeshIt:   {len(meshit_result['triangles'])} triangles in {meshit_time:.4f}s\n")
                
                std_tri_count = len(std_result['triangles'])
                meshit_tri_count = len(meshit_result['triangles'])
                diff_pct = ((meshit_tri_count - std_tri_count) / std_tri_count) * 100
                f.write(f"          Difference from standard: {diff_pct:+.2f}% triangles\n")
            else:
                f.write("MeshIt:   Not available or failed\n")
    
    print(f"\nComparison test complete. Results saved to '{output_dir}'")

def main():
    parser = argparse.ArgumentParser(description='Test custom triangle refinement in MeshIt')
    
    parser.add_argument('--point-type', choices=['grid', 'random', 'circle'], default='grid',
                        help='Type of points (default: grid)')
    parser.add_argument('--gradient-values', type=str, default='0.5,1.0,2.0',
                        help='Comma-separated list of gradient values to test')
    parser.add_argument('--output-dir', type=str, default='triangulation_comparison',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Convert gradient values to list of floats
    gradient_values = [float(val) for val in args.gradient_values.split(',')]
    
    # Run comparison test
    run_comparison_test(gradient_values, args.point_type, args.output_dir)

if __name__ == '__main__':
    main() 