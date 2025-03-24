#!/usr/bin/env python
"""
Test script to demonstrate the behavior of the updated Triangle wrapper 
that matches MeshIt's core C++ implementation.

This script generates meshes using the TriangleWrapper with various gradient values
to show how it replicates the behavior of MeshIt's core C++ triunsuitable function.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import argparse
import time

try:
    from meshit.triangle_wrapper import TriangleWrapper
    HAS_TRIANGLE_WRAPPER = True
except ImportError:
    print("Error: Could not import triangle_wrapper. Make sure meshit is installed.")
    HAS_TRIANGLE_WRAPPER = False

def generate_test_points(num_points=100, point_type='grid'):
    """Generate a set of test points based on the specified type"""
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
    
    return points

def generate_feature_points(num_features=3, feature_size=0.1):
    """Generate feature points with specified sizes"""
    # Position features at specific locations for better visualization
    if num_features == 1:
        feature_points = np.array([[0.0, 0.0]])
        feature_sizes = np.array([feature_size])
    elif num_features == 2:
        feature_points = np.array([[-0.5, 0.0], [0.5, 0.0]])
        feature_sizes = np.array([feature_size, feature_size])
    else:
        # Generate in a triangle pattern for 3 or more
        angles = np.linspace(0, 2*np.pi, num_features, endpoint=False)
        radius = 0.5  # Distance from center
        
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        feature_points = np.column_stack((x, y))
        
        # Vary sizes slightly for multiple feature points
        if num_features > 1:
            base_size = feature_size
            size_variation = 0.05
            feature_sizes = np.random.uniform(
                base_size - size_variation, 
                base_size + size_variation, 
                num_features
            )
        else:
            feature_sizes = np.array([feature_size])
            
    return feature_points, feature_sizes

def compute_convex_hull(points):
    """Compute the convex hull of a set of points"""
    hull = ConvexHull(points)
    hull_vertices = hull.vertices
    hull_points = points[hull_vertices]
    segments = np.array([[i, (i + 1) % len(hull_points)] for i in range(len(hull_points))])
    
    return hull_points, segments

def run_triangle_test(gradient_values, point_type='grid', num_points=100, 
                     num_features=3, feature_size=0.1, output_dir='triangle_test'):
    """
    Run triangle mesh generation tests with different gradient values
    
    Args:
        gradient_values: List of gradient values to test
        point_type: Type of points to generate ('grid', 'random', 'circle')
        num_points: Number of points to generate
        num_features: Number of feature points
        feature_size: Size of feature points
        output_dir: Directory to save output files
    """
    if not HAS_TRIANGLE_WRAPPER:
        print("Error: Triangle wrapper is not available.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate test points
    points = generate_test_points(num_points, point_type)
    
    # Generate feature points
    feature_points, feature_sizes = generate_feature_points(num_features, feature_size)
    
    # Compute convex hull and segments
    hull_points, segments = compute_convex_hull(points)
    
    # Calculate bounding box for consistent plotting
    min_x, max_x = min(points[:, 0].min(), feature_points[:, 0].min()), max(points[:, 0].max(), feature_points[:, 0].max())
    min_y, max_y = min(points[:, 1].min(), feature_points[:, 1].min()), max(points[:, 1].max(), feature_points[:, 1].max())
    
    # Ensure square aspect for better visualization
    max_range = max(max_x - min_x, max_y - min_y) * 1.1
    mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    plot_min_x, plot_max_x = mid_x - max_range/2, mid_x + max_range/2
    plot_min_y, plot_max_y = mid_y - max_range/2, mid_y + max_range/2
    
    # Create a combined figure for comparison
    fig, axes = plt.subplots(1, len(gradient_values), figsize=(5*len(gradient_values), 5), squeeze=False)
    axes = axes.flatten()
    
    # Run tests with different gradient values
    results = []
    
    for i, gradient in enumerate(gradient_values):
        print(f"\nTesting gradient = {gradient}")
        
        # Initialize triangle wrapper
        wrapper = TriangleWrapper(gradient=gradient)
        wrapper.set_feature_points(feature_points, feature_sizes)
        
        # Get base_size from bounding box of all points
        all_points = np.vstack([points, feature_points])
        min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
        min_y, max_y = all_points[:, 1].min(), all_points[:, 1].max()
        diagonal = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        wrapper.base_size = diagonal / 15.0
        
        # Triangulate
        start_time = time.time()
        result = wrapper.triangulate(points, segments)
        end_time = time.time()
        
        # Extract results
        triangle_count = len(result['triangles'])
        elapsed_time = end_time - start_time
        
        # Store results
        results.append({
            'gradient': gradient,
            'triangles': triangle_count,
            'time': elapsed_time
        })
        
        # Plot the triangulation
        ax = axes[i]
        ax.triplot(result['vertices'][:, 0], result['vertices'][:, 1], result['triangles'], 'b-', lw=0.5)
        ax.plot(feature_points[:, 0], feature_points[:, 1], 'ro', markersize=8)
        ax.set_title(f"Gradient = {gradient}\n{triangle_count} triangles")
        ax.set_xlim(plot_min_x, plot_max_x)
        ax.set_ylim(plot_min_y, plot_max_y)
        ax.set_aspect('equal')
        
        # Save individual plot
        plt.figure(figsize=(8, 8))
        plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], result['triangles'], 'b-', lw=0.5)
        plt.plot(feature_points[:, 0], feature_points[:, 1], 'ro', markersize=8)
        plt.title(f"MeshIt Triangle (Gradient = {gradient})\n{triangle_count} triangles")
        plt.xlim(plot_min_x, plot_max_x)
        plt.ylim(plot_min_y, plot_max_y)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"triangle_gradient_{gradient}.png"))
        plt.close()
    
    # Save combined plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "triangle_comparison.png"))
    plt.close()
    
    # Print results table
    print("\nResults Summary:")
    print("=" * 40)
    print(f"{'Gradient':<10} | {'Triangles':<10} | {'Time (s)':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r['gradient']:<10.1f} | {r['triangles']:<10d} | {r['time']:<10.2f}")
    print("=" * 40)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MeshIt's triangle wrapper implementation")
    parser.add_argument("--gradients", type=str, default="1.0,2.0,3.0",
                       help="Comma-separated list of gradient values to test")
    parser.add_argument("--point-type", type=str, default="grid", choices=["grid", "random", "circle"],
                       help="Type of points to generate")
    parser.add_argument("--num-points", type=int, default=100,
                       help="Number of points to generate")
    parser.add_argument("--feature-points", type=int, default=3,
                       help="Number of feature points")
    parser.add_argument("--feature-size", type=float, default=0.1,
                       help="Size parameter for feature points")
    parser.add_argument("--output-dir", type=str, default="triangle_test",
                       help="Directory to save output files")
    
    args = parser.parse_args()
    
    # Parse gradient values
    gradient_values = [float(g) for g in args.gradients.split(",")]
    
    # Run test
    run_triangle_test(
        gradient_values=gradient_values,
        point_type=args.point_type,
        num_points=args.num_points,
        num_features=args.feature_points,
        feature_size=args.feature_size,
        output_dir=args.output_dir
    ) 