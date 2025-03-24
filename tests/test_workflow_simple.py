#!/usr/bin/env python
"""
Simple test script demonstrating the core mesh generation workflow:
1. Generate points
2. Compute convex hull
3. Perform coarse segmentation
4. Apply coarse triangulation with gradient control

This script visualizes each step and shows how refinesize parameters
affect the triangulation process.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import argparse
from matplotlib.tri import Triangulation
import os

# Simple visualization helpers
def plot_points(points, title="Points", show=True, save_path=None):
    """Plot points in 2D"""
    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=50)
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_hull(points, hull_indices, title="Convex Hull", show=True, save_path=None):
    """Plot points and convex hull"""
    plt.figure(figsize=(10, 8))
    
    # Plot all points
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=50, label='Points')
    
    # Plot convex hull points
    hull_points = points[hull_indices]
    plt.scatter(hull_points[:, 0], hull_points[:, 1], 
                c='red', s=80, label='Hull Points')
    
    # Plot hull edges by connecting adjacent hull points
    hull_points_ordered = np.append(hull_points, [hull_points[0]], axis=0)
    plt.plot(hull_points_ordered[:, 0], hull_points_ordered[:, 1], 
             'r-', linewidth=2, label='Hull Edges')
    
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_triangulation(points, triangles, title="Triangulation", show=True, save_path=None, 
                     feature_points=None):
    """Plot triangulation"""
    plt.figure(figsize=(10, 8))
    
    # Plot all points
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=50, zorder=2, label='Points')
    
    # Plot triangulation
    for triangle in triangles:
        x = [points[triangle[0], 0], points[triangle[1], 0], 
             points[triangle[2], 0], points[triangle[0], 0]]
        y = [points[triangle[0], 1], points[triangle[1], 1], 
             points[triangle[2], 1], points[triangle[0], 1]]
        plt.plot(x, y, 'k-', linewidth=0.5, zorder=1)
    
    # Plot feature points if provided
    if feature_points is not None:
        plt.scatter(feature_points[:, 0], feature_points[:, 1], 
                    c='red', s=100, marker='*', zorder=3, label='Feature Points')
    
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

# Point generation
def generate_points(num_points=25, point_type='grid', noise=0.0):
    """Generate test points in 2D"""
    if point_type == 'grid':
        # Calculate grid size
        grid_size = int(np.sqrt(num_points))
        x = np.linspace(0, 10, grid_size)
        y = np.linspace(0, 10, grid_size)
        xx, yy = np.meshgrid(x, y)
        xx = xx.flatten()[:num_points]
        yy = yy.flatten()[:num_points]
        
        # Add noise if requested
        if noise > 0:
            xx += np.random.normal(0, noise, xx.shape)
            yy += np.random.normal(0, noise, yy.shape)
    
    elif point_type == 'random':
        xx = np.random.uniform(0, 10, num_points)
        yy = np.random.uniform(0, 10, num_points)
    
    elif point_type == 'circle':
        theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        radius = 5.0
        center_x, center_y = 5.0, 5.0
        
        xx = center_x + radius * np.cos(theta)
        yy = center_y + radius * np.sin(theta)
        
        # Add noise if requested
        if noise > 0:
            xx += np.random.normal(0, noise, xx.shape)
            yy += np.random.normal(0, noise, yy.shape)
    
    else:
        raise ValueError(f"Unknown point type: {point_type}")
    
    return np.column_stack((xx, yy))

# Step 1: Compute convex hull
def compute_convex_hull(points):
    """Compute the convex hull of a set of points"""
    hull = ConvexHull(points)
    return hull.vertices  # Indices of the hull points

# Step 2: Coarse segmentation 
def coarse_segmentation(hull_indices, points):
    """
    Create segments based on the convex hull.
    This is a simplified version of coarse segmentation.
    """
    segments = []
    for i in range(len(hull_indices)):
        start_idx = hull_indices[i]
        end_idx = hull_indices[(i + 1) % len(hull_indices)]
        segments.append((start_idx, end_idx))
    
    return segments

# Step 3: Triangulation with gradient control
def triangulate_with_gradient(points, segments, gradient=1.0, feature_points=None):
    """
    Triangulate the points with gradient control.
    
    Args:
        points: 2D points array
        segments: List of edge segments [(start_idx, end_idx), ...]
        gradient: Gradient control value
        feature_points: Points that should have finer triangulation around them
        
    Returns:
        Triangles as indices into the points array
    """
    # For this simplified example, we'll use scipy's Delaunay triangulation
    # and then modify the triangulation based on the gradient
    
    # First we need to calculate the base size for triangulation
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    domain_size = max(x_max - x_min, y_max - y_min)
    base_size = domain_size / 10.0  # A reasonable default
    
    # Create a triangulation using all points
    tri = Delaunay(points)
    triangles = tri.simplices
    
    # If we have feature points, we'll simulate refinesize by filtering
    # triangles based on their distance to feature points and the gradient
    if feature_points is not None and len(feature_points) > 0:
        # For each triangle, calculate its centroid
        centroids = np.mean(points[triangles], axis=1)
        
        # Remove triangles based on gradient and distance to feature points
        keep_triangles = []
        
        for i, triangle in enumerate(triangles):
            # Calculate centroid
            centroid = centroids[i]
            
            # Find distances to all feature points
            min_dist = float('inf')
            for fp in feature_points:
                dist = np.sqrt((centroid[0] - fp[0])**2 + (centroid[1] - fp[1])**2)
                min_dist = min(min_dist, dist)
            
            # Apply gradient-based threshold
            # Lower gradient means more uniform triangulation (keep more triangles)
            # Higher gradient means more variation (remove triangles farther from features)
            threshold = base_size * gradient
            
            if min_dist < threshold:
                keep_triangles.append(triangle)
        
        if not keep_triangles:  # If all triangles were removed, keep the original
            keep_triangles = triangles
        else:
            triangles = np.array(keep_triangles)
    
    return triangles

def identify_feature_points(points, hull_indices):
    """
    Identify feature points for refinesize parameters.
    In this simple example, we'll use hull points as features.
    """
    hull_points = points[hull_indices]
    return hull_points

def run_workflow(args):
    """Run the full workflow with the specified parameters"""
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate or load points
    if args.input_file:
        try:
            points = np.loadtxt(args.input_file)
            print(f"Loaded {len(points)} points from {args.input_file}")
        except Exception as e:
            print(f"Error loading points: {e}")
            return
    else:
        points = generate_points(args.num_points, args.point_type, args.noise)
        print(f"Generated {len(points)} {args.point_type} points")
    
    # Visualize input points
    plot_points(
        points,
        title=f"Input Points ({len(points)} {args.point_type} points)",
        show=not args.no_display,
        save_path=os.path.join(output_dir, "1_input_points.png") if args.save_all else None
    )
    
    # Step 2: Compute convex hull
    print("\nComputing convex hull...")
    hull_indices = compute_convex_hull(points)
    print(f"Convex hull has {len(hull_indices)} vertices")
    
    # Visualize convex hull
    plot_hull(
        points,
        hull_indices,
        title="Convex Hull",
        show=not args.no_display,
        save_path=os.path.join(output_dir, "2_convex_hull.png") if args.save_all else None
    )
    
    # Step 3: Coarse segmentation
    print("\nPerforming coarse segmentation...")
    segments = coarse_segmentation(hull_indices, points)
    print(f"Created {len(segments)} segments")
    
    # Step 4: Identify feature points
    feature_points = identify_feature_points(points, hull_indices)
    
    # Step 5: Triangulate with different gradient values
    print("\nPerforming triangulation with gradient control...")
    
    for gradient in args.gradients:
        print(f"\nTriangulating with gradient = {gradient}")
        triangles = triangulate_with_gradient(
            points, segments, gradient, feature_points
        )
        
        print(f"Created {len(triangles)} triangles")
        
        # Visualize triangulation
        plot_triangulation(
            points,
            triangles,
            title=f"Triangulation with Gradient = {gradient}",
            feature_points=feature_points,
            show=not args.no_display,
            save_path=os.path.join(output_dir, f"3_triangulation_gradient_{gradient}.png") if args.save_all else None
        )

def main():
    parser = argparse.ArgumentParser(description="Test mesh generation workflow with gradient control")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--input-file', '-i', type=str, help='Input file with point coordinates (x,y)')
    input_group.add_argument('--num-points', '-n', type=int, default=25, help='Number of points to generate')
    
    # Point generation options
    parser.add_argument('--point-type', '-t', choices=['grid', 'random', 'circle'], default='grid',
                      help='Type of points to generate')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level for point generation')
    
    # Triangulation options
    parser.add_argument('--gradients', '-g', type=float, nargs='+', default=[0.5, 1.0, 2.0, 3.0],
                      help='Gradient values to test')
    
    # Output options
    parser.add_argument('--output-dir', '-o', type=str, default='output',
                      help='Output directory for saved files')
    parser.add_argument('--save-all', action='store_true', help='Save all plots to files')
    parser.add_argument('--no-display', action='store_true', help='Do not display plots')
    
    args = parser.parse_args()
    
    # Run the workflow
    run_workflow(args)

if __name__ == "__main__":
    main() 