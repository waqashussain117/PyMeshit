#!/usr/bin/env python
"""
Simple visualization script for MeshIt's custom triangle wrapper.

This script:
1. Generates 25 random points
2. Computes their convex hull
3. Performs coarse segmentation of the hull boundary
4. Uses only the custom triangle wrapper (no Delaunay)
5. Saves visualizations to a 'custom_triangulation' folder
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import time
import traceback

try:
    from meshit.triangle_wrapper import TriangleWrapper
    print("MeshIt modules successfully imported")
except ImportError:
    print("Error: Could not import triangle_wrapper. Make sure meshit is installed.")
    sys.exit(1)

def generate_points(num_points=25, method='random', seed=42):
    """Generate test points using specified method"""
    np.random.seed(seed)
    
    if method == 'random':
        # Generate random points in a square
        points = np.random.uniform(-1, 1, (num_points, 2))
    elif method == 'circle':
        # Generate points in a circle pattern with some randomness
        theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        r = np.random.uniform(0.7, 1.0, num_points)
        points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    elif method == 'grid':
        # Generate a slightly perturbed grid
        n = int(np.sqrt(num_points))
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack((X.ravel(), Y.ravel()))
        points = points[:num_points]  # Ensure exactly num_points
        # Add slight perturbation
        points += np.random.normal(0, 0.05, points.shape)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return points

def compute_convex_hull(points):
    """Compute the convex hull of points"""
    hull = ConvexHull(points)
    hull_vertices = hull.vertices
    hull_points = points[hull_vertices]
    
    return hull_points

def perform_coarse_segmentation(hull_points, segment_length=0.2):
    """
    Perform coarse segmentation on the hull boundary.
    This divides the hull into segments of approximately equal length.
    """
    segmented_points = []
    
    # Add the hull points first
    for i in range(len(hull_points)):
        segmented_points.append(hull_points[i])
        
        # Get current point and next point
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]
        
        # Calculate distance
        distance = np.linalg.norm(p2 - p1)
        
        # Add intermediate points if segment is long enough
        if distance > segment_length * 1.5:
            num_points = int(distance / segment_length)
            for j in range(1, num_points):
                t = j / num_points
                intermediate_point = p1 * (1 - t) + p2 * t
                segmented_points.append(intermediate_point)
    
    return np.array(segmented_points)

def create_segments(points):
    """Create segments connecting adjacent points in order"""
    n = len(points)
    return np.array([[i, (i + 1) % n] for i in range(n)])

def run_custom_triangulation(output_dir='custom_triangulation', point_method='random', 
                            gradients=(2.0, 3.0, 4.0)):
    """
    Run the custom triangulation process:
    1. Generate points
    2. Compute convex hull
    3. Perform coarse segmentation
    4. Apply custom triangulation with different gradients
    5. Save visualizations
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Generate points
        print(f"Generating 25 points using {point_method} method...")
        points = generate_points(25, point_method)
        
        # Save initial points
        plt.figure(figsize=(8, 8))
        plt.scatter(points[:, 0], points[:, 1], color='blue', s=30)
        plt.title(f"Initial 25 points")
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "1_initial_points.png"))
        plt.close()
        
        # Step 2: Compute convex hull
        print("Computing convex hull...")
        hull_points = compute_convex_hull(points)
        
        # Save convex hull
        plt.figure(figsize=(8, 8))
        plt.scatter(points[:, 0], points[:, 1], color='blue', s=30)
        
        # Plot hull in circular order
        for i in range(len(hull_points)):
            plt.plot([hull_points[i, 0], hull_points[(i + 1) % len(hull_points), 0]],
                     [hull_points[i, 1], hull_points[(i + 1) % len(hull_points), 1]],
                     'r-', lw=2)
        
        plt.title(f"Convex hull ({len(hull_points)} points)")
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "2_convex_hull.png"))
        plt.close()
        
        # Step 3: Perform coarse segmentation
        print("Performing coarse segmentation...")
        segmented_hull_points = perform_coarse_segmentation(hull_points)
        segments = create_segments(segmented_hull_points)
        
        # Save segmented hull
        plt.figure(figsize=(8, 8))
        plt.scatter(points[:, 0], points[:, 1], color='blue', s=30)
        
        # Plot segmented hull
        for i in range(len(segmented_hull_points)):
            plt.plot([segmented_hull_points[i, 0], segmented_hull_points[(i + 1) % len(segmented_hull_points), 0]],
                     [segmented_hull_points[i, 1], segmented_hull_points[(i + 1) % len(segmented_hull_points), 1]],
                     'g-', lw=2)
        
        plt.scatter(segmented_hull_points[:, 0], segmented_hull_points[:, 1], 
                    color='green', s=40, marker='s')
        
        plt.title(f"Coarse segmentation ({len(segmented_hull_points)} boundary points)")
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "3_coarse_segmentation.png"))
        plt.close()
        
        # Combine points
        all_points = np.vstack([points, segmented_hull_points])
        
        # Calculate bounds for consistent plotting
        min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
        min_y, max_y = all_points[:, 1].min(), all_points[:, 1].max()
        
        # Add some margin
        width = max_x - min_x
        height = max_y - min_y
        margin = max(width, height) * 0.1
        plot_bounds = [min_x - margin, max_x + margin, min_y - margin, max_y + margin]
        
        # Calculate base_size from bounding box
        diagonal = np.sqrt(width**2 + height**2)
        base_size = diagonal / 15.0
        
        # Step 4: Apply custom TriangleWrapper with different gradients
        print("Applying custom MeshIt triangulation with gradient control...")
        
        # Create a summary figure for all gradients
        plt.figure(figsize=(12, 10))
        
        for i, gradient in enumerate(gradients):
            try:
                print(f"Processing gradient = {gradient}...")
                
                # Initialize triangle wrapper
                wrapper = TriangleWrapper(gradient=gradient)
                wrapper.base_size = base_size
                
                # Apply triangulation
                start_time = time.time()
                result = wrapper.triangulate(all_points, segments)
                elapsed = time.time() - start_time
                
                # Plot custom triangulation
                plt.figure(figsize=(10, 10))
                plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
                           result['triangles'], 'b-', lw=0.5)
                
                plt.title(f"Custom MeshIt triangulation (Gradient = {gradient})\n{len(result['triangles'])} triangles, {elapsed:.2f}s")
                plt.axis('equal')
                plt.xlim(plot_bounds[0], plot_bounds[1])
                plt.ylim(plot_bounds[2], plot_bounds[3])
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, f"4_custom_triangulation_g{gradient}.png"))
                plt.close()
                
                # Add to summary figure
                plt.subplot(1, len(gradients), i + 1)
                plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
                           result['triangles'], 'b-', lw=0.5)
                plt.title(f"Gradient = {gradient}\n{len(result['triangles'])} triangles")
                plt.axis('equal')
                plt.xlim(plot_bounds[0], plot_bounds[1])
                plt.ylim(plot_bounds[2], plot_bounds[3])
                plt.grid(True, alpha=0.3)
                
                print(f"Gradient {gradient} completed with {len(result['triangles'])} triangles in {elapsed:.2f}s")
            except Exception as e:
                print(f"Error processing gradient {gradient}: {str(e)}")
                traceback.print_exc()
        
        # Save summary figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "5_gradient_comparison.png"))
        plt.close()
        
        print(f"\nAll visualizations saved to '{output_dir}' directory")
        
    except Exception as e:
        print(f"Error in run_custom_triangulation: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize MeshIt custom triangulation with 25 points")
    parser.add_argument("--method", type=str, default="random", choices=["random", "circle", "grid"],
                       help="Method for generating points (default: random)")
    parser.add_argument("--gradients", type=str, default="2.0,3.0,4.0",
                       help="Comma-separated list of gradient values (default: 2.0,3.0,4.0)")
    parser.add_argument("--output-dir", type=str, default="custom_triangulation",
                       help="Output directory (default: custom_triangulation)")
    
    args = parser.parse_args()
    
    # Parse gradient values
    gradients = [float(g) for g in args.gradients.split(",")]
    
    # Run visualization
    run_custom_triangulation(
        output_dir=args.output_dir,
        point_method=args.method,
        gradients=gradients
    ) 