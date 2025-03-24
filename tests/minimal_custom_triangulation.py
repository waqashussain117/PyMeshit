#!/usr/bin/env python
"""
Minimal demonstration of MeshIt's custom triangle wrapper.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from meshit.triangle_wrapper import TriangleWrapper

def main():
    # Create output directory
    output_dir = "custom_triangulation_with_features"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 25 random points
    np.random.seed(42)
    points = np.random.uniform(-1, 1, (25, 2))
    
    # Define feature points
    feature_points = np.array([
        [0.0, 0.0],   # Center
        [0.5, 0.5],   # Top right
        [-0.5, -0.5]  # Bottom left
    ])
    feature_sizes = np.array([0.1, 0.1, 0.1])
    
    # Save points visualization
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=50)
    plt.scatter(feature_points[:, 0], feature_points[:, 1], color='red', s=100, marker='*')
    plt.title("25 Points with Feature Points")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "1_points_with_features.png"))
    plt.close()
    
    # Compute convex hull for boundary
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # Create boundary segments
    segments = np.array([[i, (i+1) % len(hull_points)] for i in range(len(hull_points))])
    
    # Try different gradient values
    for gradient in [2.0, 4.0]:
        try:
            print(f"Processing with gradient {gradient}...")
            
            # First triangulate without features for comparison
            wrapper_no_features = TriangleWrapper(gradient=gradient)
            wrapper_no_features.base_size = 0.5  # Set a fixed base size
            
            # Apply triangulation without features
            result_no_features = wrapper_no_features.triangulate(points, segments)
            
            # Plot result without features
            plt.figure(figsize=(10, 10))
            plt.triplot(result_no_features['vertices'][:, 0], result_no_features['vertices'][:, 1], 
                       result_no_features['triangles'], 'b-', lw=0.5)
            plt.scatter(points[:, 0], points[:, 1], color='blue', s=50)
            plt.scatter(feature_points[:, 0], feature_points[:, 1], color='red', s=100, marker='*')
            
            # Plot boundary
            for i in range(len(hull_points)):
                plt.plot([hull_points[i, 0], hull_points[(i+1) % len(hull_points), 0]],
                        [hull_points[i, 1], hull_points[(i+1) % len(hull_points), 1]],
                        'g-', lw=2)
            
            plt.title(f"Triangulation WITHOUT Feature Control (Gradient={gradient})\n{len(result_no_features['triangles'])} triangles")
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"2_triangulation_no_features_g{gradient}.png"))
            plt.close()
            
            print(f"Generated triangulation WITHOUT features: {len(result_no_features['triangles'])} triangles")
            
            # Now triangulate with features
            wrapper_with_features = TriangleWrapper(gradient=gradient)
            wrapper_with_features.base_size = 0.5  # Set a fixed base size
            wrapper_with_features.set_feature_points(feature_points, feature_sizes)
            
            # Apply triangulation with features
            result_with_features = wrapper_with_features.triangulate(points, segments)
            
            # Plot result with features
            plt.figure(figsize=(10, 10))
            plt.triplot(result_with_features['vertices'][:, 0], result_with_features['vertices'][:, 1], 
                       result_with_features['triangles'], 'b-', lw=0.5)
            plt.scatter(points[:, 0], points[:, 1], color='blue', s=50)
            plt.scatter(feature_points[:, 0], feature_points[:, 1], color='red', s=100, marker='*')
            
            # Plot boundary
            for i in range(len(hull_points)):
                plt.plot([hull_points[i, 0], hull_points[(i+1) % len(hull_points), 0]],
                        [hull_points[i, 1], hull_points[(i+1) % len(hull_points), 1]],
                        'g-', lw=2)
            
            plt.title(f"Triangulation WITH Feature Control (Gradient={gradient})\n{len(result_with_features['triangles'])} triangles")
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"3_triangulation_with_features_g{gradient}.png"))
            plt.close()
            
            print(f"Generated triangulation WITH features: {len(result_with_features['triangles'])} triangles")
            
            # Create comparison plot
            plt.figure(figsize=(15, 7))
            
            # Without features
            plt.subplot(1, 2, 1)
            plt.triplot(result_no_features['vertices'][:, 0], result_no_features['vertices'][:, 1], 
                       result_no_features['triangles'], 'b-', lw=0.5)
            plt.scatter(feature_points[:, 0], feature_points[:, 1], color='red', s=100, marker='*')
            plt.title(f"WITHOUT Features\n{len(result_no_features['triangles'])} triangles")
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            # With features
            plt.subplot(1, 2, 2)
            plt.triplot(result_with_features['vertices'][:, 0], result_with_features['vertices'][:, 1], 
                       result_with_features['triangles'], 'b-', lw=0.5)
            plt.scatter(feature_points[:, 0], feature_points[:, 1], color='red', s=100, marker='*')
            plt.title(f"WITH Features\n{len(result_with_features['triangles'])} triangles")
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            plt.suptitle(f"Comparison of Triangulation with Gradient={gradient}", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"4_comparison_g{gradient}.png"))
            plt.close()
            
            increase_pct = ((len(result_with_features['triangles']) - len(result_no_features['triangles'])) / 
                           len(result_no_features['triangles']) * 100)
            print(f"Triangle increase with features: {increase_pct:.1f}%")
            
        except Exception as e:
            print(f"Error with gradient {gradient}: {str(e)}")
    
    print(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    main() 