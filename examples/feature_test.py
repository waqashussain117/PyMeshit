#!/usr/bin/env python
"""
Test for feature point refinement in triangle_mesh module.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from meshit.triangle_mesh import TriangleWrapper, create_hull_segments

def main():
    print("Testing feature point refinement...")
    
    # Generate random points
    np.random.seed(42)
    points = np.random.rand(20, 2) * 2 - 1  # Range: -1 to 1
    
    # Create hull segments
    hull_indices, segments = create_hull_segments(points)
    
    # Create feature points (3 points with different sizes)
    feature_points = np.array([
        [0.0, 0.0],    # Center
        [0.5, 0.5],    # Upper right
        [-0.5, -0.5]   # Lower left
    ])
    
    # Use smaller feature sizes to avoid numerical issues
    feature_sizes = np.array([0.1, 0.08, 0.12])
    
    # First triangulate without feature points
    print("Triangulating without feature points...")
    wrapper_no_features = TriangleWrapper(gradient=1.5, base_size=0.3)
    result_no_features = wrapper_no_features.triangulate(points, segments)
    
    # Then triangulate with feature points
    print("Triangulating with feature points...")
    wrapper_with_features = TriangleWrapper(gradient=1.5, base_size=0.3)
    wrapper_with_features.set_feature_points(feature_points, feature_sizes)
    result_with_features = wrapper_with_features.triangulate(points, segments)
    
    # Report statistics
    print(f"Without features: {len(result_no_features['triangles'])} triangles")
    print(f"With features:    {len(result_with_features['triangles'])} triangles")
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot without features
    ax1.triplot(result_no_features['vertices'][:, 0], result_no_features['vertices'][:, 1], 
               result_no_features['triangles'], 'k-', alpha=0.7)
    ax1.plot(points[:, 0], points[:, 1], 'ko', markersize=4)
    ax1.set_title(f"Without Feature Refinement\n({len(result_no_features['triangles'])} triangles)")
    ax1.set_aspect('equal')
    ax1.grid(alpha=0.3)
    
    # Plot with features
    ax2.triplot(result_with_features['vertices'][:, 0], result_with_features['vertices'][:, 1], 
               result_with_features['triangles'], 'k-', alpha=0.7)
    ax2.plot(points[:, 0], points[:, 1], 'ko', markersize=4)
    
    # Plot feature points with their influence radius
    for i, (point, size) in enumerate(zip(feature_points, feature_sizes)):
        ax2.plot(point[0], point[1], 'r*', markersize=10, label=f"Feature {i+1}" if i==0 else "")
        # Draw influence circle
        circle = plt.Circle((point[0], point[1]), size, fill=False, color='r', linestyle='--', alpha=0.5)
        ax2.add_patch(circle)
    
    ax2.set_title(f"With Feature Refinement\n({len(result_with_features['triangles'])} triangles)")
    ax2.set_aspect('equal')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("feature_refinement_test.png")
    print("Plot saved to feature_refinement_test.png")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main() 