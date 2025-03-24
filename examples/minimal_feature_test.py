#!/usr/bin/env python
"""
Minimal feature point triangulation test for the triangle_mesh module.
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
    # Generate a few points
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5]
    ])
    
    # Calculate convex hull
    hull_indices, segments = create_hull_segments(points)
    print(f"Hull segments: {segments}")
    
    # Define one feature point
    feature_points = np.array([[0.3, 0.7]])  # Single feature point
    feature_sizes = np.array([0.1])  # Small size for refinement
    
    # Create wrapper with feature point
    wrapper = TriangleWrapper(gradient=1.5, base_size=0.3)
    wrapper.set_feature_points(feature_points, feature_sizes)
    
    # Run triangulation
    print("Running triangulation with feature point...")
    result = wrapper.triangulate(points, segments)
    
    # Print results
    print(f"Triangulation complete with {len(result['triangles'])} triangles")
    print(f"Number of vertices in result: {len(result['vertices'])}")
    
    # Plot results
    plt.figure(figsize=(8, 8))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], result['triangles'], 'k-')
    plt.plot(points[:, 0], points[:, 1], 'ko', markersize=6)
    
    # Plot feature point
    plt.plot(feature_points[0, 0], feature_points[0, 1], 'r*', markersize=12)
    
    # Draw influence radius circle
    circle = plt.Circle((feature_points[0, 0], feature_points[0, 1]), feature_sizes[0], 
                       fill=False, color='r', linestyle='--', alpha=0.7)
    plt.gca().add_patch(circle)
    
    # Plot additional vertices added during refinement
    if len(result['vertices']) > len(points):
        added_points = result['vertices'][len(points):]
        plt.plot(added_points[:, 0], added_points[:, 1], 'bx', markersize=4, 
                label='Added by refinement')
        plt.legend()
    
    plt.title(f"Feature Point Triangulation: {len(result['triangles'])} triangles")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig("minimal_feature_test.png")
    
    print("Minimal feature test completed successfully!")

if __name__ == "__main__":
    main() 