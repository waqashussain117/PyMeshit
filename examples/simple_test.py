#!/usr/bin/env python
"""
Simple test for the triangle_mesh module.
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
    print("Running simple test for triangle_mesh...")
    
    # Generate 20 random points
    np.random.seed(42)
    points = np.random.rand(20, 2) * 2 - 1  # Range: -1 to 1
    
    # Calculate convex hull and create segments
    print("Creating hull segments...")
    hull_indices, segments = create_hull_segments(points)
    
    # Create triangle wrapper
    print("Creating triangle wrapper...")
    wrapper = TriangleWrapper(gradient=1.0)
    
    # Run triangulation
    print("Running triangulation...")
    result = wrapper.triangulate(points, segments)
    
    # Print result
    print(f"Triangulation complete with {len(result['triangles'])} triangles")
    
    # Visualize
    plt.figure(figsize=(8, 8))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], result['triangles'], 'k-')
    plt.plot(points[:, 0], points[:, 1], 'ro')
    
    # Plot hull
    hull_points = points[hull_indices]
    hull_points = np.vstack([hull_points, hull_points[0]])  # Close the loop
    plt.plot(hull_points[:, 0], hull_points[:, 1], 'b-', linewidth=2)
    
    plt.title(f"Triangulation with {len(result['triangles'])} triangles")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig("simple_test_triangulation.png")
    print("Plot saved to simple_test_triangulation.png")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main() 