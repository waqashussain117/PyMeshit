#!/usr/bin/env python
"""
Minimal test for the triangle_mesh module.
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
    np.random.seed(42)
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
    
    # Create wrapper and triangulate
    wrapper = TriangleWrapper(gradient=1.0)
    result = wrapper.triangulate(points, segments)
    
    # Print results
    print(f"Triangulation complete with {len(result['triangles'])} triangles")
    print(f"Triangles: {result['triangles']}")
    print(f"Vertices: {result['vertices']}")
    
    # Plot results
    plt.figure(figsize=(8, 8))
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], result['triangles'], 'k-')
    plt.plot(points[:, 0], points[:, 1], 'ro')
    plt.title("Minimal Triangulation Test")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig("minimal_test.png")
    
    print("Minimal test completed successfully!")

if __name__ == "__main__":
    main() 