#!/usr/bin/env python
"""
MeshIt-style coarse triangulation implementation.

Implements triangulation with gradient control similar to MeshIt's C++ implementation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import triangle as tr

class TriangleWrapper:
    """
    Minimal Triangle library wrapper that mimics MeshIt's triangulation behavior.
    """
    
    def __init__(self, gradient=2.0, min_angle=20.0, base_size=0.15):
        self.gradient = gradient
        self.min_angle = min_angle
        self.base_size = base_size
    
    def triangulate(self, vertices, segments=None):
        """Perform triangulation with gradient-based refinement."""
        # Calculate area constraint 
        triangle_area = self.base_size * self.base_size * 0.5
        
        # Apply gradient-based adjustments (mimics MeshIt behavior)
        if self.gradient < 1.0:
            triangle_area *= (1.0 - 0.5 * (1.0 - self.gradient))
        elif self.gradient > 1.0:
            triangle_area *= (1.0 + 0.5 * (self.gradient - 1.0))
            
        # Adjust min angle based on gradient
        min_angle = self.min_angle
        if self.gradient > 1.0:
            min_angle = max(self.min_angle - (self.gradient - 1.0) * 7.0, 10.0)
            
        # Setup triangle options (similar to MeshIt's "pzYYu")
        triangle_opts = f'pzq{min_angle}a{triangle_area}'
        
        # Run triangulation
        tri_data = {'vertices': vertices}
        if segments is not None:
            tri_data['segments'] = segments
            
        return tr.triangulate(tri_data, triangle_opts)

def main():
    # Setup
    output_dir = "coarse_triangulation_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate points and compute hull
    np.random.seed(42)
    points = np.random.uniform(-1, 1, (25, 2))
    hull = ConvexHull(points)
    
    # Create boundary segments
    segments = np.array([[hull.vertices[i], hull.vertices[(i+1) % len(hull.vertices)]] 
                        for i in range(len(hull.vertices))])
    
    # Test with different gradients
    for gradient in [1.0, 2.0, 4.0]:
        # Triangulate
        wrapper = TriangleWrapper(gradient=gradient)
        result = wrapper.triangulate(points, segments)
        
        # Visualize
        plt.figure(figsize=(10, 10))
        plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
                   result['triangles'], 'b-', lw=0.5)
        plt.scatter(points[:, 0], points[:, 1], color='blue', s=50)
        
        # Draw boundary
        for i, j in segments:
            plt.plot([points[i, 0], points[j, 0]], 
                    [points[i, 1], points[j, 1]], 
                    'g-', lw=2)
        
        plt.title(f"Triangulation (Gradient={gradient})\n{len(result['triangles'])} triangles")
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"triangulation_g{gradient}.png"))
        plt.close()
        
        print(f"Gradient {gradient}: {len(result['triangles'])} triangles")
    
    print(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    main() 