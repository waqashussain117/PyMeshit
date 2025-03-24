#!/usr/bin/env python
"""
Minimal demonstration of MeshIt's coarse triangulation approach.

This script demonstrates the basic workflow of:
1. Generating random points
2. Computing convex hull
3. Coarse segmentation of the hull boundary
4. Custom triangulation using the Triangle library's options
   similar to MeshIt's C++ implementation

This version focuses on the coarse triangulation process without adding feature points.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import triangle as tr

# Create a custom wrapper for the Triangle library that uses the same
# refinement logic as the triunsuitable function in MeshIt's C++ implementation
class TriangleWrapper:
    """
    Custom wrapper for the Triangle library to mimic MeshIt's triangulation approach.
    """
    
    def __init__(self, gradient=2.0, min_angle=20.0, max_area=None):
        """
        Initialize the TriangleWrapper with refinement parameters.
        
        Args:
            gradient: Gradient control parameter (default: 2.0, matches MeshIt core)
            min_angle: Minimum angle for triangle quality (default: 20.0)
            max_area: Maximum area constraint (if None, calculated from input)
        """
        self.gradient = gradient
        self.min_angle = min_angle
        self.max_area = max_area
        self.base_size = None
        self.initial_triangles = 0
    
    def triangulate(self, vertices, segments=None, holes=None):
        """
        Triangulate points with options similar to MeshIt's approach.
        
        Args:
            vertices: Vertex coordinates (N, 2)
            segments: Segment indices for constraining the triangulation (optional)
            holes: Coordinates of holes in the triangulation (optional)
            
        Returns:
            Dictionary with triangulation results (vertices, triangles)
        """
        # Calculate base size if not set
        if self.base_size is None:
            # Estimate from bounding box
            min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
            min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
            diagonal = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
            self.base_size = diagonal / 15.0
            
        # Calculate max area if not set
        if self.max_area is None:
            # Use base_size to estimate area
            self.max_area = self.base_size * self.base_size * 0.5
            
        # Adjust area constraint based on gradient - mimic MeshIt behavior
        triangle_area = self.max_area
        if self.gradient < 1.0:
            # Smaller triangles for smaller gradients
            triangle_area *= (1.0 - 0.5 * (1.0 - self.gradient))
        elif self.gradient > 1.0:
            # For higher gradients, create larger triangles
            triangle_area *= (1.0 + 0.5 * (self.gradient - 1.0))
            
        # Adjust min angle based on gradient - mimic MeshIt behavior
        min_angle = self.min_angle
        if self.gradient > 1.0:
            # For higher gradients, allow smaller angles
            min_angle = max(self.min_angle - (self.gradient - 1.0) * 7.0, 10.0)
            
        # Setup triangle options - similar to MeshIt's "pzYYu" options
        # Note: Python Triangle library doesn't directly support 'u', so we use q+a constraints
        triangle_opts = f'pzq{min_angle}a{triangle_area}'
        
        # Print the options being used (similar to MeshIt logging)
        print(f"Triangle options: {triangle_opts}")
        
        # Prepare triangle input
        tri_data = {'vertices': vertices}
        if segments is not None:
            tri_data['segments'] = segments
        if holes is not None:
            tri_data['holes'] = holes
            
        # Run triangulation
        result = tr.triangulate(tri_data, triangle_opts)
        
        # In MeshIt's C++ version, the 'u' option would call triunsuitable
        # to do gradient-based refinement. The Python version approximates this
        # with min_angle and area constraints.
        
        # Store number of triangles for reporting
        self.initial_triangles = len(result['triangles'])
        print(f"Triangulation complete: {self.initial_triangles} triangles")
        
        return result

def main():
    # Create output directory
    output_dir = "coarse_triangulation_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 25 random points (same as in your example)
    np.random.seed(42)
    points = np.random.uniform(-1, 1, (25, 2))
    
    # Save points visualization
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=50)
    plt.title("25 Random Points")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "1_points.png"))
    plt.close()
    
    # Compute convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # Visualize convex hull
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=50)
    
    # Plot the convex hull
    for i in range(len(hull_points)):
        plt.plot([hull_points[i, 0], hull_points[(i+1) % len(hull_points), 0]],
                [hull_points[i, 1], hull_points[(i+1) % len(hull_points), 1]],
                'g-', lw=2)
    
    plt.title("Convex Hull")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "2_convex_hull.png"))
    plt.close()
    
    # Create boundary segments from the convex hull (coarse segmentation)
    segments = np.array([[i, (i+1) % len(hull_points)] for i in range(len(hull_points))])
    
    # Create mapping from hull vertices to original point indices
    hull_to_original = {i: hull.vertices[i] for i in range(len(hull.vertices))}
    
    # Convert segments to use original point indices
    original_segments = np.array([[hull.vertices[i], hull.vertices[(i+1) % len(hull.vertices)]] 
                                 for i in range(len(hull.vertices))])
    
    # Try different gradient values to show the effect
    for gradient in [1.0, 2.0, 4.0]:
        print(f"\nProcessing with gradient {gradient}...")
        
        # Triangulate using our wrapper with behavior similar to MeshIt
        wrapper = TriangleWrapper(gradient=gradient)
        wrapper.base_size = 0.150  # Set a smaller base size for denser triangulation
        
        # Apply triangulation 
        result = wrapper.triangulate(points, original_segments)
        
        # Plot triangulation result
        plt.figure(figsize=(10, 10))
        
        # Plot the triangulation
        plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
                   result['triangles'], 'b-', lw=0.5)
        
        # Overlay the original points
        plt.scatter(points[:, 0], points[:, 1], color='blue', s=50)
        
        # Plot the boundary segments
        for segment in original_segments:
            p1, p2 = segment
            plt.plot([points[p1, 0], points[p2, 0]], 
                    [points[p1, 1], points[p2, 1]], 
                    'g-', lw=2)
        
        plt.title(f"Coarse Triangulation (Gradient={gradient}, base_size=0.165)\n{len(result['triangles'])} triangles")
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"3_triangulation_g{gradient}.png"))
        plt.close()
        
        print(f"Generated triangulation with gradient {gradient}: {len(result['triangles'])} triangles")
    
    print(f"\nResults saved to {output_dir}/")

if __name__ == "__main__":
    main() 