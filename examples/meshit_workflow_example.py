#!/usr/bin/env python
"""
MeshIt Workflow Example

This script demonstrates how to use the reimplemented MeshIt workflow functions
to process geometric models similar to the original MeshIt software.

Usage:
    python meshit_workflow_example.py [--input MODEL_FILE] [--output OUTPUT_FILE]
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path

# Add the project root to the Python path if running the script directly
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Import our custom implementation
from meshit.workflow import run_complete_workflow, run_coarse_triangulation

# Simple MeshItModel implementation for testing
class SimpleVector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class SimpleSurface:
    def __init__(self, name, vertices=None, triangles=None):
        self.name = name
        self.vertices = vertices or []
        self.triangles = triangles or []
        self.convex_hull = []
        
    def calculate_convex_hull(self):
        """Calculate the convex hull in 2D (XY plane)"""
        if len(self.vertices) < 3:
            self.convex_hull = self.vertices.copy()
            return
            
        # Extract 2D points
        points_2d = np.array([[v.x, v.y] for v in self.vertices])
        
        # Calculate convex hull using Graham scan
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # collinear
            return 1 if val > 0 else 2  # clockwise or counterclockwise
        
        # Find the bottom-most point
        p0_idx = np.argmin(points_2d[:, 1])
        p0 = points_2d[p0_idx]
        
        # Sort points by polar angle with respect to p0
        def polar_angle(p):
            return np.arctan2(p[1] - p0[1], p[0] - p0[0])
        
        sorted_indices = sorted(range(len(points_2d)), key=lambda i: 
                            (polar_angle(points_2d[i]), 
                             np.linalg.norm(points_2d[i] - p0)))
        
        # Build hull
        hull_indices = [sorted_indices[0], sorted_indices[1]]
        
        for i in range(2, len(sorted_indices)):
            while len(hull_indices) > 1 and orientation(
                points_2d[hull_indices[-2]], 
                points_2d[hull_indices[-1]], 
                points_2d[sorted_indices[i]]) != 2:
                hull_indices.pop()
            hull_indices.append(sorted_indices[i])
        
        # Convert back to 3D vertices
        self.convex_hull = [self.vertices[i] for i in hull_indices]

class SimpleModel:
    def __init__(self):
        self.surfaces = []
        self.model_polylines = []
        self.intersections = []
        
    def add_surface(self, surface):
        self.surfaces.append(surface)
        return self
        
    def add_polyline(self, polyline):
        self.model_polylines.append(polyline)
        return self

def create_test_model():
    """Create a simple test model with two intersecting surfaces"""
    model = SimpleModel()
    
    # Create a horizontal surface (XY plane)
    surface1 = SimpleSurface("Horizontal Surface")
    
    # Add vertices for a square in the XY plane
    surface1.vertices = [
        SimpleVector(-5.0, -5.0, 0.0),
        SimpleVector(5.0, -5.0, 0.0),
        SimpleVector(5.0, 5.0, 0.0),
        SimpleVector(-5.0, 5.0, 0.0)
    ]
    
    # Create a vertical surface (XZ plane)
    surface2 = SimpleSurface("Vertical Surface")
    
    # Add vertices for a square in the XZ plane
    surface2.vertices = [
        SimpleVector(-3.0, 0.0, -3.0),
        SimpleVector(3.0, 0.0, -3.0),
        SimpleVector(3.0, 0.0, 3.0),
        SimpleVector(-3.0, 0.0, 3.0)
    ]
    
    # Add surfaces to the model
    model.add_surface(surface1)
    model.add_surface(surface2)
    
    # Precalculate convex hulls
    for surface in model.surfaces:
        surface.calculate_convex_hull()
    
    return model

def visualize_model(model, output_file=None):
    """
    Visualize the model using matplotlib.
    
    Args:
        model: SimpleModel instance
        output_file: Optional path to save the visualization
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("Matplotlib not available. Cannot visualize the model.")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up colors for surfaces
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    
    # Plot each surface
    for i, surface in enumerate(model.surfaces):
        color = colors[i % len(colors)]
        
        # Plot vertices
        xs = [v.x for v in surface.vertices]
        ys = [v.y for v in surface.vertices]
        zs = [v.z for v in surface.vertices]
        ax.scatter(xs, ys, zs, color='black', s=20)
        
        # Plot triangles
        if surface.triangles:
            try:
                for triangle in surface.triangles:
                    if len(triangle) != 3 or max(triangle) >= len(surface.vertices):
                        # Skip invalid triangles
                        continue
                        
                    verts = [
                        (surface.vertices[triangle[0]].x, 
                         surface.vertices[triangle[0]].y, 
                         surface.vertices[triangle[0]].z),
                        (surface.vertices[triangle[1]].x, 
                         surface.vertices[triangle[1]].y, 
                         surface.vertices[triangle[1]].z),
                        (surface.vertices[triangle[2]].x, 
                         surface.vertices[triangle[2]].y, 
                         surface.vertices[triangle[2]].z)
                    ]
                    tri = Poly3DCollection([verts], alpha=0.4)
                    tri.set_color(color)
                    ax.add_collection3d(tri)
            except Exception as e:
                print(f"Warning: Error plotting triangles for surface {surface.name}: {e}")
        
        # Plot convex hull
        if surface.convex_hull:
            hull_xs = [v.x for v in surface.convex_hull]
            hull_ys = [v.y for v in surface.convex_hull]
            hull_zs = [v.z for v in surface.convex_hull]
            
            # Add the first point again to close the loop
            hull_xs.append(hull_xs[0])
            hull_ys.append(hull_ys[0])
            hull_zs.append(hull_zs[0])
            
            ax.plot(hull_xs, hull_ys, hull_zs, 'r-', linewidth=2)
    
    # Plot intersections
    if hasattr(model, 'intersections') and model.intersections:
        for intersection in model.intersections:
            if hasattr(intersection, 'points') and intersection.points:
                intr_xs = [p.x for p in intersection.points]
                intr_ys = [p.y for p in intersection.points]
                intr_zs = [p.z for p in intersection.points]
                ax.plot(intr_xs, intr_ys, intr_zs, 'r-', linewidth=3)
    
    # Set equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Get all vertices to determine axis limits
    all_verts = []
    for surface in model.surfaces:
        all_verts.extend([(v.x, v.y, v.z) for v in surface.vertices])
    
    if all_verts:
        all_verts = np.array(all_verts)
        max_range = np.max([
            np.max(all_verts[:, 0]) - np.min(all_verts[:, 0]),
            np.max(all_verts[:, 1]) - np.min(all_verts[:, 1]),
            np.max(all_verts[:, 2]) - np.min(all_verts[:, 2])
        ])
        
        # Set equal aspect ratio
        mid_x = (np.max(all_verts[:, 0]) + np.min(all_verts[:, 0])) / 2
        mid_y = (np.max(all_verts[:, 1]) + np.min(all_verts[:, 1])) / 2
        mid_z = (np.max(all_verts[:, 2]) + np.min(all_verts[:, 2])) / 2
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.title('MeshIt Model Visualization')
    
    if output_file:
        plt.savefig(output_file)
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='MeshIt Workflow Example')
    parser.add_argument('--input', type=str, help='Input model file (not implemented)')
    parser.add_argument('--output', type=str, help='Output visualization file')
    parser.add_argument('--triangulation-only', action='store_true', 
                        help='Run only triangulation without intersections')
    
    args = parser.parse_args()
    
    print("Creating test model...")
    model = create_test_model()
    
    print(f"Model created with {len(model.surfaces)} surfaces:")
    for i, surface in enumerate(model.surfaces):
        print(f"  Surface {i+1}: {surface.name} with {len(surface.vertices)} vertices")
    
    # Process the model
    if args.triangulation_only:
        print("\nRunning coarse triangulation only...")
        processed_model = run_coarse_triangulation(model, gradient=2.0)
    else:
        print("\nRunning complete workflow...")
        processed_model = run_complete_workflow(model, gradient=2.0)
    
    print("\nProcessing complete!")
    print(f"Model now has {len(processed_model.surfaces)} surfaces:")
    for i, surface in enumerate(processed_model.surfaces):
        print(f"  Surface {i+1}: {surface.name} with {len(surface.vertices)} vertices and {len(surface.triangles)} triangles")
    
    if hasattr(processed_model, 'intersections') and processed_model.intersections:
        print(f"\nFound {len(processed_model.intersections)} intersections")
    
    # Visualize the result
    print("\nVisualizing the model...")
    visualize_model(processed_model, args.output)
    
    print("Done!")

if __name__ == "__main__":
    main() 