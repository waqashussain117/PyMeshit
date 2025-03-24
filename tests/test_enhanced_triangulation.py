#!/usr/bin/env python
"""
Test the enhanced triangle wrapper with the exact parameters
used in MeshIt for direct comparison.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import traceback

def main():
    try:
        print("Starting enhanced triangulation test...")
        
        # Create output directory
        output_dir = "enhanced_triangulation"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        
        # First, check if we can even create a simple output file
        try:
            with open(os.path.join(output_dir, "test.txt"), "w") as f:
                f.write("Testing file creation")
            print("File creation test successful")
        except Exception as e:
            print(f"ERROR: Cannot create files in output directory: {str(e)}")
            return
        
        # Load the test points from the CSV file
        try:
            points = np.loadtxt("vtu_output/points.csv", delimiter=",", skiprows=1)[:, :2]
            print(f"Loaded {len(points)} points from CSV file")
        except Exception as e:
            print(f"Error loading points CSV: {str(e)}")
            # Fallback to hard-coded points if needed
            print("Using sample points instead...")
            np.random.seed(42)
            points = np.random.uniform(-1, 1, (25, 2))
        
        # Compute convex hull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        print(f"Computed convex hull with {len(hull_points)} vertices")
        
        # Create and save a simple plot first to verify matplotlib is working
        print("Creating simple test plot...")
        plt.figure(figsize=(8, 8))
        plt.scatter(points[:, 0], points[:, 1], c='blue')
        plt.title("Test Plot")
        plt.savefig(os.path.join(output_dir, "test_plot.png"))
        plt.close()
        print("Test plot saved successfully")
        
        # Setup triangle wrapper with the exact MeshIt parameters
        # We'll make a simplified version without the custom wrapper to check if it's causing issues
        # First, let's visualize just the points without triangulation
        print("Creating points-only visualization...")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot points and convex hull
        ax.scatter(points[:, 0], points[:, 1], c='blue', s=50, label='Original Points')
        ax.scatter(hull_points[:, 0], hull_points[:, 1], c='red', s=80, label='Hull Points (Features)')
        
        # Draw convex hull
        for i in range(len(hull.vertices)):
            j = (i + 1) % len(hull.vertices)
            ax.plot([points[hull.vertices[i], 0], points[hull.vertices[j], 0]],
                     [points[hull.vertices[i], 1], points[hull.vertices[j], 1]], 'r-')
        
        ax.set_title("Original Points with Convex Hull")
        ax.grid(True)
        ax.axis('equal')
        ax.legend()
        
        # Save the figure
        output_file = os.path.join(output_dir, "points_only.png")
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Points-only visualization saved to {output_file}")
        
        # Now try to perform triangulation with standard Delaunay
        from scipy.spatial import Delaunay
        print("Performing standard Delaunay triangulation...")
        tri = Delaunay(points)
        
        # Create visualization of standard Delaunay
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot triangles
        for simplex in tri.simplices:
            x = [points[simplex[0], 0], points[simplex[1], 0], 
                 points[simplex[2], 0], points[simplex[0], 0]]
            y = [points[simplex[0], 1], points[simplex[1], 1], 
                 points[simplex[2], 1], points[simplex[0], 1]]
            ax.plot(x, y, 'b-', linewidth=0.5)
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], c='red', s=30)
        
        ax.set_title(f"Standard Delaunay Triangulation ({len(tri.simplices)} triangles)")
        ax.grid(True)
        ax.axis('equal')
        
        # Save the figure
        output_file = os.path.join(output_dir, "delaunay.png")
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Delaunay visualization saved to {output_file}")
        
        # Now try to use our custom wrapper with scaled-down parameters
        try:
            from meshit.triangle_wrapper import TriangleWrapper
            wrapper = TriangleWrapper(gradient=2.0)
            print("Created TriangleWrapper with gradient=2.0")
            
            # Set feature points as the hull points with the exact size used in MeshIt (0.165)
            feature_points = hull_points
            feature_size = 0.165
            feature_sizes = np.ones(len(feature_points)) * feature_size
            wrapper.set_feature_points(feature_points, feature_sizes)
            print(f"Set {len(feature_points)} feature points with size {feature_size}")
            
            # Create segments from hull vertices
            segments = []
            for i in range(len(hull.vertices)):
                segments.append([hull.vertices[i], hull.vertices[(i+1) % len(hull.vertices)]])
            segments = np.array(segments)
            print(f"Created {len(segments)} boundary segments")
            
            # IMPORTANT: Temporarily modify _direct_dense_refinement method to use fewer points
            # This monkey patch will reduce the complexity for debugging purposes
            original_method = wrapper._direct_dense_refinement
            
            def simplified_refinement(self, vertices, triangles):
                print("Using simplified refinement with fewer points")
                if self.feature_points is None or len(self.feature_points) == 0:
                    return np.empty((0, 2))
                    
                extra_points = []
                
                # For each feature point - simplified version with fewer points
                for i, feature in enumerate(self.feature_points):
                    size = self.feature_sizes[i]
                    num_points = 8  # Fixed smaller number
                    
                    # Just one ring for simplicity
                    for j in range(num_points):
                        angle = 2.0 * np.pi * j / num_points
                        distance = size * 2.0  # Fixed distance
                        
                        dx = distance * np.cos(angle)
                        dy = distance * np.sin(angle)
                        
                        new_point = feature + np.array([dx, dy])
                        extra_points.append(new_point)
                
                result = np.array(extra_points) if extra_points else np.empty((0, 2))
                print(f"Created {len(result)} simplified refinement points")
                return result
            
            # Monkey patch the method
            wrapper._direct_dense_refinement = simplified_refinement.__get__(wrapper, type(wrapper))
            
            # Triangulate using our wrapper with simplified refinement
            print("Starting custom triangulation with simplified refinement...")
            triangulation = wrapper.triangulate(points, segments)
            print("Triangulation completed successfully")
            
            # Extract results
            vertices = triangulation['vertices']
            triangles = triangulation['triangles']
            num_triangles = len(triangles)
            
            print(f"Enhanced triangulation completed with {num_triangles} triangles")
            print(f"Original test points: {len(points)}")
            print(f"Final mesh points: {len(vertices)}")
            
            # Create custom triangulation visualization
            print("Creating custom triangulation visualization...")
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot triangles
            for i in range(len(triangles)):
                tri = triangles[i]
                x = [vertices[tri[0], 0], vertices[tri[1], 0], vertices[tri[2], 0], vertices[tri[0], 0]]
                y = [vertices[tri[0], 1], vertices[tri[1], 1], vertices[tri[2], 1], vertices[tri[0], 1]]
                ax.plot(x, y, 'b-', linewidth=0.5)
            
            # Plot vertices
            ax.scatter(vertices[:, 0], vertices[:, 1], c='red', s=10)
            
            # Highlight original points
            ax.scatter(points[:, 0], points[:, 1], c='blue', s=30)
            
            ax.set_title(f"Custom Triangulation (Gradient: 2.0, {num_triangles} triangles)")
            ax.grid(False)
            ax.axis('equal')
            
            # Save the figure
            output_file = os.path.join(output_dir, "custom_triangulation.png")
            plt.savefig(output_file, dpi=150)
            plt.close()
            
            print(f"Custom triangulation visualization saved to {output_file}")
            
        except ImportError as e:
            print(f"ERROR: Could not import TriangleWrapper: {str(e)}")
        except Exception as e:
            print(f"ERROR in custom triangulation: {str(e)}")
            traceback.print_exc()
        
        print("Test completed!")
        
    except Exception as e:
        print(f"Unexpected error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 