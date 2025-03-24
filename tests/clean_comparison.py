#!/usr/bin/env python
"""
Compare standard Delaunay triangulation with MeshIt's triangulation pattern.
This script visualizes the key differences WITHOUT showing feature points.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import traceback

def main():
    try:
        print("Starting clean triangulation comparison...")
        
        # Create output directory
        output_dir = "clean_comparison"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        
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
        
        # Perform standard Delaunay triangulation
        print("Performing standard Delaunay triangulation...")
        delaunay = Delaunay(points)
        delaunay_triangles = delaunay.simplices
        
        # Calculate statistics for Delaunay
        num_delaunay_triangles = len(delaunay_triangles)
        print(f"Delaunay triangulation: {num_delaunay_triangles} triangles")
        
        # Create visualization of standard Delaunay - CLEAN VERSION (no feature points)
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot only triangles
        for simplex in delaunay_triangles:
            x = [points[simplex[0], 0], points[simplex[1], 0], 
                 points[simplex[2], 0], points[simplex[0], 0]]
            y = [points[simplex[0], 1], points[simplex[1], 1], 
                 points[simplex[2], 1], points[simplex[0], 1]]
            ax.plot(x, y, 'b-', linewidth=0.5)
        
        ax.set_title(f"Standard Delaunay ({num_delaunay_triangles} triangles)")
        ax.grid(False)
        ax.axis('equal')
        ax.axis('off')  # Turn off axes
        
        # Save the figure
        output_file = os.path.join(output_dir, "delaunay_clean.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Clean Delaunay visualization saved to {output_file}")
        
        # Create visualization of MeshIt triangulation (from your image)
        # Since we don't have direct access to MeshIt's triangulation,
        # we'll create a visual representation based on key observations:
        
        print("Creating MeshIt pattern visualization...")
        
        # First, generate a finer point set to represent MeshIt's refinement
        refined_points = list(points)
        
        # Add more points around hull points to simulate refinement
        for hp in hull_points:
            num_refined = 15  # Number of refinement points per hull point
            for i in range(num_refined):
                angle = 2.0 * np.pi * i / num_refined
                # Distance decreases as gradient increases
                distance = 0.165 * 1.5  # Feature size * factor
                
                dx = distance * np.cos(angle)
                dy = distance * np.sin(angle)
                
                refined_points.append(hp + np.array([dx, dy]))
        
        # Add some interior points with larger spacing
        interior_center = np.mean(points, axis=0)
        num_interior = 12
        for i in range(num_interior):
            angle = 2.0 * np.pi * i / num_interior
            distance = 0.25  # Larger spacing for interior
            
            dx = distance * np.cos(angle)
            dy = distance * np.sin(angle)
            
            refined_points.append(interior_center + np.array([dx, dy]))
        
        refined_points = np.array(refined_points)
        
        # Re-triangulate with the refined points
        refined_delaunay = Delaunay(refined_points)
        refined_triangles = refined_delaunay.simplices
        
        # Calculate statistics for refined triangulation
        num_refined_triangles = len(refined_triangles)
        print(f"Refined triangulation: {num_refined_triangles} triangles")
        
        # Create visualization of refined triangulation - CLEAN VERSION (no feature points)
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot only triangles
        for simplex in refined_triangles:
            x = [refined_points[simplex[0], 0], refined_points[simplex[1], 0], 
                 refined_points[simplex[2], 0], refined_points[simplex[0], 0]]
            y = [refined_points[simplex[0], 1], refined_points[simplex[1], 1], 
                 refined_points[simplex[2], 1], refined_points[simplex[0], 1]]
            ax.plot(x, y, 'b-', linewidth=0.5)
        
        ax.set_title(f"MeshIt-like Refined ({num_refined_triangles} triangles)")
        ax.grid(False)
        ax.axis('equal')
        ax.axis('off')  # Turn off axes
        
        # Save the figure
        output_file = os.path.join(output_dir, "meshit_like_clean.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Clean MeshIt-like visualization saved to {output_file}")
        
        # Create a side-by-side comparison - CLEAN VERSION (no feature points)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot Delaunay on the left - only triangles
        for simplex in delaunay_triangles:
            x = [points[simplex[0], 0], points[simplex[1], 0], 
                 points[simplex[2], 0], points[simplex[0], 0]]
            y = [points[simplex[0], 1], points[simplex[1], 1], 
                 points[simplex[2], 1], points[simplex[0], 1]]
            ax1.plot(x, y, 'b-', linewidth=0.5)
        
        ax1.set_title(f"Standard Delaunay ({num_delaunay_triangles} triangles)")
        ax1.axis('equal')
        ax1.axis('off')  # Turn off axes
        
        # Plot refined on the right - only triangles
        for simplex in refined_triangles:
            x = [refined_points[simplex[0], 0], refined_points[simplex[1], 0], 
                 refined_points[simplex[2], 0], refined_points[simplex[0], 0]]
            y = [refined_points[simplex[0], 1], refined_points[simplex[1], 1], 
                 refined_points[simplex[2], 1], refined_points[simplex[0], 1]]
            ax2.plot(x, y, 'b-', linewidth=0.5)
        
        ax2.set_title(f"MeshIt-like Refinement ({num_refined_triangles} triangles)")
        ax2.axis('equal')
        ax2.axis('off')  # Turn off axes
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, "clean_comparison.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Clean comparison visualization saved to {output_file}")
        
        print("Clean triangulation comparison completed!")
        
    except Exception as e:
        print(f"Unexpected error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 