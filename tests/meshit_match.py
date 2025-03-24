#!/usr/bin/env python
"""
Create a triangulation that closely matches the MeshIt pattern shown in the screenshot.
This script focuses only on the triangulation pattern without showing any points.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import traceback

def main():
    try:
        print("Creating MeshIt-matching triangulation...")
        
        # Create output directory
        output_dir = "meshit_match"
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
        
        # Create a more accurate MeshIt-matching triangulation pattern
        # Based on careful analysis of the screenshot
        print("Creating MeshIt-matching pattern...")
        
        # Create a denser set of points to capture MeshIt's refinement pattern
        refined_points = list(points)
        
        # 1. Add multiple rings of points around hull vertices with varying density
        for hp in hull_points:
            # First ring (closest to feature point) - very dense
            num_points_inner = 24
            for i in range(num_points_inner):
                angle = 2.0 * np.pi * i / num_points_inner
                distance = 0.165 * 0.6  # Very close to feature
                dx = distance * np.cos(angle)
                dy = distance * np.sin(angle)
                refined_points.append(hp + np.array([dx, dy]))
            
            # Second ring - still dense
            num_points_mid = 16
            for i in range(num_points_mid):
                angle = 2.0 * np.pi * i / num_points_mid
                distance = 0.165 * 1.2
                dx = distance * np.cos(angle)
                dy = distance * np.sin(angle)
                refined_points.append(hp + np.array([dx, dy]))
                
            # Third ring - transitional
            num_points_outer = 12
            for i in range(num_points_outer):
                angle = 2.0 * np.pi * i / num_points_outer
                distance = 0.165 * 2.2
                dx = distance * np.cos(angle)
                dy = distance * np.sin(angle)
                refined_points.append(hp + np.array([dx, dy]))
        
        # 2. Add points along the hull edges to ensure edge refinement
        for i in range(len(hull.vertices)):
            j = (i + 1) % len(hull.vertices)
            start = hull_points[i]
            end = hull_points[j]
            
            # Calculate direction and length
            edge = end - start
            edge_length = np.linalg.norm(edge)
            num_edge_points = int(edge_length / 0.15) + 1
            
            # Add points along edge
            for k in range(1, num_edge_points):
                t = k / (num_edge_points + 1)
                point = start + t * edge
                refined_points.append(point)
                
                # Also add points slightly inside from the edge
                inward = np.array([-edge[1], edge[0]])  # Perpendicular to edge
                inward = inward / np.linalg.norm(inward) * 0.1  # Normalize and scale
                refined_points.append(point + inward)
        
        # 3. Add some interior points with larger spacing
        interior_center = np.mean(points, axis=0)
        interior_radius = 0.7  # Covers most of the interior
        num_interior_rings = 3
        
        for ring in range(num_interior_rings):
            radius = interior_radius * (ring + 1) / num_interior_rings
            num_points = 8 + ring * 4  # More points in outer rings
            
            for i in range(num_points):
                angle = 2.0 * np.pi * i / num_points
                dx = radius * np.cos(angle)
                dy = radius * np.sin(angle)
                refined_points.append(interior_center + np.array([dx, dy]))
        
        # 4. Add some random points in the interior for natural look
        num_random = 25
        for _ in range(num_random):
            # Random angle and radius (biased toward the interior)
            angle = 2.0 * np.pi * np.random.random()
            radius = interior_radius * np.random.random() * 0.7
            
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            refined_points.append(interior_center + np.array([dx, dy]))
        
        refined_points = np.array(refined_points)
        
        # Re-triangulate with the refined points
        print(f"Triangulating with {len(refined_points)} points...")
        refined_delaunay = Delaunay(refined_points)
        refined_triangles = refined_delaunay.simplices
        
        # Calculate statistics for refined triangulation
        num_refined_triangles = len(refined_triangles)
        print(f"Generated triangulation: {num_refined_triangles} triangles")
        
        # Create visualization that matches MeshIt screenshot - CLEAN VERSION (no points)
        print("Creating visualization...")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot only triangles - with same blue color as screenshot
        for simplex in refined_triangles:
            x = [refined_points[simplex[0], 0], refined_points[simplex[1], 0], 
                 refined_points[simplex[2], 0], refined_points[simplex[0], 0]]
            y = [refined_points[simplex[0], 1], refined_points[simplex[1], 1], 
                 refined_points[simplex[2], 1], refined_points[simplex[0], 1]]
            ax.plot(x, y, color='#80b0ff', linewidth=0.6)  # Light blue like screenshot
        
        # Set dark blue background like screenshot
        ax.set_facecolor('#2c3a5e')  # Dark blue background
        fig.patch.set_facecolor('#2c3a5e')
        
        # Clean visualization
        ax.grid(False)
        ax.axis('equal')
        ax.axis('off')  # Turn off axes completely
        
        # Save the figure
        output_file = os.path.join(output_dir, "meshit_match.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0, 
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"MeshIt-matching visualization saved to {output_file}")
        
        # Also save a white background version for easier viewing
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot only triangles - regular blue color
        for simplex in refined_triangles:
            x = [refined_points[simplex[0], 0], refined_points[simplex[1], 0], 
                 refined_points[simplex[2], 0], refined_points[simplex[0], 0]]
            y = [refined_points[simplex[0], 1], refined_points[simplex[1], 1], 
                 refined_points[simplex[2], 1], refined_points[simplex[0], 1]]
            ax.plot(x, y, 'b-', linewidth=0.6)
        
        # Clean visualization
        ax.grid(False)
        ax.axis('equal')
        ax.axis('off')  # Turn off axes completely
        
        # Save the figure
        output_file = os.path.join(output_dir, "meshit_match_light.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Light-background version saved to {output_file}")
        
        print("MeshIt-matching triangulation completed!")
        
    except Exception as e:
        print(f"Unexpected error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 