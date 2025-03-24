#!/usr/bin/env python
"""
Compare standard Delaunay triangulation with MeshIt's triangulation pattern.
This script visualizes the key differences to understand how to better mimic MeshIt.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import traceback

def calculate_mean_edge_length(vertices, triangles):
    """Calculate the mean edge length of all triangles"""
    edge_lengths = []
    for tri in triangles:
        # Get the three vertices of the triangle
        v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        
        # Calculate the three edge lengths
        edge1 = np.linalg.norm(v1 - v2)
        edge2 = np.linalg.norm(v2 - v3)
        edge3 = np.linalg.norm(v3 - v1)
        
        edge_lengths.extend([edge1, edge2, edge3])
    
    return np.mean(edge_lengths)

def visualize_edge_length_distribution(vertices, triangles, title, save_path):
    """Visualize the distribution of edge lengths in a triangulation"""
    edge_lengths = []
    for tri in triangles:
        # Get the three vertices of the triangle
        v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        
        # Calculate the three edge lengths
        edge1 = np.linalg.norm(v1 - v2)
        edge2 = np.linalg.norm(v2 - v3)
        edge3 = np.linalg.norm(v3 - v1)
        
        edge_lengths.extend([edge1, edge2, edge3])
    
    plt.figure(figsize=(8, 6))
    plt.hist(edge_lengths, bins=30)
    plt.title(f"{title} - Edge Length Distribution")
    plt.xlabel("Edge Length")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.close()

def main():
    try:
        print("Starting triangulation comparison...")
        
        # Create output directory
        output_dir = "triangulation_comparison"
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
        mean_delaunay_edge = calculate_mean_edge_length(points, delaunay_triangles)
        print(f"Delaunay triangulation: {num_delaunay_triangles} triangles")
        print(f"Mean edge length (Delaunay): {mean_delaunay_edge:.4f}")
        
        # Visualize Delaunay edge length distribution
        visualize_edge_length_distribution(
            points, delaunay_triangles, 
            "Delaunay", 
            os.path.join(output_dir, "delaunay_edge_distribution.png")
        )
        
        # Create visualization of standard Delaunay
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot triangles
        for simplex in delaunay_triangles:
            x = [points[simplex[0], 0], points[simplex[1], 0], 
                 points[simplex[2], 0], points[simplex[0], 0]]
            y = [points[simplex[0], 1], points[simplex[1], 1], 
                 points[simplex[2], 1], points[simplex[0], 1]]
            ax.plot(x, y, 'b-', linewidth=0.5)
        
        # Plot hull points (features)
        ax.scatter(hull_points[:, 0], hull_points[:, 1], c='red', s=60, marker='*')
        
        # Plot other points
        for i, p in enumerate(points):
            if any(np.array_equal(p, hp) for hp in hull_points):
                continue  # Skip hull points already plotted
            ax.scatter(p[0], p[1], c='blue', s=30)
        
        ax.set_title(f"Standard Delaunay Triangulation ({num_delaunay_triangles} triangles)")
        ax.grid(False)
        ax.axis('equal')
        
        # Save the figure
        output_file = os.path.join(output_dir, "delaunay.png")
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Delaunay visualization saved to {output_file}")
        
        # Create visualization of MeshIt triangulation (from your image)
        # Since we don't have direct access to MeshIt's triangulation,
        # we'll create a visual representation based on key observations:
        
        # 1. MeshIt creates more triangles with smaller sizes near hull points
        # 2. It maintains a gradient of sizes from boundaries to interior
        
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
        mean_refined_edge = calculate_mean_edge_length(refined_points, refined_triangles)
        print(f"Refined triangulation: {num_refined_triangles} triangles")
        print(f"Mean edge length (Refined): {mean_refined_edge:.4f}")
        
        # Visualize refined edge length distribution
        visualize_edge_length_distribution(
            refined_points, refined_triangles, 
            "Refined (MeshIt-like)", 
            os.path.join(output_dir, "refined_edge_distribution.png")
        )
        
        # Create visualization of refined triangulation
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot triangles
        for simplex in refined_triangles:
            x = [refined_points[simplex[0], 0], refined_points[simplex[1], 0], 
                 refined_points[simplex[2], 0], refined_points[simplex[0], 0]]
            y = [refined_points[simplex[0], 1], refined_points[simplex[1], 1], 
                 refined_points[simplex[2], 1], refined_points[simplex[0], 1]]
            ax.plot(x, y, 'b-', linewidth=0.5)
        
        # Plot hull points (features)
        ax.scatter(hull_points[:, 0], hull_points[:, 1], c='red', s=60, marker='*')
        
        # Plot original points
        ax.scatter(points[:, 0], points[:, 1], c='blue', s=20)
        
        ax.set_title(f"MeshIt-like Refined Triangulation ({num_refined_triangles} triangles)")
        ax.grid(False)
        ax.axis('equal')
        
        # Save the figure
        output_file = os.path.join(output_dir, "meshit_like.png")
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"MeshIt-like visualization saved to {output_file}")
        
        # Create a side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot Delaunay on the left
        for simplex in delaunay_triangles:
            x = [points[simplex[0], 0], points[simplex[1], 0], 
                 points[simplex[2], 0], points[simplex[0], 0]]
            y = [points[simplex[0], 1], points[simplex[1], 1], 
                 points[simplex[2], 1], points[simplex[0], 1]]
            ax1.plot(x, y, 'b-', linewidth=0.5)
        
        # Plot hull points
        ax1.scatter(hull_points[:, 0], hull_points[:, 1], c='red', s=60)
        
        # Plot other points
        for i, p in enumerate(points):
            if any(np.array_equal(p, hp) for hp in hull_points):
                continue  # Skip hull points already plotted
            ax1.scatter(p[0], p[1], c='blue', s=30)
        
        ax1.set_title(f"Standard Delaunay ({num_delaunay_triangles} triangles)")
        ax1.axis('equal')
        
        # Plot refined on the right
        for simplex in refined_triangles:
            x = [refined_points[simplex[0], 0], refined_points[simplex[1], 0], 
                 refined_points[simplex[2], 0], refined_points[simplex[0], 0]]
            y = [refined_points[simplex[0], 1], refined_points[simplex[1], 1], 
                 refined_points[simplex[2], 1], refined_points[simplex[0], 1]]
            ax2.plot(x, y, 'b-', linewidth=0.5)
        
        # Plot hull points
        ax2.scatter(hull_points[:, 0], hull_points[:, 1], c='red', s=60)
        
        # Plot original points
        for i, p in enumerate(points):
            if any(np.array_equal(p, hp) for hp in hull_points):
                continue  # Skip hull points already plotted
            ax2.scatter(p[0], p[1], c='blue', s=30)
        
        ax2.set_title(f"MeshIt-like Refinement ({num_refined_triangles} triangles)")
        ax2.axis('equal')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, "comparison.png")
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Comparison visualization saved to {output_file}")
        
        print("Triangulation comparison completed!")
        
    except Exception as e:
        print(f"Unexpected error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 