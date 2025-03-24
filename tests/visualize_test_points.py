#!/usr/bin/env python
"""
Visualize the test points used in our triangulation examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

def main():
    # Load the points from the text file
    points = []
    with open("vtu_output/points.txt", "r") as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(": ")[1].split()
            points.append([float(parts[0]), float(parts[1])])
    
    points = np.array(points)
    
    # Compute the convex hull
    hull = ConvexHull(points)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Points with indices
    ax1.scatter(points[:, 0], points[:, 1], c='blue', s=50)
    
    # Label each point with its index
    for i, (x, y) in enumerate(points):
        ax1.annotate(f"{i+1}", (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Draw the convex hull
    for simplex in hull.simplices:
        ax1.plot(points[simplex, 0], points[simplex, 1], 'r-')
    
    ax1.set_title("Points with Indices and Convex Hull")
    ax1.grid(True)
    ax1.axis('equal')
    
    # Plot 2: Delaunay triangulation
    tri = Delaunay(points)
    ax2.triplot(points[:, 0], points[:, 1], tri.simplices, 'b-')
    ax2.scatter(points[:, 0], points[:, 1], c='red', s=50)
    
    # Count number of triangles
    num_triangles = len(tri.simplices)
    ax2.set_title(f"Delaunay Triangulation ({num_triangles} triangles)")
    ax2.grid(True)
    ax2.axis('equal')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("vtu_output/points_visualization.png", dpi=150)
    plt.close()
    
    print("Visualization saved to vtu_output/points_visualization.png")

if __name__ == "__main__":
    main() 