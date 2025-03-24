#!/usr/bin/env python
"""
Test script for the triangle mesh implementation with hull segments and feature point refinement.

This script demonstrates how to use the TriangleWrapper class to triangulate a set of points
with proper gradient-based refinement around feature points.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path if running the script directly
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from meshit.triangle_mesh import TriangleWrapper, create_hull_segments

def generate_test_points(num_points=100, seed=42):
    """Generate random test points for triangulation."""
    np.random.seed(seed)
    points = np.random.rand(num_points, 2) * 2 - 1  # Range: -1 to 1
    return points

def generate_feature_points(num_features=5, seed=43):
    """Generate feature points for refinement."""
    np.random.seed(seed)
    features = np.random.rand(num_features, 2) * 1.6 - 0.8  # Range: -0.8 to 0.8
    # Features have different sizes
    sizes = np.linspace(0.05, 0.2, num_features)
    return features, sizes

def visualize_triangulation(points, triangles, feature_points=None, feature_sizes=None, 
                            original_hull=None, title="Triangulation"):
    """Visualize the triangulation with optional feature points."""
    plt.figure(figsize=(10, 8))
    
    # Plot triangulation
    plt.triplot(points[:, 0], points[:, 1], triangles, 'k-', alpha=0.4, linewidth=0.5)
    
    # Plot original points
    plt.plot(points[:, 0], points[:, 1], 'ko', markersize=3)
    
    # Plot convex hull if provided
    if original_hull is not None:
        hull_points = points[original_hull]
        # Close the loop
        hull_points = np.vstack([hull_points, hull_points[0]])
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', linewidth=2, 
                label='Convex Hull')
    
    # Plot feature points if provided
    if feature_points is not None:
        for i, (point, size) in enumerate(zip(feature_points, feature_sizes)):
            plt.plot(point[0], point[1], 'r*', markersize=10)
            # Draw influence circle
            circle = plt.Circle((point[0], point[1]), size, fill=False, 
                               color='r', linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(alpha=0.3)
    if original_hull is not None:
        plt.legend()
    return plt

def test_basic_triangulation():
    """Test basic triangulation with convex hull segments."""
    print("Testing basic triangulation with convex hull segments...")
    
    # Generate random points
    points = generate_test_points(100)
    
    # Create convex hull segments
    hull_indices, segments = create_hull_segments(points)
    
    # Create triangle wrapper and triangulate
    wrapper = TriangleWrapper(gradient=1.0)
    result = wrapper.triangulate(points, segments)
    
    # Visualize
    plt = visualize_triangulation(result['vertices'], result['triangles'], 
                                 original_hull=hull_indices,
                                 title="Basic Triangulation with Convex Hull")
    plt.savefig("test_basic_triangulation.png")
    plt.close()
    
    print(f"Triangulation complete with {len(result['triangles'])} triangles")
    print(f"Output saved to test_basic_triangulation.png")
    return result

def test_gradient_triangulation():
    """Test triangulation with different gradient values."""
    print("\nTesting triangulation with different gradient values...")
    
    # Generate random points
    points = generate_test_points(100)
    
    # Create convex hull segments
    hull_indices, segments = create_hull_segments(points)
    
    # Create feature points
    feature_points, feature_sizes = generate_feature_points(5)
    
    # Test different gradient values
    gradient_values = [0.5, 1.0, 2.0, 4.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, gradient in enumerate(gradient_values):
        # Create triangle wrapper with this gradient
        wrapper = TriangleWrapper(gradient=gradient)
        wrapper.set_feature_points(feature_points, feature_sizes)
        
        # Triangulate
        result = wrapper.triangulate(points, segments)
        
        # Visualize on the corresponding subplot
        ax = axes[i]
        ax.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
                  result['triangles'], 'k-', alpha=0.4, linewidth=0.5)
        ax.plot(points[:, 0], points[:, 1], 'ko', markersize=3)
        
        # Plot hull
        hull_points = points[hull_indices]
        hull_points = np.vstack([hull_points, hull_points[0]])
        ax.plot(hull_points[:, 0], hull_points[:, 1], 'r-', linewidth=2)
        
        # Plot feature points
        for point, size in zip(feature_points, feature_sizes):
            ax.plot(point[0], point[1], 'r*', markersize=10)
            circle = plt.Circle((point[0], point[1]), size, fill=False, 
                               color='r', linestyle='--', alpha=0.5)
            ax.add_patch(circle)
        
        ax.set_title(f"Gradient = {gradient}, {len(result['triangles'])} triangles")
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("test_gradient_triangulation.png")
    plt.close()
    
    print(f"Gradient tests complete, output saved to test_gradient_triangulation.png")

def test_feature_point_refinement():
    """Test triangulation with feature point refinement."""
    print("\nTesting feature point refinement...")
    
    # Generate random points
    points = generate_test_points(80)
    
    # Create convex hull segments
    hull_indices, segments = create_hull_segments(points)
    
    # Create feature points
    feature_points, feature_sizes = generate_feature_points(5)
    
    # First triangulate without feature points
    wrapper = TriangleWrapper(gradient=2.0)
    result_no_features = wrapper.triangulate(points, segments)
    
    # Then triangulate with feature points
    wrapper = TriangleWrapper(gradient=2.0)
    wrapper.set_feature_points(feature_points, feature_sizes)
    result_with_features = wrapper.triangulate(points, segments)
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot without features
    ax1.triplot(result_no_features['vertices'][:, 0], result_no_features['vertices'][:, 1], 
               result_no_features['triangles'], 'k-', alpha=0.4, linewidth=0.5)
    ax1.plot(points[:, 0], points[:, 1], 'ko', markersize=3)
    ax1.set_title(f"Without Feature Refinement ({len(result_no_features['triangles'])} triangles)")
    ax1.set_aspect('equal')
    ax1.grid(alpha=0.3)
    
    # Plot with features
    ax2.triplot(result_with_features['vertices'][:, 0], result_with_features['vertices'][:, 1], 
               result_with_features['triangles'], 'k-', alpha=0.4, linewidth=0.5)
    ax2.plot(points[:, 0], points[:, 1], 'ko', markersize=3)
    
    # Plot feature points
    for point, size in zip(feature_points, feature_sizes):
        ax2.plot(point[0], point[1], 'r*', markersize=10)
        circle = plt.Circle((point[0], point[1]), size, fill=False, 
                           color='r', linestyle='--', alpha=0.5)
        ax2.add_patch(circle)
    
    ax2.set_title(f"With Feature Refinement ({len(result_with_features['triangles'])} triangles)")
    ax2.set_aspect('equal')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("test_feature_refinement.png")
    plt.close()
    
    print(f"Feature refinement test complete")
    print(f"Without features: {len(result_no_features['triangles'])} triangles")
    print(f"With features: {len(result_with_features['triangles'])} triangles")
    print(f"Output saved to test_feature_refinement.png")

def run_all_tests():
    """Run all tests."""
    os.makedirs('output', exist_ok=True)
    os.chdir('output')
    
    test_basic_triangulation()
    test_gradient_triangulation()
    test_feature_point_refinement()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_all_tests() 