"""
Test script for the direct Triangle callback implementation.

This script tests the new direct Triangle callback implementation and compares
it with the old Python-based approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from meshit.extensions import create_surface_from_points, triangulate_with_triangle
import meshit
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def create_test_surface():
    """Create a test surface with a complex shape"""
    # Create a clover-shaped surface
    theta = np.linspace(0, 2*np.pi, 100)
    r = 2 + np.sin(3*theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)
    
    # Create 3D points
    points = np.column_stack((x, y, z))
    
    # Add some interior points for better triangulation
    inner_points = []
    for _ in range(20):
        angle = np.random.random() * 2 * np.pi
        radius = np.random.random() * 1.5
        inner_points.append([radius * np.cos(angle), radius * np.sin(angle), 0])
    
    # Combine all points
    all_points = np.vstack((points, inner_points))
    
    # Create surface
    surface = create_surface_from_points(all_points, "TestSurface", "Test")
    
    return surface

def add_feature_points(surface):
    """Add some feature points to the surface"""
    # Add feature points at specific locations
    feature_points = [
        [0, 0, 0],  # Center
        [1, 1, 0],  # Upper right
        [-1, -1, 0],  # Lower left
        [1, -1, 0],  # Lower right
        [-1, 1, 0]   # Upper left
    ]
    
    # Create feature point vector
    for point in feature_points:
        surface.add_feature_point(point[0], point[1], point[2], 0.1)
    
    return surface

def plot_triangulation(vertices, triangles, title):
    """Plot a triangulation"""
    plt.figure(figsize=(10, 10))
    plt.triplot(vertices[:, 0], vertices[:, 1], triangles)
    plt.scatter(vertices[:, 0], vertices[:, 1], color='red', s=10)
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    return plt

def test_triangulation():
    """Test the triangulation with direct callback"""
    print("Creating test surface...")
    surface = create_test_surface()
    
    # Check if we have direct Triangle callback support
    has_direct = False
    try:
        from meshit.triangle_direct import DirectTriangleWrapper
        has_direct = True
        print("DirectTriangleWrapper is available!")
    except ImportError:
        print("DirectTriangleWrapper is not available, using fallback")
    
    # Test with different gradient values
    gradient_values = [0.5, 1.0, 2.0]
    
    for gradient in gradient_values:
        print(f"\nTesting with gradient {gradient}...")
        
        # Triangulate with the extensions module
        vertices, triangles = triangulate_with_triangle(surface, gradient=gradient)
        
        # Plot the results
        plot_triangulation(vertices, triangles, f"Triangulation with gradient={gradient}")
        
        print(f"  Generated {len(triangles)} triangles")
        print(f"  Mesh has {len(vertices)} vertices")
        
    plt.show()

if __name__ == "__main__":
    test_triangulation() 