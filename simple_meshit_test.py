#!/usr/bin/env python
"""
Simple test script to verify MeshIt installation and functionality.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

print("=== MeshIt Installation Test ===")

# Try to import MeshIt
try:
    import meshit
    print(f"✓ Successfully imported meshit module (version: {meshit.__version__})")
    
    # Try to import core components
    from meshit.core import Surface, Vector3D, GradientControl
    print("✓ Successfully imported core components: Surface, Vector3D, GradientControl")
    
    # Print available attributes in meshit module
    print("\nAvailable attributes in meshit module:")
    for attr in dir(meshit):
        if not attr.startswith('__'):
            print(f"- {attr}")
    
    # Try to import extensions
    try:
        from meshit import extensions
        print("\n✓ Successfully imported meshit.extensions")
    except ImportError as e:
        print(f"\n✗ Error importing extensions: {e}")
    
    # Create a simple surface with triangulation
    print("\n=== Testing Surface Creation and Triangulation ===")
    
    # Create a simple grid of points
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.ravel(), Y.ravel(), np.zeros(len(X.ravel()))))
    
    # Create a surface from these points
    surface = Surface()
    for point in points:
        v = Vector3D(float(point[0]), float(point[1]), float(point[2]))
        surface.add_vertex(v)
    
    print(f"Created surface with {len(surface.vertices)} vertices")
    
    # Calculate convex hull
    try:
        surface.enhanced_calculate_convex_hull()
        print(f"✓ Calculated convex hull with {len(surface.convex_hull)} points")
    except AttributeError:
        print("✗ enhanced_calculate_convex_hull not available, trying regular convex hull")
        try:
            surface.calculate_convex_hull()
            print(f"✓ Calculated convex hull with {len(surface.convex_hull)} points")
        except Exception as e:
            print(f"✗ Failed to calculate convex hull: {e}")
    
    # Try triangulation
    try:
        # Try with gradient control
        gc = GradientControl.get_instance()
        print("✓ Got GradientControl instance")
        
        # Try triangulation
        gradient = 1.0
        print(f"Triangulating with gradient={gradient}")
        
        try:
            triangles = extensions.triangulate_with_triangle(surface, gradient=gradient)
            if isinstance(triangles, tuple) and len(triangles) == 2:
                print(f"✓ Triangulation successful: {len(triangles[1])} triangles")
                
                # Visualize
                vertices, tris = triangles
                plt.figure(figsize=(8, 8))
                plt.triplot(vertices[:, 0], vertices[:, 1], tris, 'b-')
                plt.plot(vertices[:, 0], vertices[:, 1], 'ro', markersize=3)
                plt.title(f"Triangulation with gradient={gradient}")
                plt.grid(True)
                plt.axis('equal')
                plt.savefig("test_triangulation.png")
                print("✓ Saved triangulation visualization to test_triangulation.png")
            else:
                print(f"✓ Triangulation successful: {len(triangles)} triangles")
        except Exception as e:
            print(f"✗ Triangulation failed: {e}")
            
    except Exception as e:
        print(f"✗ Failed to use gradient control: {e}")
    
except ImportError as e:
    print(f"✗ Error importing meshit: {e}")
    print("\nChecking Python paths:")
    for p in sys.path:
        print(f"- {p}")

print("\n=== Test Complete ===") 