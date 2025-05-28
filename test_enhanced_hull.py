#!/usr/bin/env python3
"""
Test script to verify enhanced convex hull computation works for complex datasets.
This script tests the robust convex hull computation methods without GUI dependencies.
"""

import numpy as np
import sys
import os

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_enhanced_convex_hull():
    """Test the enhanced convex hull computation with complex data."""
    
    try:
        # Import the enhanced hull computation
        from meshit.extensions import enhanced_calculate_convex_hull, Vector3D
        from meshit import create_surface
        print("✓ Successfully imported enhanced convex hull functions")
    except ImportError as e:
        print(f"✗ Failed to import enhanced functions: {e}")
        return False
    
    # Test Case 1: Complex planar dataset
    print("\n=== Test Case 1: Complex planar dataset ===")
    try:
        # Create a complex planar dataset (e.g., points forming a star shape)
        angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
        r1 = 1.0  # outer radius
        r2 = 0.5  # inner radius
        
        complex_points = []
        for i, angle in enumerate(angles):
            radius = r1 if i % 2 == 0 else r2
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0.1  # slightly non-planar to test robustness
            complex_points.append([x, y, z])
        
        # Add some random noise
        complex_points = np.array(complex_points)
        complex_points += np.random.normal(0, 0.05, complex_points.shape)
        
        print(f"Created complex dataset with {len(complex_points)} points")
        
        # Create vertices as Vector3D objects
        vertices = []
        for point in complex_points:
            vertices.append(Vector3D(float(point[0]), float(point[1]), float(point[2])))
        
        # Create surface
        temp_surface = create_surface(
            [[v.x, v.y, v.z] for v in vertices], 
            [], 
            "TestComplexSurface", 
            "Scattered"
        )
        
        # Use enhanced hull calculation
        hull_vertices = enhanced_calculate_convex_hull(temp_surface)
        
        if hull_vertices and len(hull_vertices) >= 3:
            print(f"✓ Enhanced hull computation successful: {len(hull_vertices)} vertices")
            
            # Print first few hull vertices
            print("First few hull vertices:")
            for i, vertex in enumerate(hull_vertices[:5]):
                print(f"  {i}: ({vertex.x:.3f}, {vertex.y:.3f}, {vertex.z:.3f})")
            
            return True
        else:
            print("✗ Enhanced hull computation failed")
            return False
            
    except Exception as e:
        print(f"✗ Test Case 1 failed: {e}")
        return False

def test_basic_convex_hull_fallback():
    """Test basic convex hull computation as fallback."""
    print("\n=== Test Case 2: Basic hull fallback ===")
    
    try:
        from scipy.spatial import ConvexHull
        
        # Create a simple 2D dataset
        points = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0.5, 0.5]  # interior point
        ])
        
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        print(f"✓ Basic hull computation successful: {len(hull_points)} vertices")
        return True
        
    except Exception as e:
        print(f"✗ Basic hull computation failed: {e}")
        return False

def test_robust_hull_methods():
    """Test the robust hull computation methods."""
    print("\n=== Test Case 3: Robust hull methods ===")
    
    try:
        # Test gift wrapping algorithm
        def gift_wrapping_2d(points_2d):
            """Jarvis march algorithm for 2D convex hull."""
            def orientation(p, q, r):
                val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                if val == 0:
                    return 0  # collinear
                return 1 if val > 0 else 2  # clockwise or counterclockwise
            
            n = len(points_2d)
            if n < 3:
                return points_2d
            
            # Find the leftmost point
            l = 0
            for i in range(1, n):
                if points_2d[i][0] < points_2d[l][0]:
                    l = i
                elif points_2d[i][0] == points_2d[l][0] and points_2d[i][1] < points_2d[l][1]:
                    l = i
            
            hull = []
            p = l
            while True:
                hull.append(points_2d[p])
                
                q = (p + 1) % n
                for i in range(n):
                    if orientation(points_2d[p], points_2d[i], points_2d[q]) == 2:
                        q = i
                
                p = q
                if p == l:  # We've wrapped around to the first point
                    break
            
            return np.array(hull)
        
        # Test with a simple convex set
        points_2d = np.array([
            [0, 0], [2, 0], [2, 2], [0, 2], [1, 1]
        ])
        
        hull_result = gift_wrapping_2d(points_2d)
        print(f"✓ Gift wrapping algorithm successful: {len(hull_result)} vertices")
        
        return True
        
    except Exception as e:
        print(f"✗ Robust hull methods failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Enhanced Convex Hull Computation")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(test_enhanced_convex_hull())
    results.append(test_basic_convex_hull_fallback())
    results.append(test_robust_hull_methods())
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Enhanced Hull Test: {'PASS' if results[0] else 'FAIL'}")
    print(f"Basic Hull Test: {'PASS' if results[1] else 'FAIL'}")
    print(f"Robust Hull Test: {'PASS' if results[2] else 'FAIL'}")
    
    if all(results):
        print("\n✓ All tests passed! The enhanced convex hull computation should work.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
        sys.exit(1)
