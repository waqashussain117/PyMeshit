#!/usr/bin/env python3
"""
Test script for hull refinement functionality
"""

import sys
import os
import numpy as np

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from meshit.intersection_utils import Vector3D, refine_hull_with_interpolation
    print("Successfully imported hull refinement function")
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)

def test_hull_refinement():
    """Test the hull refinement with a simple synthetic surface"""
    print("\n=== Testing Hull Refinement Function ===")
    
    # Create a synthetic curved surface (a simple paraboloid z = x^2 + y^2)
    print("1. Creating synthetic surface data...")
    
    # Generate scattered data points
    np.random.seed(42)
    n_points = 50
    x_vals = np.random.uniform(-2, 2, n_points)
    y_vals = np.random.uniform(-2, 2, n_points)
    z_vals = x_vals**2 + y_vals**2 + np.random.normal(0, 0.1, n_points)  # Add some noise
    
    scattered_points = [Vector3D(x, y, z) for x, y, z in zip(x_vals, y_vals, z_vals)]
    print(f"   Created {len(scattered_points)} scattered data points")
    
    # Create a simple raw hull (just the boundary points, not on the true surface)
    hull_x = [-2, 2, 2, -2]  # Square boundary
    hull_y = [-2, -2, 2, 2]
    hull_z = [0, 0, 0, 0]   # Flat (incorrect) hull
    
    raw_hull = [Vector3D(x, y, z, point_type="DEFAULT") for x, y, z in zip(hull_x, hull_y, hull_z)]
    print(f"   Created raw hull with {len(raw_hull)} points (all at z=0)")
    
    # Test different interpolation methods
    test_configs = [
        {'interp': 'Thin Plate Spline (TPS)', 'smoothing': 0.0},
        {'interp': 'Linear (Barycentric)', 'smoothing': 0.0},
        {'interp': 'IDW (p=4)', 'smoothing': 0.0},
        {'interp': 'Cubic (Clough–Tocher)', 'smoothing': 0.0}
    ]
    
    print("\n2. Testing different interpolation methods...")
    
    for i, config in enumerate(test_configs):
        print(f"\n   Test {i+1}: {config['interp']}")
        
        try:
            refined_hull = refine_hull_with_interpolation(raw_hull, scattered_points, config)
            
            if refined_hull and len(refined_hull) == len(raw_hull):
                # Calculate improvement (expected z-values should be close to x^2 + y^2)
                improvements = []
                for orig, refined in zip(raw_hull, refined_hull):
                    expected_z = orig.x**2 + orig.y**2
                    orig_error = abs(orig.z - expected_z)
                    refined_error = abs(refined.z - expected_z)
                    improvement = orig_error - refined_error
                    improvements.append(improvement)
                
                avg_improvement = np.mean(improvements)
                print(f"      ✓ Success - Average Z improvement: {avg_improvement:.4f}")
                print(f"      ✓ Hull point types preserved: {all(hasattr(p, 'point_type') for p in refined_hull)}")
                
                # Show specific improvements
                for j, (orig, refined, imp) in enumerate(zip(raw_hull, refined_hull, improvements)):
                    expected = orig.x**2 + orig.y**2
                    print(f"         Point {j}: ({orig.x:.1f},{orig.y:.1f}) z: {orig.z:.3f}→{refined.z:.3f} (expected: {expected:.3f}, improvement: {imp:.3f})")
                    
            else:
                print(f"      ✗ Failed - Returned {len(refined_hull) if refined_hull else 0} points instead of {len(raw_hull)}")
                
        except Exception as e:
            print(f"      ✗ Failed with error: {e}")
    
    print("\n3. Testing edge cases...")
    
    # Test with insufficient data
    print("   Testing with insufficient scattered data...")
    try:
        minimal_points = scattered_points[:2]  # Only 2 points
        refined_hull = refine_hull_with_interpolation(raw_hull, minimal_points, test_configs[0])
        if refined_hull == raw_hull:
            print("      ✓ Correctly returned original hull for insufficient data")
        else:
            print("      ✗ Should have returned original hull for insufficient data")
    except Exception as e:
        print(f"      ✗ Failed with error: {e}")
    
    # Test with empty inputs
    print("   Testing with empty inputs...")
    try:
        empty_result = refine_hull_with_interpolation([], scattered_points, test_configs[0])
        if empty_result == []:
            print("      ✓ Correctly handled empty hull input")
        else:
            print("      ✗ Should have returned empty list for empty hull")
    except Exception as e:
        print(f"      ✗ Failed with error: {e}")
    
    print("\n=== Hull Refinement Test Complete ===")

if __name__ == "__main__":
    test_hull_refinement()
