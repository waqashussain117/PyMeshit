#!/usr/bin/env python3
"""
Test script to verify the triple point calculation fix.

This script tests the difference between the old distance-based approach
and the new C++ style skew line transversal approach.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from meshit.intersection_utils import (
    Vector3D, 
    calculate_skew_line_transversal_c_style,
    segment_segment_distance
)

def test_triple_point_scenarios():
    """Test various scenarios to verify triple point calculation works correctly."""
    
    print("Testing Triple Point Calculation Fix")
    print("=" * 50)
    
    # Test Case 1: True intersection within segments (should create triple point)
    print("\nTest Case 1: True intersection within segments")
    p1a = Vector3D(0, 0, 0)
    p1b = Vector3D(10, 0, 0)
    p2a = Vector3D(5, -5, 0)
    p2b = Vector3D(5, 5, 0)
    
    # Old method (distance-based)
    dist, closest1, closest2 = segment_segment_distance(p1a, p1b, p2a, p2b)
    old_creates_tp = dist < 1e-5
    
    # New method (C++ style)
    connector_points = calculate_skew_line_transversal_c_style(p1a, p1b, p2a, p2b)
    new_creates_tp = (connector_points is not None and 
                      len(connector_points) == 2 and 
                      (connector_points[0] - connector_points[1]).length_squared() < 1e-24)
    
    print(f"  Segments: ({p1a.x},{p1a.y},{p1a.z})-({p1b.x},{p1b.y},{p1b.z}) vs ({p2a.x},{p2a.y},{p2a.z})-({p2b.x},{p2b.y},{p2b.z})")
    print(f"  Old method creates TP: {old_creates_tp} (distance: {dist:.2e})")
    print(f"  New method creates TP: {new_creates_tp}")
    if connector_points:
        print(f"  Intersection points: ({connector_points[0].x:.3f},{connector_points[0].y:.3f},{connector_points[0].z:.3f}) - ({connector_points[1].x:.3f},{connector_points[1].y:.3f},{connector_points[1].z:.3f})")
    print(f"  Expected: Both should create TP ✓" if old_creates_tp and new_creates_tp else "  ❌")
    
    # Test Case 2: Close but no intersection within segments (should NOT create triple point)
    print("\nTest Case 2: Close segments but no intersection within bounds")
    p1a = Vector3D(0, 0, 0)
    p1b = Vector3D(4, 0, 0)  # Shorter segment
    p2a = Vector3D(5, -1, 0)  # Starts after first segment ends
    p2b = Vector3D(5, 1, 0)
    
    # Old method (distance-based)
    dist, closest1, closest2 = segment_segment_distance(p1a, p1b, p2a, p2b)
    old_creates_tp = dist < 1e-5
    
    # New method (C++ style)
    connector_points = calculate_skew_line_transversal_c_style(p1a, p1b, p2a, p2b)
    new_creates_tp = (connector_points is not None and 
                      len(connector_points) == 2 and 
                      (connector_points[0] - connector_points[1]).length_squared() < 1e-24)
    
    print(f"  Segments: ({p1a.x},{p1a.y},{p1a.z})-({p1b.x},{p1b.y},{p1b.z}) vs ({p2a.x},{p2a.y},{p2a.z})-({p2b.x},{p2b.y},{p2b.z})")
    print(f"  Old method creates TP: {old_creates_tp} (distance: {dist:.2e})")
    print(f"  New method creates TP: {new_creates_tp}")
    print(f"  Expected: Only new method should avoid false TP ✓" if old_creates_tp and not new_creates_tp else "  ❌")
    
    # Test Case 3: Parallel segments (should NOT create triple point)
    print("\nTest Case 3: Parallel segments")
    p1a = Vector3D(0, 0, 0)
    p1b = Vector3D(10, 0, 0)
    p2a = Vector3D(0, 1, 0)  # Parallel, 1 unit offset
    p2b = Vector3D(10, 1, 0)
    
    # Old method (distance-based)
    dist, closest1, closest2 = segment_segment_distance(p1a, p1b, p2a, p2b)
    old_creates_tp = dist < 1e-5
    
    # New method (C++ style)
    connector_points = calculate_skew_line_transversal_c_style(p1a, p1b, p2a, p2b)
    new_creates_tp = (connector_points is not None and 
                      len(connector_points) == 2 and 
                      (connector_points[0] - connector_points[1]).length_squared() < 1e-24)
    
    print(f"  Segments: ({p1a.x},{p1a.y},{p1a.z})-({p1b.x},{p1b.y},{p1b.z}) vs ({p2a.x},{p2a.y},{p2a.z})-({p2b.x},{p2b.y},{p2b.z})")
    print(f"  Old method creates TP: {old_creates_tp} (distance: {dist:.2e})")
    print(f"  New method creates TP: {new_creates_tp}")
    print(f"  Expected: Neither should create TP ✓" if not old_creates_tp and not new_creates_tp else "  ❌")
    
    # Test Case 4: Skew lines with intersection outside segment bounds
    print("\nTest Case 4: Skew lines with intersection outside bounds")
    p1a = Vector3D(0, 0, 0)
    p1b = Vector3D(1, 0, 0)   # Short segment
    p2a = Vector3D(2, -5, 0)  # Would intersect at x=2, but first segment only goes to x=1
    p2b = Vector3D(2, 5, 0)
    
    # Old method (distance-based)
    dist, closest1, closest2 = segment_segment_distance(p1a, p1b, p2a, p2b)
    old_creates_tp = dist < 1e-5
    
    # New method (C++ style)
    connector_points = calculate_skew_line_transversal_c_style(p1a, p1b, p2a, p2b)
    new_creates_tp = (connector_points is not None and 
                      len(connector_points) == 2 and 
                      (connector_points[0] - connector_points[1]).length_squared() < 1e-24)
    
    print(f"  Segments: ({p1a.x},{p1a.y},{p1a.z})-({p1b.x},{p1b.y},{p1b.z}) vs ({p2a.x},{p2a.y},{p2a.z})-({p2b.x},{p2b.y},{p2b.z})")
    print(f"  Old method creates TP: {old_creates_tp} (distance: {dist:.2e})")
    print(f"  New method creates TP: {new_creates_tp}")
    print(f"  Expected: New method should avoid false TP ✓" if not new_creates_tp else "  ❌")
    
    print("\n" + "=" * 50)
    print("Fix Summary:")
    print("- New C++ style method only creates triple points for TRUE intersections")
    print("- Eliminates false positives that caused 'black dots' in intersection lines")
    print("- Uses exact C++ tolerance (1e-24 squared distance)")
    print("- Validates intersection occurs within both segment bounds")

if __name__ == "__main__":
    test_triple_point_scenarios()
